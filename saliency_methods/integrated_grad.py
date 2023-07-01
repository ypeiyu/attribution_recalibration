import functools
import operator

import torch
from torch.autograd import grad
import torch.nn.functional as F
from utils import preprocess

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers. 
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]

            params   indices   output

            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i]*m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)


class IntegratedGradients(object):
    def __init__(self, model, k=10, exp_obj='logit'):
        self.model = model
        self.model.eval()
        self.k = k
        self.exp_obj = exp_obj

    def _get_samples_input(self, input_tensor, reference_tensor):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions. 
            reference_tensor: A tensor of shape (batch, k, ...) where ... 
                indicates dimensions, and k represents the number of background 
                reference samples to draw per input in the batch.
        Returns: 
            samples_input: A tensor of shape (batch, k, ...) with the 
                interpolated points between input and ref.
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)

        batch_size = reference_tensor.size()[0]
        k_ = self.k

        # Grab a [batch_size, k]-sized interpolation sample
        if k_ == 1:
            t_tensor = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size)]).to(DEFAULT_DEVICE)
        else:
            t_tensor = torch.cat([torch.linspace(0, 1, k_) for _ in range(batch_size)]).to(DEFAULT_DEVICE)

        shape = [batch_size, k_] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor
        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult

        # A fine Affine Combine
        samples_input = end_point_input + end_point_ref

        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd

    def _get_samples_inter_delta(self, input_tensor, reference_tensor):
        sd = input_tensor - reference_tensor
        return sd

    def _get_grads(self, samples_input, sparse_labels=None):
        samples_input.requires_grad = True

        grad_tensor = torch.zeros(samples_input.shape).float().to(DEFAULT_DEVICE)

        for i in range(self.k):
            particular_slice = samples_input[:, i]
            output = self.model(particular_slice)

            if self.exp_obj == 'logit':
                batch_output = output
            elif self.exp_obj == 'prob':
                batch_output = torch.log_softmax(output, 1)
            elif self.exp_obj == 'contrast':
                b_num, c_num = output.shape[0], output.shape[1]
                mask = torch.ones(b_num, c_num, dtype=torch.bool)
                mask[torch.arange(b_num), sparse_labels] = False
                neg_cls_output = output[mask].reshape(b_num, c_num - 1)
                neg_weight = F.softmax(neg_cls_output, dim=1)
                weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
                pos_cls_output = output[torch.arange(b_num), sparse_labels]
                output = pos_cls_output - weighted_neg_output
                batch_output = output.unsqueeze(1)

            # should check that users pass in sparse labels
            # Only look at the user-specified label
            if sparse_labels is not None and batch_output.size(1) > 1:
                sample_indices = torch.arange(0, batch_output.size(0)).to(DEFAULT_DEVICE)
                indices_tensor = torch.cat([
                    sample_indices.unsqueeze(1),
                    sparse_labels.unsqueeze(1)], dim=1)
                batch_output = gather_nd(batch_output, indices_tensor)

            self.model.zero_grad()
            model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(DEFAULT_DEVICE),
                    create_graph=True)
            grad_tensor[:, i, :] = model_grads[0].detach().data
        return grad_tensor

    def shap_values(self, input_tensor, sparse_labels=None):
        """
        Calculate expected gradients approximation of Shapley values for the 
        sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        shape = list(input_tensor.shape)
        shape.insert(1, self.k)

        reference_tensor = torch.zeros(*input_tensor.shape).cuda().to(DEFAULT_DEVICE)
        reference_tensor = preprocess(reference_tensor)
        reference_tensor = reference_tensor.repeat([self.k, 1, 1, 1])
        reference_tensor = reference_tensor.view(shape)

        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_inter_delta(samples_input, reference_tensor)
        grad_tensor = self._get_grads(samples_input, sparse_labels)

        mult_grads = samples_delta * grad_tensor
        attribution = mult_grads.sum(1)

        return attribution
