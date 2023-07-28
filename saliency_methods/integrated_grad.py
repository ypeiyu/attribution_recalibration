import torch
import torch.nn.functional as F
from utils import preprocess

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntegratedGradients(object):
    def __init__(self, model, k=10, exp_obj='logit', dataset_name='imagenet'):
        self.model = model
        self.model.eval()
        self.k = k
        self.exp_obj = exp_obj
        self.dataset_name = dataset_name

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

        grad_tensor = torch.zeros(samples_input.shape).float().to(DEFAULT_DEVICE)

        for i in range(self.k):
            particular_slice = samples_input[:, i]
            particular_slice.requires_grad = True

            output = self.model(particular_slice)

            batch_output = None
            if self.exp_obj == 'logit':
                batch_output = -1 * F.nll_loss(output, sparse_labels.flatten(), reduction='sum')
            elif self.exp_obj == 'prob':
                batch_output = -1 * F.nll_loss(F.log_softmax(output, dim=1), sparse_labels.flatten(), reduction='sum')
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

            self.model.zero_grad()
            batch_output.backward()
            gradients = particular_slice.grad.clone()
            particular_slice.grad.zero_()
            gradients.detach()

            grad_tensor[:, i, :] = gradients

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
        reference_tensor = preprocess(reference_tensor, self.dataset_name)
        reference_tensor = reference_tensor.repeat([self.k, 1, 1, 1])
        reference_tensor = reference_tensor.view(shape)

        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_inter_delta(samples_input, reference_tensor)
        grad_tensor = self._get_grads(samples_input, sparse_labels)

        mult_grads = samples_delta * grad_tensor
        attribution = mult_grads.sum(1)

        return attribution
