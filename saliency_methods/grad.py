import functools
import operator
import torch
from torch.autograd import grad
import torch.nn.functional as F

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


class Gradients(object):
    def __init__(self, model, exp_obj='logit'):
        self.model = model
        self.model.eval()
        self.exp_obj = exp_obj

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
        input_tensor.requires_grad = True

        output = self.model(input_tensor)

        if self.exp_obj == 'logit':
            batch_output = output
        elif self.exp_obj == 'prob':
            batch_output = torch.log_softmax(output, 1)
        elif self.exp_obj == 'contrast':
            neg_cls_indices = torch.arange(output.size(1))[
                ~torch.eq(torch.unsqueeze(output, dim=1), sparse_labels)]
            neg_cls_output = torch.index_select(output, dim=1, index=neg_cls_indices)
            neg_weight = F.softmax(neg_cls_output, dim=1)
            weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
            pos_cls_indices = torch.arange(output.size(1))[torch.eq(torch.unsqueeze(output, dim=1), sparse_labels)]
            neg_cls_output = torch.index_select(output, dim=1, index=pos_cls_indices)
            output = neg_cls_output - weighted_neg_output
            batch_output = output

        # should check that users pass in sparse labels
        # Only look at the user-specified label
        if sparse_labels is not None and batch_output.size(1) > 1 and self.exp_obj != 'contrast':
            sample_indices = torch.arange(0, batch_output.size(0)).to(DEFAULT_DEVICE)
            indices_tensor = torch.cat([
                sample_indices.unsqueeze(1),
                sparse_labels.unsqueeze(1)], dim=1)
            batch_output = gather_nd(batch_output, indices_tensor)

        self.model.zero_grad()
        grads = grad(
            outputs=batch_output,
            inputs=input_tensor,
            grad_outputs=torch.ones_like(batch_output).to(DEFAULT_DEVICE),
            create_graph=True)

        grads = grads[0]

        return grads
