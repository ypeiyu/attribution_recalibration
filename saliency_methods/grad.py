import torch
import torch.nn.functional as F

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        if sparse_labels is None:
            sparse_labels = output.max(1, keepdim=False)[1]
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
            batch_output = output.sum()

        # should check that users pass in sparse labels
        # Only look at the user-specified label

        self.model.zero_grad()
        batch_output.backward()
        gradients = input_tensor.grad.clone()
        input_tensor.grad.zero_()
        gradients.detach()

        return gradients
