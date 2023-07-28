import torch
import torch.nn.functional as F

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SmoothGrad():
    """
    Compute smoothgrad
    """

    def __init__(self, model, bg_size=100, exp_obj='logit', std_spread=0.15):
        self.model = model
        self.num_samples = bg_size
        self.exp_obj = exp_obj
        self.std_spread = std_spread

    def _getGradients(self, image, sparse_labels=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        output = self.model(image)

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
        gradients = image.grad.clone()
        image.grad.zero_()
        gradients.detach()

        return gradients

    def shap_values(self, image, sparse_labels=None):
        # SmoothGrad saliency

        self.model.eval()

        # grad = self._getGradients(image, target_class=target_class)
        std_dev = self.std_spread * (image.max().item() - image.min().item())

        cam = torch.zeros_like(image).to(image.device)

        # add gaussian noise to image multiple times
        for i in range(self.num_samples):
            noise = torch.normal(mean=torch.zeros_like(image).to(image.device), std=std_dev)
            # cam += (self._getGradients(image + noise, target_class=sparse_labels)) / self.num_samples

            saliency = self._getGradients(image + noise, sparse_labels=sparse_labels) / self.num_samples
            cam += saliency
        return cam
