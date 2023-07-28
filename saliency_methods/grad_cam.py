import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM():
    """
    Compute GradCAM
    """

    def __init__(self, model, exp_obj='prob', post_process=True):
        self.model = model
        self.exp_obj = exp_obj

        self.features = None
        self.feat_grad = None
        prev_module = None
        self.target_module = None
        self.post_process = post_process

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                prev_module = m
            elif isinstance(m, nn.Linear):
                self.target_module = prev_module
                break

        if self.target_module is not None:
            # Register feature-gradient and feature hooks for each layer
            handle_g = self.target_module.register_backward_hook(self._extract_layer_grads)
            handle_f = self.target_module.register_forward_hook(self._extract_layer_features)

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        self.feature_grads = out_grad[0]

    def _extract_layer_features(self, module, input, output):
        # function to collect the layer outputs
        self.features = output

    def getFeaturesAndGrads(self, x, sparse_labels):

        out = self.model(x)

        if sparse_labels is None:
            sparse_labels = out.data.max(1, keepdim=True)[1].squeeze(1)

        output_scalar = None
        if self.exp_obj == 'prob':
            output_scalar = -1. * F.nll_loss(F.log_softmax(out, dim=1), sparse_labels.flatten(), reduction='sum')
        elif self.exp_obj == 'logit':
            output_scalar = -1. * F.nll_loss(out, sparse_labels.flatten(), reduction='sum')
        elif self.exp_obj == 'contrast':
            b_num, c_num = out.shape[0], out.shape[1]
            mask = torch.ones(b_num, c_num, dtype=torch.bool)
            mask[torch.arange(b_num), sparse_labels] = False
            neg_cls_output = out[mask].reshape(b_num, c_num - 1)
            neg_weight = F.softmax(neg_cls_output, dim=1)
            weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
            pos_cls_output = out[torch.arange(b_num), sparse_labels]
            output = pos_cls_output - weighted_neg_output
            output_scalar = output

            output_scalar = torch.sum(output_scalar)

        # Compute gradients
        self.model.zero_grad()
        output_scalar.backward()

        return self.features, self.feature_grads

    def shap_values(self, image, sparse_labels=None):
        # Simple FullGrad saliency

        self.model.eval()
        features, intermed_grad = self.getFeaturesAndGrads(image, sparse_labels=sparse_labels)

        # GradCAM computation
        grads = intermed_grad.mean(dim=(2, 3), keepdim=True)

        if self.post_process:
            cam = (features * grads).sum(1, keepdim=True)
            cam_resized = F.interpolate(F.relu(cam), size=image.size(2), mode='bilinear', align_corners=True)
        else:
            cam = (features * grads).sum(1, keepdim=True)
            cam_resized = F.interpolate(cam, size=image.size(2), mode='bilinear', align_corners=True)

        return cam_resized
