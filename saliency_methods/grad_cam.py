#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

"""
    Implement GradCAM

    Original Paper:
    Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks
    via gradient-based localization." ICCV 2017.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
import operator

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

class GradCAM():
    """
    Compute GradCAM
    """

    def __init__(self, model, exp_obj='prob'):
        self.model = model
        self.exp_obj = exp_obj

        self.features = None
        self.feat_grad = None
        prev_module = None
        self.target_module = None

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
            sparse_labels = out.data.max(1, keepdim=True)[1]

        output_scalar = out
        if self.exp_obj == 'prob':
            output_scalar = -1. * F.nll_loss(out, sparse_labels.flatten(), reduction='sum')
        elif self.exp_obj == 'logit':
            sample_indices = torch.arange(0, out.size(0)).cuda()
            indices_tensor = torch.cat([
                sample_indices.unsqueeze(1),
                sparse_labels.unsqueeze(1)], dim=1)
            output_scalar = gather_nd(out, indices_tensor)
            output_scalar = torch.sum(output_scalar)
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
        cam = (F.relu(features) * grads).sum(1, keepdim=True)
        cam_resized = F.interpolate(F.relu(cam), size=image.size(2), mode='bilinear', align_corners=True)
        return cam_resized