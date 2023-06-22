#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement SmoothGrad saliency algorithm

    Original paper:
    Smilkov, Daniel, et al. "Smoothgrad: removing noise by adding noise." 
    arXiv preprint arXiv:1706.03825 (2017).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
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

    def _getGradients(self, image, target_class=None):
        """
        Compute input gradients for an image
        """

        image = image.requires_grad_()
        output = self.model(image)

        # ---------------------------------------------
        if self.exp_obj == 'logit':
            batch_output = output
        elif self.exp_obj == 'prob':
            batch_output = torch.log_softmax(output, 1)
        elif self.exp_obj == 'contrast':
            neg_cls_indices = torch.arange(output.size(1))[
                ~torch.eq(torch.unsqueeze(output, dim=1), target_class)]
            neg_cls_output = torch.index_select(output, dim=1, index=neg_cls_indices)
            neg_weight = F.softmax(neg_cls_output, dim=1)
            weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
            pos_cls_indices = torch.arange(output.size(1))[torch.eq(torch.unsqueeze(output, dim=1), target_class)]
            neg_cls_output = torch.index_select(output, dim=1, index=pos_cls_indices)
            output = neg_cls_output - weighted_neg_output
            batch_output = output
        out = batch_output
        loss = out

        # ---------------------------------------------
        # if target_class is None:
        #     target_class = out.data.max(1, keepdim=True)[1]
        #     target_class = target_class.flatten()
        # loss = -1. * F.nll_loss(out, target_class, reduction='sum')

        self.model.zero_grad()
        # Gradients w.r.t. input and features
        input_gradient = torch.autograd.grad(outputs=loss, inputs=image, only_inputs=True)[0]

        return input_gradient

    def shap_values(self, image, sparse_labels=None):
        #SmoothGrad saliency
        
        self.model.eval()

        # grad = self._getGradients(image, target_class=target_class)
        std_dev = self.std_spread * (image.max().item() - image.min().item())

        cam = torch.zeros_like(image).to(image.device)

        # add gaussian noise to image multiple times
        for i in range(self.num_samples):
            noise = torch.normal(mean=torch.zeros_like(image).to(image.device), std=std_dev)
            # cam += (self._getGradients(image + noise, target_class=sparse_labels)) / self.num_samples

            saliency = self._getGradients(image + noise, target_class=sparse_labels) /self.num_samples
            cam += saliency
        return cam
