#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Implement FullGrad saliency algorithm """

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
import functools
import operator

from .tensor_extractor import FullGradExtractor
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


class FullGrad():
    """
    Compute FullGrad saliency map and full gradient decomposition
    """

    def __init__(self, model, exp_obj='logit', im_size=(3, 224, 224)):
        self.model = model
        self.exp_obj = exp_obj
        self.im_size = (1,) + im_size
        self.model_ext = FullGradExtractor(model, im_size)
        self.biases = self.model_ext.getBiases()
        self.checkCompleteness()

    def checkCompleteness(self):
        """
        Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases of non-linearities).

        """

        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")

        #Random input image
        input = torch.randn(self.im_size).to(device)

        # Get raw outputs
        self.model.eval()
        raw_output = self.model(input)

        # Compute full-gradients and add them up
        input_grad, bias_grad = self.fullGradientDecompose(input, target_class=None, check=True)

        fullgradient_sum = (input_grad * input).sum()
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum()

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients."
        err_string = "Completeness test failed! Raw output = " + str(raw_output.max().item()) + " Full-gradient sum = " + str(fullgradient_sum.item())
        assert isclose(raw_output.max().item(), fullgradient_sum.item(), rel_tol=1e-4), err_string + err_message
        print('Completeness test passed for FullGrad.')

    def fullGradientDecompose(self, image, target_class=None, check=False):
        """
        Compute full-gradient decomposition for an image
        """

        self.model.eval()
        image = image.requires_grad_()
        output = self.model(image)

        if target_class is None:
            target_class = output.data.max(1, keepdim=False)[1]

        if self.exp_obj == 'prob' or check is True:
            batch_output = -1. * F.nll_loss(output, target_class.flatten(), reduction='sum')
        elif self.exp_obj == 'logit':
            sample_indices = torch.arange(0, output.size(0)).cuda()
            indices_tensor = torch.cat([
                sample_indices.unsqueeze(1),
                target_class.unsqueeze(1)], dim=1)
            output_scalar = gather_nd(output, indices_tensor)
            batch_output = torch.sum(output_scalar)

        elif self.exp_obj == 'contrast':
            b_num, c_num = output.shape[0], output.shape[1]
            mask = torch.ones(b_num, c_num, dtype=torch.bool)
            mask[torch.arange(b_num), target_class] = False
            neg_cls_output = output[mask].reshape(b_num, c_num - 1)
            neg_weight = F.softmax(neg_cls_output, dim=1)
            weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
            pos_cls_output = output[torch.arange(b_num), target_class]
            output = pos_cls_output - weighted_neg_output
            output_scalar = output
            batch_output = torch.sum(output_scalar)

        out = batch_output
        output_scalar = out

        # ---------------------------------------------
        # if target_class is None:
        #     target_class = out.data.max(1, keepdim=True)[1]
        # output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')  # -1 * extract and negative

        input_gradient, feature_gradients = self.model_ext.getFeatureGrads(image, output_scalar)

        # Compute feature-gradients \times bias 
        bias_times_gradients = []
        L = len(self.biases)

        for i in range(L):

            # feature gradients are indexed backwards
            # because of backprop
            g = feature_gradients[L-1-i]

            # reshape bias dimensionality to match gradients
            bias_size = [1] * len(g.size())
            bias_size[1] = self.biases[i].size(0)
            b = self.biases[i].view(tuple(bias_size))

            bias_times_gradients.append(g * b.expand_as(g))

        return input_gradient, bias_times_gradients

    def _postProcess(self, input, eps=1e-6):
        # Absolute value
        input = abs(input)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.min(1, keepdim=True)
        input = input - temp.unsqueeze(1).unsqueeze(1)

        flatin = input.view((input.size(0),-1))
        temp, _ = flatin.max(1, keepdim=True)
        input = input / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return input

    def shap_values(self, image, sparse_labels=None):
        #FullGrad saliency

        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, target_class=sparse_labels)

        # Input-gradient * image
        grd = input_grad * image
        gradient = self._postProcess(grd).sum(1, keepdim=True)
        cam = gradient

        im_size = image.size()

        # Aggregate Bias-gradients
        for i in range(len(bias_grad)):

            # Select only Conv layers
            if len(bias_grad[i].size()) == len(im_size):
                temp = self._postProcess(bias_grad[i])
                gradient = F.interpolate(temp, size=(im_size[2], im_size[3]), mode='bilinear', align_corners=True)
                cam += gradient.sum(1, keepdim=True)
        return cam
