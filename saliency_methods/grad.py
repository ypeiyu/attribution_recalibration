#!/usr/bin/env python
import functools
import operator

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
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
    def __init__(self, model):
        self.model = model
        self.model.eval()

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
        # if self.loss:
        #     outputs = torch.log_softmax(outputs, 1)
        #     agg = F.nll_loss(outputs, target_class, reduction='sum')
        # else:
        #     agg = -1. * F.nll_loss(outputs, target_class, reduction='sum')
        batch_output = -output

        # should check that users pass in sparse labels
        # Only look at the user-specified label
        if sparse_labels is not None and batch_output.size(1) > 1:
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

# positive
# ================= Input Gradients ==================
# 	time: 	118.370
# deletion 10-90 logit scores
# [0.589, 0.478, 0.398, 0.33, 0.269, 0.197, 0.078, 0.053, 0.048]
# deletion accu scores
# [0.368, 0.24, 0.153, 0.099, 0.059, 0.033, 0.013, 0.001, 0.001]
#
#
# insertion 10-90 logit scores
# [1.0, 0.994, 0.636, 0.461, 0.366, 0.273, 0.166, 0.11, 0.051]
# insertion accu scores
# [0.716, 0.708, 0.34, 0.183, 0.127, 0.081, 0.045, 0.021, 0.005]
#
#
# Diff logit scores
# [0.411, 0.517, 0.238, 0.131, 0.097, 0.077, 0.088, 0.056, 0.003]
# Diff accu scores
# [0.348, 0.468, 0.187, 0.084, 0.068, 0.049, 0.032, 0.02, 0.004]
# positive
# deletion 10-90 logit scores
# [0.628, 0.528, 0.43, 0.351, 0.296, 0.216, 0.101, 0.054, 0.048]
# deletion accu scores
# [0.394, 0.261, 0.17, 0.105, 0.062, 0.036, 0.014, 0.001, 0.001]
#
#
# insertion 10-90 logit scores
# [1.0, 0.993, 0.583, 0.412, 0.317, 0.218, 0.155, 0.087, 0.01]
# insertion accu scores
# [0.716, 0.707, 0.319, 0.167, 0.115, 0.073, 0.04, 0.022, 0.005]
#
#
# Diff logit scores
# [0.372, 0.465, 0.153, 0.062, 0.021, 0.002, 0.054, 0.033, -0.038]
# Diff accu scores
# [0.322, 0.446, 0.149, 0.062, 0.053, 0.037, 0.026, 0.021, 0.004]

# no absolute values
# ================= Input Gradients ==================
# 	time: 	118.490
# deletion 10-90 logit scores
# [0.588, 0.492, 0.415, 0.355, 0.275, 0.189, 0.103, 0.056, -0.003]
# deletion accu scores
# [0.375, 0.251, 0.168, 0.111, 0.073, 0.045, 0.027, 0.013, 0.003]
#
#
# insertion 10-90 logit scores
# [0.672, 0.54, 0.452, 0.376, 0.297, 0.224, 0.118, 0.059, 0.017]
# insertion accu scores
# [0.401, 0.276, 0.188, 0.123, 0.079, 0.05, 0.029, 0.014, 0.004]
#
#
# Diff logit scores
# [0.084, 0.048, 0.037, 0.021, 0.022, 0.035, 0.016, 0.003, 0.02]
# Diff accu scores
# [0.026, 0.024, 0.02, 0.012, 0.007, 0.004, 0.003, 0.001, 0.001]

# original
# ================= Input Gradients ==================
# 	time: 	119.651
# deletion 10-90 logit scores
# [0.649, 0.489, 0.41, 0.349, 0.282, 0.21, 0.142, 0.068, 0.038]
# deletion accu scores
# [0.405, 0.275, 0.185, 0.116, 0.07, 0.036, 0.018, 0.008, 0.003]
#
#
# insertion 10-90 logit scores
# [0.859, 0.77, 0.652, 0.576, 0.481, 0.387, 0.294, 0.169, 0.063]
# insertion accu scores
# [0.596, 0.511, 0.432, 0.338, 0.251, 0.169, 0.101, 0.05, 0.011]
#
#
# Diff logit scores
# [0.21, 0.281, 0.242, 0.228, 0.199, 0.177, 0.153, 0.101, 0.026]
# Diff accu scores
# [0.191, 0.236, 0.247, 0.221, 0.182, 0.133, 0.083, 0.042, 0.009]
