#!/usr/bin/env python
import functools
import operator

import torch
from torch.autograd import grad
import torch.utils.data

import numpy as np
from utils import undo_preprocess_input_function
import cv2

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


class ExpectedGradients(object):
    def __init__(self, model, k, bg_dataset, bg_size, batch_size, random_alpha=True, scale_by_inputs=True, cal_type=['nano', 'valid_ref', 'valid_intp'][0]):
        self.model = model
        self.model.eval()
        self.k = k
        self.scale_by_inputs = scale_by_inputs
        self.bg_size = bg_size
        self.random_alpha = random_alpha
        self.ref_sampler = torch.utils.data.DataLoader(
                dataset=bg_dataset,
                batch_size=bg_size*batch_size,
                shuffle=True,
                pin_memory=False,
                drop_last=False)

        self.cal_type = cal_type

    def _get_ref_batch(self):
        return next(iter(self.ref_sampler))[0].float()

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
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_*self.bg_size).uniform_(0, 1).to(DEFAULT_DEVICE)
        else:
            if k_ == 1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for i in range(batch_size*k_*self.bg_size)]).to(DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0, 1, k_) for i in range(batch_size*self.bg_size)]).to(DEFAULT_DEVICE)

        shape = [batch_size, k_*self.bg_size] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor
        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult
        # A fine Affine Combine

        samples_input = end_point_input + end_point_ref
        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        if input_tensor.shape != reference_tensor.shape:
            input_expand_mult = input_tensor.unsqueeze(1)
            sd = input_expand_mult - reference_tensor[:, ::self.k, :]
        else:
            sd = input_tensor - reference_tensor
        return sd

    def _get_grads(self, samples_input, sparse_labels=None):
        samples_input.requires_grad = True
        shape = list(samples_input.shape)
        shape[1] = self.bg_size

        if self.cal_type == 'valid_intp':
            shape.insert(2, self.k)

        grad_tensor = torch.zeros(shape).float().to(DEFAULT_DEVICE)

        for b_id in range(self.bg_size):
            for k_id in range(self.k):
                particular_slice = samples_input[:, b_id*self.k+k_id]
                output = self.model(particular_slice)
                batch_output = output

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

                if self.cal_type == 'valid_intp':
                    grad_tensor[:, b_id, k_id, :] = model_grads[0].detach().data/self.k
                else:
                    grad_tensor[:, b_id, :] += model_grads[0].detach().data / self.k
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
        shape.insert(1, self.k*self.bg_size)

        from utils import preprocess_input_function
        ref_lst = [preprocess_input_function(torch.rand(*input_tensor.shape)) for _ in range(self.k*self.bg_size)]
        ref = torch.cat(ref_lst)
        reference_tensor = ref.view(*shape).cuda()

        samples_input = self._get_samples_input(input_tensor, reference_tensor)

        if self.cal_type == 'valid_intp':
            samples_delta = self._get_samples_delta(samples_input, reference_tensor)
            grad_ori_tensor = self._get_grads(samples_input, sparse_labels)
            grad_ori_tensor = grad_ori_tensor.squeeze(2)
            grad_ori_tensor = grad_ori_tensor.reshape(samples_delta.shape)
            mult_grads = grad_ori_tensor * samples_delta
            grad_sign = torch.where(mult_grads >= 0., 1., 0.)
            mult_grads = mult_grads * grad_sign

            counts = torch.sum(grad_sign, dim=1)
            mult_grads = mult_grads.sum(1) / torch.where(counts == 0., torch.ones(counts.shape).cuda(), counts)
            expected_grads = mult_grads
        elif self.cal_type == 'valid_ref':
            samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            zeros = torch.zeros(grad_tensor.shape).cuda()
            ones = torch.ones(grad_tensor.shape).cuda()
            mult_grads = grad_tensor * samples_delta
            sign = torch.where(mult_grads >= 0., ones, zeros)
            mult_grads = mult_grads * sign

            counts = torch.sum(sign, dim=1)
            mult_grads = mult_grads.sum(1) / torch.where(counts == 0., ones[:, 0], counts)
            expected_grads = mult_grads
        else:
            samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor

            expected_grads = mult_grads.mean(1)

        return expected_grads
