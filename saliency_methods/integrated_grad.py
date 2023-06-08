#!/usr/bin/env python
import functools
import operator

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

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


from utils import undo_preprocess_input_function
import cv2
def single_img_inspection(img, file_name):
    image_set = undo_preprocess_input_function(img).detach().cpu().numpy()
    for i in range(image_set.shape[0]):
        image = image_set[i] * 255
        image = image.astype(np.uint8)
        image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_name+'img'+str(i)+'.jpg', image)


class IntegratedGradients(object):
    def __init__(self, model, k=1, scale_by_inputs=True):
        self.model = model
        self.model.eval()
        self.k = k
        self.scale_by_inputs = scale_by_inputs

        self.f_std = 0.
        self.b_std = 0.
        self.img_num = 0

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
        # if k_ == 1:
        #     t_tensor = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size)]).to(DEFAULT_DEVICE)
        # else:
        t_tensor = torch.cat([torch.linspace(0, 1, k_) for _ in range(batch_size)]).to(DEFAULT_DEVICE)

        shape = [batch_size, k_] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor
        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult

        # A fine Affine Combine
        samples_input = end_point_input + end_point_ref

        # single_img_inspection(samples_input.view(-1,3,224,224), 'exp_fig/test/')
        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd

    def _get_samples_inter_delta(self, input_tensor, reference_tensor):
        sd = input_tensor - reference_tensor
        return sd

    def _get_grads(self, samples_input, sparse_labels=None):
        samples_input.requires_grad = True

        grad_tensor = torch.zeros(samples_input.shape).float().to(DEFAULT_DEVICE)

        for i in range(self.k):
            particular_slice = samples_input[:, i]
            # output, _, proto_output, _ = model(particular_slice)  # [5, 200] [5, 2000]

            output = self.model(particular_slice)  # [5, 200] [5, 2000]
            # origin
            # output = - torch.softmax(output, 1)
            # soft: [0.134, 0.174, 0.198, 0.238, 0.24, 0.256, 0.259, 0.208, 0.095]
            # ori: [0.126, 0.153, 0.164, 0.232, 0.255, 0.245, 0.279, 0.195, 0.098]

            # output = - torch.log_softmax(output, 1)

            # import torch.nn.functional as F
            # agg = - F.nll_loss(outputs, target_class, reduction='sum')

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
            grad_tensor[:, i, :] = model_grads[0].detach().data  # detach
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

        # =============== original ==================
        # reference_tensor = torch.zeros(*input_tensor.shape).cuda().to(DEFAULT_DEVICE)
        # from utils import preprocess_input_function
        # reference_tensor = preprocess_input_function(reference_tensor)
        # reference_tensor = reference_tensor.repeat([self.k, 1, 1, 1])
        # reference_tensor = reference_tensor.view(shape)

        # =============== Gaussian noise (x) ==================
        # # k equal Gaussian noise for each input
        # std_dev = 0.15 * (input_tensor.max().item() - input_tensor.min().item())
        # noise = torch.normal(mean=torch.zeros_like(input_tensor).cuda(), std=std_dev)
        # noise = noise.repeat(1, self.k, 1, 1, 1)
        # reference_tensor = noise.view(*shape).cuda()

        # =============== Uniform noise (v) ==================
        # k equal Uniform noise for each input
        from utils import preprocess_input_function
        noise = preprocess_input_function(torch.rand(*input_tensor.shape))
        noise = noise.repeat(1, self.k, 1, 1, 1)
        reference_tensor = noise.view(*shape).cuda()

        # =============== smoothgrad (v) ==================
        # k equal Gaussian noise on input for each input
        std_dev = 0.15 * (input_tensor.max().item() - input_tensor.min().item())
        noise = torch.normal(mean=torch.zeros_like(input_tensor).cuda(), std=std_dev)
        reference_tensor = (input_tensor + noise).repeat(1, self.k, 1, 1, 1)
        reference_tensor = reference_tensor.view(*shape).cuda()


        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        # samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        samples_delta = self._get_samples_inter_delta(samples_input, reference_tensor)
        grad_tensor = self._get_grads(samples_input, sparse_labels)

        # ------------ original ------------
        # mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        # expected_grads = mult_grads.mean(1)

        # ----------------------- calculate values that are calculated from same direction ------------------------
        zeros = torch.zeros(grad_tensor.shape).cuda()
        ones = torch.ones(grad_tensor.shape).cuda()
        # ==================== case 0: All =======================
        mult_grads = grad_tensor * samples_delta
        # =================== case 1: overlap (resides on path) ===================
        sign = torch.where(mult_grads >= 0., ones, zeros)
        # =================== case 2: negative numbers ===================
        all_neg = torch.where(mult_grads < 0., ones, zeros)

        mult_grads = mult_grads * sign
        counts = torch.sum(all_neg, dim=1)
        mult_grads = mult_grads.sum(1) / torch.where(counts == 0., ones[:, 0], counts)
        # [0.643, 0.712, 0.735, 0.73, 0.698, 0.666, 0.558, 0.391, 0.194] new sum
        # [0.647, 0.727, 0.732, 0.729, 0.696, 0.641, 0.557, 0.395, 0.216] new mean yeah its actually have similar effect
        # [0.569, 0.624, 0.668, 0.688, 0.679, 0.613, 0.555, 0.388, 0.229] mean by count
        # [0.731, 0.793, 0.797, 0.782, 0.725, 0.633, 0.535, 0.42, 0.201] mean by neg count

        # [0.703, 0.78, 0.81, 0.796, 0.769, 0.689, 0.595, 0.444, 0.236] old sum
        # [0.611, 0.7, 0.721, 0.727, 0.727, 0.689, 0.602, 0.45, 0.246] old / neg_num
        # [0.691, 0.772, 0.793, 0.798, 0.766, 0.715, 0.598, 0.431, 0.223]

        # [0.637, 0.731, 0.783, 0.784, 0.743, 0.694, 0.631, 0.483, 0.276] uniform noise
        # [0.676, 0.752, 0.799, 0.775, 0.758, 0.686, 0.63, 0.462, 0.268] uniform noise / neg_num

        # [0.705, 0.778, 0.796, 0.797, 0.757, 0.674, 0.578, 0.456, 0.252] uniform noise * new delta / neg_num
        # [0.583, 0.67, 0.711, 0.722, 0.72, 0.671, 0.575, 0.456, 0.216] uniform noise * new delta / pos_num

        # [0.682, 0.757, 0.764, 0.758, 0.731, 0.696, 0.581, 0.451, 0.235] uniform different noise * new delta / neg_num
        # [0.601, 0.668, 0.696, 0.711, 0.702, 0.67, 0.587, 0.447, 0.237] uniform different noise * new delta / pos_num

        # [0.678, 0.764, 0.798, 0.808, 0.781, 0.683, 0.595, 0.489, 0.274] uniform / neg_num
        expected_grads = mult_grads

        return expected_grads
