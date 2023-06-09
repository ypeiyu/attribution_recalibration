#!/usr/bin/env python
import functools
import operator

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
import random
# DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

from utils.preprocess import preprocess, undo_preprocess


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

# count = 0

def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    # grad_lab_norm = torch.norm(data_grad_lab, p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)

    delta = perturbed_rect - image

    #######################################################
    # global count
    # count += 1
    # deltas = delta  # .squeeze(0)
    # # deltas = undo_preprocess_input_function(deltas).detach()  # .cpu().numpy()
    # file_name = '/home/peiyu/PROJECT/grad-saliency-master/exp_fig/vis_deltas/'
    # from exp_fig import visualize
    # for bth in range(deltas.shape[0]):
    #     img = deltas[bth].cpu().numpy()
    #     f_name = file_name + 'img_' + str(count).zfill(2) + str(int(bth))
    #     visualize(image=img, saliency_map=deltas[bth], filename=f_name, method_name='delta')
    #####################################################

    delta = - data_grad_lab * delta

    ##############################################
    # from utils import undo_preprocess_input_function
    # samples_input = perturbed_rect.detach()  # .squeeze(0)
    # samples_input = undo_preprocess_input_function(samples_input)  # .cpu().numpy()
    # file_name = '/home/peiyu/PROJECT/grad-saliency-master/exp_fig/vis_interpolation/'
    # from exp_fig import visualize
    # for bth in range(samples_input.shape[0]):
    #     img = samples_input[bth].cpu().numpy()
    #     f_name = file_name + 'img_' + str(count).zfill(2) + str(int(bth))
    #     visualize(image=img, saliency_map=samples_input[bth], filename=f_name, method_name='inter')
    #
    # grad_tensor = data_grad_adv  # .squeeze(0)
    # file_name = '/home/peiyu/PROJECT/grad-saliency-master/exp_fig/vis_grads/'
    # from exp_fig import visualize
    # for bth in range(grad_tensor.shape[0]):
    #     img = grad_tensor[bth].cpu().numpy()
    #     f_name = file_name + 'img_' + str(count).zfill(2) + str(int(bth))
    #     visualize(image=img, saliency_map=grad_tensor[bth], filename=f_name, method_name='grads')
    #
    # deltas = delta  # .squeeze(0)
    # # deltas = undo_preprocess_input_function(deltas).detach()  # .cpu().numpy()
    # file_name = '/home/peiyu/PROJECT/grad-saliency-master/exp_fig/vis_integration/'
    # from exp_fig import visualize
    # for bth in range(deltas.shape[0]):
    #     img = deltas[bth].cpu().numpy()
    #     f_name = file_name + 'img_' + str(count).zfill(2) + str(int(bth))
    #     visualize(image=img, saliency_map=deltas[bth], filename=f_name, method_name='delta')
    ##############################################

    eff_sign = torch.where(delta>=0., 1., 0.)
    delta = torch.where(delta>=0., delta, torch.zeros(*delta.shape).cuda())
    return perturbed_rect, delta, eff_sign
    # return perturbed_image, delta


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    c_delta = 0  # cumulative delta
    sign = 0
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # if attack is successful, then break
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if False not in (pred == targeted.view(-1, 1)):
            break

        output = F.softmax(output, dim=1)
        sample_indices = torch.arange(0, output.size(0)).cuda()
        indices_tensor = torch.cat([
            sample_indices.unsqueeze(1),
            targeted.unsqueeze(1)], dim=1)
        loss = gather_nd(output, indices_tensor)
        model.zero_grad()
        model_grads = grad(
            outputs=loss,
            inputs=perturbed_image,
            grad_outputs=torch.ones_like(loss).cuda(),
            create_graph=True)
        data_grad_adv = model_grads[0].detach().data

        sample_indices = torch.arange(0, output.size(0)).cuda()
        indices_tensor = torch.cat([
            sample_indices.unsqueeze(1),
            init_pred.unsqueeze(1)], dim=1)
        loss = gather_nd(output, indices_tensor)
        model.zero_grad()
        model_grads = grad(
            outputs=loss,
            inputs=perturbed_image,
            grad_outputs=torch.ones_like(loss).cuda(),
            create_graph=True)
        data_grad_lab = model_grads[0].detach().data

        # perturbed_image, delta = fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
        perturbed_image, delta, eff_sign = fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
        sign += eff_sign
        c_delta += delta

    c_delta = c_delta / torch.where(sign == 0., 1., sign)
    return c_delta, perturbed_image

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


# def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
#     """target here is the targeted class to be perturbed to"""
#     perturbed_image = image.clone()
#     c_delta = 0  # cumulative delta
#     delta = 0. # image.clone()  # new
#     sign = 0
#     for i in range(max_iter):
#         # requires grads
#         perturbed_image.requires_grad = True
#
#         perturbed_image = preprocess(perturbed_image)
#         output = model(perturbed_image)
#         # if attack is successful, then break
#         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         if False not in (pred == targeted.view(-1, 1)):
#             break
#
#         output = F.softmax(output, dim=1)
#         sample_indices = torch.arange(0, output.size(0)).cuda()
#         indices_tensor = torch.cat([
#             sample_indices.unsqueeze(1),
#             targeted.unsqueeze(1)], dim=1)
#         loss = gather_nd(output, indices_tensor)
#         model.zero_grad()
#         model_grads = grad(
#             outputs=loss,
#             inputs=perturbed_image,
#             grad_outputs=torch.ones_like(loss).cuda(),
#             create_graph=True)
#         data_grad_adv = model_grads[0].detach().data
#
#         # ----------- new ------------
#         multi_grad = delta * data_grad_adv
#         # sign += torch.where(multi_grad>=0., torch.ones(*multi_grad.shape).cuda(), torch.zeros(*multi_grad.shape).cuda())
#         # multi_grad = torch.where(multi_grad>=0., multi_grad, torch.zeros(*multi_grad.shape).cuda())
#         c_delta += multi_grad
#
#         # ------------ new ------------
#         delta = epsilon * data_grad_adv.sign()
#         perturbed_image = preprocess(image.clone()) + delta # epsilon * data_grad_adv.sign()
#         perturbed_image = torch.clamp(undo_preprocess(perturbed_image), min=0, max=1)
#         delta = preprocess(image - perturbed_image)
#
#     return c_delta/sign, perturbed_image

# -------------------------------------------------------------------------------------------

# def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
#     """target here is the targeted class to be perturbed to"""
#     perturbed_image = image.clone()
#     c_delta = 0  # cumulative delta
#     delta = 0. # image.clone()  # new
#     sign = 0
#     for i in range(max_iter):
#         # requires grads
#         perturbed_image.requires_grad = True
#
#         output = model(perturbed_image)
#         # if attack is successful, then break
#         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         if False not in (pred == targeted.view(-1, 1)):
#             break
#
#         output = F.softmax(output, dim=1)
#         sample_indices = torch.arange(0, output.size(0)).cuda()
#         indices_tensor = torch.cat([
#             sample_indices.unsqueeze(1),
#             targeted.unsqueeze(1)], dim=1)
#         loss = gather_nd(output, indices_tensor)
#         model.zero_grad()
#         model_grads = grad(
#             outputs=loss,
#             inputs=perturbed_image,
#             grad_outputs=torch.ones_like(loss).cuda(),
#             create_graph=True)
#         data_grad_adv = model_grads[0].detach().data
#
#         # ----------- new ------------
#         multi_grad = delta * data_grad_adv
#         sign += torch.where(multi_grad>=0., torch.ones(*multi_grad.shape).cuda(), torch.zeros(*multi_grad.shape).cuda())
#         # multi_grad = torch.where(multi_grad>=0., multi_grad, torch.zeros(*multi_grad.shape).cuda())
#         c_delta += multi_grad
#
#         # ------------ new ------------
#         delta = epsilon * data_grad_adv.sign()
#         perturbed_image = image.clone() + delta  # epsilon * data_grad_adv.sign()
#         perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
#         delta = image - perturbed_image
#     return c_delta, perturbed_image


class AGI(object):
    def __init__(self, model, k, top_k, cls_num, eps=0.05):  # 0.05 # 0.005
        self.model = model
        self.cls_num = cls_num - 1
        self.eps = eps
        self.k = k
        self.top_k = top_k
        # self.selected_ids = random.sample(list(range(0, cls_num-1)), top_k)

    def select_id(self, label):
        while True:
            top_ids = random.sample(list(range(0, self.cls_num - 1)), self.top_k)
            if label not in top_ids:
                break
            else:
                continue
        return torch.as_tensor(random.sample(list(range(0, self.cls_num - 1)), self.top_k)).view([1, -1])

    def shap_values(self, input_tensor, sparse_labels=None):

        # Send the data and label to the device
        # data = input_tensor.cuda()
        # data = data.to(device)

        # Forward pass the data through the model
        output = self.model(input_tensor)
        self.model.eval()
        init_pred = output.max(1, keepdim=True)[1].squeeze(1)  # get the index of the max log-probability
        # init_pred = sparse_labels

        # initialize the step_grad towards all target false classes
        step_grad = 0
        # num_class = 1000 # number of total classes
        top_ids_lst = []
        for bth in range(input_tensor.shape[0]):
            top_ids_lst.append(self.select_id(sparse_labels[bth]))  # only for predefined ids
        top_ids = torch.cat(top_ids_lst, dim=0).cuda()

        # top_ids = torch.ones(*[input_tensor.shape[0], 1]).cuda()

        for l in range(top_ids.shape[1]):
            targeted = top_ids[:, l].cuda()  # .to(device)
            delta, perturbed_image = pgd_step(undo_preprocess(input_tensor), self.eps, self.model, init_pred, targeted, self.k)
            # delta, perturbed_image = pgd_step(input_tensor, self.eps, self.model, init_pred, targeted, self.k)

            step_grad += delta

        attribution = step_grad

        return attribution
