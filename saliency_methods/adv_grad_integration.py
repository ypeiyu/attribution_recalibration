import functools
import operator

import torch
from torch.autograd import grad
import random
import torch.nn.functional as F

from utils.preprocess import undo_preprocess


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


class AGI(object):
    def __init__(self, model, k, top_k, cls_num, eps=0.05, scale_by_input=False, est_method='vanilla', exp_obj='logit'):
        self.model = model
        self.cls_num = cls_num - 1
        self.eps = eps
        self.k = k
        self.top_k = top_k
        self.scale_by_input = scale_by_input
        self.est_method = est_method
        self.exp_obj = exp_obj

    def select_id(self, label):
        while True:
            top_ids = random.sample(list(range(0, self.cls_num - 1)), self.top_k)
            if label not in top_ids:
                break
            else:
                continue
        return torch.as_tensor(random.sample(list(range(0, self.cls_num - 1)), self.top_k)).view([1, -1])

    def fgsm_step(self, image, epsilon, data_grad_adv, data_grad_lab):
        # generate the perturbed image based on steepest descent
        delta = epsilon * data_grad_adv.sign()

        # + delta because we are ascending
        perturbed_image = image + delta
        perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)

        delta = perturbed_rect - image
        delta = - data_grad_lab * delta

        if self.est_method == 'valid_ip':
            valid_num = torch.where(delta >= 0., 1., 0.)
            valid_delta = torch.where(delta >= 0., delta, torch.zeros(*delta.shape).cuda())
            return perturbed_rect, valid_delta, valid_num
        else:
            return perturbed_image, delta

    def pgd_step(self, image, epsilon, model, init_pred, targeted, max_iter):
        """target here is the targeted class to be perturbed to"""
        perturbed_image = image.clone()
        c_delta = 0  # cumulative delta
        sign = 0
        for i in range(max_iter):
            # requires grads
            perturbed_image.requires_grad = True
            output = model(perturbed_image)
            # if attack is successful, then break
            # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # if False not in (pred == targeted.view(-1, 1)):
            #     break

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

            if self.est_method == 'valid_ip':
                perturbed_image, delta, eff_sign = self.fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
                sign += eff_sign
                c_delta += delta
            else:
                perturbed_image, delta = self.fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)

        if self.est_method == 'valid_ip':
            c_delta = c_delta / torch.where(sign == 0., 1., sign)
        else:
            c_delta = c_delta

        return c_delta

    def shap_values(self, input_tensor, sparse_labels=None):

        # Forward pass the data through the model
        output = self.model(input_tensor)
        self.model.eval()
        init_pred = output.max(1, keepdim=True)[1].squeeze(1)  # get the index of the max log-probability
        # init_pred = sparse_labels

        # initialize the step_grad towards all target false classes
        step_grad = 0
        top_ids_lst = []
        for bth in range(input_tensor.shape[0]):
            top_ids_lst.append(self.select_id(sparse_labels[bth]))  # only for predefined ids
        top_ids = torch.cat(top_ids_lst, dim=0).cuda()

        for l in range(top_ids.shape[1]):
            targeted = top_ids[:, l].cuda()
            delta = self.pgd_step(undo_preprocess(input_tensor), self.eps, self.model, init_pred, targeted, self.k)

            if self.est_method == 'valid_ip':
                delta = delta / torch.where(delta == 0., 1., sign)
                step_grad += delta
                attribution = step_grad
            elif self.est_method == 'valid_ref':
                samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
                grad_tensor = self._get_grads(samples_input, sparse_labels)
                zeros = torch.zeros(grad_tensor.shape).cuda()
                ones = torch.ones(grad_tensor.shape).cuda()

                grad_tensor = delta
                mult_grads = grad_tensor * samples_delta
                sign = torch.where(mult_grads >= 0., ones, zeros)
                mult_grads = torch.pow(mult_grads, 2.) * sign

                counts = torch.sum(sign, dim=1)
                mult_grads = mult_grads.sum(1) / torch.where(counts == 0., ones[:, 0], counts)
                attribution = mult_grads
            else:
                step_grad += delta
                attribution = step_grad

        return attribution
