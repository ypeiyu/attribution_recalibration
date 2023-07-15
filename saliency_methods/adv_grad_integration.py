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


cls_num_dict = {'imagenet':1000, 'cifar10':10, 'cifar100':100}


class AGI(object):
    def __init__(self, model, k, top_k, eps=0.05, scale_by_input=False, est_method='vanilla', exp_obj='logit',
                 dataset_name='imagenet'):
        self.model = model
        self.cls_num = cls_num_dict[dataset_name] - 1
        self.eps = eps
        self.k = k
        self.top_k = top_k
        self.scale_by_input = scale_by_input
        self.est_method = est_method
        self.exp_obj = exp_obj
        self.dataset_name = dataset_name

    def select_id(self, label):
        while True:
            top_ids = random.sample(list(range(0, self.cls_num - 1)), self.top_k)
            if label not in top_ids:
                break
            else:
                continue
        return torch.as_tensor(random.sample(list(range(0, self.cls_num - 1)), self.top_k)).view([1, -1])

    def fgsm_step(self, image, epsilon, data_grad_label):
        # generate the perturbed image based on steepest descent
        delta = epsilon * data_grad_label.sign()

        # + delta because we are ascending
        perturbed_image = image + delta
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)

        delta = image - perturbed_image
        # delta = data_grad_label

        return perturbed_image, delta

    def pgd_step(self, image, epsilon, model, targeted, max_iter):
        """target here is the targeted class to be perturbed to"""
        perturbed_image = image.clone()
        c_delta = 0  # cumulative delta
        c_sign = 0
        curr_grad = 0
        for i in range(max_iter):
            # requires grads
            perturbed_image.requires_grad = True
            output = model(perturbed_image)
            batch_output = F.softmax(output, dim=1)

            if self.exp_obj == 'logit':
                batch_output = batch_output
            elif self.exp_obj == 'prob':
                batch_output = torch.log_softmax(output, 1)
            elif self.exp_obj == 'contrast':
                b_num, c_num = output.shape[0], output.shape[1]
                mask = torch.ones(b_num, c_num, dtype=torch.bool)
                mask[torch.arange(b_num), targeted] = False
                neg_cls_output = output[mask].reshape(b_num, c_num - 1)
                neg_weight = F.softmax(neg_cls_output, dim=1)
                weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
                pos_cls_output = output[torch.arange(b_num), targeted]
                output = pos_cls_output - weighted_neg_output
                batch_output = output.unsqueeze(1)

            if targeted is not None and batch_output.size(1) > 1:
                sample_indices = torch.arange(0, batch_output.size(0)).cuda()
                indices_tensor = torch.cat([
                    sample_indices.unsqueeze(1),
                    targeted.unsqueeze(1)], dim=1)
                target_output = gather_nd(batch_output, indices_tensor)

            model.zero_grad()
            model_grads = grad(
                outputs=target_output,
                inputs=perturbed_image,
                grad_outputs=torch.ones_like(target_output).cuda(),
                create_graph=True)
            data_grad_label = model_grads[0].detach().data

            if self.est_method == 'valid_ip':
                perturbed_image, delta, valid_num = self.fgsm_step(image, epsilon, data_grad_label)
                mul_grad_delta = curr_grad * delta
                valid_num = torch.where(mul_grad_delta >= 0., 1., 0.)
                mul_grad_delta = torch.where(mul_grad_delta >= 0., mul_grad_delta, torch.zeros(*mul_grad_delta.shape).cuda())
                c_sign += valid_num
                c_delta += mul_grad_delta
            else:
                perturbed_image, delta = self.fgsm_step(image, epsilon, data_grad_label)
                delta = curr_grad * delta
                c_delta += delta
            curr_grad = data_grad_label

        if self.est_method == 'valid_ip':
            c_delta = c_delta / torch.where(c_sign == 0., 1., c_sign)

        return c_delta * (image-perturbed_image)

    def shap_values(self, input_tensor, sparse_labels=None):

        # Forward pass the data through the model
        self.model.eval()

        # initialize the step_grad towards all target false classes
        step_grad = 0
        valid_ref_num = 0
        top_ids_lst = []
        for bth in range(input_tensor.shape[0]):
            top_ids_lst.append(self.select_id(sparse_labels[bth]))  # only for predefined ids
        top_ids = torch.cat(top_ids_lst, dim=0).cuda()

        for l in range(top_ids.shape[1]):
            targeted = top_ids[:, l].cuda()
            delta = self.pgd_step(undo_preprocess(input_tensor, self.dataset_name), self.eps, self.model, targeted, self.k)

            if self.est_method == 'valid_ref':
                ref_mask = torch.where(delta >= 0., 1., 0.)
                delta = delta * ref_mask
                valid_ref_num += ref_mask
                step_grad += delta
            else:
                step_grad += delta

        if self.est_method == 'valid_ref':
            step_grad /= torch.where(valid_ref_num == 0., torch.ones(valid_ref_num.shape).cuda(), valid_ref_num)

        return step_grad
