import torch
import random
import torch.nn.functional as F

from utils.preprocess import undo_preprocess

cls_num_dict = {'imagenet': 1000, 'cifar10': 10, 'cifar100': 100, 'gtsrb': 43}


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

    def fgsm_step(self, image, epsilon, data_grad_label, data_grad_pred):
        # generate the perturbed image based on steepest descent
        delta = epsilon * data_grad_label.sign()

        # + delta because we are ascending
        perturbed_image = image + delta
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1)

        delta = perturbed_image - image
        delta = - data_grad_pred * delta

        if self.est_method == 'valid_ip':
            valid_ip_mask = torch.where(delta >= 0., 1., 0.)
            delta = delta * valid_ip_mask
            return perturbed_image, valid_ip_mask, delta

        return perturbed_image, delta

    def pgd_step(self, image, epsilon, model, init_pred, targeted, max_iter):
        """target here is the targeted class to be perturbed to"""
        perturbed_image = image.clone()
        c_delta = 0  # cumulative delta
        c_mask = 0
        curr_grad = 0
        for i in range(max_iter):
            # requires grads
            perturbed_image.requires_grad = True
            output = model(perturbed_image)

            # ---------------------- data_grad_label -----------------------
            batch_output = None
            if self.exp_obj == 'logit':
                batch_output = -1. * F.nll_loss(output, targeted.flatten(), reduction='sum')

            elif self.exp_obj == 'prob':
                batch_output = -1. * F.nll_loss(F.log_softmax(output, dim=1), targeted.flatten(), reduction='sum')

            elif self.exp_obj == 'contrast':
                b_num, c_num = output.shape[0], output.shape[1]
                mask = torch.ones(b_num, c_num, dtype=torch.bool)
                mask[torch.arange(b_num), targeted] = False
                neg_cls_output = output[mask].reshape(b_num, c_num - 1)
                neg_weight = F.softmax(neg_cls_output, dim=1)
                weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
                pos_cls_output = output[torch.arange(b_num), targeted]
                batch_output = (pos_cls_output - weighted_neg_output).unsqueeze(1)

            self.model.zero_grad()
            batch_output.backward(retain_graph=True)
            gradients = perturbed_image.grad.clone()
            perturbed_image.grad.zero_()
            gradients.detach()

            data_grad_label = gradients

            # ---------------------- data_grad_pred -----------------------
            batch_output = None
            if self.exp_obj == 'logit':
                batch_output = -1. * F.nll_loss(output, init_pred.flatten(), reduction='sum')

            elif self.exp_obj == 'prob':
                batch_output = -1. * F.nll_loss(F.log_softmax(output, dim=1), init_pred.flatten(), reduction='sum')

            elif self.exp_obj == 'contrast':
                b_num, c_num = output.shape[0], output.shape[1]
                mask = torch.ones(b_num, c_num, dtype=torch.bool)
                mask[torch.arange(b_num), init_pred] = False
                neg_cls_output = output[mask].reshape(b_num, c_num - 1)
                neg_weight = F.softmax(neg_cls_output, dim=1)
                weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
                pos_cls_output = output[torch.arange(b_num), init_pred]
                batch_output = (pos_cls_output - weighted_neg_output).unsqueeze(1)

            self.model.zero_grad()
            batch_output.backward()
            gradients = perturbed_image.grad.clone()
            perturbed_image.grad.zero_()
            gradients.detach()

            data_grad_pred = gradients

            if self.est_method == 'valid_ip':
                perturbed_image, valid_ip_mask, delta = self.fgsm_step(image, epsilon, data_grad_label, data_grad_pred)
                c_mask += valid_ip_mask
                c_delta += delta
            else:
                perturbed_image, delta = self.fgsm_step(image, epsilon, data_grad_label, data_grad_pred)
                c_delta += delta

        if self.est_method == 'valid_ip':
            return c_delta, c_mask
        else:
            return c_delta

    def shap_values(self, input_tensor, sparse_labels=None):

        # Forward pass the data through the model
        self.model.eval()

        output = self.model(input_tensor)
        init_pred = output.max(1, keepdim=True)[1].squeeze(1)

        # initialize the step_grad towards all target false classes
        step_grad = 0

        c_valid_ref_mask = 0
        c_valid_ip_mask = 0
        top_ids_lst = []
        for bth in range(input_tensor.shape[0]):
            top_ids_lst.append(self.select_id(sparse_labels[bth]))  # only for predefined ids
        top_ids = torch.cat(top_ids_lst, dim=0).cuda()

        for l in range(top_ids.shape[1]):
            targeted = top_ids[:, l].cuda()

            if self.est_method == 'valid_ip':
                delta, valid_mask = self.pgd_step(undo_preprocess(input_tensor, self.dataset_name), self.eps,
                                                  self.model, init_pred, targeted, self.k)
                c_valid_ip_mask += valid_mask

            else:
                delta = self.pgd_step(undo_preprocess(input_tensor, self.dataset_name), self.eps, self.model, init_pred,
                                      targeted, self.k)

            if self.est_method == 'valid_ref':
                ref_mask = torch.where(delta >= 0., 1., 0.)
                delta = delta * ref_mask
                c_valid_ref_mask += ref_mask

            step_grad += delta

        if self.est_method == 'valid_ref':
            step_grad /= torch.where(c_valid_ref_mask == 0., torch.ones(c_valid_ref_mask.shape).cuda(), c_valid_ref_mask)
        if self.est_method == 'valid_ip':
            step_grad /= torch.where(c_valid_ip_mask == 0., torch.ones(c_valid_ip_mask.shape).cuda(), c_valid_ip_mask)

        return step_grad
