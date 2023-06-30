import random
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from utils import undo_preprocess_input_function
from utils import visualize

import functools
import operator


def normalize_saliency_map(saliency_map):
    saliency_map = torch.sum(torch.abs(saliency_map), dim=1, keepdim=True)
    # saliency_map = torch.abs(saliency_map)

    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.min(1, keepdim=True)
    saliency_map = saliency_map - temp.unsqueeze(1).unsqueeze(1)
    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.max(1, keepdim=True)
    saliency_map = saliency_map / (temp.unsqueeze(1).unsqueeze(1) + 1e-10)

    saliency_map = saliency_map.repeat(1, 3, 1, 1)

    return saliency_map


def gather_nd(params, indices):
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


class Evaluator(object):
    def __init__(self, model, explainer, dataloader, log=print):
        self.model = model
        self.explainer = explainer
        self.dataloader = dataloader
        self.log = log

        self.model.eval()

    def DiffID(self, ratio_lst, centers=None):
        log = self.log
        n_examples = 0
        n_correct = 0
        loc_ind = 0
        ratio_len = len(ratio_lst)
        n_pert_correct_del_ins_lst = [[0 for _ in range(ratio_len)], [0 for _ in range(ratio_len)]]
        logit_change_del_ins_lst = [[torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(ratio_len)],
                                    [torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(ratio_len)]]

        start = time.time()
        for batch_num, (batch_image, label) in enumerate(tqdm(self.dataloader)):
            batch_image = batch_image.cuda()
            target = label.cuda()

            batch_size = batch_image.shape[0]

            output = self.model(batch_image).detach()
            _, predicted = torch.max(output.data, 1)
            n_correct += (predicted == target).sum().item()
            n_examples += batch_size

            # ------------------ attribution estimation -------------------------
            if centers is not None:
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in range(output.shape[0])]
                saliency_map = self.explainer.shap_values(batch_image, sparse_labels=target, centers=clu_lst)
            else:
                saliency_map = self.explainer.shap_values(batch_image, sparse_labels=target)

            # -------------------------------- saliency map normalization -----------------------------------------
            saliency_map = normalize_saliency_map(saliency_map.detach())

            self.model.eval()
            num_elements = batch_image[0].numel()

            for r_ind, ratio in enumerate(ratio_lst):
                for is_del in [False, True]:
                    del_ratio = ratio

                    flat_image = batch_image.view(batch_size, -1)
                    flat_s_map = saliency_map.view(batch_size, -1)
                    # order by attributions
                    sorted_ind = torch.argsort(flat_s_map, dim=1, descending=is_del)
                    # preserve pixels
                    num_delete = int(num_elements * del_ratio)
                    preserve_ind = sorted_ind[:, num_delete:]
                    mask = torch.zeros_like(flat_image)
                    mean_preserve_lst = []
                    for b_num in range(batch_size):
                        mask[b_num][preserve_ind[b_num]] = 1.
                        mean_preserve_lst.append(torch.mean(flat_image[b_num][preserve_ind[b_num]]))
                    mean_preserve = torch.stack(mean_preserve_lst).unsqueeze(1)
                    perturb_img = mask*flat_image + (1-mask)*mean_preserve
                    perturb_img = perturb_img.view(batch_image.size())

                    output_pert = self.model(perturb_img).detach()

                    isd = int(is_del)
                    _, predicted_pert = torch.max(output_pert.data, 1)
                    n_pert_correct_del_ins_lst[isd][r_ind] += (predicted_pert == target).sum().item()
                    for bth in range(batch_size):
                        t = target[bth]
                        logit_change_del_ins_lst[isd][r_ind][loc_ind+bth:loc_ind+bth+1] = output_pert[bth, t] / output[bth, t]

            loc_ind += batch_size

        end = time.time()
        log('\ttime: \t{:.3f}'.format(end - start))
        insertion_logit = []
        insertion_acc = []

        deletion_logit = []
        deletion_acc = []

        DiffID_logit = []
        DiffID_acc = []

        for r_ind in range(ratio_len):
            mean_accu_del = n_pert_correct_del_ins_lst[1][r_ind] / n_examples
            var_del, mean_del = torch.var_mean(logit_change_del_ins_lst[1][r_ind], unbiased=False)
            mean_del = mean_del.item()
            deletion_logit.append(round(mean_del, 3))
            deletion_acc.append(round(mean_accu_del, 3))

            mean_accu_ins = n_pert_correct_del_ins_lst[0][r_ind] / n_examples
            var_ins, mean_ins = torch.var_mean(logit_change_del_ins_lst[0][r_ind], unbiased=False)
            mean_ins = mean_ins.item()
            insertion_logit.append(round(mean_ins, 3))
            insertion_acc.append(round(mean_accu_ins, 3))

            del_accu = mean_accu_ins - mean_accu_del
            del_logit = mean_ins - mean_del
            DiffID_logit.append(round(del_logit, 3))
            DiffID_acc.append(round(del_accu, 3))
        self.log('deletion logit scores')
        self.log(deletion_logit)
        self.log('deletion accu scores')
        self.log(deletion_acc)
        self.log('\n')
        self.log('insertion logit scores')
        self.log(insertion_logit)
        self.log('insertion accu scores')
        self.log(insertion_acc)
        self.log('\n')
        self.log('Diff logit scores')
        self.log(DiffID_logit)
        self.log('Diff accu scores')
        self.log(DiffID_acc)

    def visual_inspection(self, file_name, num_vis, method_name):
        for batch_num, (image, label) in enumerate(self.dataloader):
            if (batch_num*image.shape[0]) >= num_vis:
                break
            image = image.cuda()
            target = label.cuda()

            saliency_map = self.explainer.shap_values(image, sparse_labels=target)

            if 'MLP' in file_name:
                image = image.detach().cpu().numpy()
            else:
                image = undo_preprocess_input_function(image).detach().cpu().numpy()

            if not os.path.exists(file_name):
                os.mkdir(file_name)
            for bth in range(image.shape[0]):
                img = image[bth]
                f_name = file_name + 'img_' + str(int(batch_num*self.dataloader.batch_size)+bth)
                visualize(image=img, saliency_map=saliency_map[bth], filename=f_name, method_name=method_name)

    def sanity_inspection(self, num_vis):
        saliency_norm_lst = []
        for batch_num, (image, label) in enumerate(self.dataloader):
            if (batch_num*image.shape[0]) >= num_vis:
                break
            image = image.cuda()
            target = label.cuda()

            saliency_map = self.explainer.shap_values(image, sparse_labels=target)
            saliency_map = saliency_map.data.cpu().numpy()
            saliency_map = normalize_saliency_map(saliency_map.detach())

            saliency_norm_lst.append(saliency_map)
        return np.array(saliency_norm_lst)

    def sensitivity_n(self, baseline_name, ratio_lst, centers=None, post_hoc='abs'):
        start = time.time()
        n_examples = 0
        pcc_lst = [[] for _ in range(len(ratio_lst))]

        for batch_num, (image, label) in enumerate(self.dataloader):
            image = image.cuda()
            target = label.cuda()
            batch_size = image.shape[0]

            output = self.model(image).detach()
            _, predicted = torch.max(output.data, 1)
            n_examples += batch_size

            if centers is not None:
                # ---------------- LPI ---------------------
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in range(output.shape[0])]
                saliency_map = self.explainer.shap_values(image, sparse_labels=target, centers=clu_lst)
            else:
                saliency_map = self.explainer.shap_values(image, sparse_labels=target)

            if post_hoc == 'abs':
                saliency_map = torch.sum(torch.abs(saliency_map), dim=1, keepdim=True)
            elif post_hoc == 'sum':
                saliency_map = torch.sum(saliency_map, dim=1, keepdim=True)

            self.model.eval()

            results_all = []
            for q_ind, q_ratio in enumerate(ratio_lst):
                attr_lst = []
                output_diff_lst = []
                for sample_ind in range(10):
                    mask = torch.ones(saliency_map.shape).cuda()
                    mask *= q_ratio
                    mask = torch.bernoulli(mask)

                    sum_attr = torch.sum((saliency_map * mask).view([saliency_map.size(0), -1]), -1)

                    if baseline_name == 'rand':
                        rand_ref = torch.rand(*image.shape).cuda()
                        img_ref = torch.where(mask==1., rand_ref, image)
                    elif baseline_name == 'mean':
                        mean_ref = (image*(1.-mask)).view((image.shape[0], -1)) / torch.sum(mask.view((mask.shape[0], -1)))
                        img_ref = mean_ref.view(*image.shape)
                    elif baseline_name == 'zero':
                        img_ref = image * (1.-mask)

                    output_pert = self.model(img_ref).detach()
                    indices_tensor = torch.cat([torch.arange(0, output.shape[0]).cuda().unsqueeze(1), target.unsqueeze(1)], dim=1)
                    output_pert = gather_nd(output_pert, indices_tensor)
                    output_ori = gather_nd(output, indices_tensor)
                    output_diff = output_ori - output_pert

                    attr_lst.append(sum_attr.unsqueeze(0))
                    output_diff_lst.append(output_diff.unsqueeze(0))
                attr_tensor = torch.cat(attr_lst, dim=0)
                output_diff_tensor = torch.cat(output_diff_lst, dim=0)
                results_all.append([attr_tensor, output_diff_tensor])

            for r_ind , res in enumerate(results_all):
                attr_tensor, output_diff_tensor = res[0], res[1]
                # ------------------- computing pcc correlation -----------------
                for i in range(output.shape[0]):
                    attr_temp, out_temp = attr_tensor[:, i].reshape(-1), output_diff_tensor[:, i].reshape(-1)
                    from scipy.stats import pearsonr as pr
                    pr_corr, _ = pr(np.array(attr_temp.cpu()).flatten(), np.array(out_temp.cpu()).flatten())
                    if not np.isnan(pr_corr):
                        pcc_lst[r_ind].append(pr_corr)
        pcc_arr = np.array(pcc_lst)
        pc_mean = np.around(np.mean(pcc_arr, axis=1), 3)
        pc_std = np.around(np.std(pcc_arr, axis=1), 3)

        self.log(list(pc_mean))
        self.log(list(pc_std))
        end = time.time()
        self.log('\ttime: \t{:.3f}'.format(end - start))
