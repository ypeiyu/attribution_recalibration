import random
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import undo_preprocess_input_function
from utils import visualize

import functools
import operator


def single_img_inspection(img, file_name):
    image = undo_preprocess_input_function(img).detach().cpu().numpy()
    image = image[0] * 255
    image = image.astype(np.uint8)
    image = np.transpose(image, [1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_name, image)


def normalize_saliency_map(saliency_map, absolute=True):
    if absolute:
        saliency_map = torch.abs(saliency_map)
    saliency_map = saliency_map.sum(1, keepdim=True)

    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.min(1, keepdim=True)
    saliency_map = saliency_map - temp.unsqueeze(1).unsqueeze(1)
    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.max(1, keepdim=True)
    saliency_map = saliency_map / (temp.unsqueeze(1).unsqueeze(1) + 1e-10)
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
        self.n_examples = 0

        self.n_correct = 0
        self.n_pert_correct = 0
        self.pos_num = 0

        self.NL_difference = []

        self.model.eval()

    def DiffID(self, baseline_name, q_ratio_lst, centers=None):
        log = self.log
        self.n_examples = 0
        self.n_correct = 0
        self.pos_num = 0
        n_pert_correct_top_lst = [0 for _ in range(len(q_ratio_lst))]
        n_pert_correct_bot_lst = [0 for _ in range(len(q_ratio_lst))]
        start_loc_top = [0 for _ in range(len(q_ratio_lst))]
        start_loc_bot = [0 for _ in range(len(q_ratio_lst))]
        # logit_change_top_lst = [torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(len(q_ratio_lst))]
        # logit_change_bot_lst = [torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(len(q_ratio_lst))]

        ####################################
        ##### test for single images #######
        ####################################
        # logit_change_top_lst = [torch.zeros(1) for _ in range(len(q_ratio_lst))]
        # logit_change_bot_lst = [torch.zeros(1) for _ in range(len(q_ratio_lst))]
        logit_change_top_lst = [torch.zeros(10000) for _ in range(len(q_ratio_lst))]
        logit_change_bot_lst = [torch.zeros(10000) for _ in range(len(q_ratio_lst))]
        start = time.time()
        for batch_num, (image, label) in enumerate(self.dataloader):
            image = image.cuda()
            target = label.cuda()

            ####################################
            ##### test for single images #######
            ####################################
            # image_id = 1
            # # target = target * 0 + image_id  # figure 1
            # if image_id in target:
            #     index = (target == image_id).nonzero(as_tuple=True)
            #     target = torch.tensor([image_id]).cuda()
            #     image = image[int(index[0]):int(index[0])+1]
            # else: continue

            batch_size = image.shape[0]

            output = self.model(image).detach()
            _, predicted = torch.max(output.data, 1)
            self.n_correct += (predicted == target).sum().item()
            self.n_examples += batch_size

            if centers is not None:
                # ------------------ LPI -------------------------
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in range(output.shape[0])]
                saliency_map = self.explainer.shap_values(image, sparse_labels=target, centers=clu_lst)
            else:
                # saliency_map, pos_num = self.explainer.shap_values(image, sparse_labels=target)
                # self.pos_num += pos_num
                saliency_map = self.explainer.shap_values(image, sparse_labels=target)

            # -------------------------------- saliency map normalization -----------------------------------------
            saliency_map = normalize_saliency_map(saliency_map.detach())

            self.model.eval()
            zero_tensor = torch.zeros(*image[0].shape).cuda()
            perturb_img_batch = torch.zeros(*image.shape).cuda()
            for q_ind, q_ratio in enumerate(q_ratio_lst):
                # ========================================================================
                for perturb_top in [False, True]:

                    q_r = 1-q_ratio if perturb_top else q_ratio  # is_top 0.9 else 0.1
                    threshold = torch.quantile(saliency_map.reshape(saliency_map.shape[0], -1), q=q_r, dim=1, interpolation='midpoint')  # < top90

                    for b_num in range(batch_size):
                        sat = image.detach()[b_num] if perturb_top else zero_tensor
                        dis_sat = zero_tensor if perturb_top else image.detach()[b_num]

                        perturb = torch.where(saliency_map[b_num] < threshold[b_num], sat, dis_sat).unsqueeze(0)
                        mean_insertion = torch.sum(perturb) / (torch.count_nonzero(perturb) + 1e-10)
                        dis_sat_t = dis_sat+mean_insertion if perturb_top else dis_sat
                        sat_t = sat if perturb_top else sat+mean_insertion
                        perturb_img = torch.where(saliency_map[b_num] < threshold[b_num], sat_t, dis_sat_t).unsqueeze(0)

                        # flag = 'insertion' if perturb_top else 'deletion'
                        # single_img_inspection(perturb_img, file_name='exp_fig/img_inspection/img' + str(
                        #     self.n_examples + b_num) + '_' + flag + str(q_r) + '.jpg')

                        perturb_img_batch[b_num] = perturb_img

                    output_pert = self.model(perturb_img_batch).detach()

                    _, predicted_pert = torch.max(output_pert.data, 1)
                    if perturb_top:
                        n_pert_correct_top_lst[q_ind] += (predicted_pert == target).sum().item()
                        for bth in range(batch_size):
                            t = target[bth]
                            logit_change_top_lst[q_ind][start_loc_top[q_ind]:(start_loc_top[q_ind]+1)] = output_pert[bth, t]/output[bth, t]
                            start_loc_top[q_ind] += 1
                    else:
                        n_pert_correct_bot_lst[q_ind] += (predicted_pert == target).sum().item()
                        for bth in range(batch_size):
                            t = target[bth]
                            logit_change_bot_lst[q_ind][start_loc_bot[q_ind]:(start_loc_bot[q_ind]+1)] = output_pert[bth, t]/output[bth, t]
                            start_loc_bot[q_ind] += 1
        end = time.time()
        log('\ttime: \t{:.3f}'.format(end - start))
        insertion_logit = []
        insertion_acc = []

        deletion_logit = []
        deletion_acc = []

        DiffID_logit = []
        DiffID_acc = []

        for q_ind, q_ratio in enumerate(q_ratio_lst):
            # print('baseline: {} | ratio: {}'.format(baseline_name, q_ratio))

            # ========================================================================
            # log('\taccu: \t{:.3f}%'.format(self.n_correct/self.n_examples))

            # log('\t Perturb TOP')
            mean_accu_top = n_pert_correct_top_lst[q_ind]/self.n_examples
            # log('\tperturbed accu: \t{:.3f}%'.format(mean_accu_top))
            var_top, mean_top = torch.var_mean(logit_change_top_lst[q_ind], unbiased=False)
            mean_top = mean_top.item()
            # log('\toutput logit changes mean: \t{:.3f}'.format(mean_top))
            # log('\toutput logit changes variation: \t{:.3f}'.format(torch.sqrt(var_top)))
            deletion_logit.append(round(mean_top, 3))
            deletion_acc.append(round(mean_accu_top, 3))

            # og('\t Perturb BOTTOM')
            mean_accu_bot = n_pert_correct_bot_lst[q_ind]/self.n_examples
            # log('\tperturbed accu: \t{:.3f}%'.format(mean_accu_bot))
            var_bot, mean_bot = torch.var_mean(logit_change_bot_lst[q_ind], unbiased=False)
            mean_bot = mean_bot.item()
            # log('\toutput logit changes mean: \t{:.3f}'.format(mean_bot))
            # log('\toutput logit changes variation: \t{:.3f}'.format(torch.sqrt(var_bot)))
            insertion_logit.append(round(mean_bot, 3))
            insertion_acc.append(round(mean_accu_bot, 3))

            # log('\tDiff criterion')
            del_accu = mean_accu_bot - mean_accu_top
            del_logit = mean_bot - mean_top
            # log('\tDifferent accu: \t{:.3f}'.format(del_accu))
            # log('\tDifferent logit mean: \t{:.3f}'.format(del_logit))
            DiffID_logit.append(round(del_logit, 3))
            DiffID_acc.append(round(del_accu, 3))
        print('deletion 10-90 logit scores')
        print(deletion_logit)
        # print(np.mean(np.array(deletion_logit)))

        print('deletion accu scores')
        print(deletion_acc)
        print('\n')
        print('insertion 10-90 logit scores')
        print(insertion_logit)
        print('insertion accu scores')
        print(insertion_acc)
        print('\n')
        print('Diff logit scores')
        print(DiffID_logit)
        print('Diff accu scores')
        print(DiffID_acc)
        # print('valid interpolations')
        # print(self.pos_num/(224*224*3*1000))

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

    def converge_exp(self, baseline_name, q_ratio_lst, centers=None, post_hoc='abs'):
        log = self.log
        self.n_examples = 0
        self.n_correct = 0
        pcc_lst = [[] for _ in range(len(q_ratio_lst))]

        start = time.time()

        for batch_num, (image, label) in enumerate(self.dataloader):
            image = image.cuda()
            target = label.cuda()
            batch_size = image.shape[0]

            output = self.model(image).detach()
            ####################################
            ######### for softmax loss #########
            ####################################
            # output = torch.softmax(output, 1)
            _, predicted = torch.max(output.data, 1)
            self.n_correct += (predicted == target).sum().item()
            self.n_examples += batch_size

            if centers is not None:
                # ---------------- LPI ---------------------
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in range(output.shape[0])]
                saliency_map = self.explainer.shap_values(image, sparse_labels=target, centers=clu_lst)
            else:
                saliency_map = self.explainer.shap_values(image, sparse_labels=target)

            # -------------------------------- saliency map normalization -----------------------------------------
            # saliency_map = normalize_saliency_map(saliency_map.detach())
            # saliency_map = saliency_map.detach()
            if post_hoc == 'abs':
                saliency_map = torch.sum(torch.abs(saliency_map), dim=1, keepdim=True)
            else:
                saliency_map = torch.sum(saliency_map, dim=1, keepdim=True)

            self.model.eval()

            results_all = []
            for q_ind, q_ratio in enumerate(q_ratio_lst):
                attr_lst = []
                output_diff_lst = []
                for sample_ind in range(10):
                    mask = torch.ones(saliency_map.shape).cuda()
                    mask *= q_ratio
                    mask = torch.bernoulli(mask)

                    sum_attr = torch.sum((saliency_map * mask).view([saliency_map.size(0), -1]), -1)
                    # sum_attr = sum_attr.view([sum_attr.size(0), -1])
                    # sum_attr = torch.sum(sum_attr, dim=1)

                    # val = torch.mean(image[1-mask])
                    # mean_val = torch.mean()
                    # img = torch.where(mask == 1., image, mean_val)

                    if baseline_name == 'rand':
                        rand_ref = torch.rand(*image.shape).cuda()
                        ### if mnist
                        # from utils import preprocess_input_function
                        # rand_ref = preprocess_input_function(rand_ref)
                        img_ref = torch.where(mask==1., rand_ref, image)
                    elif baseline_name == 'mean':
                        mean_ref = (image*(1.-mask)).view((image.shape[0], -1)) / torch.sum(mask.view((mask.shape[0], -1)))
                        img_ref = mean_ref.view(*image.shape)
                    elif baseline_name == 'zero':
                        img_ref = image * (1.-mask)
                    output_pert = self.model(img_ref).detach()
                    ####################################
                    ######### for softmax loss #########
                    ####################################
                    # output_pert = torch.softmax(output_pert, 1)

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
                    from scipy.stats import spearmanr as spr
                    from scipy.stats import pearsonr as pr
                    # spr_corr, _ = spr(np.array(attr_temp.cpu()).flatten(), np.array(out_temp.cpu()).flatten())
                    pr_corr, _ = pr(np.array(attr_temp.cpu()).flatten(), np.array(out_temp.cpu()).flatten())
                    if not np.isnan(pr_corr):
                        pcc_lst[r_ind].append(pr_corr)
        pcc_arr = np.array(pcc_lst)
        pc_mean = np.around(np.mean(pcc_arr, axis=1), 3)
        pc_std = np.around(np.std(pcc_arr, axis=1), 3)

        print(list(pc_mean))
        print(list(pc_std))
        print()

        end = time.time()
        log('\ttime: \t{:.3f}'.format(end - start))

        # for ind, pcc in enumerate(pcc_lst):
        #     print(ind)
        #     print(pcc/self.n_examples)
        #     print('\n')

    #     self.NL_difference = []
    #     if ref_dataset is not None:
    #         ref_dataloader = DataLoader(
    #             dataset=ref_dataset,
    #             batch_size=1,
    #             shuffle=False,
    #             drop_last=False)
    #     for batch_num, (image, label) in enumerate(self.dataloader):
    #         image = image.cuda()
    #         target = label.cuda()
    #
    #         # img = cv2.imread('/home/zeyiwen/peiyu/grad-saliency-master/exp_fig/IG_2/img_85_origin.jpg')
    #         # img = np.float32(img.transpose(2, 0, 1))
    #         # img = img - img.min()
    #         # img = img / img.max()
    #         # img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
    #         # from utils import preprocess_input_function
    #         # img_tensor = preprocess_input_function(img_tensor)
    #         # image = img_tensor
    #         # target = torch.tensor([16], dtype=torch.int64).cuda()
    #         # saliency_map = self.explainer.shap_values(image, sparse_labels=target)
    #
    #         output_start = self.model(image)[:, target].detach()
    #
    #         # reference_tensor = torch.zeros(*list(image.shape)).cuda()
    #         # reference_tensor = preprocess_input_function(reference_tensor)
    #         output_end = self.model(reference_tensor)[:, target].detach()
    #
    #         change = output_end - output_start
    #         print(change)
    #
    #         # ----------------------- plot shap for specific class --------------------------------
    #         saliency_map = self.explainer.shap_values(image, sparse_labels=target)
    #         saliency_map = torch.sum(saliency_map)
    #         if isinstance(self.explainer, IntegratedGradients):
    #             output_start = self.model(image)[:, target]
    #             image = image*0.
    #             output_end = self.model(image)[:, target]
    #             output_change = output_start-output_end
    #
    #         else:
    #             output_start = self.model(image)[:, target]
    #             # output_end
    #             output_change = 0.
    #             for bth_n, (img_ref, lbl_ref) in enumerate(ref_dataloader):
    #                 # lbl_ref = lbl_ref.cuda()
    #                 img_ref = img_ref.cuda()
    #                 output_end = self.model(img_ref)[:, target]
    #                 output_change += output_start-output_end
    #             output_change = output_change/k
    #         output_change = torch.abs(output_change - saliency_map).data.detach().cpu().numpy()
    #         self.NL_difference.append(output_change)
        # print('the number of segments: {}'.format(k))
        # NL_difference_mean = np.mean(np.array(self.NL_difference))
        # NL_difference_std = np.std(np.array(self.NL_difference))
        # print('difference mean {:.4}'.format(NL_difference_mean))
        # print('difference standard variation {:.4}'.format(NL_difference_std))
        # print('----------------')
