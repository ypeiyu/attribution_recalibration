import os
import argparse
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision import models
from utils.settings import img_size_dict
from utils.preprocess import mean_std_dict
from saliency_methods import RandomBaseline, Gradients, SmoothGrad, FullGrad, IntegratedGradients, ExpectedGradients,\
    AGI, IntGradUniform, IntGradSG, IntGradSQ, GradCAM
from evaluator import Evaluator
from networks.MLP import Model

from torch.utils.data import DataLoader
import torchvision


from utils.settings import parser_choices, parser_default

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-batch_size', type=int,
                    default=parser_default['batch_size'])
parser.add_argument('-attr_method', type=str,
                    choices=parser_choices['attr_method'],
                    default=parser_default['attr_method'])
parser.add_argument('-model', type=str,
                    choices=parser_choices['model'],
                    default=parser_default['model'])
parser.add_argument('-dataset', type=str,
                    choices=parser_choices['dataset'],
                    default=parser_default['dataset'])
parser.add_argument('-metric', type=str,
                    choices=parser_choices['metric'],
                    default=parser_default['metric'])
parser.add_argument('-k', type=int,
                    default=parser_default['k'])
parser.add_argument('-bg_size', type=int,
                    default=parser_default['bg_size'])
parser.add_argument('-est_method', type=str,
                    choices=parser_choices['est_method'],
                    default=parser_default['est_method'])
parser.add_argument('-exp_obj', type=str,
                    choices=parser_choices['exp_obj'],
                    default=parser_default['exp_obj'])
args = parser.parse_args()


def load_explainer(model, **kwargs):
    method_name = kwargs['method_name']
    if method_name == 'Random':
        random = RandomBaseline()
        return random
    # -------------------- gradient based -------------------------
    elif method_name == 'InputGrad':
        input_grad = Gradients(model, exp_obj=kwargs['exp_obj'])
        return input_grad
    elif method_name == 'FullGrad':
        im_size = img_size_dict[kwargs['dataset_name']]
        full_grad = FullGrad(model, exp_obj=kwargs['exp_obj'], im_size=im_size)
        return full_grad
    elif method_name == 'SmoothGrad':
        smooth_grad = SmoothGrad(model, bg_size=kwargs['bg_size'], exp_obj=kwargs['exp_obj'], std_spread=0.15)
        return smooth_grad

    # -------------------- integration based -------------------------
    elif method_name == 'IntGrad':
        integrated_grad = IntegratedGradients(model, k=kwargs['k'], exp_obj=kwargs['exp_obj'], dataset_name=kwargs['dataset_name'])
        return integrated_grad
    elif method_name == 'ExpGrad':
        expected_grad = ExpectedGradients(model, k=kwargs['k'], bg_size=kwargs['bg_size'], bg_dataset=kwargs['bg_dataset'],
                                          batch_size=kwargs['bg_batch_size'], random_alpha=kwargs['random_alpha'],
                                          est_method=kwargs['est_method'], exp_obj=kwargs['exp_obj'])
        return expected_grad

    # -------------------- IG based -------------------------
    elif method_name == 'IG_Uniform':
        int_grad_uni = IntGradUniform(model, k=kwargs['k'], bg_size=kwargs['bg_size'], random_alpha=kwargs['random_alpha'],
                                      est_method=kwargs['est_method'], exp_obj=kwargs['exp_obj'], dataset_name=kwargs['dataset_name'])
        return int_grad_uni
    elif method_name == 'IG_SG':
        int_grad_sg = IntGradSG(model, k=kwargs['k'], bg_size=kwargs['bg_size'], random_alpha=kwargs['random_alpha'],
                                est_method=kwargs['est_method'], exp_obj=kwargs['exp_obj'])
        return int_grad_sg
    elif method_name == 'IG_SQ':
        int_grad_sq = IntGradSQ(model, k=kwargs['k'], bg_size=kwargs['bg_size'], random_alpha=kwargs['random_alpha'],
                                est_method=kwargs['est_method'], exp_obj=kwargs['exp_obj'])
        return int_grad_sq

    elif method_name == 'AGI':
        agi = AGI(model, k=kwargs['k'], top_k=kwargs['top_k'], est_method=kwargs['est_method'],
                  exp_obj=kwargs['exp_obj'], dataset_name=kwargs['dataset_name'])
        return agi
    elif method_name == 'GradCAM':
        grad_cam = GradCAM(model, exp_obj=kwargs['exp_obj'])
        return grad_cam

    else:
        raise NotImplementedError('%s is not implemented.' % method_name)


def load_dataset(dataset_name, test_batch_size):
    # ---------------------------- imagenet train ---------------------------
    if 'imagenet' in dataset_name:
        img_size = img_size_dict[dataset_name][1]
        mean, std = mean_std_dict[dataset_name]
        imagenet_train_dataset = datasets.ImageNet(
            root='datasets',
            split='train',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))

        # ---------------------------- imagenet eval ---------------------------
        imagenet_val_dataset = datasets.ImageNet(
            root='datasets',
            split='val',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))

        imagenet_val_loader = torch.utils.data.DataLoader(
            imagenet_val_dataset, batch_size=test_batch_size,
            shuffle=False, num_workers=4, pin_memory=False)

        return imagenet_train_dataset, imagenet_val_loader
    elif dataset_name == 'mnist':
        # -------------------------- MNIST dataset ----------------------------------------
        data_pth = 'datasets/mnist'
        mnist_tr_dataset = torchvision.datasets.MNIST(data_pth,
                                                      train=True,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                      ]),
                                                      download=True)
        mnist_te_dataset = torchvision.datasets.MNIST(data_pth,
                                                      train=False,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                      ]),
                                                      download=True)

        mnist_te_loader = DataLoader(mnist_te_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

        return mnist_tr_dataset, mnist_te_loader

    elif dataset_name == 'cifar10':
        # -------------------------- CIFAR-10 dataset ----------------------------------------
        mean, std = mean_std_dict[dataset_name]
        cifar10_tr_dataset = datasets.CIFAR10('datasets/cifar10',
                                              train=True,
                                              transform=transforms.Compose([
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std),
                                              ]),
                                              download=True)
        cifar10_te_dataset = datasets.CIFAR10('datasets/cifar10',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean, std=std),
                                              ]),
                                              download=True)
        cifar10_te_loader = DataLoader(cifar10_te_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
        return cifar10_tr_dataset, cifar10_te_loader
    elif dataset_name == 'cifar100':
        mean, std = mean_std_dict[dataset_name]

        cifar100_tr_dataset = datasets.CIFAR100('datasets/cifar100',
                                                train=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std),
                                                ]),
                                                download=True)
        cifar100_te_dataset = datasets.CIFAR100('datasets/cifar100',
                                                train=False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=mean, std=std),
                                                ]),
                                                download=True)
        cifar100_te_loader = DataLoader(cifar100_te_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
        return cifar100_tr_dataset, cifar100_te_loader


def evaluate(method_name, model_name, dataset_name, metric, k=None, bg_size=None, est_method='vanilla', exp_obj='logit'):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'MLP':
        model = Model(i_c=1, n_c=10)
        model_pth = 'saved_models/MLP/checkpoint_MNIST_MLP_Permuted_best.pth'
        pretrained_model = torch.load(model_pth)
        model.load_state_dict(pretrained_model, strict=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)

    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    model.eval()

    # =================== load train dataset & test loader ========================
    test_bth = args.batch_size

    train_dataset, test_loader = load_dataset(dataset_name=dataset_name, test_batch_size=test_bth)

    # =================== load explainer ========================
    explainer_args = {
        'Random': {'method_name': method_name},
        'InputGrad': {'method_name': method_name, 'exp_obj': exp_obj},
        'GradCAM': {'method_name': method_name, 'exp_obj': exp_obj},

        'FullGrad': {'method_name': method_name, 'exp_obj': exp_obj, 'dataset_name': dataset_name},
        'SmoothGrad': {'method_name': method_name, 'bg_size':bg_size, 'exp_obj': exp_obj},

        'IntGrad': {'method_name': method_name, 'k': k, 'exp_obj': exp_obj, 'dataset_name': dataset_name},
        'ExpGrad': {'method_name': method_name, 'k': k, 'bg_size': bg_size, 'bg_dataset': train_dataset,
                    'bg_batch_size': test_bth, 'random_alpha': True, 'est_method': est_method, 'exp_obj': exp_obj},

        'IG_Uniform': {'method_name': method_name, 'k': k, 'bg_size': bg_size, 'random_alpha': False,
                       'est_method': est_method, 'exp_obj': exp_obj, 'dataset_name': dataset_name},
        'IG_SG': {'method_name': method_name, 'k': k, 'bg_size': bg_size, 'random_alpha': False,
                  'est_method': est_method, 'exp_obj': exp_obj},
        'IG_SQ': {'method_name': method_name, 'k': k, 'bg_size': bg_size, 'random_alpha': False,
                  'est_method': est_method, 'exp_obj': exp_obj},
        'AGI': {'method_name': method_name, 'k': k, 'top_k': bg_size, 'est_method': est_method, 'exp_obj': exp_obj,
                'dataset_name': dataset_name},
    }

    explainer = load_explainer(model=model, **explainer_args[method_name])
    evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)

    if metric == 'DiffID':
        if method_name == 'LPI':
            cent_num = explainer_args['LPI']['num_centers']
            centers = None
            if cent_num > 1:
                centers = np.load(
                    'dataset_distribution/' + model_name +
                    '/kmeans_' + model_name + '_n' + str(cent_num) + '_centers.npy')
            evaluator.DiffID(ratio_lst=[step * 0.1 for step in range(1, 10)], centers=centers)
        else:
            evaluator.DiffID(ratio_lst=[step * 0.1 for step in range(1, 10)])

    elif metric == 'visualize':
        num_vis_samples = 50
        f_name = 'exp_fig/' + method_name + '_' + model_name + '/'
        evaluator.visual_inspection(file_name=f_name, num_vis=num_vis_samples, method_name=method_name)

    elif metric == 'sensitivity_n':
        baseline_names = ['rand', 'zero']
        ratio_lst = [step * 0.1 for step in range(1, 10)]
        post_hoc_lst = ['abs', 'sum']

        for b_name in baseline_names:
            for post_hoc in post_hoc_lst:
                evaluator.sensitivity_n(baseline_name=b_name, ratio_lst=ratio_lst, post_hoc=post_hoc)

    elif metric == 'sanity_check':
        if dataset_name == 'mnist':

            saliency_map_lst = []
            for model_name in ['checkpoint_MNIST_MLP_best.pth', 'checkpoint_MNIST_MLP_Permuted_best.pth']:
                model_pth = 'saved_models/MLP/' + model_name
                pretrained_model = torch.load(model_pth)
                model.load_state_dict(pretrained_model, strict=True)
                model.eval()

                explainer = load_explainer(model=model, **explainer_args[method_name])
                evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)
                num_checks = 10000

                saliency_map_lst.append(evaluator.sanity_inspection(num_vis=num_checks))
            file_name = 'exp_fig/Fig_sanity_data/' + method_name + '.npy'
            np.save(file_name, np.array(saliency_map_lst))

        else:
            # --------------------- Sanity checks ----------------------
            from utils.utils import inception_block_names
            import torch.nn as nn
            saliency_map_lst = []
            randomization_order = inception_block_names()
            for ind in range(len(randomization_order)):
                model = models.inception_v3(pretrained=True, aux_logits=True)
                model = model.to('cuda')
                model = torch.nn.DataParallel(model)
                model.eval()

                rand_order = randomization_order[:ind+1]
                for name, param in model.named_parameters():
                    for p_name in rand_order:
                        if p_name in name:
                            std, mean = torch.std_mean(param.data)
                            std_factor = 1.0
                            std *= std_factor
                            if name.endswith('conv.weight') or name.endswith('bn.bias') or name.endswith('fc.weight') or name.endswith('fc.bias'):
                                param.data = param.data.normal_(mean=mean, std=std)

                            break
                model = model.to('cuda')
                model = torch.nn.DataParallel(model)
                model.eval()
                # ------------ experimentor ---------------
                explainer = load_explainer(model=model, **explainer_args[method_name])
                evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)
                num_checks = 20
                num_vis_checks = 50
                if model_name == 'inception_v3_vis':
                    param_ind_f = str(ind).zfill(2)
                    evaluator.visual_inspection(file_name='exp_fig/Figure_sanity/'+method_name + '/',
                                                num_vis=num_vis_checks, method_name=param_ind_f+method_name + '_' + str(randomization_order[ind]))
                if model_name == 'inception_v3':
                    saliency_map_lst.append(evaluator.sanity_inspection(num_vis=num_checks))

            if model_name == 'inception_v3':
                file_name = 'exp_data/' + method_name+'.npy'
                saliency_map_set = np.array(saliency_map_lst)
                print(saliency_map_set.shape)
                np.save(file_name, saliency_map_set)

    else:
        raise NotImplementedError('%s is not implemented.' % metric)


if __name__ == '__main__':
    evaluate(method_name=args.attr_method, model_name=args.model, dataset_name=args.dataset, metric=args.metric,
             k=args.k, bg_size=args.bg_size, est_method=args.est_method, exp_obj=args.exp_obj)
