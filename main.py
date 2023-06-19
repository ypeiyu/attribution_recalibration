import os
import argparse
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision import models
from utils.settings import img_size
from utils.preprocess import LSVRC_mean, LSVRC_std
mean, std = LSVRC_mean, LSVRC_std
from saliency_methods import RandomBaseline, Gradients, SmoothGrad, FullGrad, IntegratedGradients, ExpectedGradients, AGI, IntGradUniform, IntGradSG, IntGradSQ
from evaluator import Evaluator
from networks.MLP import Model

from torch.utils.data import DataLoader
import torchvision

cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


from utils.settings import parser_choices, parser_default

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-attr_method', type=str, required=False,
                    choices=parser_choices['attr_method'],
                    default=parser_default['attr_method'])
parser.add_argument('-model', type=str, required=False,
                    choices=parser_choices['model'],
                    default=parser_default['model'])
parser.add_argument('-dataset', type=str, required=False,
                    choices=parser_choices['dataset'],
                    default=parser_default['dataset'])
parser.add_argument('-metric', type=str, required=False,
                    choices=parser_choices['metric'],
                    default=parser_default['metric'])
parser.add_argument('-k', type=int, required=False,
                    default=parser_default['k'])
parser.add_argument('-bg_size', type=int, required=False,
                    default=parser_default['bg_size'])
args = parser.parse_args()


def load_explainer(model, **kwargs):
    method_name = kwargs['method_name']
    if method_name == 'Random':
        print('================= Random Baseline ==================')
        random = RandomBaseline()
        return random
    elif method_name == 'InputGrad':
        print('================= Input Gradients ==================')
        input_grad = Gradients(model)
        return input_grad
    elif method_name == 'FullGrad':
        print('================= Full Gradients ==================')
        full_grad = FullGrad(model)
        return full_grad
    elif method_name == 'SmoothGrad':
        print('================= Smooth Gradients ==================')
        num_samples = kwargs['num_samples']
        smooth_grad = SmoothGrad(model, num_samples=num_samples)
        return smooth_grad
    elif method_name == 'AGI':
        print('================= AGI ==================')
        k = kwargs['k']
        top_k = kwargs['top_k']
        cls_num = kwargs['cls_num']
        agi = AGI(model, k=k, top_k=top_k, cls_num=cls_num)
        return agi
    elif method_name == 'IntGrad':
        print('============================ Integrated Gradients ============================')
        k = kwargs['k']
        integrated_grad = IntegratedGradients(model, k=k)
        return integrated_grad
    elif method_name == 'ExpGrad':
        print('============================ Expected Gradients ============================')
        k = kwargs['k']
        bg_size = kwargs['bg_size']
        train_dataset = kwargs['train_dataset']
        test_batch_size = kwargs['test_batch_size']
        random_alpha = kwargs['random_alpha']
        expected_grad = ExpectedGradients(model, k=k, bg_dataset=train_dataset, bg_size=bg_size, batch_size=test_batch_size, random_alpha=random_alpha)
        return expected_grad
    elif method_name == 'IG_Uniform':
        print('============================ IG_Uniform ============================')
        k = kwargs['k']
        bg_size = kwargs['bg_size']
        int_grad_uni = IntGradUniform(model, k=k, bg_size=bg_size)
        return int_grad_uni
    elif method_name == 'IG_SG':
        print('============================ IG_SG ============================')
        k = kwargs['k']
        bg_size = kwargs['bg_size']
        int_grad_sg = IntGradSG(model, k=k, bg_size=bg_size)
        return int_grad_sg
    elif method_name == 'IG_SQ':
        print('============================ IG_SQ ============================')
        k = kwargs['k']
        bg_size = kwargs['bg_size']
        int_grad_sq = IntGradSQ(model, k=k, bg_size=bg_size)
        return int_grad_sq

    else:
        raise NotImplementedError('%s is not implemented.' % method_name)


def load_dataset(dataset_name, test_batch_size):
    # ---------------------------- imagenet train ---------------------------
    if 'ImageNet' in dataset_name:
        imagenet_train_dataset = datasets.ImageNet(
            root='datasets',
            split='train',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))
        if dataset_name == 'ImageNet':
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
    elif dataset_name == 'MNIST':
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
    elif dataset_name == 'CIFAR-10':
        # -------------------------- CIFAR-10 dataset ----------------------------------------
        cifar10_tr_dataset = datasets.CIFAR10('datasets/cifar10',
                                              train=True,
                                              transform=transforms.Compose([
                                                  # transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                                              ]),
                                              download=True)
        cifar10_te_dataset = datasets.CIFAR10('datasets/cifar10',
                                              train=False,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                                              ]),
                                              download=True)
        cifar10_te_loader = DataLoader(cifar10_te_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
        return cifar10_tr_dataset, cifar10_te_loader
    elif dataset_name == 'CIFAR-100':
        cifar100_tr_dataset = datasets.CIFAR100('datasets/cifar100',
                                                train=True,
                                                transform=transforms.Compose([
                                                    # transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
                                                ]),
                                                download=True)
        cifar100_te_dataset = datasets.CIFAR100('datasets/cifar100',
                                                train=False,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
                                                ]),                                                download=True)
        cifar100_te_loader = DataLoader(cifar100_te_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
        return cifar100_tr_dataset, cifar100_te_loader


def evaluate(method_name, model_name, dataset_name, metric, k=None, bg_size=None):
    model_name = model_name  # vgg16 resnet34
    method_name = method_name

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'MLP':
        model = Model(i_c=1, n_c=10)  # load model
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)


    if model_name in ['vgg16', 'resnet34', 'MLP']:
        model = model.to('cuda')
        model = torch.nn.DataParallel(model)
        model.eval()
    if model_name == 'MLP':
        # model_pth = 'saved_models/MLP/checkpoint_MNIST_MLP_best.pth'
        model_pth = 'saved_models/MLP/checkpoint_MNIST_MLP_Permuted_best.pth'

        pretrained_model = torch.load(model_pth)
        model.load_state_dict(pretrained_model, strict=True)
        model.eval()

    # =================== load train dataset & test loader ========================
    test_batch_size = {'ImageNet':
                           {'resnet34': {'Random': 150, 'InputGrad': 70, 'FullGrad': 60, 'AGI': 80, 'ExpGrad': 40, 'IG_Uniform': 20,
                                       'IntGrad': 50, 'SmoothGrad': 200, 'LPI': 40},
                            'vgg16': {'Random': 160, 'InputGrad': 60, 'FullGrad': 60, 'AGI': 40, 'ExpGrad': 40, 'IG_Uniform': 10,
                                       'IntGrad': 40, 'SmoothGrad': 120, 'LPI': 1},
                            'inception_v3': {'InputGrad': 20, 'ExpGrad': 60, 'ExpGrad_new': 60, 'IntGrad': 60, 'AGI': 20},
                            'inception_v3_vis': {'InputGrad': 20, 'ExpGrad': 20, 'ExpGrad_new': 20, 'IntGrad': 20, 'AGI': 20}
                            },
                       'MNIST':
                           {'MLP': {'Random': 150, 'InputGrad': 150, 'FullGrad': 60, 'AGI': 80, 'ExpGrad': 120, 'IG_Uniform': 120,
                                    'IntGrad': 120, 'SmoothGrad': 400, 'LPI': 120}},
                       'CIFAR-10':
                           {'preactresnet': {'Random': 256, 'InputGrad': 256, 'FullGrad': 256, 'AGI': 512, 'ExpGrad': 64,
                                             'IG_Uniform': 512, 'IntGrad': 256, 'SmoothGrad': 256, 'LPI': 256}},
                       'CIFAR-100':
                           {'preactresnet': {'Random': 256, 'InputGrad': 256, 'FullGrad': 256, 'AGI': 512, 'ExpGrad': 64,
                                             'IG_Uniform': 512, 'IntGrad': 256, 'SmoothGrad': 256, 'LPI': 256}}
                       }
    dataset_n = dataset_name
    if 'ImageNet' in dataset_name:
        dataset_n = 'ImageNet'
    test_bth = test_batch_size[dataset_n][model_name][method_name]
    train_dataset, test_loader = load_dataset(dataset_name=dataset_name, test_batch_size=test_bth)

    # =================== load explainer ========================
    explainer_args = {
        'ExpGrad': {'method_name': method_name, 'k': k, 'bg_size': bg_size, 'train_dataset': train_dataset,
                    'test_batch_size': test_bth, 'random_alpha': False},
        'IG_Uniform': {'method_name': method_name, 'k': k, 'bg_size': bg_size, 'test_batch_size': test_bth},

    }

    if metric == 'sanity_check':
        if dataset_name == 'MNIST':

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

                rand_order = randomization_order[:ind+1]  # [:ind+1]
                for name, param in model.named_parameters():
                    for p_name in rand_order:
                        if p_name in name:
                            std, mean = torch.std_mean(param.data)
                            std_factor = 1.0  # 0.2
                            std *= std_factor  # 1.0forall 0.2forAGI
                            if name.endswith('conv.weight') or name.endswith('bn.bias') or name.endswith('fc.weight') or name.endswith('fc.bias'):
                                param.data = param.data.normal_(mean=mean, std=std)

                            break
                model = model.to('cuda')
                model = torch.nn.DataParallel(model)
                model.eval()
                # ------------ experimentor ---------------
                explainer = load_explainer(model=model, **explainer_args[method_name])
                evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)
                num_checks = 20  # 20
                num_vis_checks = 50  # 20
                if model_name == 'inception_v3_vis':
                    param_ind_f = str(ind).zfill(2)
                    evaluator.visual_inspection(file_name='exp_fig/Figure_sanity/'+method_name + '/', num_vis=num_vis_checks, method_name=param_ind_f+method_name + '_' + str(randomization_order[ind]))
                if model_name == 'inception_v3':
                    saliency_map_lst.append(evaluator.sanity_inspection(num_vis=num_checks))

            if model_name == 'inception_v3':
                file_name = 'exp_data/' + method_name+'.npy'
                saliency_map_set = np.array(saliency_map_lst)
                print(saliency_map_set.shape)
                np.save(file_name, saliency_map_set)

    if metric == 'pixel_perturb':
        explainer = load_explainer(model=model, **explainer_args[method_name])
        evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)

        # --------------------- perturb experiments ----------------------
        if method_name == 'LPI':
            cent_num = explainer_args['LPI']['num_centers']
            centers = None
            if cent_num > 1:
                centers = np.load(
                    '/home/peiyu/PROJECT/grad-saliency-master/dataset_distribution/' + model_name +
                    '/kmeans_' + model_name + '_n' + str(cent_num) + '_centers.npy')
            evaluator.DiffID(baseline_name='mean', q_ratio_lst=[step * 0.1 for step in range(1, 10)], centers=centers)
        else:
            evaluator.DiffID(baseline_name='mean', q_ratio_lst=[step * 0.1 for step in range(1, 10)])

    if metric == 'visualize':
        num_vis_samples = 500
        f_name = 'exp_fig/' + method_name + '_' + model_name + '_sg_vis/'
        explainer = load_explainer(model=model, **explainer_args[method_name])
        evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)
        evaluator.visual_inspection(file_name=f_name, num_vis=num_vis_samples, method_name=method_name)


if __name__ == '__main__':
    evaluate(method_name=args.attr_method, model_name=args.model, dataset_name=args.dataset, metric=args.metric,
             k=args.k, bg_size=args.bg_size)
