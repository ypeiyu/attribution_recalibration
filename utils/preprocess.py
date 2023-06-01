import torch

LSVRC_mean = (0.485, 0.456, 0.406)
LSVRC_std = (0.229, 0.224, 0.225)

cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


def preprocess(x, mean=LSVRC_mean, std=LSVRC_std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x)


def undo_preprocess(x, mean=LSVRC_mean, std=LSVRC_std):
    assert x.size(1) == 3

    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x)
