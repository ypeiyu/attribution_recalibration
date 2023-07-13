import torch

mean_std_dict = {
    'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
    'cifar10': [(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)],
}


def preprocess(x, d_name='imagenet'):
    mean_std = mean_std_dict[d_name]
    mean, std = mean_std[0], mean_std[1]
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_preprocess(x, d_name='imagenet'):
    mean_std = mean_std_dict[d_name]
    mean, std = mean_std[0], mean_std[1]
    assert x.size(1) == 3

    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y
