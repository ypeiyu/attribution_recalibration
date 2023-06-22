img_size = 224

parser_choices = {
    'attr_method': ['InputGrad', 'IntGrad', 'ExpGrad', 'IG_SG', 'IG_SQ', 'IG_uniform', 'AGI', 'FullGrad',
                    'SmoothGrad', 'Random'],
    'model': ['resnet34', 'vgg16', 'MLP'],
    'dataset': ['ImageNet', 'CIFAR-10', 'CIFAT-100', 'MNIST'],
    'metric': ['pixel_perturb', 'sanity_check'],
    'est_method': ['vanilla', 'valid_ip', 'valid_ref'],
    'exp_obj': ['logit', 'prob', 'contrast'],
}

parser_default = {
    'attr_method': 'IG_SG',
    'model': 'resnet34',
    'dataset': 'ImageNet',
    'metric': 'DiffID',
    'k': 5,
    'bg_size': 20,
    'est_method': 'valid_ref',
    'exp_obj': 'logit',
}
