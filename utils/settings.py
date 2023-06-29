img_size = 224

parser_choices = {
    'attr_method': ['InputGrad', 'IntGrad', 'ExpGrad', 'IG_SG', 'IG_SQ', 'IG_uniform', 'AGI', 'FullGrad',
                    'SmoothGrad', 'Random'],
    'model': ['resnet34', 'vgg16', 'MLP'],
    'dataset': ['ImageNet', 'CIFAR-10', 'CIFAT-100', 'MNIST'],
    'metric': ['DiffID', 'visualize', 'sensitivity_n', 'sanity_check'],
    'est_method': ['vanilla', 'valid_ip', 'valid_ref'],
    'exp_obj': ['logit', 'prob', 'contrast'],
}

parser_default = {
    'attr_method': 'IG_SG',  # IG_Uniform
    'model': 'resnet34',
    'dataset': 'ImageNet',
    'metric': 'visualize',
    'k': 5,
    'bg_size': 20,
    'est_method': 'vanilla',
    'exp_obj': 'contrast',
}
