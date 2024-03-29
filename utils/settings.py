img_size_dict = {
    'imagenet': (3, 224, 224),
    'cifar10': (3, 32, 32),
    'cifar100': (3, 32, 32),
}

parser_choices = {
    'attr_method': ['InputGrad', 'IntGrad', 'ExpGrad', 'IG_SG', 'IG_SQ', 'IG_Uniform', 'AGI', 'FullGrad',
                    'SmoothGrad', 'Random'],
    'model': ['resnet34', 'vgg16', 'MLP'],
    'dataset': ['imagenet', 'cifar10', 'cifar100', 'mnist'],
    'metric': ['DiffID', 'visualize', 'sensitivity_n', 'sanity_check'],
    'est_method': ['vanilla', 'valid_ip', 'valid_ref'],
    'exp_obj': ['logit', 'prob', 'contrast'],
}

parser_default = {
    'batch_size': 64,
    'attr_method': 'IG_Uniform',  #
    'model': 'resnet34',
    'dataset': 'imagenet',
    'metric': 'visualize',
    'k': 5,
    'bg_size': 10,
    'est_method': 'valid_ip',
    'exp_obj': 'logit',
}
