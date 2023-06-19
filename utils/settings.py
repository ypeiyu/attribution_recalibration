img_size = 224

parser_choices = {
    'attr_method': ['InputGrad', 'IntGrad', 'ExpGrad', 'IntGradSG', 'IntGradSQ', 'IG_uniform', 'AGI', 'FullGrad',
                    'SmoothGrad', 'Random'],
    'model': ['resnet34', 'vgg16', 'MLP'],
    'dataset': ['ImageNet', 'CIFAR-10', 'CIFAT-100', 'MNIST'],
    'metric': ['pixel_perturb', 'sanity_check'],
}

parser_default = {
    'attr_method': 'IG_Uniform',
    'model': 'resnet34',
    'dataset': 'ImageNet',
    'metric': 'visualize',
    'k': 5,
    'bg_size': 10,
}
