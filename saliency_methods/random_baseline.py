#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement SmoothGrad saliency algorithm

    Original paper:
    Smilkov, Daniel, et al. "Smoothgrad: removing noise by adding noise." 
    arXiv preprint arXiv:1706.03825 (2017).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomBaseline():
    """
    Compute smoothgrad 
    """

    def __init__(self):
        pass

    def shap_values(self, image, sparse_labels=None):
        return torch.rand(*image.shape).cuda()

