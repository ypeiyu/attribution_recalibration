import torch
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomBaseline():
    """
    Compute smoothgrad 
    """

    def __init__(self):
        pass

    def shap_values(self, image, sparse_labels=None):
        return torch.rand(*image.shape).cuda()

