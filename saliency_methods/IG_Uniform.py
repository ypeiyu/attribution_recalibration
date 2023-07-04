import torch
import torch.utils.data
from .IG_SG import IntGradSG
from utils import preprocess


class IntGradUniform(IntGradSG):

    def chew_input(self, input_tensor):
        """
        Calculate IG_Uniform values for the sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        shape = list(input_tensor.shape)
        shape.insert(1, self.bg_size)

        ref_lst = [preprocess(torch.rand(*input_tensor.shape)) for _ in range(self.bg_size)]
        ref = torch.cat(ref_lst)
        reference_tensor = ref.view(*shape).cuda()
        multi_ref_tensor = reference_tensor.repeat(1, self.k, 1, 1, 1)

        samples_input = self._get_samples_input(input_tensor, multi_ref_tensor)
        return input_tensor, samples_input, reference_tensor
