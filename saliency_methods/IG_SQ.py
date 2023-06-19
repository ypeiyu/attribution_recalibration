import torch
import torch.utils.data

from .IntGrad import IntGrad


class IntGradSQ(IntGrad):

    def chew_input(self, input_tensor):
        """
        Calculate IG_SQ values for the sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        shape = list(input_tensor.shape)
        shape.insert(1, self.k * self.bg_size)

        from utils.preprocess import preprocess, undo_preprocess
        input_tensor = undo_preprocess(input_tensor)
        std_factor = 0.15
        std_dev = std_factor * (input_tensor.max().item() - input_tensor.min().item())
        ref_lst = [torch.normal(mean=torch.zeros_like(input_tensor), std=std_dev).cuda() for _ in
                   range(self.k * self.bg_size)]
        reference_tensor = torch.cat(ref_lst).view(*shape)
        reference_tensor += input_tensor.unsqueeze(1)
        reference_tensor = torch.clamp(reference_tensor, min=0., max=1.)
        reference_tensor = preprocess(reference_tensor.reshape(-1, shape[-3], shape[-2], shape[-1]))
        input_tensor = preprocess(input_tensor)
        reference_tensor = reference_tensor.view(*shape)

        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        return input_tensor, samples_input, reference_tensor
