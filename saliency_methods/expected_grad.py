import torch
import torch.utils.data

from .IG_SG import IntGradSG


class ExpectedGradients(IntGradSG):
    def __init__(self, model, k, bg_size, bg_dataset, batch_size, random_alpha=True, est_method='vanilla',
                 exp_obj='logit'):
        super(ExpectedGradients, self).__init__(model, k, bg_size, random_alpha, est_method)
        self.bg_size = bg_size
        self.random_alpha = random_alpha
        self.ref_sampler = torch.utils.data.DataLoader(
                dataset=bg_dataset,
                batch_size=bg_size*batch_size,
                shuffle=True,
                pin_memory=False,
                drop_last=False)

        self.est_method = est_method
        self.exp_obj = exp_obj

    def _get_ref_batch(self):
        return next(iter(self.ref_sampler))[0].float()

    def chew_input(self, input_tensor):
        """
        Calculate expected gradients for the sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        shape = list(input_tensor.shape)
        shape.insert(1, self.k*self.bg_size)

        # ================= expected gradients ==================
        reference_tensor = torch.zeros(shape).float().cuda()

        ref = self._get_ref_batch()
        for bth in range(shape[0]):
            for bg in range(self.bg_size):
                ref_ = ref[bth * self.bg_size + bg]
                reference_tensor[bth, bg*self.k:(bg+1)*self.k, :] = ref_
        if ref.shape[0] != input_tensor.shape[0]*self.k:
            reference_tensor = reference_tensor[:input_tensor.shape[0]*self.k]

        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        return input_tensor, samples_input, reference_tensor
