import torch
import torch.utils.data

from .IG_SG import IntGradSG


class IntGradSQ(IntGradSG):

    def shap_values(self, input_tensor, sparse_labels=None):
        input_tensor, samples_input, reference_tensor = self.chew_input(input_tensor)

        if self.est_method == 'valid_ip':
            samples_delta = self._get_samples_delta(input_tensor, samples_input)  # samples_input, reference_tensor
            grad_ori_tensor = self._get_grads(samples_input, sparse_labels)
            grad_ori_tensor = grad_ori_tensor.squeeze(2)
            grad_ori_tensor = grad_ori_tensor.reshape(samples_delta.shape)
            mult_grads = grad_ori_tensor * samples_delta
            grad_sign = torch.where(mult_grads >= 0., 1., 0.)
            mult_grads = torch.pow(mult_grads, 2.) * grad_sign

            counts = torch.sum(grad_sign, dim=1)
            mult_grads = mult_grads.sum(1) / torch.where(counts == 0., torch.ones(counts.shape).cuda(), counts)
            attribution = mult_grads
        elif self.est_method == 'valid_ref':
            samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            zeros = torch.zeros(grad_tensor.shape).cuda()
            ones = torch.ones(grad_tensor.shape).cuda()
            mult_grads = grad_tensor * samples_delta
            grad_sign = torch.where(mult_grads >= 0., ones, zeros)
            mult_grads = torch.pow(mult_grads, 2.) * grad_sign

            counts = torch.sum(grad_sign, dim=1)
            mult_grads = mult_grads.sum(1) / torch.where(counts == 0., ones[:, 0], counts)
            attribution = mult_grads
        else:
            samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            mult_grads = samples_delta * grad_tensor
            attribution = mult_grads.mean(1)

        return attribution
