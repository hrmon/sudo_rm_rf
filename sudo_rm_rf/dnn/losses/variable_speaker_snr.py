"""!
@brief Loss for variable number of speakers with explicitly inactive
output heads: PIT-SNR on active sources plus a FUSS-baseline-style
inactive-output power penalty (thresholded 20 dB below mixture power).

@author Hamidreza (adapted from repo conventions)
"""

import torch
import torch.nn as nn
import itertools


class VariableSpeakerSNRwithZeroRefs(nn.Module):
    """!
    Loss for up-to-max_num-sources separation where some reference
    channels are all-zero (inactive).

    Active sources are matched to estimates with PIT and get an SNR
    objective, exactly like PermInvariantSNRwithZeroRefs. The outputs
    that remain unmatched to an active source under the best
    permutation get an explicit energy penalty:

        inactive_loss = max(0, 10*log10(out_power / mixture_power) + T)

    where T is the threshold below the mixture power (default 20 dB,
    following the official FUSS baseline). So the penalty goes to zero
    once the output is more than 20 dB quieter than the mixture.
    """

    SNL_EPS = 1e-8

    def __init__(self,
                 n_sources=None,
                 zero_mean=False,
                 inactivity_threshold=-40.,
                 inactive_threshold_dB=20.,
                 backward_loss=True,
                 return_individual_results=False):
        super().__init__()
        self.zero_mean = zero_mean
        self.n_sources = n_sources
        self.inactivity_threshold = inactivity_threshold
        self.inactive_threshold_dB = inactive_threshold_dB
        self.backward_loss = backward_loss
        self.return_individual_results = return_individual_results
        self.permutations = list(itertools.permutations(
            torch.arange(self.n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)

    @staticmethod
    def dot(x, y):
        return torch.sum(x * y, dim=-1, keepdim=True)

    def forward(self,
                pr_batch,
                t_batch,
                input_mixture=None,
                eps=1e-9,
                return_best_permutation=False):
        """!
        :param pr_batch: Reconstructed wavs: Torch Tensors of size:
                         batch_size x self.n_sources x length_of_wavs
        :param t_batch: Target wavs (zero rows for inactive speakers):
                        Torch Tensors of size:
                        batch_size x self.n_sources x length_of_wavs
        :param input_mixture: The actual noisy input mixture of size:
                              batch_size x 1 x length_of_wavs
        :returns the combined loss and optionally the best permutation
        """
        min_len = min(pr_batch.shape[-1], t_batch.shape[-1])
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]
        if self.zero_mean:
            pr_batch = pr_batch - torch.mean(
                pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(
                t_batch, dim=-1, keepdim=True)

        target_powers = self.dot(t_batch, t_batch)  # B x S x 1
        # A target row is active only if it carries actual energy
        activity_mask = (target_powers > eps).float()
        n_active = activity_mask.sum(-2).unsqueeze(-1)  # B x 1 x 1
        n_inactive = float(self.n_sources) - n_active

        # ---- PIT over active sources ----
        # stabilizer caps the SNR at 30 dB for near-perfect estimates,
        # following PermInvariantSNRwithZeroRefs (thresh=0.001)
        stabilizer = 0.001 * target_powers
        all_snrs = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            error = permuted_pr_batch - t_batch
            snr = 10. * torch.log10(
                (target_powers + eps) / (self.dot(error, error) +
                                         stabilizer + eps))
            # zero out the contributions of inactive channels
            all_snrs.append((snr * activity_mask).sum(-2))
        all_snrs = torch.stack(all_snrs, -1)  # B x 1 x n_perm
        best_snr, best_perm_ind = torch.max(all_snrs, -1)

        best_perm = self.permutations_tensor[best_perm_ind]

        # ---- inactive outputs penalty under the best permutation ----
        # rows of the estimates matched to inactive (zero) targets
        incomplete_rows = pr_batch[:, best_perm, :] * \
            (1. - activity_mask)
        inactive_power = self.dot(incomplete_rows,
                                  incomplete_rows).sum(-2) / \
            n_inactive.clamp(min=1)
        if input_mixture is not None:
            mixture = input_mixture[:, :, :min_len]
        else:
            mixture = torch.sum(
                pr_batch[:, best_perm, :] * activity_mask, 1, keepdim=True)
        mixture_power = self.dot(mixture, mixture)

        inactive_db = 10. * torch.log10(
            inactive_power / (mixture_power + eps) + self.SNL_EPS)
        # zero once the inactive outputs are below the mixture threshold
        inactive_loss = torch.clamp(
            inactive_db + self.inactive_threshold_dB, min=0.)

        loss = -best_snr + inactive_loss

        if not self.return_individual_results:
            loss = loss.mean()

        if self.backward_loss:
            return ((loss, best_perm) if return_best_permutation
                    else loss)
        return (loss.detach(), best_perm) if return_best_permutation \
            else loss.detach()


def test_variable_speaker_loss():
    torch.manual_seed(0)
    bs = 4
    n_sources = 3
    n_samples = 1000

    loss_func = VariableSpeakerSNRwithZeroRefs(n_sources=n_sources,
                                               backward_loss=True)

    mixture = torch.randn(bs, 1, n_samples)
    targets = torch.zeros(bs, n_sources, n_samples)
    estimates = torch.zeros(bs, n_sources, n_samples)

    # 1 active speaker, perfect estimate in channel 0
    targets[:, 0] = torch.randn(bs, n_samples)
    estimates[:, 0] = targets[:, 0]
    estimates[:, 1] = 10. * torch.randn(bs, n_samples)  # loud garbage

    # Without the inactive penalty, loss=0 ; with it must be positive
    loud_loss = loss_func(estimates, targets, mixture)
    quiet_est = estimates.clone()
    quiet_est[:, 1] = 0.001 * torch.randn(bs, n_samples)
    quiet_loss = loss_func(quiet_est, targets, mixture)
    assert loud_loss > quiet_loss, (loud_loss, quiet_loss)

    # perfect separation with inactive outputs fully silent
    # active SNR is capped at ~30 dB per source by the stabilizer
    quiet_est2 = estimates.clone()
    quiet_est2[:, 1] = 0.
    quiet_est2[:, 2] = 0.
    quiet_loss2 = loss_func(quiet_est2, targets, mixture)
    assert abs(quiet_loss2 + 30.) < 0.5, quiet_loss2

    # 3 active speakers perfect separation
    targets3 = torch.randn(bs, n_sources, n_samples)
    est3 = targets3.clone()
    loss3 = loss_func(est3, targets3, mixture)
    assert abs(loss3 + 90.) < 1.5, loss3

    # gradients flow
    est_param = estimates.clone().requires_grad_(True)
    loss4 = loss_func(est_param, targets, mixture)
    loss4.backward()
    assert est_param.grad is not None

    print('VariableSpeakerSNRwithZeroRefs test passed.')


if __name__ == '__main__':
    test_variable_speaker_loss()
