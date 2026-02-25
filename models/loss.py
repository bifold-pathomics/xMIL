import torch
import torch.nn as nn
import torch.nn.functional as F


class SurvivalNegLogLikelihood(nn.Module):
    def __init__(self, epsilon=1e-7):
        """
        :param epsilon: (float) minimum value for hazard and survival scores for numerical stability
        """
        super(SurvivalNegLogLikelihood, self).__init__()
        self.epsilon = epsilon

    def forward(self, hazard_prob, true_survival_bin, censorship_status):
        """
        Custom loss function for survival analysis using negative log-likelihood.
        (c) modified from https://github.com/mahmoodlab/MCAT

        :param hazard_prob: (torch.Tensor) Predicted hazards (risk scores).
        :param true_survival_bin: (torch.Tensor): Ground truth survival bin (1, 2, ..., K).
        :param censorship_status: (torch.Tensor): Censorship status (0 or 1).
        :return: (torch.Tensor) Negative log-likelihood loss.
        """
        batch_size = len(true_survival_bin)
        true_survival_bin = true_survival_bin.to(torch.int64)
        true_survival_bin = true_survival_bin.view(
            batch_size, 1
        )  # ground truth bin, 1,2,...,k
        censorship_status = censorship_status.view(
            batch_size, 1
        ).float()  # censorship status, 0 or 1

        survival_prob = torch.cumprod(
            1 - hazard_prob, dim=1
        )  # survival is cumulative product of 1 - hazards

        # Clamp hazard and survival scores to avoid numerical issues
        hazard_prob = hazard_prob.clamp(min=self.epsilon)
        survival_prob = survival_prob.clamp(min=self.epsilon)

        # Pad S with 1 at the start (S(0) = 1)
        survival_prob_padded = torch.cat(
            [torch.ones_like(censorship_status), survival_prob], dim=1
        )

        # Adjust indexing to be zero-based for PyTorch's gather function
        true_survival_bin_zero_based = (
            true_survival_bin - 1
        )  # Now tru_survival_bin is 0-based
        true_survival_bin_zero_based = torch.clamp(true_survival_bin_zero_based, min=0)

        # Calculate negative log-likelihood
        neg_l = -censorship_status * torch.log(
            torch.gather(survival_prob_padded, 1, true_survival_bin)
        ) - (1 - censorship_status) * (
            torch.log(
                torch.gather(survival_prob_padded, 1, true_survival_bin_zero_based)
            )
            + torch.log(torch.gather(hazard_prob, 1, true_survival_bin_zero_based))
        )
        return neg_l.mean()
