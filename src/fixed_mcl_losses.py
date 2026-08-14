"""Fixed versions of src/mcl_losses.py's MCL_LOG / MCL_MAE / MCL_EXP.

See docs/mcl_explanation.md ("演算法保真度比對") for the full derivation.

Bug being fixed: the unbiased-risk-estimator scaling factor in the original
classes is `(C-1)/(C-m)` (m = number of complementary labels for that
sample). Feng et al., ICML 2020 (arXiv:2002.08053 -> PMLR 119:3072-3081)
derive, via Eq. (12) (substituting L_EXP/L_LOG into the MAE-symmetry-based
collapse of the general Theorem 3 estimator), a scaling factor of
`2*(C-1)/m` instead. The two formulas move in *opposite* directions as m
grows (paper's factor shrinks toward 2 as m increases; the original code's
factor grows toward (C-1) and diverges as m -> C), so this is not a cosmetic
rescaling -- it changes the relative per-sample gradient weighting.

The LOG/MAE/EXP loss bodies themselves (before scaling) already matched the
paper exactly, so only the scaling factor changes here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _non_complementary_mask_sum(outputs: torch.Tensor, complementary_labels: torch.Tensor):
    """Shared setup: returns (sum_probs_not_in_complementary_set, num_complementary)."""
    valid_labels_mask = (complementary_labels != -1)
    num_complementary = valid_labels_mask.sum(dim=1).float()

    batch_size, num_classes = outputs.shape
    probs_all = F.softmax(outputs, dim=1)

    mask_complementary = torch.zeros_like(probs_all, dtype=torch.bool)
    for i in range(batch_size):
        valid_labels = complementary_labels[i][valid_labels_mask[i]]
        if len(valid_labels) > 0:
            mask_complementary[i].scatter_(0, valid_labels.long(), True)

    mask_non_complementary = ~mask_complementary
    sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary.float()).sum(dim=1)
    return sum_probs_not_in_complementary_set, num_complementary


class FixedMCLLog(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, complementary_labels):
        sum_probs_not_in_complementary_set, num_complementary = _non_complementary_mask_sum(
            outputs, complementary_labels)

        epsilon = 1e-7
        loss = -torch.log(sum_probs_not_in_complementary_set + epsilon)

        # Fixed scaling factor: 2*(C-1)/m, per paper Eq. (12).
        scaling_factor = 2.0 * (self.num_classes - 1) / num_complementary
        scaled_loss = scaling_factor * loss

        return scaled_loss.mean()


class FixedMCLMae(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, complementary_labels):
        sum_probs_not_in_complementary_set, num_complementary = _non_complementary_mask_sum(
            outputs, complementary_labels)

        loss = 1.0 - sum_probs_not_in_complementary_set

        scaling_factor = 2.0 * (self.num_classes - 1) / num_complementary
        scaled_loss = scaling_factor * loss

        return scaled_loss.mean()


class FixedMCLExp(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, complementary_labels):
        sum_probs_not_in_complementary_set, num_complementary = _non_complementary_mask_sum(
            outputs, complementary_labels)

        loss = torch.exp(-sum_probs_not_in_complementary_set)

        scaling_factor = 2.0 * (self.num_classes - 1) / num_complementary
        scaled_loss = scaling_factor * loss

        return scaled_loss.mean()
