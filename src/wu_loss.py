import torch
import torch.nn as nn
import torch.nn.functional as F


class WuPLLLoss(nn.Module):
    """
    Proper Partial Label Loss.
    Wu et al., "Learning with Proper Partial Labels", 2021/2023.

    Under uniform constant-k generation (true label always in Y,
    k-1 false candidates drawn uniformly from remaining C-1 classes),
    the proper/unbiased estimator of the CE risk R(x) = -log f_{y*}(x) is:

        L̃(x, Y) = (C-1)/(C-k) * Σ_{j∈Y} (-log f_j)
                 - (k-1)/(C-k) * Σ_j  (-log f_j)

    The loss is "proper" in the sense that its minimizer over f recovers
    the true posterior P(y|x), analogous to proper scoring rules.

    Edge cases:
        k=1   → β=0, loss = ℓ_{y*}  (standard CE)
        k=C-1 → large correction; still proper and unbiased

    Per-sample values can be negative — this is expected and correct.

    Args:
        outputs:        [B, C] raw logits
        partial_labels: [B, max_k] candidate label indices; -1 = padding
    """

    def forward(self, outputs: torch.Tensor, partial_labels: torch.Tensor) -> torch.Tensor:
        B, C = outputs.shape
        device = outputs.device

        log_sm = F.log_softmax(outputs, dim=1)         # [B, C]

        # Build dense candidate mask [B, C]
        valid_mask  = partial_labels >= 0              # [B, max_k]
        k           = valid_mask.sum(dim=1).float().clamp(min=1)  # [B]
        labels_safe = partial_labels.clamp(min=0)
        max_k       = labels_safe.shape[1]

        one_hot = torch.zeros(B, max_k, C, device=device)
        one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
        one_hot    = one_hot * valid_mask.unsqueeze(2).float()
        cand_mask  = one_hot.sum(dim=1).clamp(max=1.0)            # [B, C]

        sum_cand = -(log_sm * cand_mask).sum(dim=1)               # [B]
        sum_all  = -log_sm.sum(dim=1)                             # [B]

        alpha = (C - 1.0) / (C - k)
        beta  = (k  - 1.0) / (C - k)

        return (alpha * sum_cand - beta * sum_all).mean()
