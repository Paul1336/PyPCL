import torch
import torch.nn as nn
import torch.nn.functional as F


class CLPLSquaredHingeLoss(nn.Module):
    """
    Cour, Sapp & Taskar (JMLR 2011) CLPL with squared-hinge surrogate.

    Verified against the paper's Eq. 2 term-for-term (2026-08-14) -- exact
    match, no fixed_ version needed. See docs/cour2011_explanation.md.

    For each sample i with candidate set Y_i:

        L_i = psi( mean_{a in Y_i} g_a(x_i) )
              + sum_{a not in Y_i} psi( -g_a(x_i) )

    where  psi(u) = max(0, 1 - u)^2  (squared hinge)
    and    g_a(x_i) = outputs[i, a]  (raw logit / score, NOT softmax).

    Expanded:

        positive term:  max(0, 1 - avg_candidate_score)^2
        negative term:  sum_{a not in Y_i} max(0, 1 + g_a(x_i))^2

    Args:
        outputs:        [B, C]  raw logits g_a(x).
        partial_labels: [B, L]  candidate label indices; -1 = padding.
    """

    def forward(self, outputs: torch.Tensor, partial_labels: torch.Tensor) -> torch.Tensor:
        B, C = outputs.shape
        device = outputs.device

        valid_mask = partial_labels >= 0                           # [B, L]
        count = valid_mask.sum(dim=1).float().clamp(min=1)        # [B]

        # Build binary candidate mask [B, C]
        labels_safe = partial_labels.clamp(min=0)
        L = labels_safe.shape[1]
        one_hot = torch.zeros(B, L, C, device=device, dtype=outputs.dtype)
        one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
        one_hot = one_hot * valid_mask.unsqueeze(2).to(outputs.dtype)
        candidate_mask = one_hot.sum(dim=1).clamp(max=1.0)        # [B, C]
        negative_mask  = 1.0 - candidate_mask                     # [B, C]

        # Positive term: psi( mean_{a in Y_i} g_a )
        avg_score    = (outputs * candidate_mask).sum(dim=1) / count
        positive_loss = F.relu(1.0 - avg_score).pow(2)            # [B]

        # Negative term: sum_{a not in Y_i} psi( -g_a ) = sum max(0, 1+g_a)^2
        negative_loss = (negative_mask * F.relu(1.0 + outputs).pow(2)).sum(dim=1)  # [B]

        return (positive_loss + negative_loss).mean()


# Alias matching the recommended naming in the fix request
CourCLPLSquaredHingeLoss = CLPLSquaredHingeLoss
