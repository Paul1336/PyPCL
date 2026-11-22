import torch
import torch.nn as nn
import torch.nn.functional as F


class PiCOMCLLoss(nn.Module):
    """
    MCL-LOG adapted for partial labels, used as PiCO's cls loss.

    Non-candidate labels act as complementary labels, giving the dual form:
        L = -log( sum_{j in Y} f^j(x) ) * (C-1) / k

    where k = |Y| = number of partial labels per sample.
    Unbiased risk estimator under uniform partial label generation.

    Args:
        outputs:   [B, C] raw logits
        partial_Y: [B, C] binary partial label mask (1 = candidate)
    """

    def forward(self, outputs: torch.Tensor, partial_Y: torch.Tensor) -> torch.Tensor:
        C = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)                                  # [B, C]
        candidate_sum = (probs * partial_Y).sum(dim=1).clamp(min=1e-8)    # [B]
        k = partial_Y.sum(dim=1).clamp(min=1.0)                           # [B]
        loss = -torch.log(candidate_sum) * ((C - 1.0) / k)
        return loss.mean()
