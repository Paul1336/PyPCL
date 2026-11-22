import torch
import torch.nn as nn
import torch.nn.functional as F


from src.losses.stateless import ComCoCLSLoss  # noqa: F401,E402  (moved 2026-11-22)
from src.losses.contrastive import comco_masks, masked_infonce  # noqa: E402


class ComCoContrastiveLoss(nn.Module):
    """InfoNCE with ComCo's positive / negative selection (Jiang et al. 2024, Sec. 3.4):
    positives = own key (+ top-K same-pseudo-label neighbour once warmup_pos), denominator =
    positives + pool entries whose complementary set contains c* (once warmup_neg), where c* is
    the class whose nearest complementary-set member is farthest. Core: src/losses/contrastive.py
    (comco_masks + masked_infonce), 2026-11-22."""

    def __init__(self, temperature: float = 0.17, top_k: int = 1):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k

    def forward(self, q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos: bool, warmup_neg: bool,
                return_masks: bool = False):
        sim = torch.mm(q, all_feats.t())
        pos_mask, denom_mask, _ = comco_masks(sim, pseudo_q, all_pseudo, all_comp, warmup_pos, warmup_neg,
                                              self.top_k)
        loss = masked_infonce(q, all_feats, pos_mask, denom_mask, self.temperature)
        if return_masks:
            return loss, pos_mask, denom_mask
        return loss
