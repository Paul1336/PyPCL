import torch
import torch.nn.functional as F
import torch.nn as nn
    
from src.losses.confidence import PartialLoss  # noqa: F401,E402  (moved 2026-11-22)
from src.losses.contrastive import masked_infonce, moco_infonce, self_mask  # noqa: E402


class SupConLoss(nn.Module):
    """Supervised contrastive loss (PiCO). mask mode -> masked_infonce with positives = mask
    minus self and denominator = everything but self; mask=None -> MoCo loss.
    Core in src/losses/contrastive.py (2026-11-22)."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        if mask is None:
            return moco_infonce(features, batch_size, self.temperature)
        mask = mask.float().detach().to(features.device)
        not_self = self_mask(batch_size, mask.shape[1], features.device)
        loss = masked_infonce(features[:batch_size], features, mask * not_self, not_self, self.temperature)
        scale = self.temperature / self.base_temperature
        return loss if scale == 1.0 else loss * scale
