"""Power-mean / Renyi unified cls loss (2026-10-13).

    L_rho = -(1/rho) log sum_{j in S} s_j^(1-rho) p_j^rho        (0 < rho <= 1)
    L_0   = -sum_{j in S} s_j log p_j                             (PartialLoss)

times ((C-1)/k)^rho, so that
  * rho = 0 is exactly PiCO's PartialLoss (s = its prototype-EMA confidence),
  * rho = 1 is exactly the ORIGINAL ComCo cls loss (src/comco/utils_loss.py
    ComCoCLSLoss == swap experiment's MCLLOG): (C-1)/k * -log sum_{j in S} p_j.
Gradient: ((C-1)/k)^rho * (p - t), t_j proportional to s_j^(1-rho) p_j^rho on S.
rho is passed per call (`alpha=` keyword, to fit the existing switch-epoch
loops), so a bandit can change it every epoch. s keeps being EMA-updated by
the caller every batch regardless of rho -- same as the ComCo-Weighted
switch/bandit runs.

Also here: PiCOFixedComCoMixClsLoss, the PiCO-architecture mirror of
src/comco/weighted_cls_loss.py's ComCoWeightedClsLoss (alpha * SCL-NL +
(1-alpha) * PartialLoss), so the earlier {PartialLoss, ComCo-Fixed} bandit
can be rerun on PiCO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.comco.fixed_utils_loss import FixedComCoCLSLoss
from src.pico.utils_loss import PartialLoss


class UnifiedClsLoss(nn.Module):
    """PiCO-style interface: forward(outputs, index, partial_Y, alpha=rho)."""

    def __init__(self, init_conf, rho: float = 0.0, comco_scale: bool = True):
        super().__init__()
        self.rho = rho
        self.comco_scale = comco_scale
        self.partial_loss = PartialLoss(init_conf)
        self.frozen = False  # True: confidence_update is a no-op (s-freezing intervention)

    def set_conf_ema_m(self, epoch, args):
        self.partial_loss.set_conf_ema_m(epoch, args)

    def confidence_update(self, **kwargs):
        if not self.frozen:
            self.partial_loss.confidence_update(**kwargs)

    def forward(self, outputs, index, partial_Y, alpha: float = None):
        rho = self.rho if alpha is None else float(alpha)
        log_p = F.log_softmax(outputs, dim=1)
        s = self.partial_loss.confidence[index, :]
        if rho <= 0.0:
            per_sample = -(s * log_p).sum(dim=1)
        else:
            w = torch.where(partial_Y > 0,
                            (1.0 - rho) * torch.log(s.clamp_min(1e-12)) + rho * log_p,
                            torch.full_like(log_p, float('-inf')))
            per_sample = -torch.logsumexp(w, dim=1) / rho
        if self.comco_scale and rho > 0.0:
            C = outputs.shape[1]
            k = partial_Y.sum(dim=1).clamp(min=1.0)
            per_sample = per_sample * ((C - 1.0) / k) ** rho
        return per_sample.mean()


class UnifiedClsLossComCo(UnifiedClsLoss):
    """ComCo-style interface: forward(outputs, index, comp_mask, alpha=rho)."""

    def forward(self, outputs, index, comp_mask, alpha: float = None):
        return super().forward(outputs, index, 1.0 - comp_mask, alpha=alpha)


class PiCOFixedComCoMixClsLoss(nn.Module):
    """alpha * FixedComCoCLSLoss (SCL-NL) + (1 - alpha) * PartialLoss, PiCO
    interface forward(outputs, index, partial_Y, alpha)."""

    def __init__(self, init_conf, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.comco_loss = FixedComCoCLSLoss()
        self.partial_loss = PartialLoss(init_conf)

    def set_conf_ema_m(self, epoch, args):
        self.partial_loss.set_conf_ema_m(epoch, args)

    def confidence_update(self, **kwargs):
        self.partial_loss.confidence_update(**kwargs)

    def forward(self, outputs, index, partial_Y, alpha: float = None):
        a = self.alpha if alpha is None else float(alpha)
        return a * self.comco_loss(outputs, 1.0 - partial_Y) + (1 - a) * self.partial_loss(outputs, index)
