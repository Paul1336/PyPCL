import torch
import torch.nn as nn
import torch.nn.functional as F

from .fixed_utils_loss import FixedComCoCLSLoss
from src.pico.utils_loss import PartialLoss


class ComCoWeightedClsLoss(nn.Module):
    """ComCo-weighted-cls-loss's cls term -- the ComCo-side mirror of
    src/pico/weighted_cls_loss.py's PiCOWeightedClsLoss:

        L_cls = alpha * FixedComCoCLSLoss(outputs, comp_mask)
                + (1 - alpha) * PartialLoss(outputs, index)

    alpha weights ComCo-Fixed's OWN cls loss (SCL-NL over the complementary
    set); (1 - alpha) weights PartialLoss (PiCO-Fixed's candidate-set,
    prototype-confidence-driven loss), borrowed onto ComCo's architecture --
    alpha=1 reduces to plain ComCo-Fixed, alpha=0 to a "PartialLoss on
    ComCo's dual-encoder backbone" variant. PartialLoss is the only stateful
    half (its per-sample confidence buffer is EMA-updated across epochs,
    driven by prototype similarity -- see ComCoWeightedModel); FixedComCoCLSLoss
    is stateless. set_conf_ema_m/confidence_update are delegated straight
    through to the wrapped PartialLoss.
    """

    def __init__(self, init_conf, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.comco_loss = FixedComCoCLSLoss()
        self.partial_loss = PartialLoss(init_conf)

    def set_conf_ema_m(self, epoch, args):
        self.partial_loss.set_conf_ema_m(epoch, args)

    def confidence_update(self, **kwargs):
        self.partial_loss.confidence_update(**kwargs)

    def forward(self, outputs, index, comp_mask, alpha: float = None):
        """`alpha`, if given, overrides self.alpha for this call only -- used
        by the per-batch alpha-switching training loops (see
        src/fixed_comco_engine.py::train_comco_weighted_switch_epoch), which
        pick alpha in {0.0, 1.0} fresh each batch instead of using one fixed
        value for the whole run."""
        a = self.alpha if alpha is None else alpha
        if torch.is_tensor(a):
            # Per-sample alpha [B] (SampleCond): blend the two losses sample
            # by sample, then average. Same per-sample terms as
            # FixedComCoCLSLoss / PartialLoss's own forward, before their .mean().
            probs = F.softmax(outputs, dim=1)
            m = comp_mask.sum(dim=1).clamp(min=1.0)
            comco_ps = (-torch.log1p(-probs.clamp(max=1.0 - 1e-7)) * comp_mask).sum(dim=1) / m
            partial_ps = -(F.log_softmax(outputs, dim=1) * self.partial_loss.confidence[index, :]).sum(dim=1)
            return (a * comco_ps + (1 - a) * partial_ps).mean()
        return (a * self.comco_loss(outputs, comp_mask)
                + (1 - a) * self.partial_loss(outputs, index))
