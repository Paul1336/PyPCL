import torch.nn as nn

from .mcl_cls_loss import PiCOMCLLoss
from .utils_loss import PartialLoss


class PiCOWeightedClsLoss(nn.Module):
    """PiCO-weighted-cls-loss's cls term:
        L_cls = alpha * PartialLoss(outputs, index) + (1 - alpha) * PiCOMCLLoss(outputs, partial_Y)

    PartialLoss (PiCO-Fixed's cls loss) is the only stateful half (its
    per-sample confidence buffer is EMA-updated across epochs); PiCOMCLLoss
    is stateless. set_conf_ema_m/confidence_update are delegated straight
    through to the wrapped PartialLoss.
    """

    def __init__(self, init_conf, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.partial_loss = PartialLoss(init_conf)
        self.mcl_loss = PiCOMCLLoss()

    def set_conf_ema_m(self, epoch, args):
        self.partial_loss.set_conf_ema_m(epoch, args)

    def confidence_update(self, **kwargs):
        self.partial_loss.confidence_update(**kwargs)

    def forward(self, outputs, index, partial_Y):
        return (self.alpha * self.partial_loss(outputs, index)
                + (1 - self.alpha) * self.mcl_loss(outputs, partial_Y))
