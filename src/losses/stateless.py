"""nn.Module wrappers around src.losses.functional, one per historical loss class.

Constructor signatures and call signatures are kept so the registry, verify_scripts and old
results keep their meaning; the arithmetic is the canonical one in functional.py.

Input conventions (unchanged from the originals):
  padded-CL  forward(logits, comp_labels_padded)    MCL_*, FixedMCL*, SCL_NL, OPLoss, OPWLoss, CPELoss
  padded-PL  forward(logits, cand_labels_padded)    CLPLSquaredHingeLoss, UniformCandidateCE, WuPLLLoss
  dense-Y    forward(logits, partial_Y)             PiCOMCLLoss
  dense-comp forward(logits, comp_mask)             ComCoCLSLoss, FixedComCoCLSLoss
"""

import torch.nn as nn

from . import functional as Fn
from .masks import padded_to_dense


class _PaddedCL(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes

    def _Y(self, logits, comp_labels):
        return 1.0 - padded_to_dense(comp_labels, logits.shape[1], logits.dtype)


class _PaddedPL(nn.Module):
    def _Y(self, logits, cand_labels):
        return padded_to_dense(cand_labels, logits.shape[1], logits.dtype)


# --- MCL family (padded CL) -----------------------------------------------------------------

class MCL_LOG(_PaddedCL):
    """Original MCL-LOG (scale (C-1)/(C-m), see docs/mcl_explanation.md)."""
    def forward(self, outputs, complementary_labels):
        return Fn.mcl(outputs, self._Y(outputs, complementary_labels), 'log', 'c1_over_k')


class MCL_MAE(_PaddedCL):
    def forward(self, outputs, complementary_labels):
        return Fn.mcl(outputs, self._Y(outputs, complementary_labels), 'mae', 'c1_over_k')


class MCL_EXP(_PaddedCL):
    def forward(self, outputs, complementary_labels):
        return Fn.mcl(outputs, self._Y(outputs, complementary_labels), 'exp', 'c1_over_k')


class FixedMCLLog(_PaddedCL):
    """MCL-LOG with the paper's Eq. 12 scale 2(C-1)/m."""
    def forward(self, outputs, complementary_labels):
        return Fn.mcl(outputs, self._Y(outputs, complementary_labels), 'log', 'feng_fixed')


class FixedMCLMae(_PaddedCL):
    def forward(self, outputs, complementary_labels):
        return Fn.mcl(outputs, self._Y(outputs, complementary_labels), 'mae', 'feng_fixed')


class FixedMCLExp(_PaddedCL):
    def forward(self, outputs, complementary_labels):
        return Fn.mcl(outputs, self._Y(outputs, complementary_labels), 'exp', 'feng_fixed')


class SCL_NL(_PaddedCL):
    def forward(self, outputs, complementary_labels):
        return Fn.scl_nl(outputs, self._Y(outputs, complementary_labels))


class OPLoss(_PaddedCL):
    def forward(self, outputs, complementary_labels):
        return Fn.op(outputs, self._Y(outputs, complementary_labels))


class OPWLoss(_PaddedCL):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, outputs, complementary_labels):
        return Fn.opw(outputs, self._Y(outputs, complementary_labels), self.eps)


class CPELoss(_PaddedCL):
    """Predict with argmin (see runners._evaluate_argmin)."""
    def forward(self, outputs, complementary_labels):
        return Fn.cpe(outputs, self._Y(outputs, complementary_labels))


# --- PLL (padded PL) ------------------------------------------------------------------------

class CLPLSquaredHingeLoss(_PaddedPL):
    def forward(self, outputs, partial_labels):
        return Fn.clpl(outputs, self._Y(outputs, partial_labels))


class UniformCandidateCrossEntropyLoss(_PaddedPL):
    def forward(self, outputs, partial_labels):
        return Fn.uce(outputs, self._Y(outputs, partial_labels))


class WuPLLLoss(_PaddedPL):
    def forward(self, outputs, partial_labels):
        return Fn.wu2022(outputs, self._Y(outputs, partial_labels))


# --- dense-mask framework losses ------------------------------------------------------------

class PiCOMCLLoss(nn.Module):
    """PiCO-MCL cls loss: MCL-LOG (C-1)/k on the dense candidate mask."""
    def forward(self, outputs, partial_Y):
        return Fn.mcl(outputs, partial_Y, 'log', 'c1_over_k', eps='clamp')


class ComCoCLSLoss(nn.Module):
    """Original ComCo cls loss (scaled MCL-NL == MCL-LOG (C-1)/(C-m)) on the dense comp mask."""
    def forward(self, logits, comp_mask):
        return Fn.mcl(logits, 1.0 - comp_mask, 'log', 'c1_over_k', eps='clamp')


class FixedComCoCLSLoss(nn.Module):
    """ComCo-Fixed cls loss = SCL-NL on the dense comp mask."""
    def forward(self, logits, comp_mask):
        return Fn.scl_nl(logits, 1.0 - comp_mask)
