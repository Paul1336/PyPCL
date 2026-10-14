"""Hybrid PiCO <-> ComCo framework switching (2026-10-14).

Arm 0 = PiCO (PiCOModel's SupCon contrastive branch) + PartialLoss.
Arm 1 = ComCo (ComCoWeightedModel's ComCo contrastive branch) + original ComCo cls loss (MCL-LOG).
Both cls losses are the endpoints of src/unified_loss.py's UnifiedClsLoss (rho=0 / rho=1), so
they share ONE confidence buffer s, EMA-updated every batch whichever arm is active.

Shared: encoder_q, momentum encoder_k (same module objects in both sub-models), s.
Per arm: MoCo queue (+ queue labels), contrastive loss. Prototypes are kept in both
sub-models and synced after every training call, so either arm always starts from the
latest prototypes. The inactive arm's queue is not updated while the other arm trains.

PiCO and ComCo use identical weak/strong augmentations, so the ComCo arm consumes the
PiCO loader's batches with comp_mask = 1 - partial_Y (ComCoBatchView): in the paired
lookahead both arms see exactly the same images.
"""

import torch.nn as nn

from src.comco.model import ComCoWeightedModel
from src.pico.model import PiCOModel


class HybridModel(nn.Module):
    def __init__(self, pico_args: dict, comco_args: dict):
        super().__init__()
        self.pico = PiCOModel(pico_args)
        self.comco = ComCoWeightedModel(comco_args)
        self.comco.encoder_q = self.pico.encoder_q
        self.comco.encoder_k = self.pico.encoder_k

    @property
    def encoder_q(self):  # used by holdout scoring / src/switch_diag.py
        return self.pico.encoder_q

    @property
    def prototypes(self):  # kept in sync, either copy is current
        return self.pico.prototypes

    def sync_prototypes(self, from_arm: str):
        src, dst = (self.pico, self.comco) if from_arm == 'pico' else (self.comco, self.pico)
        dst.prototypes.copy_(src.prototypes)

    def forward(self, x, *args, eval_only: bool = True, **kwargs):
        logits, _ = self.pico.encoder_q(x)
        return logits


class ComCoBatchView:
    """Iterate a PiCO-format loader (or list of batches) as ComCo-format batches."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for (w, s, partial_Y, y, idx) in self.batches:
            yield w, s, 1.0 - partial_Y, y, idx

    def __len__(self):
        return len(self.batches)


class ComCoLossView:
    """ComCo-interface view (comp_mask) of a PiCO-interface UnifiedClsLoss sharing its state."""

    def __init__(self, unified):
        self.unified = unified

    def __call__(self, outputs, index, comp_mask, alpha=None):
        return self.unified(outputs, index, 1.0 - comp_mask, alpha=alpha)

    def confidence_update(self, **kwargs):
        self.unified.confidence_update(**kwargs)
