"""ComCo-Weighted's training epoch -- the ComCo-side mirror of
src/fixed_pico_engine.py's train_pico_weighted_epoch.

ComCo-Fixed's own architecture/loop (src/engine.py::train_comco_epoch) needs
no warm-up: FixedComCoCLSLoss is stateless. ComCoWeightedClsLoss
(src/comco/weighted_cls_loss.py) mixes in PartialLoss, though, which is
driven by a prototype-similarity confidence signal (PiCO Eq. 6/7) that isn't
meaningful until the (newly-added, see ComCoWeightedModel) prototypes have
accumulated for a while -- so this loop gates PartialLoss's
confidence_update behind `comco_args['prot_start']`, same rationale as
PiCO-Fixed's warm-up, while leaving everything else (loss_weight, the
contrastive loss's own warmup_pos/warmup_neg schedule) identical to
train_comco_epoch, active from epoch 0 as in ComCo-Fixed. Unlike PiCO-Fixed,
L_cont is never omitted here -- that omission was specific to reproducing
the PiCO paper's stated warm-up behavior, and has no analogue in ComCo's own
design.
"""

import random
import torch

from tqdm import tqdm


from src.train.comco import SPECS as _CS, comco_epoch as _comco_epoch

def train_comco_weighted_epoch(comco_args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch, device):
    """ComCo-Weighted (fixed alpha) -> src/train/comco.py (2026-11-22 refactor)."""
    return _comco_epoch(comco_args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch, device, _CS['ComCo-Weighted'])


# ─── per-batch alpha switching (variable-average-k experiment) ────────────
#
# Under variable-size candidate sets (--only_q, mean size != fixed k), a
# batch's OWN realized average candidate-set size fluctuates around the
# dataset's target mean. These two strategies pick a HARD alpha in {0.0,
# 1.0} (never a blend) fresh each batch instead of using one fixed alpha for
# the whole run -- see train_comco_weighted_switch_epoch below. Threshold is
# a fixed absolute size (10), not scaled to whatever mean the dataset was
# built with, matching the experiment's own design: at mean=10 this is a
# genuine ~50/50 adaptive switch, at mean=5/15 it degenerates toward
# almost-always-A000/almost-always-A100.

BATCH_COND_THRESHOLD = 10.0


def batch_cond_alpha(partial_Y, threshold: float = BATCH_COND_THRESHOLD) -> float:
    """alpha=1.0 (ComCo-Fixed's own loss) if this batch's average candidate-
    set size is ABOVE `threshold`, else 0.0 (PartialLoss)."""
    avg_size = partial_Y.sum(dim=1).mean().item()
    return 1.0 if avg_size > threshold else 0.0


def sample_cond_alpha(partial_Y, threshold: float = BATCH_COND_THRESHOLD):
    """Per-sample version of batch_cond_alpha: a [B] tensor, 1.0 for samples
    whose OWN candidate-set size is above `threshold` (ComCo-Fixed loss),
    else 0.0 (PartialLoss). Independent of how batches are composed."""
    return (partial_Y.sum(dim=1) > threshold).float()


def batch_random_alpha(partial_Y) -> float:
    """alpha in {0.0, 1.0}, independent 50/50 coin flip each batch --
    `partial_Y` is unused, kept only so this has the same signature as
    batch_cond_alpha (both are called as `alpha_fn(partial_Y)`)."""
    del partial_Y
    return 1.0 if random.random() < 0.5 else 0.0


def train_comco_weighted_switch_epoch(comco_args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch,
                                       device, alpha_fn, alpha_trace: list = None):
    """ComCo-Weighted with per-batch alpha = alpha_fn(Y) (switch / bandit arms) -> src/train/comco.py."""
    return _comco_epoch(comco_args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch, device,
                        _CS['ComCo-Weighted-Switch'], alpha_fn=alpha_fn, alpha_trace=alpha_trace)
