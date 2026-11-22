"""Training loop for PiCOOracleModel (src/pico/model.py) -- graduated
precision-controlled contrastive pair-selection oracle.

Aligned with PiCO-Fixed (src.fixed_pico_engine.train_pico_epoch_fixed), not
plain PiCO: warm-up (epoch < pico_args['prot_start']) omits L_cont entirely
(loss = loss_cls only), rather than falling back to an unsupervised MoCo
variant the way plain PiCO's warm-up does.

Post-warmup, the contrastive `mask` starts as the model's own natural
pseudo-label-driven mask (identical construction to PiCO-Fixed's). This
batch's selected-positive-pair precision against ground truth is then
measured: if it's already >= `precision_threshold`, the mask is used
unchanged. Otherwise, just enough of the WRONG selected-positive pairs
(mask==1 but not actually same true class) are picked at random and flipped
to negative to bring precision up to the threshold -- pairs the model
already got right (mask==1 and same true class) are never touched, and no
pair the model selected as negative is ever promoted to positive (so this
only ever removes precision-hurting false positives, it never recovers
recall). threshold=0 is a no-op (precision is always >= 0), reproducing
plain PiCO-Fixed exactly. threshold=1 corrects every false positive out of
the model's own selected set -- an upper bound on what precision-only
correction can achieve, but NOT identical to the old ground-truth-mask
PiCO-Oracle (mask = true-label equality), since a true-class pair the model
never selected as positive in the first place is still never added.
"""

import math

import torch
from tqdm import tqdm


from src.train.pico import SPECS as _PS, OracleAddTP, OracleRemoveFP, pico_epoch as _pico_epoch

def train_pico_oracle_graded_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                                    precision_threshold):
    """PiCO-Oracle: remove false-positive pairs until precision >= threshold -> src/train/pico.py."""
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                       _PS['PiCO-Oracle'], mask_transform=OracleRemoveFP(precision_threshold))


def train_pico_oracle_add_graded_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                                        precision_threshold, max_add_ratio=1.0):
    """PiCO-Oracle-Add: add true-positive pairs (capped) until precision >= threshold -> src/train/pico.py."""
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                       _PS['PiCO-Oracle-Add'], mask_transform=OracleAddTP(precision_threshold, max_add_ratio))
