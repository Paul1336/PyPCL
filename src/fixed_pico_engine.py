"""Fixed version of src/engine.py's train_pico_epoch.

See docs/pico_explanation.md ("演算法保真度比對" -> 第 4 項 warm-up 機制) for the
full comparison. Wang et al., ICLR 2022 (PiCO) describes the warm-up period
(Appendix B.1: 1 epoch by default, 100 epochs for the CIFAR-100 q=0.1 setting)
as simply *omitting L_cont from the total loss* for the first N epochs --
Algorithm 1's pseudocode runs prototype updates, pseudo-target updates, and
L_cls unconditionally from epoch 0; only L_cont is described as disabled
during warm-up.

The original train_pico_epoch (src/engine.py) instead keeps L_cont active
throughout, but swaps SupConLoss from its masked (label-similarity) mode to
a plain unsupervised MoCo InfoNCE mode during warm-up, and additionally
gates the confidence/pseudo-target EMA update (Eq. 6) off until prot_start.
That is a reasonable engineering choice but not what the paper describes.

This module reproduces the paper's literal warm-up behavior: during
epoch < prot_start, L_cont is omitted entirely (loss = loss_cls only); the
confidence/pseudo-target update stays gated off during warm-up too, since
Eq. 6's prototype-similarity signal is not meaningful before the prototypes
have been updated for a while (consistent with the paper's own stated
intuition, even though Algorithm 1's box doesn't show an explicit gate for
it) -- this part of the original code is kept unchanged.
"""

import torch
from tqdm import tqdm


from src.train.pico import SPECS as _PS, pico_epoch as _pico_epoch

def train_pico_epoch_fixed(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                            conf_source: str = 'prototype', conf_hard: bool = True):
    """PiCO-Fixed (paper-faithful warm-up: L_cont omitted before prot_start). conf_source /
    conf_hard: A/B confidence-update ablation -> src/train/pico.py (2026-11-22 refactor)."""
    if conf_source not in ('prototype', 'classifier'):
        raise ValueError(f"conf_source must be 'prototype' or 'classifier', got {conf_source!r}")
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                       _PS['PiCO-Fixed'].with_(conf_hard=conf_hard), conf_source=conf_source)


def train_pico_weighted_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """PiCO-Weighted (alpha mix cls loss) -> src/train/pico.py (2026-11-22 refactor)."""
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, _PS['PiCO-Weighted'])


def train_pico_mcl_epoch_fixed(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """PiCO-MCL-Fixed -> src/train/pico.py (2026-11-22 refactor)."""
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, _PS['PiCO-MCL-Fixed'])


def train_pico_switch_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, alpha_fn):
    """PiCO switch / bandit arm epoch (per-batch alpha) -> src/train/pico.py (2026-11-22 refactor)."""
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, _PS['PiCO-Switch'], alpha_fn=alpha_fn)
