"""Imports + helpers shared by every runner module (split out of runners.py 2026-11-22)."""


import gc
import json
import math
import os
import time
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms

from src.collate import solar_collate_fn
from src.comco.model import ComCoModel, ComCoWeightedModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss
from src.comco.fixed_utils_loss import FixedComCoCLSLoss
from src.comco.weighted_cls_loss import ComCoWeightedClsLoss
from src.clpl_loss import CLPLSquaredHingeLoss
from src.cpe_loss import CPELoss
from src.data_utils import SoLarDataset
from src.engine import (evaluate_model, train_algorithm, train_comco_epoch,
                         train_pico_epoch, train_pico_mclloss_epoch,
                         train_pico_moco_epoch, train_pico_sc_epoch, train_solar)
from src.fixed_comco_engine import (batch_cond_alpha, batch_random_alpha, sample_cond_alpha, train_comco_weighted_epoch,
                                     train_comco_weighted_switch_epoch)
from src.fixed_pico_engine import train_pico_epoch_fixed, train_pico_mcl_epoch_fixed, train_pico_weighted_epoch
from src.oracle_pico_engine import train_pico_oracle_add_graded_epoch, train_pico_oracle_graded_epoch
from src.mcl_losses import MCL_LOG
from src.fixed_mcl_losses import FixedMCLLog
from src.model_setup import setup_solar
from src.models import create_model_for_spec
from src.op_loss import OPLoss, OPWLoss
from src.pico.mcl_cls_loss import PiCOMCLLoss
from src.pico.model import PiCOModel, PiCOOracleModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.pico.weighted_cls_loss import PiCOWeightedClsLoss
from src.pico_cls_loss import PiCOCLSLoss
from src.proden_loss import ProdenLoss
from src.scl_loss import SCL_NL
from src.wu_loss import WuPLLLoss

from src.core.io import write_trace
from src.pipeline.algorithms.families import (candidate_init_conf, comco_args as _comco_args, fmt_eta as _fmt_eta,  # noqa: F401
                                              print_eta as _print_eta, run_comco_family, run_pico_family)
from src.pipeline.algorithms.families import pico_args as _pico_args  # noqa: E402
from src.pipeline import biasvariance, detail

from ..hparams import make_optimizer

from src.data.transforms import CIFAR_MEAN as _MEAN, CIFAR_STD as _STD  # noqa: E402  (single source, 2026-11-22)

# ─── shared helpers ────────────────────────────────────────────────────────






class _IndexedDataset(Dataset):
    """Wraps raw uint8 image array (or, for tabular datasets, raw feature
    vectors); returns (img_or_features, index) for index-based losses
    (PRODEN, PiCO-CLS).

    image_size/mean/std default to the original CIFAR values so existing call
    sites (no args passed) are unaffected. Previously cached the transform as
    a class-level singleton; now built per-instance so different datasets
    (different image_size/mean/std) don't share a stale cached transform.

    modality='tabular' skips the image transform pipeline entirely and just
    returns the raw feature vector as a float tensor -- Image.fromarray()
    doesn't apply to a feature vector."""

    def __init__(self, data, image_size=32, mean=_MEAN, std=_STD, modality='image'):
        self.modality = modality
        self.data = data
        if modality == 'image':
            self._train_tf = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.modality == 'tabular':
            return torch.as_tensor(self.data[idx], dtype=torch.float32), idx
        img = Image.fromarray(self.data[idx])
        return self._train_tf(img), idx


def _candidate_masked_init_conf(pl_ds, C: int, device) -> torch.Tensor:
    """Paper-faithful pseudo-target initialization (PiCO Eq. 6): s_j =
    1/|Y| * I(j in Y) -- uniform WITHIN the candidate set only, zero outside
    it. See docs/pico_explanation.md's newly-found discrepancy: run_pico's
    `torch.ones(N, C) / C` is uniform over ALL C classes regardless of the
    candidate set, which the paper's own text ("we first initialize the
    pseudo targets with a uniform distribution, s_j = 1/|Y| * I(j in Y)")
    does not support -- it gives equal initial weight even to classes
    provably not the true label, diluting the classification loss's signal
    for the whole warm-up period. Same construction as ProdenLoss.__init__
    (src/proden_loss.py), which already does this correctly."""
    from src.pll_init import candidate_masked_init
    return candidate_masked_init(pl_ds.targets, C).to(device)


def _uniform_all_init_conf(pl_ds, C: int, device) -> torch.Tensor:
    """Uniform over ALL C classes regardless of candidate-set membership --
    same as plain run_pico's init, but usable with PiCO-Fixed's warm-up
    config (see PiCO-Fixed-UniformInit) to isolate the effect of the initial
    confidence *shape* from the warm-up-length/L_cont-omission fix."""
    from src.pll_init import uniform_all_init
    return uniform_all_init(len(pl_ds), C).to(device)


def _biased_oracle_init_conf(pl_ds, orig_targets, C: int, device) -> torch.Tensor:
    """True class gets 0.2, one (fixed-seed-random, decided once) other
    candidate from the sample's own partial-label set gets 0.8, everything
    else 0 -- see PiCO-Fixed-BiasedInit and src/pll_init.py.biased_oracle_init."""
    from src.pll_init import biased_oracle_init
    return biased_oracle_init(pl_ds.targets, orig_targets, C).to(device)


_write_trace = write_trace  # src.core.io (trace file layout lives there)


def _save_alpha_trace(raw_cfg, algorithm, alpha_per_epoch, C):
    """Per epoch, per batch, the realized mean alpha the switch picked."""
    _write_trace(raw_cfg, 'alpha_trace', algorithm, C, alpha_per_epoch)
