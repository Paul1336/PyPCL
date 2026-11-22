"""cls-loss swap experiment (2026-10-06): framework epoch loops. The losses moved to
src/losses/swap.py (2026-11-22) and are re-exported here for old imports."""

import torch
import torch.nn.functional as F
from tqdm import tqdm  # noqa: F401

from src.train.pico import SPECS as _PS, pico_epoch as _pico_epoch
from src.train.comco import SPECS as _CS, comco_epoch as _comco_epoch

from src.losses.swap import (CLS_LOSSES, FRAMEWORKS, CandidatePruneLoss, CLPLLoss, MCLLOGFixedLoss,  # noqa: F401
                             MCLLOGLoss, MCLNLLoss, OPLoss, OPWLoss, PiCOEMALoss, PRODENLoss, SCLNLLoss,
                             SoLarLoss, SwapMixClsLoss, UCELoss, Wu2022Loss, _conf_buffer, _Stateless,
                             build_cls_loss, swap_algorithm_name)


def train_pico_swap_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """SwapPiCO-<loss> -> src/train/pico.py (2026-11-22 refactor)."""
    return _pico_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, _PS['SwapPiCO'])


def train_comco_swap_epoch(comco_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """SwapComCo-<loss> -> src/train/comco.py (2026-11-22 refactor)."""
    return _comco_epoch(comco_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, _CS['SwapComCo'])
