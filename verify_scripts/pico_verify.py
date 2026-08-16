"""Standalone, paper-exact verification script for PiCO (Wang, Xiao, Li,
Feng, Niu, Chen, Zhao, ICLR 2022) on CIFAR-10, q=0.5.

Reference: "PiCO: Contrastive Label Disambiguation for Partial Label
Learning", ICLR 2022 (https://openreview.net/forum?id=EhYjZy6e1gJ). Settings
below were confirmed by directly extracting text from the paper PDF (not
reconstructed from memory) -- see docs/pico_explanation.md for the full
paper-vs-code fidelity audit this script is built on.

Paper-exact settings used here (Appendix B.1, "default setting"):
    - Architecture:  18-layer ResNet + 2-layer MLP projection head, 128-dim
                      (src/pico/resnet.py::SupConResNet via src/pico/model.py
                      ::PiCOModel -- already exact, no changes needed).
    - batch_size:     256
    - epochs:         800
    - LR schedule:    cosine annealing (torch.optim.lr_scheduler.CosineAnnealingLR)
    - optimizer:      SGD, momentum=0.9
    - q:              0.5 (each false label independently included w.p. q)
    - moco_queue=8192, moco_m=0.999, proto_m=0.99, tau=0.07, loss_weight=0.5,
      conf_ema_range=[0.95, 0.8] -- all match config.yaml's `pico:` block.
    - prot_start=1 (paper's general default; CIFAR-10 q=0.5 is NOT the
      special CIFAR-100 q=0.1 case that uses prot_start=100).
    - paper_target_accuracy = 93.58% (Table 1, CIFAR-10 q=0.5).

UNRESOLVED PAPER AMBIGUITY: the paper never states its base learning rate or
weight decay anywhere in the text (confirmed via PDF extraction, not just an
oversight in this script). This script uses lr=0.001, weight_decay=1e-4 as a
placeholder, taken from this repo's config.yaml `training:` block -- the
same SGD defaults this project already uses for its other SGD-optimized
algorithms (PRODEN, SoLar) absent a paper-stated override. Flagged in the
output CSV's `notes` column.

Paper-faithful mechanisms used (per docs/pico_explanation.md's fidelity
audit, NOT the original pipeline's plain `PiCO` algorithm ID, which is known
to deviate from the paper in these two respects):
    1. Warm-up: src/fixed_pico_engine.py::train_pico_epoch_fixed omits
       L_cont from the total loss entirely while epoch < prot_start (the
       paper's literal Appendix B.1 description), instead of the original
       train_pico_epoch's "keep L_cont active as plain MoCo" behavior.
    2. Pseudo-target init: verify_scripts.pico_helpers.candidate_masked_init_conf
       initializes confidence uniformly WITHIN each sample's candidate set
       only (paper Eq. 6: s_j = 1/|Y| * I(j in Y)), not
       torch.ones(N, C) / C (uniform over all C classes) like the original
       pipeline's run_pico.

CLI:
    python verify_scripts/pico_verify.py [--epochs N] [--seed N]
        [--batch_size N] [--q FLOAT] [--data_dir ./data]

GPU handling: uses torch.device('cuda' if available else 'cpu'). No
--gpu_id flag -- a unified launcher pins GPUs externally via
CUDA_VISIBLE_DEVICES.

Output: appends one row to verify_results/pico.csv (created with header if
new). Paper evaluates 5 independent seeds and reports mean+-std; this script
runs a single seed per invocation (noted in the CSV's `notes` column) -- run
it multiple times with different --seed values and average externally if a
mean+-std comparable to the paper's is wanted.
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Repo root on sys.path so `src.*`/`verify_scripts.*` imports work regardless
# of cwd (mirrors scripts/run_pipeline.py's own sys.path handling).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.collate import pico_collate_fn
from src.data_utils import ComparisonDataGenerator, PicoDataset
from src.engine import evaluate_model
from src.fixed_pico_engine import train_pico_epoch_fixed
from src.pico.model import PiCOModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.pipeline.datasets.cifar10 import _CIFAR_MEAN, _CIFAR_STD, _get_raw

from verify_scripts.pico_helpers import ArrayDatasetShim, ArrayTestDataset, candidate_masked_init_conf

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO_ROOT / 'verify_results' / 'pico.csv'
CSV_FIELDNAMES = ['dataset', 'config', 'seed', 'epochs', 'final_accuracy', 'paper_target_accuracy',
                   'training_time_s', 'timestamp', 'notes']

PAPER_TARGET_ACCURACY = 93.58  # Table 1, CIFAR-10 q=0.5.

# Paper-stated PiCO hyperparameters (Appendix B.1 / Eq. 5-7), matching
# config.yaml's `pico:` block exactly (see module docstring).
LOW_DIM = 128
MOCO_QUEUE = 8192
MOCO_M = 0.999
PROTO_M = 0.99
TAU = 0.07
LOSS_WEIGHT = 0.5
CONF_EMA_RANGE = [0.95, 0.8]
PROT_START = 1  # paper's general default (not the CIFAR-100 q=0.1 special case).

# LR / weight decay: NOT stated in the paper (see module docstring) --
# placeholder taken from this repo's config.yaml `training:` block.
PLACEHOLDER_LR = 0.001
PLACEHOLDER_WEIGHT_DECAY = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pico_args(C: int, epochs: int, prot_start: int = PROT_START) -> dict:
    return {
        'num_class': C,
        'epochs': epochs,
        'low_dim': LOW_DIM,
        'moco_queue': MOCO_QUEUE,
        'moco_m': MOCO_M,
        'proto_m': PROTO_M,
        'prot_start': prot_start,
        'loss_weight': LOSS_WEIGHT,
        'conf_ema_range': CONF_EMA_RANGE,
    }


def build_dataloaders(data_dir: str, q: float, batch_size: int, seed: int):
    """Builds the PiCO (weak/strong-aug pair) train loader and the plain
    test loader for real CIFAR-10, q-based candidate label generation.
    Reuses src/pipeline/datasets/cifar10.py for raw data, src/data_utils.py
    for candidate-set generation and the PicoDataset wrapper -- CIFAR-10's
    own mean/std already match PicoDataset's defaults, but they're passed
    explicitly here for clarity."""
    raw = _get_raw(data_dir)
    C = 10
    classes = [str(c) for c in range(C)]
    shim = ArrayDatasetShim(raw['train_data'], list(raw['train_targets']), classes)
    generator = ComparisonDataGenerator(shim, noise_type='clean', eta=0.0)
    pl_dataset_raw, _cl_dataset_raw = generator.generate_variable_pl_cl_datasets(q=q, num_classes=C)
    original_targets = generator.original_targets

    pico_dataset = PicoDataset(pl_dataset_raw, original_targets, image_size=32,
                                mean=_CIFAR_MEAN, std=_CIFAR_STD)
    pico_loader = DataLoader(pico_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              collate_fn=pico_collate_fn, pin_memory=torch.cuda.is_available())

    test_dataset = ArrayTestDataset(raw['test_data'], list(raw['test_targets']), mean=_CIFAR_MEAN, std=_CIFAR_STD)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return pico_loader, test_loader, pl_dataset_raw, C


def train_and_evaluate(pico_loader, test_loader, pl_dataset_raw, C: int, epochs: int, device,
                        lr: float = PLACEHOLDER_LR, weight_decay: float = PLACEHOLDER_WEIGHT_DECAY,
                        prot_start: int = PROT_START, eval_every: int = 1):
    """The actual paper-faithful PiCO training loop: SGD + cosine LR,
    train_pico_epoch_fixed (paper-faithful warm-up), candidate-masked
    pseudo-target init (Eq. 6). Kept as its own function (not inlined in
    main()) so a synthetic-data smoke test can call it directly with fake
    DataLoaders, bypassing real CIFAR-10 download/iteration entirely."""
    pico_args = build_pico_args(C, epochs, prot_start=prot_start)
    model = PiCOModel(pico_args).to(device)
    init_conf = candidate_masked_init_conf(pl_dataset_raw, C, device)
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss(temperature=TAU, base_temperature=TAU)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    final_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        cls_loss.set_conf_ema_m(epoch, pico_args)
        train_pico_epoch_fixed(pico_args, model, pico_loader, cls_loss, cont_loss, optimizer, epoch, device)
        scheduler.step()

        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            final_acc = evaluate_model(model, test_loader, device)
            print(f'  [PiCO-verify] epoch {epoch + 1}/{epochs}  test_acc={final_acc:.2f}%', flush=True)

    training_time_s = time.perf_counter() - t0
    return final_acc, training_time_s


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=800, help='Paper default: 800.')
    parser.add_argument('--seed', type=int, default=1, help='Paper runs 5 seeds and reports mean+-std; default 1.')
    parser.add_argument('--batch_size', type=int, default=256, help='Paper default: 256.')
    parser.add_argument('--q', type=float, default=0.5, help='Candidate label inclusion probability.')
    parser.add_argument('--data_dir', type=str, default='./data', help='CIFAR-10 root (downloaded if missing).')
    args = parser.parse_args()

    if MOCO_QUEUE % args.batch_size != 0:
        raise ValueError(f'moco_queue={MOCO_QUEUE} must be divisible by --batch_size={args.batch_size} '
                          f'(PiCOModel._dequeue_and_enqueue asserts this).')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[PiCO-verify] device={device} epochs={args.epochs} batch_size={args.batch_size} '
          f'q={args.q} seed={args.seed}', flush=True)

    pico_loader, test_loader, pl_dataset_raw, C = build_dataloaders(
        args.data_dir, args.q, args.batch_size, args.seed)

    final_acc, training_time_s = train_and_evaluate(
        pico_loader, test_loader, pl_dataset_raw, C, args.epochs, device,
        lr=PLACEHOLDER_LR, weight_decay=PLACEHOLDER_WEIGHT_DECAY, prot_start=PROT_START)

    row = {
        'dataset': 'cifar10',
        'config': f'q={args.q},prot_start={PROT_START}',
        'seed': args.seed,
        'epochs': args.epochs,
        'final_accuracy': round(final_acc, 2),
        'paper_target_accuracy': PAPER_TARGET_ACCURACY,
        'training_time_s': round(training_time_s, 1),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'notes': ('Single seed (paper reports mean+-std over 5 seeds). '
                  f'lr={PLACEHOLDER_LR}, weight_decay={PLACEHOLDER_WEIGHT_DECAY} are a placeholder '
                  '(paper Appendix B.1 states SGD momentum=0.9 + cosine LR but never states base '
                  'LR/weight_decay) taken from config.yaml training: block. Uses paper-faithful '
                  'train_pico_epoch_fixed (warm-up omits L_cont) + candidate-masked pseudo-target '
                  'init (Eq. 6), not the original pipeline PiCO algorithm ID\'s known deviations.'),
    }
    append_csv_row(RESULTS_CSV, row)

    print(f'[PiCO] dataset=cifar10 q={args.q} final_accuracy={final_acc:.2f}%  '
          f'paper_target={PAPER_TARGET_ACCURACY:.2f}%', flush=True)


if __name__ == '__main__':
    main()
