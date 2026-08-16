#!/usr/bin/env python
"""Standalone, paper-exact verification script for MCL-LOG.

Feng, Kaneko, Han, Niu, An, Sugiyama, ICML 2020, "Learning with Multiple
Complementary Labels". This script reproduces the paper's CIFAR-10 +
ResNet-34 setting (Table 4 target: 75.38 +/- 0.34% test accuracy) as
closely as the (appendix-less, 10-page) source PDF allows, independent of
the main comparison pipeline (`scripts/run_pipeline.py`) -- see
`verify_scripts/mcl_log_model.py` for the architecture and complementary-
label-generation code this script builds on, and the module docstrings
there for the specific judgment calls made where the paper is silent.

Confirmed-from-paper settings used here:
    - Architecture: ResNet-34 (CIFAR stem, see mcl_log_model.py)
    - Complementary-label generation: paper's own p(s) = C(k,s)/(2^k-2)
      combinatorial scheme (k=10 for CIFAR-10), NOT this repo's default
      independent-inclusion-probability `q` mechanism.
    - Optimizer: Adam, batch_size=256 (default), epochs=250 (default).
    - lr / weight_decay: paper grid-searches {1e-6,...,1e-1} x 7 for each,
      selected by accuracy on a 10%-held-out split of the complementarily
      labeled training data. SCOPE-DOWN (deliberate, documented in the CSV
      `notes` column and printed at the end of each run): a full 7x7=49
      combo search is impractical for a lightweight verification run, so
      the *default* grid here is a smaller representative subset
      (lr in {1e-2,1e-3,1e-4}, wd in {1e-4,1e-5} = 6 combos), each trained
      for only `--val_epochs` epochs before being scored on the held-out
      split. The grid-search machinery itself (`run_grid_search` below) is
      general -- pass a larger `--lr_grid` / `--wd_grid` for a full,
      paper-exact sweep on a GPU server.
    - Evaluation: paper reports mean +/- std over 5 trials with a paired
      t-test; this script runs a single seed (see `--seed`) and notes the
      discrepancy in the CSV `notes` column.

CLI:
    python verify_scripts/mcl_log_verify.py [--epochs N] [--seed N]
        [--batch_size N] [--lr_grid v1,v2,...] [--wd_grid v1,v2,...]
        [--val_epochs N] [--data_dir ./data]

GPU handling: device is chosen automatically via
`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. There is
no `--gpu_id` flag by design -- an external unified launcher is expected
to pin the visible GPU via CUDA_VISIBLE_DEVICES.
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fixed_mcl_losses import FixedMCLLog as MCL_LOG  # noqa: E402
# NOTE (corrected 2026-08-16): the ORIGINAL src.mcl_losses.MCL_LOG has a
# confirmed unbiased-risk-estimator scaling bug -- paper Eq. 12 scales by
# 2*(C-1)/m, the original code uses (C-1)/(C-m) instead (off by a factor of
# ~18x at C=10, m=1). FixedMCLLog (src/fixed_mcl_losses.py) has the correct
# paper scaling and the same (num_classes) / forward(outputs, labels)
# signature, so it's aliased to the name MCL_LOG here as a drop-in swap
# rather than renaming every call site below. See docs/mcl_explanation.md.
from verify_scripts.mcl_log_model import create_model, generate_complementary_labels  # noqa: E402

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.247, 0.2435, 0.2616)

PAPER_TARGET_ACCURACY = 75.38  # Table 4, MCL-LOG on CIFAR-10 + ResNet (paper reports 75.38+-0.34, mean of 5 trials)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV_PATH = os.path.join(_REPO_ROOT, 'verify_results', 'mcl_log.csv')
CSV_HEADER = [
    'dataset', 'config', 'seed', 'epochs', 'final_accuracy',
    'paper_target_accuracy', 'training_time_s', 'timestamp', 'notes',
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ComplementaryLabelDataset(Dataset):
    """Images + padded (-1-filled) complementary-label index tensors, the
    format `src/mcl_losses.py::MCL_LOG` expects."""

    def __init__(self, images, comp_labels, transform):
        self.images = images
        self.comp_labels = comp_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        img = self.transform(img)
        return img, self.comp_labels[idx]


class EvalDataset(Dataset):
    """Images + true integer class labels, for accuracy evaluation (used
    both for the held-out grid-search validation split and the test set --
    neither ever exposes complementary labels to the model)."""

    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        img = self.transform(img)
        return img, int(self.targets[idx])


def build_transforms(image_size=32):
    train_tf = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Data loading (reuses the raw-array-loading half of
# src/pipeline/datasets/cifar10.py; label generation is entirely our own,
# see mcl_log_model.py)
# ---------------------------------------------------------------------------

def load_cifar10_arrays(data_dir):
    from src.pipeline.datasets.cifar10 import _get_raw
    raw = _get_raw(data_dir)
    return raw['train_data'], raw['train_targets'], raw['test_data'], raw['test_targets']


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for _epoch in range(epochs):
        for imgs, comp_labels in loader:
            imgs = imgs.to(device)
            comp_labels = comp_labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, comp_labels)
            if torch.isnan(loss):
                raise RuntimeError("MCL_LOG loss went NaN during training")
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Grid-search machinery (general-purpose: works for any lr_grid/wd_grid,
# the "smaller representative grid" is only the *default* CLI value, see
# module docstring)
# ---------------------------------------------------------------------------

def run_grid_search(lr_grid, wd_grid, num_classes, gs_train_ds, gs_val_ds,
                     val_epochs, batch_size, device, num_workers=0):
    gs_train_loader = DataLoader(gs_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    gs_val_loader = DataLoader(gs_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results = []
    best = None
    for lr in lr_grid:
        for wd in wd_grid:
            model = create_model(num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = MCL_LOG(num_classes=num_classes)
            train_model(model, gs_train_loader, optimizer, criterion, device, val_epochs)
            acc = evaluate(model, gs_val_loader, device)
            results.append({'lr': lr, 'wd': wd, 'val_accuracy': acc})
            print(f"[MCL-LOG][grid-search] lr={lr:g} wd={wd:g} -> "
                  f"held_out_val_accuracy={acc:.2f}%", flush=True)
            if best is None or acc > best['val_accuracy']:
                best = {'lr': lr, 'wd': wd, 'val_accuracy': acc}
            del model
    return best, results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_result_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def build_arg_parser():
    p = argparse.ArgumentParser(
        description='Paper-exact MCL-LOG verification: CIFAR-10 + ResNet-34.')
    p.add_argument('--epochs', type=int, default=250,
                    help='Final-training epoch budget (paper: 250).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=256, help='Paper: 256.')
    p.add_argument('--lr_grid', type=str, default='1e-2,1e-3,1e-4',
                    help='Comma-separated learning rates for the grid search. '
                         'Paper searched {1e-6,...,1e-1} (7 values); default here '
                         'is a smaller representative subset -- pass a larger list '
                         'for the full paper-exact sweep.')
    p.add_argument('--wd_grid', type=str, default='1e-4,1e-5',
                    help='Comma-separated weight-decay values for the grid search. '
                         'Paper searched {1e-6,...,1e-1} (7 values); default here '
                         'is a smaller representative subset.')
    p.add_argument('--val_epochs', type=int, default=5,
                    help='Short training budget used only during grid-search model '
                         'selection (not the final full run).')
    p.add_argument('--data_dir', type=str, default='./data')
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr_grid = parse_float_list(args.lr_grid)
    wd_grid = parse_float_list(args.wd_grid)

    num_classes = 10
    train_images, train_targets, test_images, test_targets = load_cifar10_arrays(args.data_dir)

    # Paper's own p(s) = C(k,s)/(2^k-2) combinatorial complementary-label
    # generation scheme (see mcl_log_model.py), applied once to the whole
    # training set.
    comp_labels_all = generate_complementary_labels(train_targets, num_classes, args.seed)

    # 10%-held-out validation split of the complementarily-labeled training
    # data, used only for grid-search model selection (paper's own protocol).
    n = len(train_targets)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    val_size = int(0.1 * n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_tf, eval_tf = build_transforms()

    gs_train_ds = ComplementaryLabelDataset(train_images[train_idx], comp_labels_all[train_idx], train_tf)
    gs_val_ds = EvalDataset(train_images[val_idx], train_targets[val_idx], eval_tf)

    t_grid_start = time.time()
    best, _grid_results = run_grid_search(
        lr_grid, wd_grid, num_classes, gs_train_ds, gs_val_ds,
        args.val_epochs, args.batch_size, device)
    grid_search_time_s = time.time() - t_grid_start
    print(f"[MCL-LOG] grid search selected lr={best['lr']:g} wd={best['wd']:g} "
          f"(held_out_val_accuracy={best['val_accuracy']:.2f}%)", flush=True)

    # Final training. JUDGMENT CALL (not specified by the paper): the 10%
    # validation split is merged back into the training set for the final
    # run (common practice once hyperparameters are fixed), rather than
    # withheld throughout -- documented here and in the CSV `notes` column.
    full_train_ds = ComplementaryLabelDataset(train_images, comp_labels_all, train_tf)
    full_train_loader = DataLoader(full_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_ds = EvalDataset(test_images, test_targets, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    final_model = create_model(num_classes).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best['lr'], weight_decay=best['wd'])
    criterion = MCL_LOG(num_classes=num_classes)

    t_final_start = time.time()
    train_model(final_model, full_train_loader, optimizer, criterion, device, args.epochs)
    final_train_time_s = time.time() - t_final_start

    final_accuracy = evaluate(final_model, test_loader, device)
    total_time_s = grid_search_time_s + final_train_time_s

    config_str = f"resnet34,lr={best['lr']:g},wd={best['wd']:g}"
    notes = (
        f"Grid-search scope-down: paper grid-searches lr,wd over a 7x7=49-combo "
        f"{{1e-6,...,1e-1}} grid selected via 10%-held-out validation accuracy; "
        f"this run used a reduced {len(lr_grid)}x{len(wd_grid)}-combo grid "
        f"(lr_grid={lr_grid}, wd_grid={wd_grid}) with val_epochs={args.val_epochs} "
        f"per combo for tractability (grid_search_time_s={grid_search_time_s:.1f}). "
        f"Paper reports mean+-std over 5 trials with a paired t-test; this run is "
        f"a single seed ({args.seed}). Held-out val split merged back into the "
        f"training set for final training (judgment call, not specified by paper)."
    )

    row = {
        'dataset': 'cifar10',
        'config': config_str,
        'seed': args.seed,
        'epochs': args.epochs,
        'final_accuracy': f"{final_accuracy:.4f}",
        'paper_target_accuracy': PAPER_TARGET_ACCURACY,
        'training_time_s': f"{total_time_s:.2f}",
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'notes': notes,
    }
    write_result_row(RESULTS_CSV_PATH, row)

    print(f"[MCL-LOG] dataset=cifar10 arch=resnet34 final_accuracy={final_accuracy:.2f}%  "
          f"paper_target=75.38%  selected_lr={best['lr']:g} selected_wd={best['wd']:g}")

    return row


if __name__ == '__main__':
    main()
