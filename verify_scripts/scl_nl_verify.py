#!/usr/bin/env python
"""Standalone, paper-exact verification script for SCL-NL.

Chou, Niu, Lin, Sugiyama, ICML 2020, "Unbiased Risk Estimators Can Mislead:
A Case Study of Learning with Complementary Labels". This script reproduces
the paper's Table 1 CIFAR-10 + ResNet-34 classification-accuracy setting as
closely as the source PDF allows, independent of the main comparison
pipeline (`scripts/run_pipeline.py`) -- see `verify_scripts/scl_nl_model.py`
for the architecture and complementary-label-generation code this script
builds on, and its module docstrings for the specific judgment calls made
where the paper is silent.

Confirmed-from-paper settings used here (Table 1 target: SCL-NL on
CIFAR-10 + ResNet(34) = 0.4713 test accuracy):
    - Architecture: ResNet-34 (CIFAR stem -- see scl_nl_model.py for the
      conv1-only modification, mirroring src/models.py::ResNet18's own
      precedent, same judgment call independently made by
      verify_scripts/mcl_log_model.py for consistency between the two).
    - Complementary-label generation: paper's "Uniform Assumption" -- exactly
      ONE uniformly-random complementary label per sample, drawn via
      `src/data_utils.py::ComparisonDataGenerator.generate_cl_dataset(m=1)`
      reused directly and unmodified (see scl_nl_model.py docstring).
    - Loss: `src/scl_loss.py::SCL_NL` (Eq. 11), reused directly and
      unmodified -- already verified faithful for the single-CL case.
    - Optimizer: Adam (paper doesn't state betas/weight_decay -- left at
      PyTorch defaults, no weight_decay, since the paper names only the
      optimizer family).
    - batch_size=256: NOT stated by the paper -- user-selected value, since
      the paper's Appendix E.1 gives no batch size for this experiment.
    - epochs=300 at EACH of 5 candidate learning rates
      {1e-1, 1e-2, 1e-3, 1e-4, 1e-5} (paper: "learning rate selected from
      {1e-1, ..., 1e-5}" -- vague about the selection *procedure*, so per
      user decision this script runs a full 300-epoch training at every
      candidate lr and reports the lr with the best final test accuracy as
      the headline result, logging ALL 5 runs to the results CSV).
    - Evaluation: paper reports a single number (no stated seed count /
      variance) for this Table 1 row; this script runs a single seed (see
      `--seed`) per lr, noted in the CSV `notes` column.

CLI:
    python verify_scripts/scl_nl_verify.py [--epochs N] [--seed N]
        [--batch_size N] [--lrs v1,v2,...] [--data_dir ./data]

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

from src.scl_loss import SCL_NL  # noqa: E402  (paper-verified as-is, reused unmodified)
from verify_scripts.scl_nl_model import create_model, generate_single_complementary_labels  # noqa: E402

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.247, 0.2435, 0.2616)

# Table 1: SCL-NL on CIFAR-10 + ResNet(34) = 0.4713 (fraction, not percent --
# stored as a fraction throughout this script's CSV/columns so it is directly
# comparable to `final_accuracy`, also stored as a fraction).
PAPER_TARGET_ACCURACY = 0.4713

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV_PATH = os.path.join(_REPO_ROOT, 'verify_results', 'scl_nl.csv')
CSV_HEADER = [
    'dataset', 'config', 'seed', 'epochs', 'final_accuracy',
    'paper_target_accuracy', 'training_time_s', 'timestamp', 'notes',
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ComplementaryLabelDataset(Dataset):
    """Images + single-complementary-label index tensors (shape [1] each --
    no -1 padding needed since every sample has exactly one complementary
    label), the format `src/scl_loss.py::SCL_NL` expects (it also accepts
    the general padded multi-CL shape, but degenerates correctly here)."""

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
    """Images + true integer class labels, for test-accuracy evaluation
    (never exposes complementary labels to the model)."""

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
# src/pipeline/datasets/cifar10.py; single-CL label generation is
# scl_nl_model.py's generate_single_complementary_labels, see its docstring)
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


def train_model(model, loader, optimizer, criterion, device, epochs, lr_tag='', report_every=50):
    # report_every kept as a parameter (still accepted, e.g. by callers) but
    # no longer gates whether a line prints -- every epoch reports now. A
    # silent multi-epoch gap was easy to mistake for a hang, per user
    # feedback after a real interruption incident.
    model.train()
    run_t0 = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for imgs, comp_labels in loader:
            imgs = imgs.to(device)
            comp_labels = comp_labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, comp_labels)
            if torch.isnan(loss):
                raise RuntimeError(f"SCL_NL loss went NaN during training (lr={lr_tag}, epoch={epoch})")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_s = time.time() - t0
        avg_s_per_ep = (time.time() - run_t0) / (epoch + 1)
        eta_min = avg_s_per_ep * (epochs - epoch - 1) / 60
        print(f"  [SCL-NL][lr={lr_tag}] epoch {epoch + 1}/{epochs} "
              f"avg_loss={epoch_loss / max(n_batches, 1):.4f}  {epoch_s:.1f}s/ep  ETA {eta_min:.1f}min", flush=True)
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
    return correct / max(total, 1)  # fraction, not percent


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
# One learning rate's full training + evaluation run (also the unit the
# smoke test exercises directly with synthetic DataLoaders).
# ---------------------------------------------------------------------------

def run_one_lr(lr, num_classes, train_loader, test_loader, device, epochs, in_channels=3, report_every=50):
    model = create_model(num_classes, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SCL_NL()

    t0 = time.time()
    train_model(model, train_loader, optimizer, criterion, device, epochs, lr_tag=f"{lr:g}", report_every=report_every)
    training_time_s = time.time() - t0

    accuracy = evaluate(model, test_loader, device)
    del model, optimizer
    return accuracy, training_time_s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def build_arg_parser():
    p = argparse.ArgumentParser(
        description='Paper-exact SCL-NL verification: CIFAR-10 + ResNet-34 (Table 1).')
    p.add_argument('--epochs', type=int, default=300, help='Paper: 300.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=256,
                    help='Paper does not state a batch size -- user-selected default.')
    p.add_argument('--lrs', type=str, default='1e-1,1e-2,1e-3,1e-4,1e-5',
                    help="Comma-separated candidate learning rates. Paper: "
                         "'learning rate selected from {1e-1, ..., 1e-5}' -- this "
                         "script trains a full run at EVERY candidate and reports "
                         "the best final test accuracy (matching the paper's own "
                         "post-hoc-selection phrasing literally).")
    p.add_argument('--report_every', type=int, default=50,
                    help='Print a progress line every N epochs (0 to disable).')
    p.add_argument('--data_dir', type=str, default='./data')
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lrs = parse_float_list(args.lrs)
    num_classes = 10

    train_images, train_targets, test_images, test_targets = load_cifar10_arrays(args.data_dir)

    # Paper's Uniform Assumption: exactly ONE uniformly-random complementary
    # label per sample (see scl_nl_model.py). Generated once, reused for
    # every candidate lr (same complementarily-labeled training set across
    # the whole sweep, only lr varies).
    cl_dataset_raw, _original_targets = generate_single_complementary_labels(
        train_images, train_targets, num_classes, args.seed)
    comp_label_widths = {len(t) for t in cl_dataset_raw.targets}
    assert comp_label_widths == {1}, (
        f"Expected exactly 1 complementary label/sample (paper's Uniform Assumption), "
        f"got widths {comp_label_widths}")
    print(f"[SCL-NL] complementary labels generated: n_complementary_labels=1/sample "
          f"(confirmed over {len(cl_dataset_raw.targets)} samples), dataset=cifar10 C={num_classes}", flush=True)

    train_tf, eval_tf = build_transforms()
    train_ds = ComplementaryLabelDataset(cl_dataset_raw.data, cl_dataset_raw.targets, train_tf)
    test_ds = EvalDataset(test_images, test_targets, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = []
    for lr in lrs:
        print(f"[SCL-NL] starting lr={lr:g} ({args.epochs} epochs, batch_size={args.batch_size})...", flush=True)
        accuracy, training_time_s = run_one_lr(
            lr, num_classes, train_loader, test_loader, device, args.epochs,
            report_every=args.report_every)
        results.append({'lr': lr, 'accuracy': accuracy, 'training_time_s': training_time_s})
        print(f"[SCL-NL] lr={lr:g} accuracy={accuracy * 100:.2f}%", flush=True)

    best = max(results, key=lambda r: r['accuracy'])

    base_notes = (
        "batch_size=256 (paper unspecified, user-selected); optimizer=Adam with "
        "default betas and no weight_decay (paper names only 'Adam', no further "
        "hyperparameters given); lr selected post-hoc from "
        f"{{{','.join(f'{l:g}' for l in lrs)}}} by best final test accuracy, "
        "matching the paper's own vague 'learning rate selected from {...}' "
        "phrasing literally (full 300-epoch run at every candidate, not an "
        "early-stopped search); complementary labels: exactly 1/sample via "
        "the paper's Uniform Assumption "
        "(ComparisonDataGenerator.generate_cl_dataset(m=1)); single seed "
        f"({args.seed}) run, paper does not state a trial count for Table 1."
    )

    for r in results:
        is_best = (r is best)
        row = {
            'dataset': 'cifar10',
            'config': f"resnet34,lr={r['lr']:g}",
            'seed': args.seed,
            'epochs': args.epochs,
            'final_accuracy': f"{r['accuracy']:.4f}",
            'paper_target_accuracy': PAPER_TARGET_ACCURACY,
            'training_time_s': f"{r['training_time_s']:.2f}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'notes': base_notes + (' BEST' if is_best else ''),
        }
        write_result_row(RESULTS_CSV_PATH, row)

    print(f"[SCL-NL] BEST: lr={best['lr']:g} final_accuracy={best['accuracy'] * 100:.2f}%  "
          f"paper_target={PAPER_TARGET_ACCURACY * 100:.2f}%")

    return results, best


if __name__ == '__main__':
    main()
