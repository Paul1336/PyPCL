#!/usr/bin/env python
"""Standalone paper-exact verification script for PRODEN on real CIFAR-10.

PRODEN = Lv, Xu, Feng, Niu, Geng, Sugiyama, "Progressive Identification of
True Labels for Partial-Label Learning", ICML 2020.

This is intentionally SEPARATE from the shared comparison pipeline
(scripts/run_pipeline.py, src/pipeline/algorithms/runners.py::run_proden),
which trains PRODEN on CIFAR-100 class-subsets with a generic ResNet-18
regardless of the target dataset -- the wrong architecture for reproducing
this paper's own CIFAR-10 numbers. This script instead reproduces ONE
specific configuration from the paper as literally as possible:

    CIFAR-10 + the paper's own 12-layer ConvNet (Laine & Aila, 2017 family)
    + binomial partial-label generation with q=0.1.

Paper settings used here, confirmed by direct pypdf text extraction from
C:\\Users\\User\\Desktop\\papers\\PRODEN.pdf (quoted, not reconstructed from
memory):

  Architecture (Appendix E.1, CIFAR-10 paragraph):
    "The detailed architecture of ConvNet (Laine & Aila, 2017) is as
    follows. 0th (input) layer: (32*32*3)- 1st to 4th layers: [C(3*3,
    128)]*3-Max Pooling- 5th to 8th layers: [C(3*3, 256)]*3-Max Pooling- 9th
    to 11th layers: C(3*3, 512)-C(3*3, 256)-C(3*3, 128)- 12th layers:
    Average Pooling-10 where C(3*3, 128) means 128 channels of 3*3
    convolutions followed by Leaky-ReLU (LReLU) active function (Maas et
    al., 2013)"
    -> implemented in verify_scripts/proden_model.py::PRODENConvNet.

  Optimizer / schedule (Section 5 "Experimental setup"):
    "The optimizer is stochastic gradient descent (SGD) (Robbins & Monro,
    1951) with momentum 0.9. We train each model 500 epochs with softmax
    function and cross-entropy loss."
    Batch size 256 is stated explicitly for the MNIST/ResNet-CIFAR
    paragraphs in Appendix E.1 ("...the batch size was set to 256." /
    "...the batch size was 256.") and is used uniformly across all image
    experiments in this appendix, including the ConvNet-CIFAR cell.

  Label generation (Section 5): "we conduct the experiments under both
    less-partial circumstances q = 0.1 and strong-partial circumstances q =
    0.7" (binomial scheme: each false label independently included w.p. q).
    This script defaults to q=0.1 (the paper's "less-partial" setting).

  Evaluation convention (Section 5, "Results on benchmark datasets"): "We
    average the classification accuracy of PRODEN over the last 10 epochs
    as the results to prove that PRODEN is always stable and will not cause
    performance degradation due to overfitting when the number of epochs
    increases." -- replicated literally below (mean of the last 10 tracked
    test-accuracy values).

  Target number: the paper reports CIFAR-10 + ConvNet + q=0.1 only as an
    epoch-accuracy curve in Figure 1 ("CIFAR, ConvNet, q = 0.1"), not as a
    precise table value (Tables 1/2 are UCI + real-world PLL datasets only,
    not CIFAR). There is no precise published number for this exact
    (dataset, architecture, q) combination to validate against --
    `paper_target_accuracy` is left blank in the output CSV for that reason.

ASSUMPTIONS (paper text does not state these explicitly for this
architecture/setting; flagged rather than silently guessed -- see also the
assumptions block in verify_scripts/proden_model.py):
  - BatchNorm after every conv (paper's Appendix E.1 only explicitly
    mentions BatchNorm for the MNIST MLP model, not for the CIFAR ConvNet;
    included here as standard convention for this well-known Laine & Aila
    "conv-large" architecture family).
  - Learning rate 0.01 and weight_decay 1e-4. The paper never states a
    numeric learning rate anywhere in the main text or appendix, and only
    says "L2-regularization added" generically (in the MNIST paragraph)
    without a coefficient. These values match what this repo's OWN existing
    PRODEN hyperparameters already use for the shared pipeline
    (src/pipeline/algorithms/hparams.py: `_SGD = dict(optimizer='sgd',
    lr=0.01, momentum=0.9, weight_decay=1e-4)`, commented there as
    "following their original papers"), used here for consistency.
  - LeakyReLU negative_slope=0.1, matching the Laine & Aila (2017) "conv-large"
    network's own published LReLU slope (the paper cites Maas et al. 2013 for
    the LReLU activation function itself, whose default slope is 0.01, but
    only to name the nonlinearity -- the architecture it cites, Laine & Aila
    2017, specifies 0.1 for this exact network).

Reused, not reimplemented, from the existing repo (do not re-verify or
modify -- see CLAUDE.md and their own docstrings):
  - src/proden_loss.py::ProdenLoss -- already paper-verified faithful to
    the PRODEN paper's Algorithm 1 (cross-epoch confidence accumulation).
  - src/data_utils.py::ComparisonDataGenerator.generate_variable_pl_cl_datasets
    -- implements exactly the "each false label independently included
    w.p. q" binomial scheme used here.

Usage:
    python verify_scripts/proden_verify.py \\
        [--epochs 500] [--seed 42] [--batch_size 256] [--q 0.1] [--data_dir ./data]

GPU selection is NOT handled by a --gpu_id flag here (per this repo's
existing convention, e.g. scripts/launch_tmux_job.sh) -- a unified launcher
is expected to pin the GPU externally via CUDA_VISIBLE_DEVICES before
invoking this script; this script itself just picks
`cuda if torch.cuda.is_available() else cpu`.
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Make `src` (repo root) and `proden_model` (this script's own directory)
# importable regardless of the invoking cwd (mirrors scripts/run_pipeline.py's
# own sys.path handling).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_utils import ComparisonDataGenerator  # noqa: E402
from src.proden_loss import ProdenLoss  # noqa: E402

from proden_model import PRODENConvNet  # noqa: E402

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.247, 0.2435, 0.2616)

_RESULTS_DIR = os.path.join(_REPO_ROOT, 'verify_results')
_RESULTS_CSV = os.path.join(_RESULTS_DIR, 'proden.csv')
_CSV_HEADER = ['dataset', 'config', 'seed', 'epochs', 'final_accuracy',
               'paper_target_accuracy', 'training_time_s', 'timestamp', 'notes']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _IndexedTrainDataset(Dataset):
    """Wraps the (weakly-labelled) training images; returns (image, index)
    since ProdenLoss needs the sample index to look up/update its persistent
    per-sample confidence buffer, not the (unusable, ambiguous) label
    itself. Mirrors src/pipeline/algorithms/runners.py::_IndexedDataset
    (image-modality branch), rebuilt standalone here per this task's
    'do not modify/import runners.py' scope (that module also pulls in
    every other algorithm's heavy deps -- PiCO, ComCo, SoLar, etc. -- which
    this standalone script has no need for)."""

    def __init__(self, images: np.ndarray):
        self.images = images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), idx


class _EvalDataset(Dataset):
    """Standard (image, true_label) test-set wrapper -- evaluation is
    always against ground-truth labels, never the weak/candidate labels."""

    def __init__(self, images: np.ndarray, targets):
        self.images = images
        self.targets = targets
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), int(self.targets[idx])


def load_cifar10(data_dir: str, download: bool = True):
    """Real CIFAR-10 train/test split via torchvision (downloads on first
    use, cached under data_dir like the rest of this repo). Returns the raw
    torchvision CIFAR10 dataset objects directly -- they already expose the
    .classes/.data/.targets/__len__/__getitem__ interface
    ComparisonDataGenerator expects (see src/data_utils.py), same as
    src/pipeline/datasets/cifar10.py's own loader.

    Factored out as its own function (rather than inlined in main()) so a
    smoke test can monkeypatch it with synthetic data without touching
    network I/O or the rest of the training pipeline.
    """
    train = CIFAR10(root=data_dir, train=True, download=download)
    test = CIFAR10(root=data_dir, train=False, download=download)
    return train, test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def append_result_row(row: dict) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(_RESULTS_CSV)
    with open(_RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='PRODEN paper-exact verification: CIFAR-10 + 12-layer ConvNet + binomial q.')
    p.add_argument('--epochs', type=int, default=500,
                    help='Training epochs (paper: 500).')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument('--batch_size', type=int, default=256,
                    help='SGD batch size (paper: 256).')
    p.add_argument('--q', type=float, default=0.1,
                    help='Binomial false-label inclusion probability (paper: 0.1 or 0.7; default 0.1).')
    p.add_argument('--data_dir', type=str, default='./data',
                    help='CIFAR-10 download/cache directory.')
    return p


def main(args=None) -> float:
    parser = build_arg_parser()
    args = parser.parse_args(args)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[PRODEN] device={device} epochs={args.epochs} seed={args.seed} '
          f'batch_size={args.batch_size} q={args.q} data_dir={args.data_dir}', flush=True)

    train_raw, test_raw = load_cifar10(args.data_dir)
    num_classes = len(train_raw.classes)

    # Paper-exact "binomial" candidate-label generation: each false label
    # independently included w.p. q (reused as-is, see module docstring).
    generator = ComparisonDataGenerator(train_raw, noise_type='clean', eta=0.0)
    pl_train, _cl_train = generator.generate_variable_pl_cl_datasets(args.q, num_classes)

    train_ds = _IndexedTrainDataset(pl_train.data)
    test_ds = _EvalDataset(test_raw.data, test_raw.targets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=max(args.batch_size, 1), shuffle=False, num_workers=0)

    model = PRODENConvNet(num_classes=num_classes).to(device)
    loss_fn = ProdenLoss(pl_train.targets, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    test_accuracies = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        n_batches = 0
        for images, indices in train_loader:
            images, indices = images.to(device), indices.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, indices)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            n_batches += 1

        acc = evaluate(model, test_loader, device)
        test_accuracies.append(acc)

        avg_loss = epoch_loss_sum / max(n_batches, 1)
        if (epoch + 1) % max(1, args.epochs // 20 or 1) == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f'  epoch {epoch + 1:>4}/{args.epochs}  loss={avg_loss:.4f}  test_acc={acc:.2f}%', flush=True)

    training_time_s = time.perf_counter() - t0

    last_k = test_accuracies[-10:] if len(test_accuracies) >= 10 else test_accuracies
    final_accuracy = sum(last_k) / len(last_k)

    config = f'12layer_convnet,q={args.q}'
    notes = ('No precise published target exists for CIFAR-10+ConvNet+q=0.1 '
             '(paper reports this cell only as an epoch curve in Figure 1, not a table '
             'value). Assumptions: BatchNorm included (not stated for ConvNet in paper), '
             'lr=0.01/weight_decay=1e-4 (not stated in paper; matches this repo\'s existing '
             'PRODEN hparams), LeakyReLU slope=0.1 (Laine & Aila 2017 conv-large convention). '
             f'final_accuracy = mean of last {len(last_k)} epochs\' test accuracy per paper convention.')

    append_result_row({
        'dataset': 'cifar10',
        'config': config,
        'seed': args.seed,
        'epochs': args.epochs,
        'final_accuracy': f'{final_accuracy:.4f}',
        'paper_target_accuracy': '',
        'training_time_s': f'{training_time_s:.2f}',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'notes': notes,
    })

    print(f'[PRODEN] dataset=cifar10 arch=12layer_convnet q={args.q} '
          f'final_accuracy={final_accuracy:.2f}% (mean of last {len(last_k)} epochs)', flush=True)

    return final_accuracy


if __name__ == '__main__':
    main()
