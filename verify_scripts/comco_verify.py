"""Paper-exact verification script for ComCo (Jiang, Sun & Tian, Neural
Networks 2024), CIFAR-10, unbiased single-complementary-label setting
(paper Table 1; headline reported accuracy 96.70%, mean over 5 trials --
this script runs a single seed).

Architecture: 18-layer ResNet (CIFAR-tuned stem) + MLP projection head
512->128, i.e. src/pico/resnet.py::SupConResNet, reused unmodified by
src/comco/model.py::ComCoModel (ComCo reuses PiCO's backbone class).

CL generation: uniform random draw of exactly 1 complementary label from the
C-1 non-true classes per sample (Ishida et al. 2017 scheme) via
src/data_utils.py::ComparisonDataGenerator.generate_cl_dataset(m=1).

Classification loss: FixedComCoCLSLoss (src/comco/fixed_utils_loss.py), NOT
the original ComCoCLSLoss -- a separate fidelity audit of this repo found the
original has an incorrect (C-1)/(C-m) classification-loss scaling for the
multi-CL case (borrowed from a different paper, Feng et al. 2020's MCL-NL);
the paper's own Section 5.1 states it uses plain unscaled SCL-NL. For the
single-CL setting run here the two are numerically identical (scale=1 when
m=1), but FixedComCoCLSLoss is used regardless for consistency with the
rest of this verification effort. Contrastive loss (ComCoContrastiveLoss)
is already paper-verified correct and reused as-is.

See verify_scripts/comco_helpers.py for the data/model/loss construction and
CSV/summary output helpers -- this file is CLI parsing + the epoch loop.

Usage:
    python verify_scripts/comco_verify.py [--epochs 1000] [--seed 42]
        [--batch_size 256] [--data_dir ./data]

All defaults are paper-exact. GPU selection is external (CUDA_VISIBLE_DEVICES
pinned by the unified verify_scripts launcher) -- no --gpu_id flag here;
device is chosen via torch.device('cuda' if torch.cuda.is_available() else
'cpu').

Smoke-testing note: a real 1000-epoch CIFAR-10 GPU run is not something to
attempt locally (Windows/CPU-only here). To verify this script's training
loop end-to-end without downloading CIFAR-10 or waiting on real training,
import build_model_and_losses / train_one_epoch / evaluate / write_result_row
from comco_helpers directly and swap in a tiny synthetic DataLoader that
yields (img_w, img_s, comp_mask, true_label, index) batches of the same
shapes ComCoDataset + comco_collate_fn would produce -- main()'s body below
only calls build_dataloaders() for the real-data path, so nothing else needs
to change to run the rest of this file's logic on fake data.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from comco_helpers import (
    BATCH_SIZE_DEFAULT, EPOCHS_DEFAULT, build_dataloaders, build_model_and_losses,
    evaluate, print_summary, train_one_epoch, write_result_row,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(
        description='ComCo paper-exact verification (CIFAR-10, unbiased single-complementary-label).')
    p.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument('--data_dir', type=str, default='./data')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[ComCo] device={device} epochs={args.epochs} batch_size={args.batch_size} '
          f'seed={args.seed} data_dir={args.data_dir}', flush=True)

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.seed)
    model, cls_loss, cont_loss, optimizer, comco_args = build_model_and_losses(args.epochs, device)

    report_every = max(1, args.epochs // 20)
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(comco_args, model, train_loader, cls_loss, cont_loss,
                                    optimizer, epoch, device)
        if (epoch + 1) % report_every == 0 or epoch + 1 == args.epochs:
            print(f'  [ComCo] epoch {epoch + 1}/{args.epochs}  avg_loss={avg_loss:.4f}', flush=True)
    training_time_s = time.perf_counter() - t0

    final_accuracy = evaluate(model, test_loader, device)

    csv_path = os.path.join(_REPO_ROOT, 'verify_results', 'comco.csv')
    write_result_row(csv_path, args.seed, args.epochs, final_accuracy, training_time_s,
                      notes='paper reports mean+-std over 5 trials; this run is a single seed')

    print_summary(final_accuracy)


if __name__ == '__main__':
    main()
