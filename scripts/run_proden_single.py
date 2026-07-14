"""
PRODEN + Cour2011 single run: C=10, k=7, Adam lr=3e-4 bs=512 wd=1e-4.

Usage:
    python scripts/run_proden_single.py
    python scripts/run_proden_single.py --epochs 500 --data_dir data/
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import prepare_cifar100_subset, get_subset_dataloaders
from src.clpl_loss import CLPLSquaredHingeLoss
from src.engine import train_algorithm
from src.models import create_model
from src.proden_loss import proden

CHUNK = 10


def _fmt_eta(s):
    if s < 90:    return f'{s:.0f}s'
    if s < 3600:  return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'


def run_method(name, loss_fn, loaders, C, epochs, lr, wd, device):
    model = create_model(C).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    t_start   = time.perf_counter()
    final_acc = 0.0

    for ep_start in range(0, epochs, CHUNK):
        chunk = min(CHUNK, epochs - ep_start)
        t0    = time.perf_counter()
        accs  = train_algorithm(model, loaders['pl'], loaders['test'],
                                loss_fn, opt, chunk, device)
        elapsed   = time.perf_counter() - t0
        final_acc = accs[-1]

        ep_done = ep_start + chunk
        avg_s   = elapsed / chunk
        eta_s   = avg_s * (epochs - ep_done)
        print(f'  [{name}]  ep {ep_done:>3}/{epochs}  '
              f'{avg_s:.1f}s/ep  acc={final_acc:.2f}%  ETA {_fmt_eta(eta_s)}',
              flush=True)

    total_s = time.perf_counter() - t_start
    print(f'  [{name}] DONE  final={final_acc:.2f}%  ({total_s/60:.1f} min)\n')

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C',        type=int,   default=10)
    parser.add_argument('--k',        type=int,   default=7)
    parser.add_argument('--epochs',   type=int,   default=200)
    parser.add_argument('--lr',       type=float, default=3e-4)
    parser.add_argument('--bs',       type=int,   default=512)
    parser.add_argument('--wd',       type=float, default=1e-4)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--data_dir', type=str,   default='data/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'C={args.C}  k={args.k}  epochs={args.epochs}  '
          f'lr={args.lr}  bs={args.bs}  wd={args.wd}\n', flush=True)

    # ── Data (shared between both methods) ────────────────────────────────────
    pl_raw, cl_raw, orig_targets, test_info, log_info = prepare_cifar100_subset(
        total_classes=args.C,
        n_partial_labels=args.k,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    print(f"Classes : {log_info['selected_class_names']}")
    print(f"Samples : {log_info['n_train']} train / {log_info['n_test']} test\n")

    loaders = get_subset_dataloaders(
        pl_raw, cl_raw, orig_targets, test_info, batch_size=args.bs
    )

    # ── PRODEN ────────────────────────────────────────────────────────────────
    acc_proden = run_method('PRODEN', proden(), loaders,
                            args.C, args.epochs, args.lr, args.wd, device)

    # ── Cour2011 ──────────────────────────────────────────────────────────────
    acc_cour = run_method('Cour2011', CLPLSquaredHingeLoss(), loaders,
                          args.C, args.epochs, args.lr, args.wd, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('=' * 45)
    print(f'  C={args.C}  k={args.k}  epochs={args.epochs}')
    print(f'  PRODEN  : {acc_proden:.2f}%')
    print(f'  Cour2011: {acc_cour:.2f}%')
    print('=' * 45)


if __name__ == '__main__':
    main()
