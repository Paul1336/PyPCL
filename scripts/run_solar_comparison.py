"""
SoLar comparison script — trains SoLar and produces a comprehensive plot
combining all 11 methods across all three experiment runs.

SoLar-specific notes:
  - Optimizer: SGD  lr=0.01  momentum=0.9  wd=1e-4  bs=512
  - 2-stage training: est_epochs (pre-estimation) + epochs (final)
  - Uses SoLarDataset (weak+strong augmentation, one-hot PL vectors)
  - Results written to results/solar_comparison/gpu{id}/results.csv

Comprehensive plots (written to plots/solar_comparison/) read from:
  results/adam_comparison/   — CLPL, PRODEN, PiCO, PiCO-MCL, MCL-LOG, SCL-NL, ComCo
  results/op_cpe_comparison/ — OP, OP-W, CPE
  results/solar_comparison/  — SoLar

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_solar_comparison.py
  CUDA_VISIBLE_DEVICES=0 python scripts/run_solar_comparison.py --only_c 5 --only_k 3
"""

import argparse
import csv
import gc
import glob
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset
from src.collate import solar_collate_fn
from src.data_utils import SoLarDataset
from src.engine import evaluate_model, train_solar
from src.model_setup import setup_solar

# ─── Constants ────────────────────────────────────────────────────────────────

C_VALUES     = [5, 20]
LR_SOLAR     = 0.01        # SGD lr (same as PRODEN / original SoLar paper)
MOMENTUM     = 0.9
WD           = 1e-4
BS           = 512
REPORT_EVERY = 10

ALL_ALGOS = ['SoLar']

# ─── Combined visual styles (all 11 methods) ──────────────────────────────────

_RENAME = {'Cour2011': 'CLPL'}

STYLES = {
    # PLL methods (from adam_comparison)
    'CLPL':     dict(color='#1f77b4', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'PRODEN':   dict(color='#2ca02c', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'PiCO-MCL': dict(color='#bcbd22', marker='p', linestyle=':',  linewidth=2, markersize=6),
    # CLL baselines (from adam_comparison)
    'MCL-LOG':  dict(color='#d62728', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'SCL-NL':   dict(color='#ff7f0e', marker='D', linestyle='--', linewidth=2, markersize=6),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
    # OP / CPE (from op_cpe_comparison)
    'OP':       dict(color='#e377c2', marker='D', linestyle='-',  linewidth=2, markersize=6),
    'OP-W':     dict(color='#aa40fc', marker='P', linestyle='-',  linewidth=2, markersize=6),
    'CPE':      dict(color='#17becf', marker='s', linestyle='-',  linewidth=2, markersize=6),
    # SoLar (this script)
    'SoLar':    dict(color='#e8b13f', marker='*', linestyle='-',  linewidth=2.5, markersize=9),
}

PLL_ALGOS = ['CLPL', 'PRODEN', 'PiCO', 'PiCO-MCL']
CLL_ALGOS = ['MCL-LOG', 'SCL-NL', 'ComCo', 'OP', 'OP-W', 'CPE']

# ─── k schedule (same as all other comparison scripts) ───────────────────────

def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [C - 1]))

# ─── ETA ─────────────────────────────────────────────────────────────────────

def _fmt_eta(s: float) -> str:
    if s < 90:   return f'{s:.0f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'

# ─── CSV helpers ──────────────────────────────────────────────────────────────

_CSV_FIELDS = ['total_classes', 'n_partial_labels', 'algorithm',
               'final_accuracy', 'epochs', 'training_time_s', 'timestamp']


def _load_done(csv_path: str) -> set:
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['total_classes']),
                      int(row['n_partial_labels']),
                      row['algorithm']))
    return done


def _append_result(csv_path, C, k, alg, acc, epochs, elapsed_s):
    new_file = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'total_classes':    C,
            'n_partial_labels': k,
            'algorithm':        alg,
            'final_accuracy':   round(acc, 4),
            'epochs':           epochs,
            'training_time_s':  round(elapsed_s, 1),
            'timestamp':        datetime.now().isoformat(),
        })


def _load_all_results(base_dirs: list) -> dict:
    """Merge CSVs from multiple result directories → res[C][alg][k] = accuracy."""
    res  = {}
    seen = set()
    for base in base_dirs:
        for pat in [os.path.join(base, 'results.csv'),
                    os.path.join(base, 'gpu*', 'results.csv')]:
            for path in sorted(glob.glob(pat)):
                if not os.path.isfile(path):
                    continue
                with open(path, newline='') as f:
                    for row in csv.DictReader(f):
                        key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                        if key in seen:
                            continue
                        seen.add(key)
                        C_  = int(row['total_classes'])
                        k_  = int(row['n_partial_labels'])
                        alg = _RENAME.get(row['algorithm'], row['algorithm'])
                        acc = float(row['final_accuracy'])
                        res.setdefault(C_, {}).setdefault(alg, {})[k_] = acc
    return res

# ─── Training ─────────────────────────────────────────────────────────────────

def _train_solar(pl_dataset_raw, original_targets, test_loader,
                 C: int, solar_config: dict, epochs: int, device, tag: str) -> float:
    """
    Build SoLarDataset + DataLoader, run 2-stage SoLar training.

    Stage 1 (pre-estimation, solar_config['est_epochs']):
        Estimate empirical class distribution via EMA of model predictions.
    Stage 2 (final training, epochs):
        Use Sinkhorn-Knopp with estimated distribution for pseudo-label selection.

    Optimizer: SGD (not Adam) — SoLar requires SGD as per the original paper.
    """
    solar_ds     = SoLarDataset(pl_dataset_raw, original_targets)
    solar_loader = DataLoader(solar_ds, batch_size=BS, shuffle=True,
                              drop_last=True, collate_fn=solar_collate_fn, pin_memory=True)

    # Mock args object expected by setup_solar
    fake_args = SimpleNamespace(
        lr=LR_SOLAR, momentum=MOMENTUM, weight_decay=WD,
        epochs=epochs, batch_size=BS,
    )
    train_config = {'num_classes': C}

    model, loss_fn, optimizer, solar_args, queue = setup_solar(
        fake_args, train_config, solar_config, solar_ds, device,
    )

    total_epochs = solar_config['est_epochs'] + epochs
    print(f'  [{tag}]  SoLar: est={solar_config["est_epochs"]} + train={epochs} '
          f'= {total_epochs} total epochs', flush=True)

    t0 = time.perf_counter()
    accuracies = train_solar(
        solar_args, model, solar_loader, test_loader, loss_fn, optimizer, device, queue,
    )
    elapsed = time.perf_counter() - t0

    final_acc = accuracies[-1] if accuracies else 0.0
    print(f'  [{tag}]  done  acc={final_acc:.2f}%  total={_fmt_eta(elapsed)}', flush=True)

    del model, loss_fn, optimizer, queue, solar_ds, solar_loader
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc

# ─── Plotting ─────────────────────────────────────────────────────────────────

def _global_ymax(res: dict) -> int:
    vals = [acc for C_d in res.values()
            for alg_d in C_d.values()
            for acc in alg_d.values()]
    return 80 if not vals else int(np.ceil(max(vals) / 10) * 10)


def _draw(ax, alg: str, k_acc: dict):
    if not k_acc or alg not in STYLES:
        return
    ks, accs = zip(*sorted(k_acc.items()))
    ax.plot(ks, accs, label=alg, **STYLES[alg])


def _setup_ax(ax, title: str, y_max: int, ylabel: bool = False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('k  (# partial / complementary labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def make_plots(res: dict, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    ym = _global_ymax(res)

    # ── Figure 1: All 11 methods ──────────────────────────────────────────────
    all_algos = PLL_ALGOS + CLL_ALGOS + ['SoLar']
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('All methods  —  C=5 and C=20', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in all_algos:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'fig1_all_methods.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: SoLar vs CLL methods ───────────────────────────────────────
    cll_plus_solar = CLL_ALGOS + ['SoLar']
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('SoLar vs CLL methods  —  C=5 and C=20', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in cll_plus_solar:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'fig2_solar_vs_cll.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3: SoLar vs PLL methods ───────────────────────────────────────
    pll_plus_solar = PLL_ALGOS + ['SoLar']
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('SoLar vs PLL methods  —  C=5 and C=20', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in pll_plus_solar:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'fig3_solar_vs_pll.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  [plots] → {plots_dir}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',    type=int, default=0)
    parser.add_argument('--num_gpus',  type=int, default=1)
    parser.add_argument('--data_dir',  default='./data')
    parser.add_argument('--out_dir',   default='results/solar_comparison/')
    parser.add_argument('--adam_dir',  default='results/adam_comparison/')
    parser.add_argument('--opcpe_dir', default='results/op_cpe_comparison/')
    parser.add_argument('--plots_dir', default='plots/solar_comparison/')
    parser.add_argument('--log_dir',   default='logs/cifar100_subset')
    parser.add_argument('--config',    default='config.yaml')
    parser.add_argument('--epochs',    type=int, default=200,
                        help='Stage-2 training epochs (est_epochs from config.yaml are added on top)')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--only_c',    type=int, default=None)
    parser.add_argument('--only_k',    type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    solar_config = cfg['solar']

    gpu_dir  = os.path.join(args.out_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir,        exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    done = _load_done(csv_path)

    total_ep = solar_config['est_epochs'] + args.epochs
    print(f'Device={device}  GPU {args.gpu_id}/{args.num_gpus}', flush=True)
    print(f'SoLar: SGD lr={LR_SOLAR}  bs={BS}  '
          f'est={solar_config["est_epochs"]}+train={args.epochs}={total_ep} total epochs', flush=True)
    print(f'k schedule: C=5→{get_k_values(5)}  C=20→{get_k_values(20)}', flush=True)
    print(f'Resume: {len(done)} entries in {csv_path}\n', flush=True)

    c_values = [args.only_c] if args.only_c is not None else C_VALUES

    for C in c_values:
        k_vals = get_k_values(C)
        if args.only_k is not None:
            k_vals = [args.only_k]

        # Round-robin by (C, k) pair index across GPUs
        ck_pairs = [(C, k) for k in k_vals]
        my_pairs = [p for i, p in enumerate(ck_pairs)
                    if i % args.num_gpus == args.gpu_id]

        print(f'\n{"="*60}', flush=True)
        print(f'C = {C}   k = {[k for _, k in my_pairs]}', flush=True)
        print(f'{"="*60}', flush=True)

        for k_idx, k in enumerate(k for _, k in my_pairs):
            if (C, k, 'SoLar') in done:
                print(f'  [skip] C={C} k={k}', flush=True)
                continue

            print(f'\n--- C={C}  k={k}  ({k_idx+1}/{len(my_pairs)}) ---', flush=True)

            pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                total_classes=C, n_partial_labels=k,
                data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
            )
            loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, BS)

            tag = f'GPU{args.gpu_id} SoLar C={C} k={k}'
            t0  = time.perf_counter()

            acc = _train_solar(pl_ds, orig_targets, loaders['test'],
                               C, solar_config, args.epochs, device, tag)

            elapsed = time.perf_counter() - t0
            _append_result(csv_path, C, k, 'SoLar', acc, args.epochs, elapsed)
            done.add((C, k, 'SoLar'))

            del loaders, pl_ds, cl_ds, orig_targets, test_info
            gc.collect()
            torch.cuda.empty_cache()

            res = _load_all_results([args.out_dir, args.adam_dir, args.opcpe_dir])
            make_plots(res, args.plots_dir)

    print(f'\nGPU {args.gpu_id} finished.', flush=True)
    print(f'  CSV   → {csv_path}', flush=True)
    print(f'  Plots → {args.plots_dir}', flush=True)


if __name__ == '__main__':
    main()
