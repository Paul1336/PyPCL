"""
Plot per-class accuracy heatmap from detailed analysis results.

For each k value: one figure with one subplot per algorithm.
Color = accuracy (0-100%), green=high, red=low.

Usage:
    # all algs, k=5,10,15
    python scripts/plot_per_class_acc.py --k 5 10 15

    # specific algs
    python scripts/plot_per_class_acc.py --alg PiCO ComCo --k 10

    # show CIFAR-100 class names on Y-axis
    python scripts/plot_per_class_acc.py --k 10 --show_class_names
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_ALGS = ['PiCO', 'PiCO-CLS', 'PiCO-SC', 'ComCo']

# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_per_class_acc(csv_path: str, C: int):
    """
    Returns (epochs_list, acc_mat) where acc_mat has shape [T, C].
    Values are percentages 0-100. NaN if class missing.
    """
    if not os.path.isfile(csv_path):
        return None, None

    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return None, None

    epochs_list = [int(r['epoch']) for r in rows]
    acc_mat = np.array([
        [float(r.get(f'acc_class_{c}', 'nan')) for c in range(C)]
        for r in rows
    ])  # [T, C]

    overall_acc = [float(r.get('overall_acc', 'nan')) for r in rows]

    return epochs_list, acc_mat, overall_acc


def _get_class_names(C: int, seed: int, data_dir: str):
    """Load CIFAR-100 class names for the selected subset. Returns list[str] or None."""
    try:
        from src.cifar100_subset import select_cifar100_classes
        from torchvision.datasets import CIFAR100
        indices = select_cifar100_classes(C, seed=seed)
        ds = CIFAR100(root=data_dir, train=True, download=True)
        return [ds.classes[i] for i in indices]
    except Exception as e:
        print(f'  [warn] Could not load class names: {e}')
        return None


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, acc_mat, epochs_list, overall_acc, alg, C, k, class_names=None):
    """
    Draw one per-class accuracy heatmap onto ax.
    acc_mat: [T, C] array, values 0-100.
    """
    im = ax.imshow(
        acc_mat.T,          # [C, T]  → rows=class, cols=epoch
        aspect='auto',
        origin='lower',
        cmap='RdYlGn',
        vmin=0, vmax=100,
        interpolation='nearest',
    )

    # X-axis: epoch ticks
    ax.set_xticks(range(len(epochs_list)))
    ax.set_xticklabels(epochs_list, rotation=45, fontsize=6)
    ax.set_xlabel('Epoch', fontsize=8)

    # Y-axis: class index / names
    ax.set_yticks(range(C))
    if class_names:
        ax.set_yticklabels(class_names, fontsize=6)
    else:
        ax.set_yticklabels(range(C), fontsize=7)

    # Title: alg name + final overall acc
    final_acc = overall_acc[-1] if overall_acc else float('nan')
    ax.set_title(f'{alg}  (final {final_acc:.1f}%)', fontsize=10, fontweight='bold')

    return im


def make_plots_for_k(algs, k, C, out_dir, plots_dir, class_names=None):
    """
    One figure with len(algs) subplots for the given k.
    Saves to plots_dir/per_class_acc_C{C}_k{k}.png.
    """
    # Gather data first — skip algs with no data
    data = {}
    for alg in algs:
        csv_path = os.path.join(out_dir, alg, f'C{C}_k{k}', 'per_class_loss.csv')
        result = _load_per_class_acc(csv_path, C)
        if result[0] is not None:
            data[alg] = result   # (epochs_list, acc_mat, overall_acc)
        else:
            print(f'  [skip] {alg} C={C} k={k} — no data at {csv_path}')

    if not data:
        print(f'  [skip] k={k} — no data for any alg')
        return

    n = len(data)
    fig_w = max(6, 4 * n)
    fig_h = max(5, C * 0.28 + 1.5)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(f'Per-class Accuracy  —  C={C}  k={k}', fontsize=12, fontweight='bold', y=1.01)

    im_ref = None
    for ax, (alg, (epochs_list, acc_mat, overall_acc)) in zip(axes, data.items()):
        im_ref = _draw_heatmap(ax, acc_mat, epochs_list, overall_acc, alg, C, k, class_names)

    # Shared colorbar
    fig.colorbar(im_ref, ax=axes, label='Accuracy (%)', shrink=0.8, pad=0.02)

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f'per_class_acc_C{C}_k{k}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] → {out_path}')


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', nargs='+', default=['all'],
                        help='Algorithms to plot. Use "all" for all four. '
                             'Options: PiCO PiCO-CLS PiCO-SC ComCo')
    parser.add_argument('--k',   nargs='+', type=int, required=True,
                        help='k values to plot (e.g. 5 10 15)')
    parser.add_argument('--C',   type=int, default=20)
    parser.add_argument('--seed', type=int, default=42,
                        help='Class selection seed (for class names)')
    parser.add_argument('--out_dir',   default='results/detailed_analysis')
    parser.add_argument('--plots_dir', default='plots/detailed_analysis')
    parser.add_argument('--show_class_names', action='store_true',
                        help='Show CIFAR-100 class names on Y-axis')
    parser.add_argument('--data_dir', default='./data',
                        help='Path to CIFAR-100 data (only needed with --show_class_names)')
    args = parser.parse_args()

    # Resolve alg list
    algs = ALL_ALGS if args.alg == ['all'] else args.alg
    for a in algs:
        if a not in ALL_ALGS:
            parser.error(f'Unknown alg "{a}". Valid: {ALL_ALGS}')

    # Optionally load class names
    class_names = None
    if args.show_class_names:
        class_names = _get_class_names(args.C, args.seed, args.data_dir)

    print(f'Algs: {algs}')
    print(f'k values: {args.k}')
    print(f'C={args.C}, out_dir={args.out_dir}, plots_dir={args.plots_dir}')
    print()

    for k in args.k:
        print(f'── k={k} ──')
        make_plots_for_k(algs, k, args.C, args.out_dir, args.plots_dir, class_names)

    print('\nDone.')


if __name__ == '__main__':
    main()
