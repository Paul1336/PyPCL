"""
Combined per-class heatmap: accuracy (top) + CE loss (bottom) in one figure.

Usage:
    python scripts/plot_combined_heatmap.py --alg PiCO-CLS --k 5
    python scripts/plot_combined_heatmap.py --alg PiCO --k 10 --show_class_names
    python scripts/plot_combined_heatmap.py --alg PiCO-CLS --k 5 10 15 19
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR  = 'results/extended_analysis'
OUT_DIR   = 'plots/extended_analysis'
C_DEFAULT = 20

# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_class(alg, k, C, base_dir):
    path = os.path.join(base_dir, alg, f'C{C}_k{k}', 'per_class_loss.csv')
    if not os.path.isfile(path):
        print(f'  [skip] {alg} k={k} — not found: {path}')
        return None
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    epochs   = [int(r['epoch']) for r in rows]
    acc_mat  = np.array([[float(r.get(f'acc_class_{c}',  'nan')) for c in range(C)] for r in rows])
    loss_mat = np.array([[float(r.get(f'loss_class_{c}', 'nan')) for c in range(C)] for r in rows])
    overall  = np.array([float(r['overall_acc']) for r in rows])
    return {'epochs': epochs, 'acc_mat': acc_mat, 'loss_mat': loss_mat, 'overall': overall}


def load_class_names(C, seed, data_dir):
    try:
        from src.cifar100_subset import select_cifar100_classes
        from torchvision.datasets import CIFAR100
        indices = select_cifar100_classes(C, seed=seed)
        ds = CIFAR100(root=data_dir, train=True, download=False)
        return [ds.classes[i] for i in indices]
    except Exception as e:
        print(f'  [warn] class names unavailable: {e}')
        return None

# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, mat, epochs, cmap, vmin, vmax, ylabel, class_names, C,
                  show_xticks=True, fontsize_tick=6):
    im = ax.imshow(
        mat.T,              # [C, T] → rows=class, cols=epoch
        aspect='auto',
        origin='lower',
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation='nearest',
    )
    T = len(epochs)
    ax.set_yticks(range(C))
    ax.set_yticklabels(class_names if class_names else range(C), fontsize=fontsize_tick)
    ax.set_ylabel(ylabel, fontsize=9)

    if show_xticks:
        ax.set_xticks(range(T))
        ax.set_xticklabels(epochs, rotation=45, fontsize=fontsize_tick)
        ax.set_xlabel('Epoch checkpoint', fontsize=9)
    else:
        ax.set_xticks([])

    return im


def plot_combined(alg, k, C, base_dir, out_dir, class_names=None):
    d = load_per_class(alg, k, C, base_dir)
    if d is None:
        return

    epochs   = d['epochs']
    acc_mat  = d['acc_mat']    # [T, C]
    loss_mat = d['loss_mat']   # [T, C]
    final_acc = d['overall'][-1]

    T        = len(epochs)
    fig_w    = max(10, T * 0.32)
    fig_h    = max(8,  C * 0.32 + 2.5)

    fig, (ax_acc, ax_loss) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={'hspace': 0.08},
    )
    fig.suptitle(
        f'{alg}  C={C}  k={k}  —  final acc {final_acc:.1f}%',
        fontsize=12, fontweight='bold',
    )

    # ── Top: accuracy ──────────────────────────────────────────────────────
    im_acc = _draw_heatmap(
        ax_acc, acc_mat, epochs,
        cmap='RdYlGn', vmin=0, vmax=100,
        ylabel='Accuracy (%)',
        class_names=class_names, C=C,
        show_xticks=False,
    )
    fig.colorbar(im_acc, ax=ax_acc, label='Accuracy (%)', shrink=0.9, pad=0.01)

    # ── Bottom: CE loss ────────────────────────────────────────────────────
    loss_vmax = np.nanpercentile(loss_mat, 97)   # clip extreme outliers
    im_loss = _draw_heatmap(
        ax_loss, loss_mat, epochs,
        cmap='RdYlGn_r', vmin=0, vmax=loss_vmax,
        ylabel='CE Loss',
        class_names=class_names, C=C,
        show_xticks=True,
    )
    fig.colorbar(im_loss, ax=ax_loss, label='CE loss', shrink=0.9, pad=0.01)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, alg, f'C{C}_k{k}', 'combined_heatmap.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg',  required=True,
                        help='Algorithm name, e.g. PiCO-CLS')
    parser.add_argument('--k',    type=int, nargs='+', required=True,
                        help='k value(s), e.g. 5 10 15 19')
    parser.add_argument('--C',    type=int, default=C_DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_dir', default=BASE_DIR)
    parser.add_argument('--out_dir',  default=OUT_DIR)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--show_class_names', action='store_true')
    args = parser.parse_args()

    class_names = None
    if args.show_class_names:
        class_names = load_class_names(args.C, args.seed, args.data_dir)

    for k in args.k:
        print(f'{args.alg}  k={k}')
        plot_combined(args.alg, k, args.C, args.base_dir, args.out_dir, class_names)

    print('Done.')


if __name__ == '__main__':
    main()
