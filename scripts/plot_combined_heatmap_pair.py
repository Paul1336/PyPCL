"""
Combined heatmap comparing PiCO vs PiCO-SC (PiCO-softmax) side by side.

Layout (2 rows × 2 cols):
  [acc  PiCO] [acc  PiCO-SC]
  [loss PiCO] [loss PiCO-SC]

Usage:
    python scripts/plot_combined_heatmap_pair.py --k 5
    python scripts/plot_combined_heatmap_pair.py --k 5 10 15 19 --show_class_names
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

BASE_DIR = 'results/extended_analysis'
OUT_DIR  = 'plots/extended_analysis'
ALG_L   = 'PiCO'
ALG_R   = 'PiCO-Uniform'
LABEL_L = 'PiCO'
LABEL_R = 'PiCO-Uniform'

# ── Data loading ──────────────────────────────────────────────────────────────

def load_per_class(alg, k, C, base_dir):
    path = os.path.join(base_dir, alg, f'C{C}_k{k}', 'per_class_loss.csv')
    if not os.path.isfile(path):
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

def _draw(ax, mat, epochs, cmap, vmin, vmax, C, class_names,
          show_xlabel=False, show_ylabel=False, ylabel='', fontsize=6):
    im = ax.imshow(
        mat.T,
        aspect='auto', origin='lower',
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation='nearest',
    )
    T = len(epochs)
    ax.set_yticks(range(C))
    if show_ylabel:
        ax.set_yticklabels(class_names if class_names else range(C), fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=9)
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xticks(range(T))
        ax.set_xticklabels(epochs, rotation=45, fontsize=fontsize)
        ax.set_xlabel('Epoch checkpoint', fontsize=9)
    else:
        ax.set_xticks([])

    return im


def plot_pair(k, C, base_dir, out_dir, class_names=None):
    dL = load_per_class(ALG_L, k, C, base_dir)
    dR = load_per_class(ALG_R, k, C, base_dir)

    if dL is None and dR is None:
        print(f'  [skip] k={k} — no data for either alg')
        return

    # Use whichever epochs list is available (should be identical)
    epochs = (dL or dR)['epochs']
    T = len(epochs)

    fig_w = max(14, T * 0.32 * 2 + 2)
    fig_h = max(5,  C * 0.32 + 1.5)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={'wspace': 0.04},
    )

    acc_l = dL['acc_mat'] if dL else np.full((T, C), np.nan)
    acc_r = dR['acc_mat'] if dR else np.full((T, C), np.nan)

    fa_l = f'{dL["overall"][-1]:.1f}%' if dL else 'N/A'
    fa_r = f'{dR["overall"][-1]:.1f}%' if dR else 'N/A'

    im_acc_l = _draw(axes[0], acc_l, epochs, 'RdYlGn', 0, 100, C, class_names,
                     show_xlabel=True, show_ylabel=True, ylabel='Class')
    im_acc_r = _draw(axes[1], acc_r, epochs, 'RdYlGn', 0, 100, C, class_names,
                     show_xlabel=True, show_ylabel=False)

    axes[0].set_title(f'{LABEL_L}  (final {fa_l})', fontsize=11, fontweight='bold')
    axes[1].set_title(f'{LABEL_R}  (final {fa_r})', fontsize=11, fontweight='bold')

    fig.colorbar(im_acc_l, ax=axes, label='Accuracy (%)', shrink=0.85, pad=0.01)

    fig.suptitle(f'{LABEL_L} vs {LABEL_R}  —  C={C}  k={k}', fontsize=13, fontweight='bold')

    out_path = os.path.join(out_dir, f'{ALG_L}_vs_{ALG_R}', f'C{C}_k{k}_combined.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',    type=int, nargs='+', required=True)
    parser.add_argument('--C',    type=int, default=20)
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
        print(f'k={k}')
        plot_pair(k, args.C, args.base_dir, args.out_dir, class_names)

    print('Done.')


if __name__ == '__main__':
    main()
