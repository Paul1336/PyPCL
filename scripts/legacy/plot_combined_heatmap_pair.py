"""
Side-by-side heatmap comparison of two algorithms.

Layout:
  Default (2×2): acc row + loss row
  --acc_only   : acc row only (1×2)

Usage:
    python scripts/plot_combined_heatmap_pair.py --alg_l PiCO --alg_r ComCo --k 19
    python scripts/plot_combined_heatmap_pair.py --alg_l PiCO --alg_r PiCO-Uniform --k 5 10 15 19 --acc_only
    python scripts/plot_combined_heatmap_pair.py --alg_l PiCO --alg_r ComCo --k 19 --show_class_names
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


def plot_pair(alg_l, alg_r, k, C, base_dir, out_dir, class_names=None, acc_only=False):
    dL = load_per_class(alg_l, k, C, base_dir)
    dR = load_per_class(alg_r, k, C, base_dir)
    if dL is None and dR is None:
        print(f'  [skip] k={k} — no data for either alg')
        return

    epochs = (dL or dR)['epochs']
    T      = len(epochs)

    fa_l = f'{dL["overall"][-1]:.1f}%' if dL else 'N/A'
    fa_r = f'{dR["overall"][-1]:.1f}%' if dR else 'N/A'

    acc_l  = dL['acc_mat']  if dL else np.full((T, C), np.nan)
    acc_r  = dR['acc_mat']  if dR else np.full((T, C), np.nan)

    n_rows = 1 if acc_only else 2
    fig_w  = max(14, T * 0.32 * 2 + 2)
    fig_h  = max(5 * n_rows, C * 0.32 * n_rows + 1.5)

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={'hspace': 0.06, 'wspace': 0.04},
        squeeze=False,
    )

    # ── Accuracy row ──
    im_acc_l = _draw(axes[0, 0], acc_l, epochs, 'RdYlGn', 0, 100, C, class_names,
                     show_xlabel=acc_only, show_ylabel=True, ylabel='Accuracy (%)')
    im_acc_r = _draw(axes[0, 1], acc_r, epochs, 'RdYlGn', 0, 100, C, class_names,
                     show_xlabel=acc_only, show_ylabel=False)
    axes[0, 0].set_title(f'{alg_l}  (final {fa_l})', fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f'{alg_r}  (final {fa_r})', fontsize=11, fontweight='bold')
    fig.colorbar(im_acc_l, ax=axes[0, :], label='Accuracy (%)', shrink=0.85, pad=0.01)

    # ── Loss row (optional) ──
    if not acc_only:
        loss_l = dL['loss_mat'] if dL else np.full((T, C), np.nan)
        loss_r = dR['loss_mat'] if dR else np.full((T, C), np.nan)
        all_loss = [m for m in [loss_l, loss_r] if not np.all(np.isnan(m))]
        loss_vmax = np.nanpercentile(np.concatenate(all_loss), 97)

        im_loss_l = _draw(axes[1, 0], loss_l, epochs, 'RdYlGn_r', 0, loss_vmax, C, class_names,
                          show_xlabel=True, show_ylabel=True, ylabel='CE Loss')
        im_loss_r = _draw(axes[1, 1], loss_r, epochs, 'RdYlGn_r', 0, loss_vmax, C, class_names,
                          show_xlabel=True, show_ylabel=False)
        fig.colorbar(im_loss_l, ax=axes[1, :], label='CE loss', shrink=0.85, pad=0.01)

    fig.suptitle(f'{alg_l} vs {alg_r}  —  C={C}  k={k}', fontsize=13, fontweight='bold')

    tag  = 'acc' if acc_only else 'acc_loss'
    name = f'{alg_l}_vs_{alg_r}'
    out_path = os.path.join(out_dir, name, f'C{C}_k{k}_{tag}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_l', required=True, help='Left algorithm')
    parser.add_argument('--alg_r', required=True, help='Right algorithm')
    parser.add_argument('--k',     type=int, nargs='+', required=True)
    parser.add_argument('--C',     type=int, default=20)
    parser.add_argument('--seed',  type=int, default=42)
    parser.add_argument('--acc_only',         action='store_true', help='Only accuracy row (1×2)')
    parser.add_argument('--show_class_names', action='store_true')
    parser.add_argument('--base_dir', default=BASE_DIR)
    parser.add_argument('--out_dir',  default=OUT_DIR)
    parser.add_argument('--data_dir', default='./data')
    args = parser.parse_args()

    class_names = None
    if args.show_class_names:
        class_names = load_class_names(args.C, args.seed, args.data_dir)

    for k in args.k:
        print(f'{args.alg_l} vs {args.alg_r}  k={k}')
        plot_pair(args.alg_l, args.alg_r, k, args.C,
                  args.base_dir, args.out_dir, class_names, args.acc_only)

    print('Done.')


if __name__ == '__main__':
    main()
