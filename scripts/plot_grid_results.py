"""
Plot grid search results: per-optimizer subplots, each with 9 lines (3 batch × 3 lr).
Generates one figure per (C, algorithm).

Can be run at any time — merges all gpu*/results.csv files found under --results_dir,
so partial results from mid-sweep are plotted as-is.

Output:
    plots/grid_search/C{C}_Cour2011.png
    plots/grid_search/C{C}_MCL-LOG.png

Usage:
    python scripts/plot_grid_results.py
    python scripts/plot_grid_results.py --results_dir results/grid_search \
                                        --output_dir  plots/grid_search
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

OPTIMIZERS = ['sgd', 'adam', 'adamw']
OPT_TITLES = {'sgd': 'SGD', 'adam': 'Adam', 'adamw': 'AdamW'}

# Batch size → color family (dark/mid/light per lr index)
BS_COLORS = {
    64:  ['#08519c', '#3182bd', '#9ecae1'],   # blues
    256: ['#006d2c', '#31a354', '#a1d99b'],   # greens
    512: ['#a50f15', '#de2d26', '#fc9272'],   # reds
}

# lr index → linestyle (same order as OPTIMIZER_LR[opt_type])
LR_STYLES  = ['-', '--', ':']
LR_MARKERS = ['o', 's', '^']

ALGORITHMS  = ['Cour2011', 'MCL-LOG']
ALG_DISPLAY = {'Cour2011': 'Cour CLPL (PLL)', 'MCL-LOG': 'MCL-LOG (CLL)'}

OPTIMIZER_LR = {
    'sgd':   [0.1,  0.01,  0.001],
    'adam':  [3e-3, 1e-3,  3e-4],
    'adamw': [3e-3, 1e-3,  3e-4],
}

# ---------------------------------------------------------------------------
# Data loading — merges all gpu*/results.csv under results_dir
# ---------------------------------------------------------------------------

def load_all_csvs(results_dir: str) -> list:
    patterns = [
        os.path.join(results_dir, 'results.csv'),
        os.path.join(results_dir, 'gpu*', 'results.csv'),
        os.path.join(results_dir, '*',    'results.csv'),
    ]
    csv_files = []
    for pat in patterns:
        csv_files.extend(glob.glob(pat))
    csv_files = sorted(set(csv_files))

    rows = []
    seen = set()
    for csv_path in csv_files:
        print(f'  reading {csv_path}')
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['total_classes'], row['n_partial_labels'],
                       row['algorithm'], row['batch_size'],
                       row['optimizer'], row['lr'], row['seed'])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    'C':          int(row['total_classes']),
                    'k':          int(row['n_partial_labels']),
                    'algorithm':  row['algorithm'],
                    'acc':        float(row['final_accuracy']),
                    'epochs':     int(row['epochs']),
                    'batch_size': int(row['batch_size']),
                    'optimizer':  row['optimizer'],
                    'lr':         float(row['lr']),
                    'seed':       int(row['seed']),
                })
    return rows


def aggregate(rows: list):
    """
    Returns nested dict:
      data[C][algorithm][optimizer][batch_size][lr][k] = list of acc values (one per seed)
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(list)
                    )
                )
            )
        )
    )
    for r in rows:
        data[r['C']][r['algorithm']][r['optimizer']][r['batch_size']][r['lr']][r['k']].append(r['acc'])
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _lr_label(lr: float) -> str:
    if lr >= 0.01:
        return f'{lr:.3g}'
    return f'{lr:.0e}'


def plot_for_C_alg(C: int, alg: str, data_C_alg: dict, save_dir: str):
    """
    data_C_alg: data[C][alg]  →  {optimizer: {batch_size: {lr: {k: [acc]}}}}
    3 subplots (one per optimizer), 9 lines each (3 batch × 3 lr), x=k, y=mean acc.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'{ALG_DISPLAY[alg]}  —  C = {C} classes', fontsize=13, y=1.01)

    for ax, opt_type in zip(axes, OPTIMIZERS):
        ax.set_title(OPT_TITLES[opt_type], fontsize=12)
        ax.set_xlabel('k  (# partial labels per sample)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('Test Accuracy (%)', fontsize=10)
        ax.grid(True, alpha=0.3)

        opt_data = data_C_alg.get(opt_type, {})
        lr_list  = OPTIMIZER_LR[opt_type]

        handles = []
        for bs_idx, bs in enumerate([64, 256, 512]):
            colors = BS_COLORS[bs]
            for lr_idx, lr in enumerate(lr_list):
                k_acc = opt_data.get(bs, {}).get(lr, {})
                if not k_acc:
                    continue
                ks   = sorted(k_acc.keys())
                means = [np.mean(k_acc[k]) for k in ks]
                stds  = [np.std(k_acc[k])  for k in ks]

                color  = colors[lr_idx]
                style  = LR_STYLES[lr_idx]
                marker = LR_MARKERS[lr_idx]
                label  = f'bs={bs}, lr={_lr_label(lr)}'

                line, = ax.plot(ks, means,
                                linestyle=style, marker=marker, color=color,
                                linewidth=1.6, markersize=5, label=label)
                ax.fill_between(ks,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=color, alpha=0.12)
                handles.append(line)

        ax.legend(handles=handles, fontsize=7, loc='best', ncol=1)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    alg_safe = alg.replace('/', '-')
    path = os.path.join(save_dir, f'C{C}_{alg_safe}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot grid search accuracy vs k')
    parser.add_argument('--results_dir', default='results/grid_search',
                        help='Root dir containing gpu*/results.csv files')
    parser.add_argument('--output_dir',  default='plots/grid_search')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'[ERROR] Directory not found: {args.results_dir}')
        return

    print(f'Loading CSVs from {args.results_dir} …')
    rows = load_all_csvs(args.results_dir)
    print(f'  {len(rows)} unique rows loaded.')

    data = aggregate(rows)
    all_C = sorted(data.keys())
    print(f'  C values found: {all_C}')

    for C in all_C:
        for alg in ALGORITHMS:
            if alg not in data[C]:
                continue
            print(f'Plotting C={C}, {alg}')
            plot_for_C_alg(C, alg, data[C][alg], args.output_dir)

    print(f'\nDone. Plots → {args.output_dir}/')


if __name__ == '__main__':
    main()
