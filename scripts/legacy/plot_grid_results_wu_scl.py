"""
Plot grid search results for Wu2022 (PLL) vs SCL-NL (CLL).

Merges all gpu*/results.csv under --results_dir and generates one figure
per (C, algorithm): 2 subplots (SGD | Adam), mean ± std over 3 seeds.

Output:
    plots/grid_search_wu_scl/C{C}_Wu2022.png
    plots/grid_search_wu_scl/C{C}_SCL-NL.png

Usage:
    python scripts/plot_grid_results_wu_scl.py
    python scripts/plot_grid_results_wu_scl.py --results_dir results/grid_search_wu_scl \
                                               --output_dir  plots/grid_search_wu_scl
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
# Grid constants (must match run_sweep_grid_search_wu_scl.py)
# ---------------------------------------------------------------------------

OPTIMIZERS = ['sgd', 'adam']
OPT_TITLES = {'sgd': 'SGD  (lr=0.01)', 'adam': 'Adam'}

OPTIMIZER_LR = {
    'sgd':  [0.01],
    'adam': [3e-3, 1e-3, 3e-4],
}

ALGORITHMS  = ['Wu2022', 'SCL-NL']
ALG_DISPLAY = {
    'Wu2022':  'Wu2022 / Feng2020 (PLL)',
    'SCL-NL':  'SCL-NL (CLL)',
}

# Batch size → color family (one shade per lr index)
BS_COLORS = {
    64:  ['#08519c', '#3182bd', '#9ecae1'],   # blues
    256: ['#006d2c', '#31a354', '#a1d99b'],   # greens
    512: ['#a50f15', '#de2d26', '#fc9272'],   # reds
}

LR_STYLES  = ['-', '--', ':']
LR_MARKERS = ['o', 's', '^']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_csvs(results_dir: str) -> list:
    patterns = [
        os.path.join(results_dir, 'results.csv'),
        os.path.join(results_dir, 'gpu*', 'results.csv'),
        os.path.join(results_dir, '*',    'results.csv'),
    ]
    csv_files = sorted(set(f for pat in patterns for f in glob.glob(pat)))

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
                    'batch_size': int(row['batch_size']),
                    'optimizer':  row['optimizer'],
                    'lr':         float(row['lr']),
                    'seed':       int(row['seed']),
                })
    return rows


def aggregate(rows: list) -> dict:
    """data[C][alg][opt][bs][lr][k] = [acc, ...]"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))
    for r in rows:
        data[r['C']][r['algorithm']][r['optimizer']][r['batch_size']][r['lr']][r['k']].append(r['acc'])
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _lr_label(lr: float) -> str:
    return f'{lr:.3g}' if lr >= 0.01 else f'{lr:.0e}'


def plot_for_C_alg(C: int, alg: str, data_C_alg: dict, save_dir: str):
    """
    2 subplots: SGD (3 lines: one per bs) | Adam (9 lines: 3 bs × 3 lr).
    x = k, y = mean accuracy over seeds, shading = ±1 std.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f'{ALG_DISPLAY[alg]}  —  C = {C} classes', fontsize=13, y=1.01)

    for ax, opt_type in zip(axes, OPTIMIZERS):
        ax.set_title(OPT_TITLES[opt_type], fontsize=12)
        ax.set_xlabel('k  (# partial labels per sample)', fontsize=10)
        ax.set_ylim(5, 85)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel('Test Accuracy (%)', fontsize=10)

        opt_data = data_C_alg.get(opt_type, {})
        lr_list  = OPTIMIZER_LR[opt_type]

        handles = []
        for bs in [64, 256, 512]:
            colors = BS_COLORS[bs]
            for lr_idx, lr in enumerate(lr_list):
                k_acc = opt_data.get(bs, {}).get(lr, {})
                if not k_acc:
                    continue
                ks    = sorted(k_acc)
                means = [np.mean(k_acc[k]) for k in ks]
                stds  = [np.std(k_acc[k])  for k in ks]

                color  = colors[lr_idx]
                label  = f'bs={bs}' if opt_type == 'sgd' else f'bs={bs}, lr={_lr_label(lr)}'
                line, = ax.plot(ks, means,
                                linestyle=LR_STYLES[lr_idx],
                                marker=LR_MARKERS[lr_idx],
                                color=color, linewidth=1.6, markersize=5,
                                label=label)
                ax.fill_between(ks,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=color, alpha=0.12)
                handles.append(line)

        ax.legend(handles=handles, fontsize=8, loc='best', ncol=1)

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
    parser = argparse.ArgumentParser(description='Plot Wu2022 vs SCL-NL grid search results')
    parser.add_argument('--results_dir', default='results/grid_search_wu_scl')
    parser.add_argument('--output_dir',  default='plots/grid_search_wu_scl')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'[ERROR] Directory not found: {args.results_dir}')
        return

    print(f'Loading CSVs from {args.results_dir} …')
    rows = load_all_csvs(args.results_dir)
    print(f'  {len(rows)} unique rows loaded.')

    if not rows:
        print('No data found.')
        return

    data  = aggregate(rows)
    all_C = sorted(data.keys())
    print(f'  C values found: {all_C}')

    for C in all_C:
        for alg in ALGORITHMS:
            if alg not in data[C]:
                print(f'  [skip] C={C} {alg}: no data')
                continue
            print(f'Plotting C={C}, {alg}')
            plot_for_C_alg(C, alg, data[C][alg], args.output_dir)

    print(f'\nDone. Plots → {args.output_dir}/')


if __name__ == '__main__':
    main()
