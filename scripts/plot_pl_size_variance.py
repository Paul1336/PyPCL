"""
Standalone plot script for pl_size_variance results.
Reads results/pl_size_variance/gpu*/results.csv and saves the bar chart.

Usage:
    python scripts/plot_pl_size_variance.py
    python scripts/plot_pl_size_variance.py --out plots/pl_size_variance/pl_size_variance.png
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VAR_LEVELS  = [1, 2, 3]
VAR_COLORS  = {1: '#1f77b4', 2: '#ff7f0e', 3: '#d62728'}
BINS        = list(range(0, 101, 5))


def _load_results(base_dir):
    res  = {}
    seen = set()
    for path in sorted(glob.glob(os.path.join(base_dir, 'gpu*', 'results.csv'))):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['seed'], row['var_level'], row['algorithm'])
                if key in seen:
                    continue
                seen.add(key)
                alg = row['algorithm']
                vl  = int(row['var_level'])
                acc = float(row['final_accuracy'])
                res.setdefault(alg, {}).setdefault(vl, []).append(acc)
    return res


def make_plot(base_dir, out_path):
    res = _load_results(base_dir)
    if not res:
        print('No results found.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        'Accuracy Distribution — PL size variance effect\nC=10, mean PL size=7',
        fontsize=12,
    )

    for col, alg in enumerate(['PRODEN', 'ComCo']):
        ax       = axes[col]
        alg_data = res.get(alg, {})
        if not alg_data:
            ax.set_title(f'{alg}  (no data yet)')
            continue

        ax.set_title(alg, fontsize=11)
        ax.set_xlabel('Test Accuracy (%)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        ym = max(np.histogram(accs, bins=BINS)[0].max()
                 for accs in alg_data.values() if accs)
        ax.set_ylim(0, ym * 1.15)

        for vl in VAR_LEVELS:
            accs = alg_data.get(vl, [])
            if not accs:
                continue
            counts, edges = np.histogram(accs, bins=BINS)
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
            widths  = [(edges[i+1] - edges[i]) * 0.25 for i in range(len(counts))]
            offset  = (vl - 2) * 1.4
            ax.bar([c + offset for c in centers], counts, width=widths,
                   color=VAR_COLORS[vl], alpha=0.75,
                   label=f'Var={vl}  n={len(accs)}  μ={np.mean(accs):.1f}%  σ={np.std(accs):.1f}%')
            ax.axvline(np.mean(accs), color=VAR_COLORS[vl],
                       linestyle='--', linewidth=1.2, alpha=0.8)

        ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='results/pl_size_variance/')
    parser.add_argument('--out',      default='plots/pl_size_variance/pl_size_variance.png')
    args = parser.parse_args()
    make_plot(args.base_dir, args.out)


if __name__ == '__main__':
    main()
