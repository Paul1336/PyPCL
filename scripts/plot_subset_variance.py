"""
Standalone plot script for subset_variance results.
Reads results/subset_variance/gpu*/results.csv and saves the bar chart.

Usage:
    python scripts/plot_subset_variance.py
    python scripts/plot_subset_variance.py --out plots/subset_variance/accuracy_distribution.png
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

COLORS = {'PRODEN': '#2ca02c', 'ComCo': '#8c564b'}
BINS   = list(range(0, 101, 5))


def _load_results(base_dir):
    res  = {}
    seen = set()
    for path in sorted(glob.glob(os.path.join(base_dir, 'gpu*', 'results.csv'))):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['seed'], row['algorithm'])
                if key in seen:
                    continue
                seen.add(key)
                res.setdefault(row['algorithm'], []).append(float(row['final_accuracy']))
    return res


def make_plot(base_dir, out_path):
    res = _load_results(base_dir)
    if not res:
        print('No results found.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        'Accuracy Distribution across CIFAR-100 10-class Subsets\nC=10, k=7',
        fontsize=12,
    )

    for col, alg in enumerate(['PRODEN', 'ComCo']):
        ax   = axes[col]
        accs = res.get(alg, [])
        if not accs:
            ax.set_title(f'{alg}  (no data)')
            continue

        counts, edges = np.histogram(accs, bins=BINS)
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
        widths  = [edges[i+1] - edges[i] for i in range(len(counts))]

        ax.bar(centers, counts, width=[w * 0.8 for w in widths],
               color=COLORS[alg], alpha=0.8, edgecolor='white', linewidth=0.5)

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        ax.axvline(mean_acc, color='black', linestyle='--', linewidth=1.5,
                   label=f'mean={mean_acc:.1f}%')
        ax.set_title(f'{alg}  —  n={len(accs)}\nmean={mean_acc:.1f}%  std={std_acc:.1f}%', fontsize=11)
        ax.set_xlabel('Test Accuracy (%)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='results/subset_variance/')
    parser.add_argument('--out',      default='plots/subset_variance/accuracy_distribution.png')
    args = parser.parse_args()
    make_plot(args.base_dir, args.out)


if __name__ == '__main__':
    main()
