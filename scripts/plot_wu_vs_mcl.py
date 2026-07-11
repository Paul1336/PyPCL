"""
Standalone plot: Feng 2020 (PLL) vs MCL-LOG (CLL) comparison.

Reads:
  --feng_dir  : results/feng/          (default)
  --mcl_dir   : results/cifar100_v2/   (default)

Writes one PNG per C value found in feng_dir:
  plots/feng/comparison/C{C}_feng_vs_mcl.png

Y-axis fixed to [5, 85].

Usage:
    python scripts/plot_feng_vs_mcl.py
    python scripts/plot_feng_vs_mcl.py --feng_dir results/feng/ --mcl_dir results/cifar100_v2/
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_ALGO_STYLE = {
    'Wu2022': dict(color='darkorange', marker='o', linestyle='-',  linewidth=2,
                     markersize=6, label='Wu2022 (PLL, proper)'),
    'MCL-LOG':  dict(color='tomato',     marker='s', linestyle='--', linewidth=2,
                     markersize=6, label='MCL-LOG (CLL, unbiased)'),
}


def _load_alg(root: str, algorithm: str) -> dict:
    """Return {C: {k: acc}} for the given algorithm from all CSVs under root."""
    patterns = [
        os.path.join(root, 'results.csv'),
        os.path.join(root, 'gpu*', 'results.csv'),
        os.path.join(root, '*',    'results.csv'),
    ]
    data: dict = {}
    seen: set  = set()
    for pat in patterns:
        for csv_path in sorted(glob.glob(pat)):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    if row['algorithm'] != algorithm:
                        continue
                    key = (row['total_classes'], row['n_partial_labels'])
                    if key in seen:
                        continue
                    seen.add(key)
                    C   = int(row['total_classes'])
                    k   = int(row['n_partial_labels'])
                    acc = float(row['final_accuracy'])
                    data.setdefault(C, {})[k] = acc
    return data


def plot_one(C: int, feng_ck: dict, mcl_ck: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'Wu2022 (PLL) vs MCL-LOG (CLL)  —  C = {C} classes', fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    datasets = {'Wu2022': feng_ck, 'MCL-LOG': mcl_ck}
    plotted = False
    for alg, style in _ALGO_STYLE.items():
        kv = sorted(datasets[alg].items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, **style)
        plotted = True

    if plotted:
        ax.legend(fontsize=10)
        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'C{C}_feng_vs_mcl.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')
    else:
        print(f'[skip] C={C}: no data to plot')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot Wu2022 vs MCL-LOG comparison')
    parser.add_argument('--feng_dir', default='results/feng/')
    parser.add_argument('--mcl_dir',  default='results/cifar100_v2/')
    parser.add_argument('--out_dir',  default='plots/feng/comparison/')
    args = parser.parse_args()

    feng_all = _load_alg(args.feng_dir, 'Wu2022')
    mcl_all  = _load_alg(args.mcl_dir,  'MCL-LOG')

    all_C = sorted(set(list(feng_all.keys()) + list(mcl_all.keys())))
    if not all_C:
        print('No data found. Check --feng_dir and --mcl_dir.')
        return

    for C in all_C:
        plot_one(C, feng_all.get(C, {}), mcl_all.get(C, {}), args.out_dir)

    print(f'\nDone. {len(all_C)} plots written to {args.out_dir}')


if __name__ == '__main__':
    main()
