"""
Standalone comparison plot: PiCO vs PiCO-MCL vs ComCo.

Reads:
  --pico_comco_dir  (default results/pico_comco/)      → PiCO + ComCo results
  --mclloss_dir     (default results/pico/pico_mclloss) → PiCO-MCL results

Output: plots/pico/comparison/C{C}_comparison.png

Can be run at any time — partial results are plotted as-is.

Usage:
    python scripts/plot_pico_comparison.py
    python scripts/plot_pico_comparison.py \
        --pico_comco_dir results/pico_comco \
        --mclloss_dir    results/pico/pico_mclloss \
        --output_dir     plots/pico/comparison
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
    'PiCO':     dict(color='royalblue', marker='o', linestyle='-',  linewidth=2, markersize=6, label='PiCO (PLL)'),
    'PiCO-MCL': dict(color='steelblue', marker='s', linestyle='--', linewidth=2, markersize=6, label='PiCO-MCL (PLL)'),
    'ComCo':    dict(color='tomato',    marker='^', linestyle='-',  linewidth=2, markersize=6, label='ComCo (CLL)'),
}


def _load_dir(root: str) -> dict:
    """Returns {C: {alg: {k: acc}}} merging all gpu*/results.csv under root."""
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
                    key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                    if key in seen:
                        continue
                    seen.add(key)
                    C   = int(row['total_classes'])
                    k   = int(row['n_partial_labels'])
                    alg = row['algorithm']
                    acc = float(row['final_accuracy'])
                    data.setdefault(C, {}).setdefault(alg, {})[k] = acc
    return data


def plot_C(C: int, alg_data: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'PiCO vs PiCO-MCL vs ComCo  —  C = {C} classes', fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    plotted = False
    for alg, style in _ALGO_STYLE.items():
        kv = sorted(alg_data.get(alg, {}).items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, **style)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.legend(fontsize=10)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'C{C}_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description='Plot PiCO vs PiCO-MCL vs ComCo')
    parser.add_argument('--pico_comco_dir', default='results/pico_comco')
    parser.add_argument('--mclloss_dir',    default='results/pico/pico_mclloss')
    parser.add_argument('--output_dir',     default='plots/pico/comparison')
    args = parser.parse_args()

    print("Loading results …")
    data: dict = {}

    for root in [args.pico_comco_dir, args.mclloss_dir]:
        if not os.path.isdir(root):
            print(f"  [skip] not found: {root}")
            continue
        for C, alg_kv in _load_dir(root).items():
            for alg, kv in alg_kv.items():
                data.setdefault(C, {}).setdefault(alg, {}).update(kv)

    if not data:
        print("No results found.")
        return

    all_C = sorted(data.keys())
    print(f"C values found: {all_C}")

    for C in all_C:
        print(f"Plotting C={C}  ({sorted(data[C].keys())})")
        plot_C(C, data[C], args.output_dir)

    print(f"\nDone. Plots → {args.output_dir}/")


if __name__ == '__main__':
    main()
