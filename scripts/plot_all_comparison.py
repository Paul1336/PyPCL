"""
All-method comparison plot: PLL vs CLL across all implemented algorithms.

Reads results from each algorithm's canonical result directory and produces
one plot per C value, showing accuracy vs k for all available methods.

Methods included (skipped gracefully if data not found):
  PLL: Cour2011, Wu2022 (proper/Feng), PiCO, Proden
  CLL: MCL-LOG,  ComCo,  SCL-NL

Result directories (defaults, override with --*_dir):
  Cour2011 / MCL-LOG : results/cifar100_v2/
  Wu2022             : results/feng/
  PiCO   / ComCo     : results/pico_comco/
  Proden             : results/proden/
  SCL-NL             : results/scl/

Output: plots/all_comparison/C{C}_all.png
        Y-axis fixed [5, 85].

Usage:
    python scripts/plot_all_comparison.py
    python scripts/plot_all_comparison.py --cifar100_dir results/cifar100_v2/
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


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
# Each entry: (csv_alg_name, display_label, color, marker, linestyle, paradigm)
# paradigm: 'pll' | 'cll'

_ALGO_REGISTRY = [
    # PLL algorithms
    ('Cour2011', 'Cour2011 (PLL)',          '#1f77b4', 'o', '-',  'pll'),
    ('Wu2022',   'Wu2022 / Feng2020 (PLL)', '#17becf', 'D', '--', 'pll'),
    ('Proden',   'Proden (PLL)',            '#2ca02c', '^', '-',  'pll'),
    ('PiCO',     'PiCO (PLL)',              '#9467bd', 's', '--', 'pll'),
    # CLL algorithms
    ('MCL-LOG',  'MCL-LOG (CLL)',           '#d62728', 'o', '-',  'cll'),
    ('SCL-NL',   'SCL-NL (CLL)',            '#ff7f0e', 'D', '--', 'cll'),
    ('ComCo',    'ComCo (CLL)',             '#8c564b', '^', '-',  'cll'),
]

# Map algorithm name → canonical result dir (overridable via CLI)
_DEFAULT_DIRS = {
    'Cour2011': 'results/cifar100_v2/',
    'MCL-LOG':  'results/cifar100_v2/',
    'Wu2022':   'results/feng/',
    'PiCO':     'results/pico_comco/',
    'ComCo':    'results/pico_comco/',
    'Proden':   'results/proden/',
    'SCL-NL':   'results/scl/',
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_all(root: str, algorithm: str) -> dict:
    """Return {C: {k: acc}} for one algorithm from all CSVs under root."""
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_one(C: int, algo_data: dict, out_dir: str):
    """
    algo_data: {alg_name: {k: acc}}
    Skips algorithms with no data for this C.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f'All Methods  —  C = {C} classes', fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    plotted_pll = plotted_cll = False
    for alg_name, label, color, marker, ls, paradigm in _ALGO_REGISTRY:
        kv = sorted(algo_data.get(alg_name, {}).items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=6, label=label)
        if paradigm == 'pll':
            plotted_pll = True
        else:
            plotted_cll = True

    if not (plotted_pll or plotted_cll):
        print(f'  [skip] C={C}: no data')
        plt.close(fig)
        return

    # Add paradigm annotations in legend
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_, fontsize=9, ncol=2,
              loc='upper left', framealpha=0.85)

    # Vertical divider annotation
    if plotted_pll and plotted_cll:
        ax.text(0.99, 0.97, '─ PLL   -- CLL',
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                color='gray')

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'C{C}_all.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='All-method comparison plot')
    parser.add_argument('--cifar100_dir',  default='results/cifar100_v2/')
    parser.add_argument('--feng_dir',      default='results/feng/')
    parser.add_argument('--pico_comco_dir',default='results/pico_comco/')
    parser.add_argument('--proden_dir',    default='results/proden/')
    parser.add_argument('--scl_dir',       default='results/scl/')
    parser.add_argument('--out_dir',       default='plots/all_comparison/')
    args = parser.parse_args()

    # Override dirs from CLI
    dirs = {
        'Cour2011': args.cifar100_dir,
        'MCL-LOG':  args.cifar100_dir,
        'Wu2022':   args.feng_dir,
        'PiCO':     args.pico_comco_dir,
        'ComCo':    args.pico_comco_dir,
        'Proden':   args.proden_dir,
        'SCL-NL':   args.scl_dir,
    }

    # Load all data
    algo_data: dict[str, dict] = {}
    for alg_name, label, *_ in _ALGO_REGISTRY:
        d = dirs[alg_name]
        loaded = _load_all(d, alg_name)
        if loaded:
            algo_data[alg_name] = loaded
            n_points = sum(len(v) for v in loaded.values())
            print(f'  {alg_name:12s}: {len(loaded)} C values, {n_points} (C,k) points')
        else:
            algo_data[alg_name] = {}
            print(f'  {alg_name:12s}: [no data found in {d}]')

    # Collect all C values present in any algorithm
    all_C = sorted({C for d in algo_data.values() for C in d})
    if not all_C:
        print('\nNo data found. Check result directories.')
        return

    print(f'\nGenerating {len(all_C)} plots for C ∈ {all_C}')
    for C in all_C:
        # Slice to this C
        C_data = {alg: d.get(C, {}) for alg, d in algo_data.items()}
        plot_one(C, C_data, args.out_dir)

    print(f'\nDone. Plots → {args.out_dir}')


if __name__ == '__main__':
    main()
