"""
Separate comparison plots for PLL methods and CLL methods.

PLL plot (C{C}_pll.png): Cour2011, Wu2022, Proden, PiCO, PiCO-MCL
CLL plot (C{C}_cll.png): MCL-LOG, SCL-NL, ComCo

Output: plots/pll_comparison/  and  plots/cll_comparison/
        Y-axis fixed [5, 85].

Usage:
    python scripts/plot_pll_cll_comparison.py
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
# Registry
# ---------------------------------------------------------------------------

PLL_ALGOS = [
    # (csv_name, display_label, color, marker, linestyle)
    ('Cour2011',  'Cour2011 (CLPL)',        '#1f77b4', 'o', '-'),
    ('Wu2022',    'Wu2022 (Proper PLL)',     '#17becf', 'D', '--'),
    ('Proden',    'PRODEN',                 '#2ca02c', '^', '-'),
    ('PiCO',      'PiCO',                   '#9467bd', 's', '--'),
    ('PiCO-MCL',  'PiCO-MCL',              '#bcbd22', 'p', ':'),
]

CLL_ALGOS = [
    ('MCL-LOG',   'MCL-LOG',                '#d62728', 'o', '-'),
    ('SCL-NL',    'SCL-NL',                 '#ff7f0e', 'D', '--'),
    ('ComCo',     'ComCo',                  '#8c564b', '^', '-'),
]

_DEFAULT_DIRS = {
    'Cour2011':  'results/cifar100_v2/',
    'Wu2022':    'results/feng/',
    'Proden':    'results/proden/',
    'PiCO':      'results/pico_comco/',
    'PiCO-MCL':  'results/pico/pico_mclloss/',
    'MCL-LOG':   'results/cifar100_v2/',
    'SCL-NL':    'results/scl/',
    'ComCo':     'results/pico_comco/',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(root: str, algorithm: str) -> dict:
    """Return {C: {k: acc}} for one algorithm."""
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
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_group(C: int, algo_data: dict, algos: list,
                title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    plotted = False
    for alg_name, label, color, marker, ls in algos:
        kv = sorted(algo_data.get(alg_name, {}).get(C, {}).items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=6, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PLL vs CLL separate comparison plots')
    parser.add_argument('--cifar100_dir',     default='results/cifar100_v2/')
    parser.add_argument('--feng_dir',         default='results/feng/')
    parser.add_argument('--pico_comco_dir',   default='results/pico_comco/')
    parser.add_argument('--pico_mclloss_dir', default='results/pico/pico_mclloss/')
    parser.add_argument('--proden_dir',       default='results/proden/')
    parser.add_argument('--scl_dir',          default='results/scl/')
    parser.add_argument('--pll_out_dir',      default='plots/pll_comparison/')
    parser.add_argument('--cll_out_dir',      default='plots/cll_comparison/')
    args = parser.parse_args()

    dirs = {
        'Cour2011':  args.cifar100_dir,
        'Wu2022':    args.feng_dir,
        'Proden':    args.proden_dir,
        'PiCO':      args.pico_comco_dir,
        'PiCO-MCL':  args.pico_mclloss_dir,
        'MCL-LOG':   args.cifar100_dir,
        'SCL-NL':    args.scl_dir,
        'ComCo':     args.pico_comco_dir,
    }

    # Load all algorithms
    all_algos = PLL_ALGOS + CLL_ALGOS
    algo_data: dict[str, dict] = {}
    print('Loading data …')
    for alg_name, label, *_ in all_algos:
        loaded = _load(dirs[alg_name], alg_name)
        algo_data[alg_name] = loaded
        if loaded:
            n = sum(len(v) for v in loaded.values())
            print(f'  {alg_name:12s}: {len(loaded)} C values, {n} points')
        else:
            print(f'  {alg_name:12s}: [no data]')

    # Collect all C values that appear in any algorithm
    all_C = sorted({C for d in algo_data.values() for C in d})
    if not all_C:
        print('No data found.')
        return

    print(f'\nGenerating plots for C ∈ {all_C}')
    for C in all_C:
        _plot_group(
            C, algo_data, PLL_ALGOS,
            title=f'Partial Label Learning (PLL)  —  C = {C} classes',
            out_path=os.path.join(args.pll_out_dir, f'C{C}_pll.png'),
        )
        _plot_group(
            C, algo_data, CLL_ALGOS,
            title=f'Complementary Label Learning (CLL)  —  C = {C} classes',
            out_path=os.path.join(args.cll_out_dir, f'C{C}_cll.png'),
        )

    print(f'\nDone.')
    print(f'  PLL plots → {args.pll_out_dir}')
    print(f'  CLL plots → {args.cll_out_dir}')


if __name__ == '__main__':
    main()
