"""
Custom group comparison plots (3 figures per C value):

  1. comco_wu_mcl    : ComCo + Wu2022 + MCL-LOG
  2. pico_comco_mcl  : PiCO + ComCo + PiCO-MCL
  3. cour_proden_scl : Cour2011 + PRODEN + SCL-NL

Output: plots/custom_groups/C{C}_{group}.png
        Y-axis fixed [5, 85].

Usage:
    python scripts/plot_custom_groups.py
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
# Group definitions
# ---------------------------------------------------------------------------
# Each group: (filename_suffix, title, [(csv_name, label, color, marker, ls)])

GROUPS = [
    (
        'comco_wu_mcl',
        'ComCo vs Wu2022 vs MCL-LOG',
        [
            ('ComCo',   'ComCo (CLL)',           '#8c564b', '^', '-'),
            ('Wu2022',  'Wu2022 (PLL, proper)',   '#17becf', 'D', '--'),
            ('MCL-LOG', 'MCL-LOG (CLL, URE)',     '#d62728', 'o', ':'),
        ],
    ),
    (
        'pico_comco_picomcl',
        'PiCO vs ComCo vs PiCO-MCL',
        [
            ('PiCO',     'PiCO (PLL)',     '#9467bd', 's', '-'),
            ('ComCo',    'ComCo (CLL)',    '#8c564b', '^', '--'),
            ('PiCO-MCL', 'PiCO-MCL (PLL)', '#bcbd22', 'p', ':'),
        ],
    ),
    (
        'cour_proden_scl',
        'Cour2011 vs PRODEN vs SCL-NL',
        [
            ('Cour2011', 'Cour2011 (PLL, hinge)',  '#1f77b4', 'o', '-'),
            ('Proden',   'PRODEN (PLL, weighted)', '#2ca02c', '^', '--'),
            ('SCL-NL',   'SCL-NL (CLL, surrogate)','#ff7f0e', 'D', ':'),
        ],
    ),
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
    """Return {C: {k: acc}}."""
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
                    data.setdefault(C, {})[k] = float(row['final_accuracy'])
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_group(C: int, group_suffix: str, title: str,
               members: list, algo_data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'{title}  —  C = {C} classes', fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    plotted = False
    for alg_name, label, color, marker, ls in members:
        kv = sorted(algo_data.get(alg_name, {}).get(C, {}).items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=6, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'C{C}_{group_suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Custom group comparison plots')
    parser.add_argument('--cifar100_dir',     default='results/cifar100_v2/')
    parser.add_argument('--feng_dir',         default='results/feng/')
    parser.add_argument('--pico_comco_dir',   default='results/pico_comco/')
    parser.add_argument('--pico_mclloss_dir', default='results/pico/pico_mclloss/')
    parser.add_argument('--proden_dir',       default='results/proden/')
    parser.add_argument('--scl_dir',          default='results/scl/')
    parser.add_argument('--out_dir',          default='plots/custom_groups/')
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

    # Collect all unique algorithm names across groups
    all_alg_names = {alg for _, _, members in GROUPS for alg, *_ in members}

    print('Loading data …')
    algo_data: dict[str, dict] = {}
    for alg_name in sorted(all_alg_names):
        loaded = _load(dirs[alg_name], alg_name)
        algo_data[alg_name] = loaded
        if loaded:
            n = sum(len(v) for v in loaded.values())
            print(f'  {alg_name:12s}: {len(loaded)} C values, {n} points')
        else:
            print(f'  {alg_name:12s}: [no data]')

    all_C = sorted({C for d in algo_data.values() for C in d})
    if not all_C:
        print('No data found.')
        return

    print(f'\nGenerating plots for C ∈ {all_C}')
    for C in all_C:
        for suffix, title, members in GROUPS:
            plot_group(C, suffix, title, members, algo_data, args.out_dir)

    print(f'\nDone. Plots → {args.out_dir}')


if __name__ == '__main__':
    main()
