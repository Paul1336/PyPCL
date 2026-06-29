"""
Combined plot: Cour2011 + MCL-LOG + PiCO + ComCo, one PNG per C.

Reads all GPU sub-directories from two result folders and merges them.
For each C that has at least one data point, generates a plot with up to
four lines (missing algorithms are simply omitted).

Usage:
    # Default paths
    python scripts/plot_combined.py

    # Custom paths
    python scripts/plot_combined.py \
        --clpl_dir  results/cifar100_v2 \
        --pico_dir  results/pico_comco \
        --output_dir plots/combined

Output: plots/combined/C{C}_combined.png
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Styles for each algorithm
# ---------------------------------------------------------------------------
_ALGO_STYLE = {
    'Cour2011': dict(color='royalblue',   marker='o', linestyle='-',  label='Cour CLPL (PLL)'),
    'MCL-LOG':  dict(color='tomato',      marker='s', linestyle='-',  label='MCL-LOG (CLL)'),
    'PiCO':     dict(color='mediumblue',  marker='^', linestyle='--', label='PiCO (PLL)'),
    'ComCo':    dict(color='firebrick',   marker='v', linestyle='--', label='ComCo (CLL)'),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(root_dirs: list[str]) -> dict:
    """
    Reads all results.csv files under any gpu* subdirectory of each root_dir.
    Returns nested dict:  data[C][algorithm][k] = accuracy
    If the same (C, algorithm, k) appears in multiple files, last write wins
    (they should be identical since each GPU handles distinct C values).
    """
    data: dict = {}

    for root in root_dirs:
        # Accept both  root/results.csv  and  root/gpu*/results.csv
        patterns = [
            os.path.join(root, 'results.csv'),
            os.path.join(root, 'gpu*', 'results.csv'),
            os.path.join(root, '*', 'results.csv'),
        ]
        csv_files = []
        for pat in patterns:
            csv_files.extend(glob.glob(pat))

        for csv_path in csv_files:
            if not os.path.isfile(csv_path):
                continue
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    C   = int(row['total_classes'])
                    k   = int(row['n_partial_labels'])
                    alg = row['algorithm']
                    acc = float(row['final_accuracy'])

                    data.setdefault(C, {}).setdefault(alg, {})[k] = acc

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_combined(C: int, alg_data: dict, save_dir: str):
    """
    alg_data: {algorithm_name: {k: accuracy}}
    Plots all available algorithms on the same axes.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for alg, style in _ALGO_STYLE.items():
        if alg not in alg_data:
            continue
        kv = sorted(alg_data[alg].items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, linewidth=2, markersize=7, **style)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title(f'PLL vs CLL  —  C = {C} classes', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'C{C}_combined.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot combined 4-algorithm accuracy vs k')
    parser.add_argument('--clpl_dir',   default='results/cifar100_v2',
                        help='Root dir for Cour2011 + MCL-LOG results (contains gpu* subdirs)')
    parser.add_argument('--pico_dir',   default='results/pico_comco',
                        help='Root dir for PiCO + ComCo results (contains gpu* subdirs)')
    parser.add_argument('--output_dir', default='plots/combined')
    args = parser.parse_args()

    print("Loading results …")
    data = load_results([args.clpl_dir, args.pico_dir])

    if not data:
        print("No results found. Check --clpl_dir and --pico_dir.")
        return

    all_C = sorted(data.keys())
    print(f"Found data for C = {all_C}")

    for C in all_C:
        alg_data = data[C]
        algs_present = sorted(alg_data.keys())
        print(f"C={C}: {algs_present}")
        plot_combined(C, alg_data, args.output_dir)

    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
