"""
Plot OP-W (displayed as OP) / CPE / ComCo / MCL-LOG / PRODEN accuracy vs k for C=20.

Data sources:
  OP-W, CPE          →  results/op_cpe_comparison/gpu*/results.csv
  ComCo, MCL-LOG,
  PRODEN             →  results/adam_comparison/gpu*/results.csv

Usage:
    python scripts/plot_c20_op_cpe.py
    python scripts/plot_c20_op_cpe.py --out plots/c20_op_cpe/c20_op_cpe.png
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

# ─── Config ───────────────────────────────────────────────────────────────────

C          = 20
K_SCHEDULE = [1, 2, 3, 5, 10, 15, 19]

# OP-W stored as 'OP-W' in CSV but displayed as 'OP'
DISPLAY_NAME = {'OP-W': 'OP'}

STYLES = {
    'OP':      dict(color='#1f77b4', marker='D', linestyle='-',  linewidth=2, markersize=6),
    'CPE':     dict(color='#ff7f0e', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'ComCo':   dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'MCL-LOG': dict(color='#d62728', marker='s', linestyle='--', linewidth=2, markersize=6),
    'SoLar':   dict(color='#e8b13f', marker='*', linestyle='-',  linewidth=2, markersize=9),
}
PLOT_ALGOS = ['OP', 'CPE', 'ComCo', 'MCL-LOG', 'SoLar']

# ─── CSV loading ──────────────────────────────────────────────────────────────

def _load_csv(pattern, target_c, target_algs):
    """Return {display_name: {k: acc}} for target_c."""
    res  = {}
    seen = set()
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if int(row['total_classes']) != target_c:
                    continue
                alg = row['algorithm']
                if alg not in target_algs:
                    continue
                display = DISPLAY_NAME.get(alg, alg)
                k   = int(row['n_partial_labels'])
                key = (display, k)
                if key in seen:
                    continue
                seen.add(key)
                res.setdefault(display, {})[k] = float(row['final_accuracy'])
    return res

# ─── Plot ─────────────────────────────────────────────────────────────────────

def make_plot(res, out_path):
    vals = [acc for alg_d in res.values() for acc in alg_d.values()]
    ym   = 80 if not vals else int(np.ceil(max(vals) / 10) * 10) + 5

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'C = {C}  —  Test Accuracy vs k (partial labels)', fontsize=13)

    for alg in PLOT_ALGOS:
        k_acc = res.get(alg, {})
        if not k_acc:
            print(f'  [warn] no data for {alg}')
            continue
        ks, accs = zip(*sorted(k_acc.items()))
        ax.plot(ks, accs, label=alg, **STYLES[alg])

    ax.set_xlabel('k  (# partial labels per sample)', fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=10)
    ax.set_ylim(0, ym)
    ax.set_xticks(K_SCHEDULE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] → {out_path}')

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adam_dir',   default='results/adam_comparison/')
    parser.add_argument('--op_cpe_dir', default='results/op_cpe_comparison/')
    parser.add_argument('--solar_dir',  default='results/solar_comparison/')
    parser.add_argument('--out',        default='plots/c20_op_cpe/c20_op_cpe.png')
    args = parser.parse_args()

    res = {}

    op_pat    = os.path.join(args.op_cpe_dir, '**', 'results.csv')
    res.update(_load_csv(op_pat, C, {'OP-W', 'CPE'}))

    adam_pat  = os.path.join(args.adam_dir, '**', 'results.csv')
    res.update(_load_csv(adam_pat, C, {'ComCo', 'MCL-LOG'}))

    solar_pat = os.path.join(args.solar_dir, '**', 'results.csv')
    res.update(_load_csv(solar_pat, C, {'SoLar'}))

    make_plot(res, args.out)


if __name__ == '__main__':
    main()
