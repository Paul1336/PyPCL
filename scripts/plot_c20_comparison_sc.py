"""
Plot PRODEN / MCL-LOG / PiCO / PiCO-CLS / ComCo / PiCO-SC accuracy vs k for C=20.

Data sources:
  PRODEN, MCL-LOG, PiCO, ComCo  →  results/adam_comparison/gpu*/results.csv
  PiCO-CLS                       →  results/pico_cls/results.csv
  PiCO-SC                        →  results/pico_sc/results.csv

Usage:
    python scripts/plot_c20_comparison_sc.py
    python scripts/plot_c20_comparison_sc.py --out plots/c20_comparison/c20_comparison_sc.png
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

STYLES = {
    'PRODEN':   dict(color='#2ca02c', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'MCL-LOG':  dict(color='#d62728', marker='D', linestyle='-',  linewidth=2, markersize=6),
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'PiCO-CLS': dict(color='#e377c2', marker='*', linestyle='-',  linewidth=2, markersize=8),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'PiCO-SC':  dict(color='#FFD700', marker='h', linestyle='--', linewidth=2, markersize=7),
}
PLOT_ALGOS = ['PRODEN', 'MCL-LOG', 'PiCO', 'PiCO-CLS', 'ComCo', 'PiCO-SC']

# ─── CSV loading ──────────────────────────────────────────────────────────────

def _load_csv(pattern, target_c, target_algs):
    """Return {alg: {k: acc}} for target_c from files matching glob pattern."""
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
                k   = int(row['n_partial_labels'])
                key = (alg, k)
                if key in seen:
                    continue
                seen.add(key)
                res.setdefault(alg, {})[k] = float(row['final_accuracy'])
    return res

# ─── Plot ─────────────────────────────────────────────────────────────────────

def _draw_ax(ax, res, algs, title):
    vals = [acc for alg in algs for acc in res.get(alg, {}).values()]
    ym   = 80 if not vals else int(np.ceil(max(vals) / 10) * 10) + 5

    ax.set_title(title, fontsize=11)
    for alg in algs:
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


def _save_fig(res, algs, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'C = {C}  —  Test Accuracy vs k (partial labels)', fontsize=13)
    _draw_ax(ax, res, algs, title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] → {out_path}')


def make_plot(res, out_dir):
    _save_fig(res, ['PiCO', 'PiCO-CLS', 'PRODEN'],
              'PiCO / PiCO-CLS / PRODEN',
              os.path.join(out_dir, 'c20_base.png'))
    _save_fig(res, ['PiCO', 'PiCO-CLS', 'PRODEN', 'PiCO-SC', 'ComCo'],
              'PiCO / PiCO-CLS / PRODEN / PiCO-SC / ComCo',
              os.path.join(out_dir, 'c20_with_sc.png'))

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adam_dir',     default='results/adam_comparison/')
    parser.add_argument('--pico_cls_dir', default='results/pico_cls/')
    parser.add_argument('--pico_sc_dir',  default='results/pico_sc/')
    parser.add_argument('--out_dir',      default='plots/c20_comparison/')
    args = parser.parse_args()

    res = {}

    adam_pat = os.path.join(args.adam_dir, '**', 'results.csv')
    res.update(_load_csv(adam_pat, C, {'PRODEN', 'MCL-LOG', 'PiCO', 'ComCo'}))

    cls_pat = os.path.join(args.pico_cls_dir, 'results.csv')
    res.update(_load_csv(cls_pat, C, {'PiCO-CLS'}))

    sc_pat = os.path.join(args.pico_sc_dir, 'results.csv')
    res.update(_load_csv(sc_pat, C, {'PiCO-SC'}))

    make_plot(res, args.out_dir)


if __name__ == '__main__':
    main()
