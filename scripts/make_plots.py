"""
從既有 results/adam_comparison/gpu*/results.csv 讀取結果，
輸出 4 張圖到 plots/final/（或 --out_dir 指定的目錄）。

圖 1  fig1_pll_cll.png          — 4格  PLL/CLL × C=5/C=20
圖 2  fig2_pico_picomcl_comco.png — 2格  PiCO / PiCO-MCL / ComCo
圖 3  fig3_comco_scl_clpl.png   — 2格  ComCo / SCL-NL / Cour2011(CLPL)
圖 4  fig4_comco_mcl_proden.png — 2格  ComCo / MCL-LOG / PRODEN

Usage:
    python scripts/make_plots.py
    python scripts/make_plots.py --res_dir results/adam_comparison --out_dir plots/final
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

C_VALUES  = [5, 20]
PLL_ALGOS = ['CLPL', 'PRODEN', 'PiCO', 'PiCO-MCL']
CLL_ALGOS = ['MCL-LOG', 'SCL-NL', 'ComCo']

# CSV stores 'Cour2011'; rename to 'CLPL' for all display purposes
_RENAME = {'Cour2011': 'CLPL'}

STYLES = {
    'CLPL':     dict(color='#1f77b4', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'PRODEN':   dict(color='#2ca02c', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'PiCO-MCL': dict(color='#bcbd22', marker='p', linestyle=':',  linewidth=2, markersize=6),
    'MCL-LOG':  dict(color='#d62728', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'SCL-NL':   dict(color='#ff7f0e', marker='D', linestyle='--', linewidth=2, markersize=6),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(res_dir: str) -> dict:
    """Merge all gpu*/results.csv → res[C][alg][k] = accuracy."""
    res: dict = {}
    patterns = [
        os.path.join(res_dir, 'results.csv'),
        os.path.join(res_dir, 'gpu*', 'results.csv'),
    ]
    seen: set = set()
    total = 0
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                    if key in seen:
                        continue
                    seen.add(key)
                    C   = int(row['total_classes'])
                    k   = int(row['n_partial_labels'])
                    alg = _RENAME.get(row['algorithm'], row['algorithm'])
                    acc = float(row['final_accuracy'])
                    res.setdefault(C, {}).setdefault(alg, {})[k] = acc
                    total += 1
    print(f'Loaded {total} entries from {res_dir}')
    return res


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _global_ymax(res: dict) -> int:
    vals = [acc for C_d in res.values()
            for alg_d in C_d.values()
            for acc in alg_d.values()]
    return 80 if not vals else int(np.ceil(max(vals) / 10) * 10)


def _draw(ax, alg: str, k_acc: dict):
    if not k_acc:
        return
    ks, accs = zip(*sorted(k_acc.items()))
    ax.plot(ks, accs, label=alg, **STYLES[alg])


def _setup_ax(ax, title: str, y_max: int, ylabel: bool = False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {path}')


# ── 4 figures ─────────────────────────────────────────────────────────────────

def make_fig1(res, ym, out_dir):
    """Figure 1 — 2×2  PLL/CLL × C=5/C=20"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PLL vs CLL  —  C=5 and C=20', fontsize=13)
    for row, C in enumerate(C_VALUES):
        for col, (algos, paradigm) in enumerate([(PLL_ALGOS, 'PLL'), (CLL_ALGOS, 'CLL')]):
            ax = axes[row][col]
            _setup_ax(ax, f'{paradigm}  —  C = {C}', ym, ylabel=(col == 0))
            for alg in algos:
                _draw(ax, alg, res.get(C, {}).get(alg, {}))
            ax.legend(fontsize=8, loc='best')
    _save(fig, os.path.join(out_dir, 'fig1_pll_cll.png'))


def make_fig2(res, ym, out_dir):
    """Figure 2 — 2格  PiCO / PiCO-MCL / ComCo"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('PiCO vs PiCO-MCL vs ComCo', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['PiCO', 'PiCO-MCL', 'ComCo']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    _save(fig, os.path.join(out_dir, 'fig2_pico_picomcl_comco.png'))


def make_fig3(res, ym, out_dir):
    """Figure 3 — 2格  ComCo / SCL-NL / CLPL"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('ComCo vs SCL-NL vs CLPL', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['ComCo', 'SCL-NL', 'CLPL']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    _save(fig, os.path.join(out_dir, 'fig3_comco_scl_clpl.png'))


def make_fig4(res, ym, out_dir):
    """Figure 4 — 2格  ComCo / MCL-LOG / PRODEN"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('ComCo vs MCL-LOG vs PRODEN', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['ComCo', 'MCL-LOG', 'PRODEN']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    _save(fig, os.path.join(out_dir, 'fig4_comco_mcl_proden.png'))


def make_fig5(res, ym, out_dir):
    """Figure 5 — 2格  全部 7 個 algorithms"""
    all_algos = PLL_ALGOS + CLL_ALGOS  # CLPL, PRODEN, PiCO, PiCO-MCL, MCL-LOG, SCL-NL, ComCo
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('All Methods  —  C=5 and C=20', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in all_algos:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=8, loc='best')
    _save(fig, os.path.join(out_dir, 'fig5_all_methods.png'))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='results/adam_comparison',
                        help='Directory containing gpu*/results.csv files')
    parser.add_argument('--out_dir', default='plots/final',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    res = load_results(args.res_dir)

    if not res:
        print('No results found. Check --res_dir path.')
        return

    ym = _global_ymax(res)
    print(f'y-axis max: {ym}%\n')

    make_fig1(res, ym, args.out_dir)
    make_fig2(res, ym, args.out_dir)
    make_fig3(res, ym, args.out_dir)
    make_fig4(res, ym, args.out_dir)
    make_fig5(res, ym, args.out_dir)

    print(f'\nDone. Figures saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
