#!/usr/bin/env python
"""Per-(AB-variant, EMA level) figures + CSV tables for the ab_sweep_0915
confidence-update-mechanism ablation (see scripts/run_ab_sweep.py's
docstring) -- same 2-panel layout as scripts/plot_ema_sweep.py (TC-PLS
accuracy-vs-W on the left, TC-n-PLS accuracy-vs-N on the right, one line per
k, dashed = that k's unbiased baseline), just with the AB-variant name and
EMA level fixed per figure/table instead of {PiCO-Fixed, PRODEN}.

    python scripts/plot_ab_sweep.py --run_name ab_sweep_0915

For each of the 3 trained variants (PiCO-Fixed-SrcSoftmax, -SoftUpdate,
-SrcSoftmax-SoftUpdate) x 3 EMA levels (EMA100/050/000), writes:
    plots/<run_name>/<base>/<base>-<EMA_TAG>.png
    results/<run_name>/tables/<base>-<EMA_TAG>.csv
PLUS the same for the 4th, reference combination 'PiCO-Fixed' (native A+B,
not retrained here) -- its rows are read straight from --reference_run
(default ema_sweep_0915) instead, so all four combinations end up with
directly comparable figures/tables in the SAME new run_name folder. 4 x 3 =
12 pictures + 12 tables total. Safe to re-run at any time on partial results.
"""

import argparse
import csv
import os
import statistics
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.pipeline.results import load_results_by_seed
from src.pll_init import (AB_SWEEP_BASES, AB_SWEEP_SCALES, biased_rand_all_variant_name,
                           biased_variant_name, ema_tag, ema_variant_name, weight_pct_str)

# Must match scripts/run_ab_sweep.py's EXP_W_* / EXP_N_* constants.
EXP_W_K_VALUES = [10, 15, 19]
EXP_W_WEIGHTS = [0.052, 0.10, 0.20]
EXP_N_K_VALUES = [5, 10, 15, 19]  # extended 2026-09-17 to match run_ab_sweep.py's EXP_N_K_VALUES
EXP_N_WEIGHT = 0.20
EXP_N_VALUES = [4, 9, 14, 19]
REFERENCE_BASE = 'PiCO-Fixed'   # native A+B, pulled from --reference_run

K_COLORS = {5: '#9467bd', 10: '#1f77b4', 15: '#2ca02c', 19: '#d62728'}


def _stats(accs):
    if not accs:
        return None, None, 0
    m = statistics.mean(accs)
    s = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return m, s, len(accs)


def _collect(base, scale, by_seed, C):
    """Returns (w_series, w_baseline, n_series, n_baseline, table_rows) --
    identical shape to plot_ema_sweep.py's _collect."""
    base_algo = ema_variant_name(base, scale)
    rows = []

    def _row(exp, k, init, accs):
        m, s, n = _stats(accs)
        rows.append(dict(exp=exp, k=k, init=init,
                          mean='' if m is None else round(m, 4),
                          std='' if s is None else round(s, 4),
                          n_seeds=n, accs=';'.join(f'{a:.2f}' for a in accs)))
        return m, s

    w_series, w_baseline = {}, {}
    for k in EXP_W_K_VALUES:
        accs = by_seed.get(C, {}).get(base_algo, {}).get(k, [])
        m, _ = _row('w', k, 'baseline', accs)
        w_baseline[k] = m
        means, stds = [], []
        for w in EXP_W_WEIGHTS:
            alg = ema_variant_name(biased_variant_name(base, 'cand', w), scale)
            accs = by_seed.get(C, {}).get(alg, {}).get(k, [])
            m, s = _row('w', k, f'W{weight_pct_str(w)}', accs)
            means.append(m if m is not None else float('nan'))
            stds.append(s if s is not None else float('nan'))
        w_series[k] = (means, stds)

    n_series, n_baseline = {}, {}
    for k in EXP_N_K_VALUES:
        accs = by_seed.get(C, {}).get(base_algo, {}).get(k, [])
        m, _ = _row('n', k, 'baseline', accs)
        n_baseline[k] = m
        means, stds = [], []
        for nv in EXP_N_VALUES:
            alg = ema_variant_name(biased_rand_all_variant_name(base, EXP_N_WEIGHT, nv), scale)
            accs = by_seed.get(C, {}).get(alg, {}).get(k, [])
            m, s = _row('n', k, f'W{weight_pct_str(EXP_N_WEIGHT)}-N{nv}', accs)
            means.append(m if m is not None else float('nan'))
            stds.append(s if s is not None else float('nan'))
        n_series[k] = (means, stds)

    return w_series, w_baseline, n_series, n_baseline, rows


def _plot(base, scale, w_series, w_baseline, n_series, n_baseline, out_path):
    ema = ema_tag(scale)
    fig, (axw, axn) = plt.subplots(1, 2, figsize=(13, 5.5))

    for k, (means, stds) in w_series.items():
        color = K_COLORS.get(k)
        axw.errorbar(range(len(EXP_W_WEIGHTS)), means, yerr=stds, marker='o', linewidth=2,
                     capsize=4, label=f'k={k}', color=color)
        if w_baseline.get(k) is not None:
            axw.axhline(w_baseline[k], color=color, linestyle='--', linewidth=1.2, alpha=0.7)
    axw.set_xticks(range(len(EXP_W_WEIGHTS)))
    axw.set_xticklabels([f'W{weight_pct_str(w)}' for w in EXP_W_WEIGHTS])
    axw.set_xlabel('True-class weight (W)  [BiasedCand init]')
    axw.set_ylabel('Test accuracy (%)')
    axw.set_title('TC-PLS: accuracy vs. W')
    axw.grid(True, alpha=0.3)
    axw.legend(fontsize=8, title='solid=BiasedCand, dashed=unbiased baseline')

    for k, (means, stds) in n_series.items():
        color = K_COLORS.get(k)
        axn.errorbar(range(len(EXP_N_VALUES)), means, yerr=stds, marker='o', linewidth=2,
                     capsize=4, label=f'k={k}', color=color)
        if n_baseline.get(k) is not None:
            axn.axhline(n_baseline[k], color=color, linestyle='--', linewidth=1.2, alpha=0.7)
    axn.set_xticks(range(len(EXP_N_VALUES)))
    axn.set_xticklabels([f'N{n}' for n in EXP_N_VALUES])
    axn.set_xlabel(f'# classes sharing (1-W) mass  [W={weight_pct_str(EXP_N_WEIGHT)}%, BiasedRandAll init]')
    axn.set_title('TC-n-PLS: accuracy vs. N')
    axn.grid(True, alpha=0.3)
    axn.legend(fontsize=8, title='solid=BiasedRandAll, dashed=unbiased baseline')

    fig.suptitle(f'{base}  —  confidence-EMA scale {scale:.2f}  ({ema})', fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _write_table(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['exp', 'k', 'init', 'mean', 'std', 'n_seeds', 'accs'])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_name', default='ab_sweep_0915')
    ap.add_argument('--reference_run', default='ema_sweep_0915',
                     help="Run to pull the 'PiCO-Fixed' (native A+B) reference column from (pass '' to skip)")
    ap.add_argument('--C', type=int, default=20)
    args = ap.parse_args()

    by_seed = load_results_by_seed([os.path.join('results', args.run_name)])
    ref_by_seed = load_results_by_seed([os.path.join('results', args.reference_run)]) if args.reference_run else {}

    bases = list(AB_SWEEP_BASES) + ([REFERENCE_BASE] if args.reference_run else [])
    for base in bases:
        source = ref_by_seed if base == REFERENCE_BASE else by_seed
        for scale in AB_SWEEP_SCALES:
            ema = ema_tag(scale)
            w_series, w_baseline, n_series, n_baseline, rows = _collect(base, scale, source, args.C)
            pic_path = os.path.join('plots', args.run_name, base, f'{base}-{ema}.png')
            tab_path = os.path.join('results', args.run_name, 'tables', f'{base}-{ema}.csv')
            _plot(base, scale, w_series, w_baseline, n_series, n_baseline, pic_path)
            _write_table(rows, tab_path)
            n_have = sum(1 for r in rows if r['n_seeds'])
            tag = f'{base:<35} {ema}'
            src = f'(from {args.reference_run})' if base == REFERENCE_BASE else ''
            print(f'{tag}: {n_have}/{len(rows)} cells with data -> {pic_path}  {tab_path}  {src}')


if __name__ == '__main__':
    main()
