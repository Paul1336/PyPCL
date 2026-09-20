#!/usr/bin/env python
"""Per-(algorithm, EMA level) figures + CSV tables for the ema_sweep_0915
init-sensitivity re-run (see scripts/run_ema_sweep.py's docstring).

Reproduces the two slide-63/64 style views -- TC-PLS "accuracy vs true-class
weight W" and TC-n-PLS "accuracy vs number of classes N sharing the (1-W)
mass" -- but with the confidence-update EMA scale held fixed per figure/table
and k as the extra per-line dimension (the original slides fixed k and swept
algorithm; here the algorithm is fixed too, since PiCO-Fixed and PRODEN each
get their own figure/table per EMA level).

    python scripts/plot_ema_sweep.py --run_name ema_sweep_0915

For each of the 2 bases (PiCO-Fixed, PRODEN) x 5 EMA levels, writes:
    plots/<run_name>/<base>/<base>-<EMA_TAG>.png   (2-panel: W sweep | N sweep)
    results/<run_name>/tables/<base>-<EMA_TAG>.csv (rows: exp, k, init)
i.e. 5 pictures + 5 tables per base, 10 of each total. Safe to re-run at any
time on partial results (missing cells show as blank rows / gaps in lines).
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
from src.pll_init import (BIAS_RAND_ALL_N_VALUES, CONF_EMA_SCALES, biased_rand_all_variant_name,
                           biased_variant_name, ema_tag, ema_variant_name, weight_pct_str)

# Must match scripts/run_ema_sweep.py's EXP_W_* / EXP_N_* constants.
EXP_W_K_VALUES = [10, 15, 19]
EXP_W_WEIGHTS = [0.052, 0.10, 0.20]
EXP_N_K_VALUES = [5]
EXP_N_WEIGHT = 0.20
EXP_N_VALUES = list(BIAS_RAND_ALL_N_VALUES)
BASES = ('PiCO-Fixed', 'PRODEN')

K_COLORS = {5: '#9467bd', 10: '#1f77b4', 15: '#2ca02c', 19: '#d62728'}


def _stats(accs):
    if not accs:
        return None, None, 0
    m = statistics.mean(accs)
    s = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return m, s, len(accs)


def _collect(base, scale, by_seed, C):
    """Returns (w_series, w_baseline, n_series, n_baseline, table_rows).
    w_series/n_series: k -> (means, stds) aligned with EXP_W_WEIGHTS/EXP_N_VALUES.
    w_baseline/n_baseline: k -> mean accuracy of the unbiased (candidate_masked)
    init at this base+scale, i.e. the dashed reference line for that k."""
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
    ap.add_argument('--run_name', default='ema_sweep_0915')
    ap.add_argument('--C', type=int, default=20)
    args = ap.parse_args()

    by_seed = load_results_by_seed([os.path.join('results', args.run_name)])

    for base in BASES:
        for scale in CONF_EMA_SCALES:
            ema = ema_tag(scale)
            w_series, w_baseline, n_series, n_baseline, rows = _collect(base, scale, by_seed, args.C)
            pic_path = os.path.join('plots', args.run_name, base, f'{base}-{ema}.png')
            tab_path = os.path.join('results', args.run_name, 'tables', f'{base}-{ema}.csv')
            _plot(base, scale, w_series, w_baseline, n_series, n_baseline, pic_path)
            _write_table(rows, tab_path)
            n_have = sum(1 for r in rows if r['n_seeds'])
            print(f'{base:<11} {ema}: {n_have}/{len(rows)} cells with data -> {pic_path}  {tab_path}')


if __name__ == '__main__':
    main()
