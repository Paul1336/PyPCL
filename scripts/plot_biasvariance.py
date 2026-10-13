#!/usr/bin/env python
"""Figures for a bias-variance-diagnostic sweep (e.g. biasvar_0921_C20), which
logs per-(algorithm, k, seed, epoch, loss_type) gradient bias^2/variance/mse/
cos_sim against an oracle gradient (see src/pipeline/biasvariance.py) into
results/<run_name>/detail/<algorithm>/C<C>_k<k>/biasvariance.csv.

Writes, per loss_type in {cls, cont}:
    plots/<run_name>/<loss_type>_vs_k.png   -- bias^2 / variance / MSE vs k at
                                                the final logged epoch, one
                                                line per algorithm (log-y)
    plots/<run_name>/<loss_type>_vs_epoch.png -- bias^2 / variance vs epoch,
                                                one small-multiple panel per k,
                                                one line per algorithm

    python scripts/plot_biasvariance.py --run_name biasvar_0921_C20
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# dataviz skill categorical palette, slots 1-8 (light mode), fixed order.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']

# Preferred color/order for algorithm names seen in past runs, so figures
# stay visually consistent across runs that reuse these names. Any
# algorithm NOT listed here (e.g. a new ComCo-Weighted-A0** sweep) still
# gets a color -- see algo_colors() below, which assigns the remaining
# PALETTE slots (in sorted-name order, for determinism) to whatever's
# actually present in that run's data.
KNOWN_ALGO_ORDER = [
    'PiCO-Fixed', 'PiCO-MCL-Fixed', 'PiCO-Weighted-A000', 'PiCO-Weighted-A050',
    'PiCO-Weighted-A100', 'ComCo-Fixed',
]


def algo_colors(present: list) -> dict:
    """Color map covering exactly the algorithms in `present` -- known names
    keep their usual PALETTE slot; anything new is assigned the remaining
    slots in sorted-name order (falls back to cycling the palette if there
    are more than 8 distinct algorithms in one run)."""
    ordered = [a for a in KNOWN_ALGO_ORDER if a in present]
    ordered += sorted(a for a in present if a not in KNOWN_ALGO_ORDER)
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(ordered)}


def load_detail(run_name):
    files = glob.glob(os.path.join('results', run_name, 'detail', '*', '*', 'biasvariance.csv'))
    if not files:
        raise SystemExit(f'no detail/*/*/biasvariance.csv found under results/{run_name}')
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset=['algorithm', 'k', 'seed', 'epoch', 'loss_type'], keep='last')
    return df


def plot_vs_k(df, loss_type, out_path):
    sub = df[df.loss_type == loss_type]
    final_epoch = sub.epoch.max()
    sub = sub[sub.epoch == final_epoch]
    agg = sub.groupby(['algorithm', 'k']).agg(
        bias2=('bias2', 'mean'), variance=('variance', 'mean'), mse=('mse', 'mean')
    ).reset_index()

    colors = algo_colors(list(agg.algorithm.unique()))
    algos = list(colors.keys())
    metrics = ['bias2', 'variance', 'mse']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, metric in zip(axes, metrics):
        for algo in algos:
            line = agg[agg.algorithm == algo].sort_values('k')
            ax.plot(line.k, line[metric], marker='o', markersize=5, linewidth=2,
                    color=colors[algo], label=algo)
        ax.set_yscale('log')
        ax.set_xlabel('k (candidate labels)')
        ax.set_title(metric)
        ax.grid(True, which='both', alpha=0.25)
    axes[0].set_ylabel(f'{loss_type} loss gradient, epoch {final_epoch}')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.suptitle(f'Gradient bias/variance vs k -- {loss_type} loss (final epoch {final_epoch})')
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vs_epoch(df, loss_type, out_path):
    sub = df[df.loss_type == loss_type]
    agg = sub.groupby(['algorithm', 'k', 'epoch']).agg(
        bias2=('bias2', 'mean'), variance=('variance', 'mean')
    ).reset_index()

    colors = algo_colors(list(agg.algorithm.unique()))
    algos = list(colors.keys())
    k_values = sorted(agg.k.unique())
    fig, axes = plt.subplots(2, len(k_values), figsize=(3.0 * len(k_values), 6.0), sharex=True)
    for col, k in enumerate(k_values):
        for row, metric in enumerate(['bias2', 'variance']):
            ax = axes[row, col]
            for algo in algos:
                line = agg[(agg.algorithm == algo) & (agg.k == k)].sort_values('epoch')
                ax.plot(line.epoch, line[metric], linewidth=2, color=colors[algo], label=algo)
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.25)
            if row == 0:
                ax.set_title(f'k={k}')
            if col == 0:
                ax.set_ylabel(metric)
            if row == 1:
                ax.set_xlabel('epoch')
    axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.suptitle(f'Gradient bias/variance vs epoch -- {loss_type} loss')
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    args = p.parse_args()

    df = load_detail(args.run_name)
    out_dir = os.path.join('plots', args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    for loss_type in ['cls', 'cont']:
        if loss_type not in df.loss_type.unique():
            continue
        plot_vs_k(df, loss_type, os.path.join(out_dir, f'{loss_type}_vs_k.png'))
        plot_vs_epoch(df, loss_type, os.path.join(out_dir, f'{loss_type}_vs_epoch.png'))
        print(f'wrote {out_dir}/{loss_type}_vs_k.png and {out_dir}/{loss_type}_vs_epoch.png')


if __name__ == '__main__':
    main()
