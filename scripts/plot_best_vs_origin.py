"""
Plot best vs origin hyperparameter settings for Cour2011 and MCL-LOG.

Draws 4 lines per C:
  - Cour2011 (origin): SGD, lr=0.001, bs=512
  - Cour2011 (best):   best mean accuracy across all (optimizer, lr, bs) configs
  - MCL-LOG  (origin): SGD, lr=0.001, bs=512
  - MCL-LOG  (best):   best mean accuracy across all (optimizer, lr, bs) configs

One figure per C, x-axis = k, y-axis = accuracy.

Usage:
    python scripts/plot_best_vs_origin.py
    python scripts/plot_best_vs_origin.py --results_dir results/grid_search \
                                          --output_dir  plots/best_vs_origin
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Origin config (matches run_sweep_cifar100.py defaults from config.yaml)
# ---------------------------------------------------------------------------

ORIGIN_OPT = 'sgd'
ORIGIN_LR  = 0.001
ORIGIN_BS  = 512

ALGORITHMS = ['Cour2011', 'MCL-LOG']

LINE_STYLES = {
    ('Cour2011', 'origin'): dict(color='royalblue',  linestyle='-',  marker='o', linewidth=2, markersize=6),
    ('Cour2011', 'best'):   dict(color='royalblue',  linestyle='--', marker='s', linewidth=2, markersize=6),
    ('MCL-LOG',  'origin'): dict(color='tomato',     linestyle='-',  marker='o', linewidth=2, markersize=6),
    ('MCL-LOG',  'best'):   dict(color='tomato',     linestyle='--', marker='s', linewidth=2, markersize=6),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_csvs(results_dir: str) -> list:
    patterns = [
        os.path.join(results_dir, 'results.csv'),
        os.path.join(results_dir, 'gpu*', 'results.csv'),
        os.path.join(results_dir, '*',    'results.csv'),
    ]
    csv_files = []
    for pat in patterns:
        csv_files.extend(glob.glob(pat))
    csv_files = sorted(set(csv_files))

    rows = []
    seen = set()
    for csv_path in csv_files:
        print(f'  reading {csv_path}')
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['total_classes'], row['n_partial_labels'],
                       row['algorithm'], row['batch_size'],
                       row['optimizer'], row['lr'], row['seed'])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    'C':          int(row['total_classes']),
                    'k':          int(row['n_partial_labels']),
                    'algorithm':  row['algorithm'],
                    'acc':        float(row['final_accuracy']),
                    'batch_size': int(row['batch_size']),
                    'optimizer':  row['optimizer'],
                    'lr':         float(row['lr']),
                    'seed':       int(row['seed']),
                })
    return rows


def _lr_close(a: float, b: float) -> bool:
    return abs(a - b) / (abs(b) + 1e-12) < 1e-4


def aggregate(rows: list) -> dict:
    """
    Returns:
      data[C][alg][k][(opt, lr, bs)] = list of acc per seed
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )
    for r in rows:
        config_key = (r['optimizer'], r['lr'], r['batch_size'])
        data[r['C']][r['algorithm']][r['k']][config_key].append(r['acc'])
    return data


def get_origin_and_best(data_alg_k: dict) -> tuple[dict, dict, str]:
    """
    data_alg_k: {k: {(opt, lr, bs): [acc, ...]}}

    Returns:
      origin_k_acc : {k: mean_acc}   for the origin config
      best_k_acc   : {k: mean_acc}   for the best config (same config across all k)
      best_label   : str description of the best config
    """
    # Collect all k values
    all_k = sorted(data_alg_k.keys())

    # --- Origin ---
    origin_k_acc = {}
    for k in all_k:
        for (opt, lr, bs), accs in data_alg_k[k].items():
            if opt == ORIGIN_OPT and _lr_close(lr, ORIGIN_LR) and bs == ORIGIN_BS:
                origin_k_acc[k] = float(np.mean(accs))
                break

    # --- Best: find config with highest mean acc averaged over all k ---
    # First collect all configs that appear in at least one k
    all_configs = set()
    for k in all_k:
        all_configs.update(data_alg_k[k].keys())

    best_config = None
    best_score  = -1.0
    for cfg in all_configs:
        means = []
        for k in all_k:
            if cfg in data_alg_k[k]:
                means.append(float(np.mean(data_alg_k[k][cfg])))
        if not means:
            continue
        score = float(np.mean(means))
        if score > best_score:
            best_score  = score
            best_config = cfg

    best_k_acc = {}
    if best_config is not None:
        opt, lr, bs = best_config
        for k in all_k:
            if best_config in data_alg_k[k]:
                best_k_acc[k] = float(np.mean(data_alg_k[k][best_config]))
        best_label = f'{opt.upper()}, lr={lr:.0e}, bs={bs}'
    else:
        best_label = 'N/A'

    return origin_k_acc, best_k_acc, best_label


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _lr_label(lr: float) -> str:
    if lr >= 0.01:
        return f'{lr:.3g}'
    return f'{lr:.0e}'


def plot_C(C: int, data_C: dict, save_dir: str):
    """
    data_C: data[C] = {alg: {k: {config: [acc]}}}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'PLL vs CLL  —  C = {C} classes', fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    origin_label = f'SGD, lr={_lr_label(ORIGIN_LR)}, bs={ORIGIN_BS}'

    plotted = False
    for alg in ALGORITHMS:
        if alg not in data_C:
            continue

        origin_k_acc, best_k_acc, best_cfg_label = get_origin_and_best(data_C[alg])

        # Origin line
        if origin_k_acc:
            ks   = sorted(origin_k_acc.keys())
            accs = [origin_k_acc[k] for k in ks]
            style = LINE_STYLES[(alg, 'origin')]
            ax.plot(ks, accs, label=f'{alg} ({origin_label}) (origin)', **style)
            plotted = True

        # Best line
        if best_k_acc:
            ks   = sorted(best_k_acc.keys())
            accs = [best_k_acc[k] for k in ks]
            style = LINE_STYLES[(alg, 'best')]
            ax.plot(ks, accs, label=f'{alg} ({best_cfg_label})', **style)
            plotted = True

    if not plotted:
        plt.close(fig)
        print(f'  [skip] C={C} — no data')
        return

    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'C{C}_best_vs_origin.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot best vs origin hyperparameter configs')
    parser.add_argument('--results_dir', default='results/grid_search',
                        help='Root dir containing gpu*/results.csv files')
    parser.add_argument('--output_dir',  default='plots/best_vs_origin')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'[ERROR] Directory not found: {args.results_dir}')
        return

    print(f'Loading CSVs from {args.results_dir} …')
    rows = load_all_csvs(args.results_dir)
    print(f'  {len(rows)} unique rows loaded.')

    if not rows:
        print('No data found.')
        return

    data = aggregate(rows)
    all_C = sorted(data.keys())
    print(f'  C values found: {all_C}\n')

    for C in all_C:
        print(f'Plotting C={C}')
        plot_C(C, data[C], args.output_dir)

    print(f'\nDone. Plots → {args.output_dir}/')


if __name__ == '__main__':
    main()
