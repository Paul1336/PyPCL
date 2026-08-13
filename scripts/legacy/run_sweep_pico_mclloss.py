"""
Sweep: PiCO-MCL on CIFAR-100 subsets.

PiCO-MCL = PiCO architecture with MCL-LOG cls loss instead of EMA PartialLoss.
Same C/k schedule as run_sweep_pico_comco.py.

Results : results/pico/pico_mclloss/results.csv
Plots   : plots/pico/pico_mclloss/   (per-C progress)
          plots/pico/comparison/     (PiCO vs PiCO-MCL vs ComCo, updated each k)

Usage:
    python scripts/run_sweep_pico_mclloss.py --data_dir data/ --epochs 200

4-GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_pico_mclloss.py --data_dir data/ --class_counts 5,27,100
    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep_pico_mclloss.py --data_dir data/ --class_counts 8,40
    CUDA_VISIBLE_DEVICES=2 python scripts/run_sweep_pico_mclloss.py --data_dir data/ --class_counts 12,58
    CUDA_VISIBLE_DEVICES=3 python scripts/run_sweep_pico_mclloss.py --data_dir data/ --class_counts 18,84

Smoke test:
    python scripts/run_sweep_pico_mclloss.py --data_dir data/ --epochs 5 --class_counts 5
"""

import argparse
import csv
import gc
import glob
import os
import sys
from datetime import datetime

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset
from src.training_pipelines import run_pico_mclloss_training


# ---------------------------------------------------------------------------
# C / k schedule (mirrors run_sweep_pico_comco.py)
# ---------------------------------------------------------------------------

def get_class_count_sequence() -> list:
    return [5, 8, 12, 18, 27, 40, 58, 84, 100]


def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    all_k = sorted(set(fixed + prop + [C - 1]))
    return [k for k in all_k if 1 <= k <= C - 1]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'total_classes', 'n_partial_labels', 'n_complementary_labels',
    'algorithm', 'final_accuracy', 'epochs', 'seed', 'training_time_s', 'timestamp',
]


def _load_done(csv_path: str) -> set:
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['total_classes']), int(row['n_partial_labels']), row['algorithm']))
    return done


def append_result(csv_path, C, k, algorithm, accuracy, epochs, seed, training_time_s):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            'total_classes':          C,
            'n_partial_labels':       k,
            'n_complementary_labels': C - k,
            'algorithm':              algorithm,
            'final_accuracy':         round(accuracy, 4),
            'epochs':                 epochs,
            'seed':                   seed,
            'training_time_s':        round(training_time_s, 1),
            'timestamp':              datetime.now().isoformat(),
        })


# ---------------------------------------------------------------------------
# Comparison plot: PiCO / ComCo (existing) vs PiCO-MCL (current run)
# ---------------------------------------------------------------------------

_ALGO_STYLE = {
    'PiCO':     dict(color='royalblue',  marker='o', linestyle='-',  linewidth=2, markersize=6, label='PiCO (PLL)'),
    'PiCO-MCL': dict(color='steelblue',  marker='s', linestyle='--', linewidth=2, markersize=6, label='PiCO-MCL (PLL)'),
    'ComCo':    dict(color='tomato',     marker='^', linestyle='-',  linewidth=2, markersize=6, label='ComCo (CLL)'),
}


def _load_csv_dir(root: str) -> dict:
    """Load {C: {alg: {k: acc}}} from a results dir (supports gpu* subdirs)."""
    patterns = [
        os.path.join(root, 'results.csv'),
        os.path.join(root, 'gpu*', 'results.csv'),
        os.path.join(root, '*',    'results.csv'),
    ]
    data = {}
    seen = set()
    for pat in patterns:
        for csv_path in glob.glob(pat):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                    if key in seen:
                        continue
                    seen.add(key)
                    C   = int(row['total_classes'])
                    k   = int(row['n_partial_labels'])
                    alg = row['algorithm']
                    acc = float(row['final_accuracy'])
                    data.setdefault(C, {}).setdefault(alg, {})[k] = acc
    return data


def update_comparison_plot(C: int, pico_comco_dir: str, mclloss_csv: str, plots_dir: str):
    """Regenerate plots/pico/comparison/C{C}_comparison.png."""
    # Load PiCO + ComCo from existing sweep
    ref_data = _load_csv_dir(pico_comco_dir).get(C, {})

    # Load PiCO-MCL from current run's CSV
    mcl_data: dict[int, float] = {}
    if os.path.isfile(mclloss_csv):
        with open(mclloss_csv, newline='') as f:
            for row in csv.DictReader(f):
                if int(row['total_classes']) == C and row['algorithm'] == 'PiCO-MCL':
                    mcl_data[int(row['n_partial_labels'])] = float(row['final_accuracy'])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'PiCO vs PiCO-MCL vs ComCo  —  C = {C} classes', fontsize=13)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_ylim(5, 85)
    ax.grid(True, alpha=0.3)

    plotted = False
    for alg, style in _ALGO_STYLE.items():
        if alg == 'PiCO-MCL':
            kv = sorted(mcl_data.items())
        else:
            kv = sorted(ref_data.get(alg, {}).items())
        if not kv:
            continue
        ks, accs = zip(*kv)
        ax.plot(ks, accs, **style)
        plotted = True

    if plotted:
        ax.legend(fontsize=10)
        fig.tight_layout()
        os.makedirs(plots_dir, exist_ok=True)
        path = os.path.join(plots_dir, f'C{C}_comparison.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Comparison plot → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PiCO-MCL sweep on CIFAR-100 subsets')
    parser.add_argument('--data_dir',       default='./data')
    parser.add_argument('--output_dir',     default='results/pico/pico_mclloss/')
    parser.add_argument('--pico_comco_dir', default='results/pico_comco/',
                        help='Root dir of existing PiCO + ComCo results for comparison plot')
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--lr',             type=float, default=0.01)
    parser.add_argument('--momentum',       type=float, default=0.9)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--log_dir',        default='logs/cifar100_subset')
    parser.add_argument('--config',         default='config.yaml')
    parser.add_argument('--class_counts',   default=None,
                        help='Comma-separated C values to run (default: full sequence)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config) as f:
        pico_config = yaml.safe_load(f)['pico']

    csv_path       = os.path.join(args.output_dir, 'results.csv')
    progress_dir   = os.path.join('plots', 'pico', 'pico_mclloss')
    comparison_dir = os.path.join('plots', 'pico', 'comparison')
    os.makedirs(args.output_dir, exist_ok=True)

    class_counts = (
        [int(c.strip()) for c in args.class_counts.split(',')]
        if args.class_counts else get_class_count_sequence()
    )
    print(f"Class counts: {class_counts}\n")

    train_ns = argparse.Namespace(
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    done = _load_done(csv_path)
    print(f"Already done: {len(done)} entries in {csv_path}\n")

    for C in class_counts:
        k_values = get_k_values(C)
        print(f"\n{'=' * 60}")
        print(f"C = {C} classes   |   k values: {k_values}")
        print(f"{'=' * 60}")

        for k in k_values:
            dk = (C, k, 'PiCO-MCL')
            if dk in done:
                print(f"  [skip] C={C} k={k} PiCO-MCL")
                continue

            print(f"\n--- C={C}, k={k} ---")
            train_config = {'num_classes': C}

            try:
                pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                    total_classes=C, n_partial_labels=k,
                    data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
                )
            except Exception as exc:
                print(f"  [SKIP] Data prep failed: {exc}")
                continue

            loaders = get_subset_dataloaders_full(
                pl_ds, cl_ds, orig_targets, test_info, args.batch_size
            )

            t0    = time.perf_counter()
            accs  = run_pico_mclloss_training(train_ns, loaders, train_config, pico_config, device)
            elapsed = time.perf_counter() - t0
            final = accs[-1]
            append_result(csv_path, C, k, 'PiCO-MCL', final, args.epochs, args.seed, elapsed)
            done.add(dk)
            print(f"  PiCO-MCL final accuracy: {final:.2f}%  ({elapsed/3600:.2f}h)")

            update_comparison_plot(C, args.pico_comco_dir, csv_path, comparison_dir)

    print(f"\nSweep finished.")
    print(f"  Results: {csv_path}")
    print(f"  Plots  : {comparison_dir}/")


if __name__ == '__main__':
    main()
