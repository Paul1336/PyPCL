"""
Sweep script: PiCO (PLL) vs ComCo (CLL) on CIFAR-100 class subsets.

Same C / k schedule as run_sweep_cifar100.py.
PiCO uses the PL loader; ComCo uses the CL loader (complement of PL).

Usage (full sweep):
    python scripts/run_sweep_pico_comco.py --data_dir data/ --epochs 200

4-GPU example:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_pico_comco.py --data_dir data/ --epochs 200 --output_dir results/pico_comco/gpu0 --class_counts 5,27,100
    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep_pico_comco.py --data_dir data/ --epochs 200 --output_dir results/pico_comco/gpu1 --class_counts 8,40
    CUDA_VISIBLE_DEVICES=2 python scripts/run_sweep_pico_comco.py --data_dir data/ --epochs 200 --output_dir results/pico_comco/gpu2 --class_counts 12,58
    CUDA_VISIBLE_DEVICES=3 python scripts/run_sweep_pico_comco.py --data_dir data/ --epochs 200 --output_dir results/pico_comco/gpu3 --class_counts 18,84

Smoke test:
    python scripts/run_sweep_pico_comco.py --data_dir data/ --epochs 5 --class_counts 5
"""

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset
from src.plotting import plot_accuracy_vs_k
from src.training_pipelines import run_comco_training, run_pico_training


# ---------------------------------------------------------------------------
# C and k schedule  (mirrors run_sweep_cifar100.py)
# ---------------------------------------------------------------------------

def get_class_count_sequence() -> list:
    return [5, 8, 12, 18, 27, 40, 58, 84, 100]


def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    all_k = sorted(set(fixed + prop + [C - 1]))
    return [k for k in all_k if 1 <= k <= C - 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h: return f"{h}h {m}m"
    if m: return f"{m}m {s}s"
    return f"{s}s"


_CSV_FIELDS = [
    'total_classes', 'n_partial_labels', 'n_complementary_labels',
    'algorithm', 'final_accuracy', 'epochs', 'seed', 'timestamp',
]


def append_result(csv_path, total_classes, n_partial_labels, algorithm,
                  final_accuracy, epochs, seed):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            'total_classes':          total_classes,
            'n_partial_labels':       n_partial_labels,
            'n_complementary_labels': total_classes - n_partial_labels,
            'algorithm':              algorithm,
            'final_accuracy':         round(final_accuracy, 4),
            'epochs':                 epochs,
            'seed':                   seed,
            'timestamp':              datetime.now().isoformat(),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PiCO vs ComCo sweep on CIFAR-100 subsets')
    parser.add_argument('--data_dir',     default='./data')
    parser.add_argument('--output_dir',   default='results/pico_comco/')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_dir',      default='logs/cifar100_subset')
    parser.add_argument('--config',       default='config.yaml')
    parser.add_argument('--class_counts', default=None,
                        help='Comma-separated C values to run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pico_config  = cfg['pico']
    comco_config = cfg.get('comco', {
        'low_dim':      128,
        'moco_queue':   8192,
        'moco_m':       0.999,
        'loss_weight':  0.3,
        'temperature':  0.17,
        'top_k':        1,
        'warmup_neg':   1,
        'warmup_pos':   100,
    })

    csv_path  = os.path.join(args.output_dir, 'results.csv')
    plots_dir = os.path.join('plots', 'pico_comco')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if args.class_counts:
        class_counts = [int(c.strip()) for c in args.class_counts.split(',')]
    else:
        class_counts = get_class_count_sequence()
    print(f"Class counts: {class_counts}\n")

    train_ns = argparse.Namespace(
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    for C in class_counts:
        k_values = get_k_values(C)
        print(f"\n{'=' * 60}")
        print(f"C = {C} classes   |   k values: {k_values}")
        print(f"{'=' * 60}")

        results_for_C = []

        for k in k_values:
            print(f"\n--- C={C}, k={k}  (PL candidates={k}, CL complements={C - k}) ---")
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

            # --- PiCO (PLL) ---
            pico_accs  = run_pico_training(train_ns, loaders, train_config,
                                           pico_config, pl_ds, orig_targets, device)
            pico_final = pico_accs[-1]
            append_result(csv_path, C, k, 'PiCO', pico_final, args.epochs, args.seed)
            print(f"  PiCO   final accuracy: {pico_final:.2f}%")

            # --- ComCo (CLL) ---
            comco_accs  = run_comco_training(train_ns, loaders, train_config,
                                             comco_config, device)
            comco_final = comco_accs[-1]
            append_result(csv_path, C, k, 'ComCo', comco_final, args.epochs, args.seed)
            print(f"  ComCo  final accuracy: {comco_final:.2f}%")

            results_for_C.append({'k': k, 'pico': pico_final, 'comco': comco_final})
            plot_accuracy_vs_k(
                C, results_for_C, plots_dir,
                alg1_key='pico',  alg1_label='PiCO (PLL)',  alg1_style='b-o',
                alg2_key='comco', alg2_label='ComCo (CLL)', alg2_style='g-^',
                filename=f'C{C}_pico_comco.png',
            )
            print(f"  Plot updated → {plots_dir}/C{C}_pico_comco.png")

    print(f"\nSweep finished.")
    print(f"  Results: {csv_path}")
    print(f"  Plots  : {plots_dir}/")


if __name__ == '__main__':
    main()
