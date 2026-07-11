"""
Sweep: Proden (PLL) on CIFAR-100 subsets.

Same C/k schedule as run_sweep_pico_comco.py.
Uses pl_loader (partial label dataloader).

Results : results/proden/results.csv
Plots   : plots/proden/

4-GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_proden.py --data_dir data/ --class_counts 5,27,100
    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep_proden.py --data_dir data/ --class_counts 8,40
    CUDA_VISIBLE_DEVICES=2 python scripts/run_sweep_proden.py --data_dir data/ --class_counts 12,58
    CUDA_VISIBLE_DEVICES=3 python scripts/run_sweep_proden.py --data_dir data/ --class_counts 18,84

Smoke test:
    python scripts/run_sweep_proden.py --data_dir data/ --epochs 5 --class_counts 5
"""

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders, prepare_cifar100_subset
from src.plotting import plot_accuracy_vs_k
from src.training_pipelines import run_proden_training


def get_class_count_sequence() -> list:
    return [5, 8, 12, 18, 27, 40, 58, 84, 100]


def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    all_k = sorted(set(fixed + prop + [C - 1]))
    return [k for k in all_k if 1 <= k <= C - 1]


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h: return f"{h}h {m}m"
    if m: return f"{m}m {s}s"
    return f"{s}s"


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


def main():
    parser = argparse.ArgumentParser(description='Proden sweep on CIFAR-100 subsets')
    parser.add_argument('--data_dir',     default='./data')
    parser.add_argument('--output_dir',   default='results/proden/')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=512)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_dir',      default='logs/cifar100_subset')
    parser.add_argument('--class_counts', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    csv_path  = os.path.join(args.output_dir, 'results.csv')
    plots_dir = os.path.join('plots', 'proden')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

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
    print(f"Already done: {len(done)} entries\n")

    for C in class_counts:
        k_values = get_k_values(C)
        print(f"\n{'=' * 60}")
        print(f"C = {C} classes   |   k values: {k_values}")
        print(f"{'=' * 60}")

        results_for_C = []

        # Pre-populate results_for_C with already-completed k values
        for k in k_values:
            if (C, k, 'Proden') in done:
                with open(csv_path, newline='') as f:
                    for row in csv.DictReader(f):
                        if (int(row['total_classes']) == C
                                and int(row['n_partial_labels']) == k
                                and row['algorithm'] == 'Proden'):
                            results_for_C.append({'k': k, 'proden': float(row['final_accuracy'])})

        for k in k_values:
            if (C, k, 'Proden') in done:
                print(f"  [skip] C={C} k={k} Proden")
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

            loaders = get_subset_dataloaders(pl_ds, cl_ds, orig_targets, test_info, args.batch_size)

            t0      = time.perf_counter()
            accs    = run_proden_training(train_ns, loaders, train_config, device)
            elapsed = time.perf_counter() - t0
            final   = accs[-1]

            append_result(csv_path, C, k, 'Proden', final, args.epochs, args.seed, elapsed)
            done.add((C, k, 'Proden'))
            results_for_C.append({'k': k, 'proden': final})
            print(f"  Proden final accuracy: {final:.2f}%  ({_fmt(elapsed)})")

            plot_accuracy_vs_k(
                C, results_for_C, plots_dir,
                alg1_key='proden', alg1_label='Proden (PLL)', alg1_style='b-o',
                filename=f'C{C}_proden.png',
            )
            print(f"  Plot updated → {plots_dir}/C{C}_proden.png")

    print(f"\nSweep finished.")
    print(f"  Results: {csv_path}")
    print(f"  Plots  : {plots_dir}/")


if __name__ == '__main__':
    main()
