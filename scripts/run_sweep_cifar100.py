"""
Sweep script: Cour 2011 (PLL) vs MCL-LOG (CLL) on CIFAR-100 class subsets.

For each class count C in the sequence [5, 6, 8, 10, ..., 100]:
  - Select C classes from CIFAR-100 (fixed by seed).
  - For each k in {1, 10%C, 20%C, 30%C, 50%C, 70%C, 80%C, 90%C, C-1}:
      - Train Cour 2011 on PL labels (k candidates per sample).
      - Train MCL-LOG  on CL labels (C-k complementary, complement of PL).
      - Append result to CSV.
      - Redraw the per-C accuracy-vs-k plot (incremental update).

Usage (full sweep):
    python scripts/run_sweep_cifar100.py --data_dir data/ --epochs 200

Smoke test:
    python scripts/run_sweep_cifar100.py --data_dir data/ --epochs 5 --class_counts 5,10
"""

import argparse
import csv
import gc
import math
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders, prepare_cifar100_subset
from src.plotting import plot_accuracy_vs_k
from src.training_pipelines import run_cour_training, run_mcl_training


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def get_class_count_sequence() -> list:
    """
    Generates integer class counts from 5 to 100 via ×1.2 growth (ceiling).
    Sequence: [5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40, 48, 58, 70, 84, 100].
    """
    counts = [5]
    while True:
        nxt = math.ceil(counts[-1] * 1.2)
        if nxt >= 100:
            if counts[-1] != 100:
                counts.append(100)
            break
        counts.append(nxt)
    return counts


def get_k_values(C: int) -> list:
    """
    Returns the sorted, deduplicated list of k values to test for class count C.
    Includes:
      - absolute endpoints {1, C-1}
      - proportional values at 10%/20%/30%/50%/70%/80%/90% of C
      - fixed small-k values {2, 3, 4, 5} (clamped to [1, C-1])
    All values are filtered to [1, C-1].
    """
    proportional = [max(1, round(r * C)) for r in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]]
    fixed_small  = [k for k in [2, 3, 4, 5] if k <= C - 1]
    all_k = sorted(set([1, C - 1] + proportional + fixed_small))
    return [k for k in all_k if 1 <= k <= C - 1]


# ---------------------------------------------------------------------------
# Training time estimation
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def estimate_training_time(loaders, num_classes: int, epochs: int, device,
                           n_warmup: int = 2, n_bench: int = 5) -> float:
    """
    Runs a few timed forward+backward passes through the PL loader to estimate
    how long the full sweep (Cour 2011 + MCL-LOG) will take, then returns the
    estimate in seconds.

    Uses Cour loss and a fresh ResNet18 for benchmarking (deleted after).
    n_warmup passes are discarded; n_bench passes are timed.
    """
    from src.clpl_loss import CLPLSquaredHingeLoss
    from src.models import create_model

    model = create_model(num_classes).to(device)
    loss_fn = CLPLSquaredHingeLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    loader_iter = iter(loaders['pl'])

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loaders['pl'])
            return next(loader_iter)

    def step(images, labels):
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        loss_fn(model(images), labels).backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Warm-up (not timed)
    for _ in range(n_warmup):
        step(*next_batch())

    # Timed passes
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        step(*next_batch())
        times.append(time.perf_counter() - t0)

    del model, loss_fn, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    avg_batch_s = sum(times) / len(times)
    # 2 algorithms; both loaders have the same length
    batches_per_epoch = len(loaders['pl'])
    return avg_batch_s * batches_per_epoch * epochs * 2


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'total_classes', 'n_partial_labels', 'n_complementary_labels',
    'algorithm', 'final_accuracy', 'epochs', 'seed', 'timestamp',
]


def append_result(csv_path: str, total_classes, n_partial_labels, algorithm,
                  final_accuracy, epochs, seed):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
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
    parser = argparse.ArgumentParser(description='Cour 2011 vs MCL-LOG sweep on CIFAR-100 subsets')
    parser.add_argument('--data_dir',     default='./data',
                        help='Directory for CIFAR-100 download / cache')
    parser.add_argument('--output_dir',   default='results/cifar100_sweep/',
                        help='Directory for the results CSV')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_dir',      default='logs/cifar100_subset',
                        help='Directory for per-experiment JSON logs')
    parser.add_argument('--class_counts', default=None,
                        help='Comma-separated class counts to override the auto sequence')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    csv_path  = os.path.join(args.output_dir, 'results.csv')
    plots_dir = os.path.join('plots', 'cifar100_sweep')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Build class-count list
    if args.class_counts:
        class_counts = [int(c.strip()) for c in args.class_counts.split(',')]
    else:
        class_counts = get_class_count_sequence()
    print(f"Class counts to sweep: {class_counts}\n")

    # Minimal args namespace expected by setup_* / train_algorithm
    train_ns = argparse.Namespace(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    for C in class_counts:
        k_values = get_k_values(C)
        print(f"\n{'=' * 60}")
        print(f"C = {C} classes   |   k values: {k_values}")
        print(f"{'=' * 60}")

        results_for_C = []  # accumulates {'k', 'cour', 'mcl'} as k values complete

        for k in k_values:
            print(f"\n--- C={C}, k={k} (CL complement m={C - k}) ---")
            train_config = {'num_classes': C}

            # Prepare data
            try:
                pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                    total_classes=C,
                    n_partial_labels=k,
                    data_dir=args.data_dir,
                    seed=args.seed,
                    log_dir=args.log_dir,
                )
            except Exception as exc:
                print(f"  [SKIP] Data preparation failed: {exc}")
                continue

            loaders = get_subset_dataloaders(pl_ds, cl_ds, orig_targets, test_info, args.batch_size)

            # --- Time estimate ---
            try:
                est_s = estimate_training_time(loaders, C, args.epochs, device)
                batches = len(loaders['pl'])
                print(f"  Estimated time  : {_fmt(est_s)}  "
                      f"({batches} batches/epoch × {args.epochs} epochs × 2 algs)")
            except Exception as e:
                print(f"  [time estimate skipped: {e}]")

            # --- Cour 2011 (PLL) ---
            cour_accs  = run_cour_training(train_ns, loaders, train_config, device)
            cour_final = cour_accs[-1]
            append_result(csv_path, C, k, 'Cour2011', cour_final, args.epochs, args.seed)
            print(f"  Cour 2011  final accuracy: {cour_final:.2f}%")

            # --- MCL-LOG (CLL) ---
            mcl_accs  = run_mcl_training(train_ns, loaders, train_config, device, loss_type='log')
            mcl_final = mcl_accs[-1]
            append_result(csv_path, C, k, 'MCL-LOG', mcl_final, args.epochs, args.seed)
            print(f"  MCL-LOG    final accuracy: {mcl_final:.2f}%")

            # Accumulate and redraw plot immediately
            results_for_C.append({'k': k, 'cour': cour_final, 'mcl': mcl_final})
            plot_accuracy_vs_k(C, results_for_C, plots_dir)
            print(f"  Plot updated → plots/cifar100_sweep/C{C}_accuracy_vs_k.png")

    print(f"\nSweep finished.")
    print(f"  Results CSV : {csv_path}")
    print(f"  Plots       : plots/cifar100_sweep/")
    print(f"  Logs        : {args.log_dir}/")


if __name__ == '__main__':
    main()
