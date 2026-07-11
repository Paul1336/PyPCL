"""
Hyperparameter grid search: Cour2011 (PLL) vs MCL-LOG (CLL) on CIFAR-100 subsets.

Grid (27 configs total):
  batch_size  : [64, 256, 512]
  optimizer   : sgd   (lr in [0.1, 0.01, 0.001])
                adam  (lr in [3e-3, 1e-3, 3e-4])
                adamw (lr in [3e-3, 1e-3, 3e-4])
  seeds       : [0, 1, 2]   stored individually; averaged at plot time

Fixed:
  C values    : [5, 18, 40, 84, 100]
  k values    : {1, 2, round(50%*C), C-1}
  epochs      : 200
  momentum    : 0.9  (SGD only)
  weight_decay: 1e-4
  class seed  : 42   (same class subset across all configs)

Single GPU:
    python scripts/run_sweep_grid_search.py --data_dir data/

Multi-GPU (4 GPUs, run in 4 separate terminals):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_grid_search.py --gpu_id 0 --num_gpus 4
    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep_grid_search.py --gpu_id 1 --num_gpus 4
    CUDA_VISIBLE_DEVICES=2 python scripts/run_sweep_grid_search.py --gpu_id 2 --num_gpus 4
    CUDA_VISIBLE_DEVICES=3 python scripts/run_sweep_grid_search.py --gpu_id 3 --num_gpus 4

Each GPU writes to results/grid_search/gpu{gpu_id}/results.csv independently.
plot_grid_results.py merges all gpu* CSVs automatically.

Smoke test:
    python scripts/run_sweep_grid_search.py --data_dir data/ --epochs 5 --class_counts 5,18
"""

import argparse
import csv
import gc
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders, prepare_cifar100_subset
from src.clpl_loss import CLPLSquaredHingeLoss
from src.engine import train_algorithm
from src.mcl_losses import MCL_LOG
from src.models import create_model

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

BATCH_SIZES = [64, 256, 512]
SEEDS       = [0, 1, 2]
CLASS_COUNTS = [5, 18, 40, 84, 100]

OPTIMIZER_LR = {
    'sgd':   [0.1,  0.01,  0.001],
    'adam':  [3e-3, 1e-3,  3e-4],
    'adamw': [3e-3, 1e-3,  3e-4],
}

MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4

# All 27 (opt_type, lr, batch_size) configs in a fixed order for stable GPU assignment
ALL_CONFIGS = [
    (opt_type, lr, batch_size)
    for opt_type, lr_list in OPTIMIZER_LR.items()
    for lr in lr_list
    for batch_size in BATCH_SIZES
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2] if k <= C - 1]
    half  = max(1, round(0.5 * C))
    return sorted(set(fixed + [half, C - 1]))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(opt_type: str, params, lr: float) -> optim.Optimizer:
    if opt_type == 'sgd':
        return optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif opt_type == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    elif opt_type == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    raise ValueError(f'Unknown optimizer: {opt_type}')


def train_single(loader_key: str, loss_fn, C: int, loaders: dict,
                 epochs: int, opt_type: str, lr: float, device) -> float:
    model     = create_model(C)
    optimizer = make_optimizer(opt_type, model.parameters(), lr)
    accs = train_algorithm(model, loaders[loader_key], loaders['test'],
                           loss_fn, optimizer, epochs, device)
    del model, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return accs[-1]


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'total_classes', 'n_partial_labels', 'n_complementary_labels',
    'algorithm', 'final_accuracy', 'epochs',
    'batch_size', 'optimizer', 'lr', 'seed',
    'training_time_s', 'timestamp',
]


def _load_done_set(csv_path: str) -> set:
    """Return set of (C, k, alg, batch_size, opt_type, lr_str, seed) already in CSV."""
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (
                int(row['total_classes']),
                int(row['n_partial_labels']),
                row['algorithm'],
                int(row['batch_size']),
                row['optimizer'],
                f"{float(row['lr']):.6g}",
                int(row['seed']),
            )
            done.add(key)
    return done


def _done_key(C, k, alg, batch_size, opt_type, lr, seed) -> tuple:
    return (C, k, alg, batch_size, opt_type, f'{lr:.6g}', seed)


def append_result(csv_path: str, C: int, k: int, algorithm: str,
                  accuracy: float, epochs: int,
                  batch_size: int, opt_type: str, lr: float, seed: int,
                  training_time_s: float):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'total_classes':          C,
            'n_partial_labels':       k,
            'n_complementary_labels': C - k,
            'algorithm':              algorithm,
            'final_accuracy':         round(accuracy, 4),
            'epochs':                 epochs,
            'batch_size':             batch_size,
            'optimizer':              opt_type,
            'lr':                     lr,
            'seed':                   seed,
            'training_time_s':        round(training_time_s, 1),
            'timestamp':              datetime.now().isoformat(),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Grid search sweep: Cour2011 vs MCL-LOG')
    parser.add_argument('--data_dir',     default='./data')
    parser.add_argument('--output_dir',   default='results/grid_search/')
    parser.add_argument('--log_dir',      default='logs/cifar100_subset')
    parser.add_argument('--epochs',       type=int, default=200)
    parser.add_argument('--gpu_id',       type=int, default=0,
                        help='GPU index for work partitioning (0-indexed)')
    parser.add_argument('--num_gpus',     type=int, default=1,
                        help='Total number of GPUs used in this sweep')
    parser.add_argument('--class_counts', default=None,
                        help='Comma-separated C values to override default list')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  gpu_id={args.gpu_id}/{args.num_gpus}")

    class_counts = (
        [int(c.strip()) for c in args.class_counts.split(',')]
        if args.class_counts else CLASS_COUNTS
    )

    # Each GPU writes to its own subdirectory to avoid file-write races
    gpu_dir  = os.path.join(args.output_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir, exist_ok=True)

    # Assign configs to this GPU by round-robin over the fixed config list
    my_configs = [cfg for i, cfg in enumerate(ALL_CONFIGS)
                  if i % args.num_gpus == args.gpu_id]

    print(f"\nThis GPU handles {len(my_configs)}/{len(ALL_CONFIGS)} configs:")
    for opt_type, lr, batch_size in my_configs:
        print(f"  {opt_type:6s}  lr={lr:.0e}  bs={batch_size}")

    total_k = sum(len(get_k_values(C)) for C in class_counts)
    total_runs = len(class_counts) * total_k * len(my_configs) * len(SEEDS) * 2
    print(f"\nEstimated training runs for this GPU: {total_runs}\n")

    # Load already-completed runs once upfront
    done = _load_done_set(csv_path)
    print(f"Already done: {len(done)} results found in {csv_path}\n")

    for C in class_counts:
        k_values = get_k_values(C)
        print(f"\n{'='*65}")
        print(f"C = {C}   k values = {k_values}")
        print(f"{'='*65}")

        for k in k_values:
            # Prepare datasets once per (C, k) — class selection fixed by seed=42
            try:
                pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                    total_classes=C,
                    n_partial_labels=k,
                    data_dir=args.data_dir,
                    seed=42,
                    log_dir=args.log_dir,
                )
            except Exception as exc:
                print(f"  [SKIP] Data prep failed C={C}, k={k}: {exc}")
                continue

            for opt_type, lr, batch_size in my_configs:
                loaders = get_subset_dataloaders(
                    pl_ds, cl_ds, orig_targets, test_info, batch_size
                )

                for seed in SEEDS:
                    tag = (f"C={C} k={k} bs={batch_size} "
                           f"opt={opt_type} lr={lr:.0e} seed={seed}")

                    # Cour2011
                    dk = _done_key(C, k, 'Cour2011', batch_size, opt_type, lr, seed)
                    if dk in done:
                        print(f"  [skip] Cour2011 | {tag}")
                    else:
                        print(f"\n  Cour2011 | {tag}")
                        set_seed(seed)
                        t0  = time.perf_counter()
                        acc = train_single('pl', CLPLSquaredHingeLoss(), C, loaders,
                                           args.epochs, opt_type, lr, device)
                        elapsed = time.perf_counter() - t0
                        append_result(csv_path, C, k, 'Cour2011', acc,
                                      args.epochs, batch_size, opt_type, lr, seed, elapsed)
                        done.add(dk)
                        print(f"    → {acc:.2f}%  ({elapsed/3600:.2f}h)")

                    # MCL-LOG
                    dk = _done_key(C, k, 'MCL-LOG', batch_size, opt_type, lr, seed)
                    if dk in done:
                        print(f"  [skip] MCL-LOG  | {tag}")
                    else:
                        print(f"\n  MCL-LOG  | {tag}")
                        set_seed(seed)
                        t0  = time.perf_counter()
                        acc = train_single('cl', MCL_LOG(num_classes=C), C, loaders,
                                           args.epochs, opt_type, lr, device)
                        elapsed = time.perf_counter() - t0
                        append_result(csv_path, C, k, 'MCL-LOG', acc,
                                      args.epochs, batch_size, opt_type, lr, seed, elapsed)
                        done.add(dk)
                        print(f"    → {acc:.2f}%  ({elapsed/3600:.2f}h)")

    print(f"\nGPU {args.gpu_id} finished. Results → {csv_path}")


if __name__ == '__main__':
    main()
