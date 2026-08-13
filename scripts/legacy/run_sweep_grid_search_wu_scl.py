"""
Hyperparameter grid search: Wu2022 (PLL) vs SCL-NL (CLL) on CIFAR-100 subsets.

Grid (12 configs total):
  batch_size : [64, 256, 512]
  optimizer  : sgd   (lr = 0.01)
               adam  (lr in [3e-3, 1e-3, 3e-4])
  seeds      : [0, 1, 2]   stored individually; averaged at plot time

Fixed:
  C values    : [5, 18, 40, 84, 100]
  k values    : {1, 2, round(50%*C), C-1}
  epochs      : 200
  momentum    : 0.9  (SGD only)
  weight_decay: 1e-4
  class seed  : 42

7-GPU (run in 7 terminals):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 0 --num_gpus 7
    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 1 --num_gpus 7
    CUDA_VISIBLE_DEVICES=2 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 2 --num_gpus 7
    CUDA_VISIBLE_DEVICES=3 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 3 --num_gpus 7
    CUDA_VISIBLE_DEVICES=4 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 4 --num_gpus 7
    CUDA_VISIBLE_DEVICES=5 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 5 --num_gpus 7
    CUDA_VISIBLE_DEVICES=6 python scripts/run_sweep_grid_search_wu_scl.py --gpu_id 6 --num_gpus 7

Each GPU writes to results/grid_search_wu_scl/gpu{gpu_id}/results.csv independently.

Smoke test:
    python scripts/run_sweep_grid_search_wu_scl.py --data_dir data/ --epochs 5 --class_counts 5,18
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
from src.engine import train_algorithm
from src.mcl_losses import MCL_LOG
from src.models import create_model
from src.scl_loss import SCL_NL
from src.wu_loss import WuPLLLoss

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

BATCH_SIZES  = [64, 256, 512]
SEEDS        = [0, 1, 2]
CLASS_COUNTS = [5, 18, 40, 84, 100]

SGD_LR   = 0.01
ADAM_LRS = [3e-3, 1e-3, 3e-4]

MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4

# 12 configs in fixed order for stable GPU assignment:
#   [0..2]  sgd  × 3 batch_sizes
#   [3..11] adam × 3 lrs × 3 batch_sizes
ALL_CONFIGS = (
    [('sgd',  SGD_LR, bs) for bs in BATCH_SIZES] +
    [('adam', lr,     bs) for lr in ADAM_LRS for bs in BATCH_SIZES]
)


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
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((
                int(row['total_classes']),
                int(row['n_partial_labels']),
                row['algorithm'],
                int(row['batch_size']),
                row['optimizer'],
                f"{float(row['lr']):.6g}",
                int(row['seed']),
            ))
    return done


def _done_key(C, k, alg, batch_size, opt_type, lr, seed) -> tuple:
    return (C, k, alg, batch_size, opt_type, f'{lr:.6g}', seed)


def append_result(csv_path, C, k, algorithm, accuracy, epochs,
                  batch_size, opt_type, lr, seed, training_time_s):
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
    parser = argparse.ArgumentParser(description='Grid search: Wu2022 vs SCL-NL')
    parser.add_argument('--data_dir',     default='./data')
    parser.add_argument('--output_dir',   default='results/grid_search_wu_scl/')
    parser.add_argument('--log_dir',      default='logs/cifar100_subset')
    parser.add_argument('--epochs',       type=int, default=200)
    parser.add_argument('--gpu_id',       type=int, default=0)
    parser.add_argument('--num_gpus',     type=int, default=1)
    parser.add_argument('--class_counts', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  gpu_id={args.gpu_id}/{args.num_gpus}")

    class_counts = (
        [int(c.strip()) for c in args.class_counts.split(',')]
        if args.class_counts else CLASS_COUNTS
    )

    gpu_dir  = os.path.join(args.output_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir, exist_ok=True)

    my_configs = [cfg for i, cfg in enumerate(ALL_CONFIGS)
                  if i % args.num_gpus == args.gpu_id]

    print(f"\nThis GPU handles {len(my_configs)}/{len(ALL_CONFIGS)} configs:")
    for opt_type, lr, batch_size in my_configs:
        print(f"  {opt_type:4s}  lr={lr:.0e}  bs={batch_size}")

    total_k    = sum(len(get_k_values(C)) for C in class_counts)
    total_runs = len(class_counts) * total_k * len(my_configs) * len(SEEDS) * 2
    print(f"\nEstimated training runs for this GPU: {total_runs}\n")

    done = _load_done_set(csv_path)
    print(f"Already done: {len(done)} results in {csv_path}\n")

    for C in class_counts:
        k_values = get_k_values(C)
        print(f"\n{'='*65}")
        print(f"C = {C}   k values = {k_values}")
        print(f"{'='*65}")

        for k in k_values:
            try:
                pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                    total_classes=C, n_partial_labels=k,
                    data_dir=args.data_dir, seed=42, log_dir=args.log_dir,
                )
            except Exception as exc:
                print(f"  [SKIP] Data prep failed C={C}, k={k}: {exc}")
                continue

            for opt_type, lr, batch_size in my_configs:
                loaders = get_subset_dataloaders(
                    pl_ds, cl_ds, orig_targets, test_info, batch_size
                )

                for seed in SEEDS:
                    tag = f"C={C} k={k} bs={batch_size} opt={opt_type} lr={lr:.0e} seed={seed}"

                    # --- Wu2022 (PLL) ---
                    dk = _done_key(C, k, 'Wu2022', batch_size, opt_type, lr, seed)
                    if dk in done:
                        print(f"  [skip] Wu2022  | {tag}")
                    else:
                        print(f"\n  Wu2022  | {tag}")
                        set_seed(seed)
                        t0  = time.perf_counter()
                        acc = train_single('pl', WuPLLLoss(), C, loaders,
                                           args.epochs, opt_type, lr, device)
                        elapsed = time.perf_counter() - t0
                        append_result(csv_path, C, k, 'Wu2022', acc,
                                      args.epochs, batch_size, opt_type, lr, seed, elapsed)
                        done.add(dk)
                        print(f"    → {acc:.2f}%  ({elapsed/3600:.2f}h)")

                    # --- SCL-NL (CLL) ---
                    dk = _done_key(C, k, 'SCL-NL', batch_size, opt_type, lr, seed)
                    if dk in done:
                        print(f"  [skip] SCL-NL  | {tag}")
                    else:
                        print(f"\n  SCL-NL  | {tag}")
                        set_seed(seed)
                        t0  = time.perf_counter()
                        acc = train_single('cl', SCL_NL(), C, loaders,
                                           args.epochs, opt_type, lr, device)
                        elapsed = time.perf_counter() - t0
                        append_result(csv_path, C, k, 'SCL-NL', acc,
                                      args.epochs, batch_size, opt_type, lr, seed, elapsed)
                        done.add(dk)
                        print(f"    → {acc:.2f}%  ({elapsed/3600:.2f}h)")

    print(f"\nGPU {args.gpu_id} finished. Results → {csv_path}")


if __name__ == '__main__':
    main()
