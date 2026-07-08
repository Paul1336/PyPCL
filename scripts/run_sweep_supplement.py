"""
Supplementary sweep: fills in k=2,3,4,5 that the original sweep missed.

Only runs (C, k) pairs that are absent from the main sweep.
Results go to results/cifar100_supplement/results.csv.
After each C finishes, regenerates the combined plot in plots/cifar100_sweep/
by merging main-sweep CSVs with supplement results.

Usage:
    # Run on whichever GPU is free (GPU0/GPU1 finish first)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_supplement.py --data_dir data/

    # Test a single C first
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep_supplement.py --data_dir data/ --class_counts 58
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
from src.training_pipelines import run_cour_training, run_mcl_training

# ---------------------------------------------------------------------------
# Missing (C, k) pairs — computed from the original get_k_values() vs {2,3,4,5}
# ---------------------------------------------------------------------------
MISSING_K: dict[int, list[int]] = {
    5:   [3],
    # 6: nothing missing
    8:   [3, 5],
    10:  [4],
    12:  [3, 5],
    15:  [5],
    18:  [3],
    22:  [3, 5],
    27:  [2, 4],
    33:  [2, 4, 5],
    40:  [2, 3, 5],
    48:  [2, 3, 4],
    58:  [2, 3, 4, 5],
    70:  [2, 3, 4, 5],
    84:  [2, 3, 4, 5],
    100: [2, 3, 4, 5],
}

# Main-sweep CSV directories to read when regenerating combined plots
_MAIN_GPU_DIRS = [
    'results/cifar100_sweep/gpu0',
    'results/cifar100_sweep/gpu1',
    'results/cifar100_sweep/gpu2',
    'results/cifar100_sweep/gpu3',
]

_CSV_FIELDS = [
    'total_classes', 'n_partial_labels', 'n_complementary_labels',
    'algorithm', 'final_accuracy', 'epochs', 'seed', 'training_time_s', 'timestamp',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def append_result(csv_path, total_classes, n_partial_labels, algorithm,
                  final_accuracy, epochs, seed, training_time_s):
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
            'training_time_s':        round(training_time_s, 1),
            'timestamp':              datetime.now().isoformat(),
        })


def load_combined_results(C: int, supplement_csv: str) -> list[dict]:
    """
    Reads all completed (C, k) pairs from main-sweep GPU CSVs + supplement CSV.
    Returns list of {'k', 'cour', 'mcl'} for k values where both algorithms exist.
    """
    cour, mcl = {}, {}

    def _read_csv(path):
        if not os.path.exists(path):
            return
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if int(row['total_classes']) != C:
                    continue
                k = int(row['n_partial_labels'])
                acc = float(row['final_accuracy'])
                if row['algorithm'] == 'Cour2011':
                    cour[k] = acc
                elif row['algorithm'] == 'MCL-LOG':
                    mcl[k] = acc

    for d in _MAIN_GPU_DIRS:
        _read_csv(os.path.join(d, 'results.csv'))
    _read_csv(supplement_csv)

    common = sorted(set(cour) & set(mcl))
    return [{'k': k, 'cour': cour[k], 'mcl': mcl[k]} for k in common]


def estimate_training_time(loaders, num_classes, epochs, device,
                           n_warmup=2, n_bench=5) -> float:
    from src.clpl_loss import CLPLSquaredHingeLoss
    from src.models import create_model
    model = create_model(num_classes).to(device)
    loss_fn = CLPLSquaredHingeLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    it = iter(loaders['pl'])

    def _next():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loaders['pl'])
            return next(it)

    def _step(imgs, lbls):
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        loss_fn(model(imgs), lbls).backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    for _ in range(n_warmup):
        _step(*_next())
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        _step(*_next())
        times.append(time.perf_counter() - t0)

    del model, loss_fn, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    avg = sum(times) / len(times)
    return avg * len(loaders['pl']) * epochs * 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Supplement sweep: k=2,3,4,5 gaps')
    parser.add_argument('--data_dir',     default='./data')
    parser.add_argument('--output_dir',   default='results/cifar100_supplement')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_dir',      default='logs/cifar100_subset')
    parser.add_argument('--class_counts', default=None,
                        help='Comma-separated C values to run (default: all with gaps)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    csv_path  = os.path.join(args.output_dir, 'results.csv')
    plots_dir = os.path.join('plots', 'cifar100_sweep')   # same dir as main sweep
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if args.class_counts:
        class_counts = [int(c.strip()) for c in args.class_counts.split(',')]
        missing_k = {C: MISSING_K[C] for C in class_counts if C in MISSING_K}
    else:
        missing_k = MISSING_K

    train_ns = argparse.Namespace(
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    total_pairs = sum(len(ks) for ks in missing_k.values())
    print(f"Will run {total_pairs} missing (C, k) pairs across {len(missing_k)} C values.\n")

    for C, k_list in missing_k.items():
        print(f"\n{'=' * 60}")
        print(f"C = {C}  |  missing k values: {k_list}")
        print(f"{'=' * 60}")
        train_config = {'num_classes': C}

        for k in k_list:
            print(f"\n--- C={C}, k={k} (CL complement m={C - k}) ---")

            try:
                pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                    total_classes=C, n_partial_labels=k,
                    data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
                )
            except Exception as exc:
                print(f"  [SKIP] Data prep failed: {exc}")
                continue

            loaders = get_subset_dataloaders(pl_ds, cl_ds, orig_targets, test_info, args.batch_size)

            try:
                est = estimate_training_time(loaders, C, args.epochs, device)
                print(f"  Estimated time: {_fmt(est)}  "
                      f"({len(loaders['pl'])} batches/epoch × {args.epochs} epochs × 2 algs)")
            except Exception as e:
                print(f"  [time estimate skipped: {e}]")

            t0 = time.perf_counter()
            cour_accs = run_cour_training(train_ns, loaders, train_config, device)
            cour_time = time.perf_counter() - t0
            cour_final = cour_accs[-1]
            append_result(csv_path, C, k, 'Cour2011', cour_final, args.epochs, args.seed, cour_time)
            print(f"  Cour 2011  final accuracy: {cour_final:.2f}%  ({_fmt(cour_time)})")

            t0 = time.perf_counter()
            mcl_accs = run_mcl_training(train_ns, loaders, train_config, device, loss_type='log')
            mcl_time = time.perf_counter() - t0
            mcl_final = mcl_accs[-1]
            append_result(csv_path, C, k, 'MCL-LOG', mcl_final, args.epochs, args.seed, mcl_time)
            print(f"  MCL-LOG    final accuracy: {mcl_final:.2f}%  ({_fmt(mcl_time)})")

        # Regenerate combined plot for this C (main sweep + supplement results)
        results_combined = load_combined_results(C, csv_path)
        if results_combined:
            plot_accuracy_vs_k(C, results_combined, plots_dir)
            print(f"\n  Plot updated → {plots_dir}/C{C}_accuracy_vs_k.png  "
                  f"({len(results_combined)} k values total)")

    print(f"\nSupplement sweep finished.")
    print(f"  Results : {csv_path}")
    print(f"  Plots   : {plots_dir}/")


if __name__ == '__main__':
    main()
