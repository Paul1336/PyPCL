#!/usr/bin/env python
"""Single entry point for the PLL/CLL algorithm comparison pipeline.

Consolidates what used to be 57 scripts (run_sweep_*, run_*_comparison,
plot_*, launch_*.sh — now kept for reference under scripts/legacy/) into
one CLI with three subcommands: run / merge / plot.

Subcommands:
    run    Train algorithms on the CIFAR-100 class-subset sweep and record
           results (supports multi-GPU sharding and resume).
    merge  Merge per-worker result shards into results/<run_name>/results.csv.
    plot   Draw accuracy-vs-k figures from one or more runs' merged results.

Examples:
    # Single GPU, a handful of algorithms
    python scripts/run_pipeline.py run --run_name demo \\
        --algorithms CLPL PRODEN MCL-LOG PiCO ComCo SoLar \\
        --c_values 5 20 --epochs 200

    # Multi-GPU: one process per GPU, same run_name, round-robin algorithm split
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py run --run_name demo --gpu_id 0 --num_gpus 8
    CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline.py run --run_name demo --gpu_id 1 --num_gpus 8
    ...

    # Merge / plot independently of training
    python scripts/run_pipeline.py merge --run demo
    python scripts/run_pipeline.py plot --runs demo --out plots/demo/summary.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.algorithms import ALL_ALGORITHM_NAMES
from src.pipeline.config import load_config
from src.pipeline import results as results_mod
from src.pipeline.plotting import plot_accuracy_vs_k


def _add_run_parser(sub):
    p = sub.add_parser('run', help='Train algorithms and record results')
    p.add_argument('--run_name', required=True, help='Name of this experiment; results go to results/<run_name>/')
    p.add_argument('--algorithms', nargs='+', default=ALL_ALGORITHM_NAMES, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--c_values', nargs='+', type=int, default=[5, 20])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--log_dir', default='logs/cifar100_subset')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--algo', default=None,
                    help='Pin this worker to one algorithm (overrides gpu_id round-robin)')
    p.add_argument('--report_every', type=int, default=10, help='Print ETA every N epochs')
    p.add_argument('--only_c', type=int, default=None, help='Run only this C value')
    p.add_argument('--only_k', type=int, default=None, help='Run only this k value')


def _add_merge_parser(sub):
    p = sub.add_parser('merge', help='Merge per-worker CSV shards into results.csv')
    p.add_argument('--run', required=True, dest='run_name')


def _add_plot_parser(sub):
    p = sub.add_parser('plot', help='Draw accuracy-vs-k figures from merged results')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one plot')
    p.add_argument('--algorithms', nargs='+', default=None, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--c_values', nargs='+', type=int, default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--group_by', default='paradigm', choices=['paradigm', 'all-in-one', 'per-algorithm'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    _add_run_parser(sub)
    _add_merge_parser(sub)
    _add_plot_parser(sub)
    args = parser.parse_args()

    if args.command == 'run':
        from src.pipeline.runner import run
        run(load_config(args))
    elif args.command == 'merge':
        out = results_mod.merge_shards(os.path.join('results', args.run_name))
        print(f'Merged -> {out}')
    elif args.command == 'plot':
        run_dirs = [os.path.join('results', r) for r in args.runs]
        plot_accuracy_vs_k(run_dirs, algorithms=args.algorithms, c_values=args.c_values,
                            out_path=args.out, group_by=args.group_by)


if __name__ == '__main__':
    main()
