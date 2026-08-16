#!/usr/bin/env python
"""Single entry point for the PLL/CLL algorithm comparison pipeline.

Consolidates what used to be 57 scripts (run_sweep_*, run_*_comparison,
plot_*, launch_*.sh — now kept for reference under scripts/legacy/) into
one CLI with three subcommands: run / merge / plot.

Subcommands:
    run              Train algorithms on the CIFAR-100 class-subset sweep and
                     record results (supports multi-GPU sharding and resume).
    merge            Merge per-worker result shards into
                     results/<run_name>/results.csv.
    plot             Draw accuracy-vs-k figures from one or more runs' merged
                     results.
    detail-plot      Per-class accuracy/loss heatmap over epoch checkpoints
                     (needs `run --detail`); one or two algorithms side by side.
    detail-plot-pico PiCO's contrastive pair-selection precision vs. ground
                     truth, over epochs (needs `run --detail`).

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
from src.pipeline.datasets import ALL_DATASET_NAMES


def _add_run_parser(sub):
    p = sub.add_parser('run', help='Train algorithms and record results')
    p.add_argument('--run_name', required=True, help='Name of this experiment; results go to results/<run_name>/')
    p.add_argument('--algorithms', nargs='+', default=ALL_ALGORITHM_NAMES, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--dataset', default='cifar100-subset', choices=ALL_DATASET_NAMES,
                    help="Which dataset to train on. 'cifar100-subset' (default) is the original "
                         "C-class-subset sweep; other values train on a fixed-class-count dataset "
                         "(see docs/00_paper_alignment_guide.md) and ignore --c_values (forced to "
                         "the dataset's native class count).")
    p.add_argument('--c_values', nargs='+', type=int, default=[5, 20],
                    help='Only used for --dataset cifar100-subset; ignored otherwise.')
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
    p.add_argument('--only_q', type=float, default=None,
                    help='Use variable (q-based) candidate-label generation instead of a fixed k: '
                         'each false label is independently included w.p. q (0-1). Replaces the '
                         'whole k-schedule with a single cell. Mutually exclusive with --only_k. '
                         'Only supported for --dataset cifar100-subset, mnist, fashion-mnist, '
                         'kmnist, and cifar10 so far (DatasetSpec.supports_q).')
    p.add_argument('--detail', action='store_true',
                    help='Enable per-class-per-epoch-checkpoint diagnostic logging (replaces '
                         'scripts/legacy/run_extended_analysis.py): adds an extra full test-set '
                         'eval every --detail_log_every epochs, written to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/per_class_loss.csv. For '
                         'PiCO specifically, also logs per-batch contrastive positive/negative '
                         'pair-selection precision against ground truth to pico_selection_stats.csv '
                         '(every batch, not gated by --detail_log_every). '
                         'Adds real overhead; off by default.')
    p.add_argument('--detail_log_every', type=int, default=10,
                    help='Epoch cadence for --detail per-class checkpoint logging.')
    p.add_argument('--tsne', action='store_true',
                    help='Every --tsne_every epochs, save a t-SNE snapshot of the contrastive '
                         'projection-head representation (raw embeddings + rendered PNG) to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/tsne/. Independent of '
                         '--detail. Only meaningful for dual-encoder models (PiCO family, ComCo '
                         'family) -- silently does nothing for other algorithms. Adds a CPU-bound '
                         't-SNE fit (seconds) each time; off by default.')
    p.add_argument('--tsne_every', type=int, default=50, help='Epoch cadence for --tsne snapshots.')
    p.add_argument('--tsne_max_points', type=int, default=2000,
                    help='Max test-set samples to embed per --tsne snapshot (t-SNE cost grows '
                         'roughly with n log n).')


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


def _add_detail_plot_parser(sub):
    p = sub.add_parser('detail-plot', help='Per-class accuracy/loss heatmap over epoch checkpoints '
                                            '(requires --detail during run); one or two algorithms side by side')
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg_l', required=True, help='Left (or only) algorithm')
    p.add_argument('--alg_r', default=None, help='Right algorithm, for a side-by-side comparison')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--acc_only', action='store_true', help='Omit the CE-loss row (accuracy heatmap only)')
    p.add_argument('--show_class_names', action='store_true')
    p.add_argument('--seed', type=int, default=42, help='Only used with --show_class_names')
    p.add_argument('--data_dir', default='./data', help='Only used with --show_class_names')


def _add_detail_plot_pico_parser(sub):
    p = sub.add_parser('detail-plot-pico', help="PiCO's contrastive positive/negative pair-selection "
                                                 "precision vs. ground truth, over epochs (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg', default='PiCO', help="Which PiCO-family algorithm's log to read (default: PiCO)")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    _add_run_parser(sub)
    _add_merge_parser(sub)
    _add_plot_parser(sub)
    _add_detail_plot_parser(sub)
    _add_detail_plot_pico_parser(sub)
    args = parser.parse_args()

    if args.command == 'run':
        if getattr(args, 'only_q', None) is not None and args.only_k is not None:
            parser.error('--only_q and --only_k are mutually exclusive')
        if getattr(args, 'only_q', None) is not None and not (0 <= args.only_q <= 1):
            parser.error('--only_q must be in [0, 1]')
        from src.pipeline.runner import run
        run(load_config(args))
    elif args.command == 'merge':
        out = results_mod.merge_shards(os.path.join('results', args.run_name))
        print(f'Merged -> {out}')
    elif args.command == 'plot':
        run_dirs = [os.path.join('results', r) for r in args.runs]
        plot_accuracy_vs_k(run_dirs, algorithms=args.algorithms, c_values=args.c_values,
                            out_path=args.out, group_by=args.group_by)
    elif args.command == 'detail-plot':
        from src.pipeline.detail import plot_heatmap
        class_names = None
        if args.show_class_names:
            from src.cifar100_subset import select_cifar100_classes
            from torchvision.datasets import CIFAR100
            indices = select_cifar100_classes(args.C, seed=args.seed)
            ds = CIFAR100(root=args.data_dir, train=True, download=False)
            class_names = [ds.classes[i] for i in indices]
        plot_heatmap(os.path.join('results', args.run_name), args.alg_l, args.C, args.k, args.out,
                     alg_r=args.alg_r, acc_only=args.acc_only, class_names=class_names)
    elif args.command == 'detail-plot-pico':
        from src.pipeline.detail import plot_pico_selection_stats
        plot_pico_selection_stats(os.path.join('results', args.run_name), args.alg, args.C, args.k, args.out)


if __name__ == '__main__':
    main()
