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
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                    help='One or more random seeds to sweep per (C, k, algorithm) cell -- each '
                         'seed gets its own row in results.csv (dedup/resume key includes seed, '
                         'so partially-completed seed sweeps resume correctly). Use `report` to '
                         'see per-seed values plus the mean/std across seeds. Diagnostic '
                         'instrumentation (--detail/--tsne/--concentration/--knn_eval) only runs '
                         'for the first seed in this list, regardless of how many are swept -- see '
                         'src/pipeline/detail.py _seed_matches. Default: 3 seeds.')
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--log_dir', default='logs/cifar100_subset')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--algo', default=None,
                    help='Pin this worker to one algorithm (overrides gpu_id round-robin)')
    p.add_argument('--report_every', type=int, default=10, help='Print ETA every N epochs')
    p.add_argument('--only_c', type=int, default=None, help='Run only this C value')
    p.add_argument('--only_k', nargs='+', type=int, default=None,
                    help='Run only this k value, or a specific list of k values (e.g. --only_k 5 8 10) '
                         'instead of the default per-C k-schedule.')
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
    p.add_argument('--concentration', action='store_true',
                    help='Log per-sample and averaged prediction-concentration (entropy + '
                         'max-softmax-prob of the model\'s own predicted distribution over the '
                         'training set) every --concentration_log_every epochs, to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/concentration_summary.csv '
                         'and concentration/ep{epoch}.npz. Independent of --detail. Works for the '
                         'PiCO family, PRODEN family, and ComCo. Adds an extra full train-set '
                         'forward pass each time it fires; off by default.')
    p.add_argument('--concentration_log_every', type=int, default=10,
                    help='Epoch cadence for --concentration logging.')
    p.add_argument('--knn_eval', action='store_true',
                    help='Once at the end of training, evaluate kNN top-1 accuracy using the '
                         'contrastive encoder (train set as reference bank, test set as queries), '
                         'written to results/<run_name>/detail/<algorithm>/C{C}_k{k}/knn_eval.csv. '
                         'Only meaningful for dual-encoder models (PiCO/ComCo family) -- no-op '
                         'otherwise; off by default.')
    p.add_argument('--knn_eval_k', type=int, default=20, help='Number of neighbors for --knn_eval.')
    p.add_argument('--knn_temperature', type=float, default=0.07,
                    help='Softmax temperature for the --knn_eval weighted vote.')


def _add_merge_parser(sub):
    p = sub.add_parser('merge', help='Merge per-worker CSV shards into results.csv')
    p.add_argument('--run', required=True, dest='run_name')


def _add_report_parser(sub):
    p = sub.add_parser('report', help='Print per-seed accuracy plus mean/std across seeds, '
                                       'one row per (C, k, algorithm)')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one report')
    p.add_argument('--algorithms', nargs='+', default=None, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--c_values', nargs='+', type=int, default=None)
    p.add_argument('--out', default=None, help='Optional: also write the table as CSV to this path')


def _add_plot_parser(sub):
    p = sub.add_parser('plot', help='Draw accuracy-vs-k figures from merged results')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one plot')
    p.add_argument('--algorithms', nargs='+', default=None, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--c_values', nargs='+', type=int, default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--group_by', default='paradigm', choices=['paradigm', 'all-in-one', 'per-algorithm'])
    p.add_argument('--rename', nargs='+', default=None,
                    help="Override legend/title labels, e.g. --rename PiCO-Fixed=PiCO ComCo-Fixed=ComCo "
                         "(data selection still uses the real algorithm name; only the shown text changes)")


def _add_detail_plot_parser(sub):
    p = sub.add_parser('detail-plot', help='Per-class accuracy/loss heatmap over epoch checkpoints '
                                            '(requires --detail during run); one or two algorithms side by side')
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg_l', required=True, help='Left (or only) algorithm')
    p.add_argument('--alg_r', default=None, help='Right algorithm, for a side-by-side comparison')
    p.add_argument('--alg_l_display', default=None, help='Override --alg_l\'s title text only')
    p.add_argument('--alg_r_display', default=None, help='Override --alg_r\'s title text only')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--acc_only', action='store_true', help='Omit the CE-loss row (accuracy heatmap only)')
    p.add_argument('--side_by_side', action='store_true',
                    help='Single algorithm only: accuracy (left) / CE loss (right) side by side, '
                         'instead of the default stacked (accuracy on top, loss below) layout')
    p.add_argument('--show_class_names', action='store_true')
    p.add_argument('--seed', type=int, default=42, help='Only used with --show_class_names')
    p.add_argument('--data_dir', default='./data', help='Only used with --show_class_names')


def _add_detail_plot_multi_parser(sub):
    p = sub.add_parser('detail-plot-multi',
                        help='Per-class accuracy/loss heatmap for N algorithms side by side '
                             '(requires --detail during run); generalizes detail-plot beyond two algorithms')
    p.add_argument('--entries', nargs='+', required=True,
                    help='RUN:ALG:LABEL triples, one per column, in order, e.g. '
                         '--entries new_main_c20_k19_pico:PiCO:PiCO thresholdoracle_c20_k19_t1:PiCO-Oracle:PiCO-oracle')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--acc_only', action='store_true', help='Omit the CE-loss row (accuracy heatmaps only)')
    p.add_argument('--show_class_names', action='store_true')
    p.add_argument('--seed', type=int, default=42, help='Only used with --show_class_names')
    p.add_argument('--data_dir', default='./data', help='Only used with --show_class_names')


def _add_detail_plot_pico_parser(sub):
    p = sub.add_parser('detail-plot-pico', help="PiCO's contrastive positive/negative pair-selection "
                                                 "precision vs. ground truth, over epochs (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg', default='PiCO', help="Which PiCO-family algorithm's log to read (default: PiCO)")
    p.add_argument('--display_name', default=None, help="Override --alg's title text only")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_pico_multik_parser(sub):
    p = sub.add_parser('detail-plot-pico-multik',
                        help="Overlay one algorithm's positive-pair selection precision across several "
                             "k values on one chart (requires --detail on all of them)")
    p.add_argument('--runs', nargs='+', required=True,
                    help='k=run_name pairs, e.g. --runs 19=new_main_c20_k19_pico_fixed 15=new_main_c20_k15_pico_fixed')
    p.add_argument('--alg', default='PiCO-Fixed', help="Which algorithm's log to read (default: PiCO-Fixed)")
    p.add_argument('--display_name', default=None, help="Override --alg's title text only")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_concentration_parser(sub):
    p = sub.add_parser('detail-plot-concentration',
                        help='Overlay per-algorithm prediction-concentration trends (mean entropy + '
                             'max-softmax-prob vs epoch, N algorithms) (requires `run --concentration`)')
    p.add_argument('--entries', nargs='+', required=True,
                    help='RUN:ALG:LABEL triples, one per line, in order, e.g. '
                         '--entries 0820_ablation:PiCO-Fixed:PiCO-Fixed '
                         '0820_ablation:PiCO-Fixed-UniformInit:UniformInit')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_knn_parser(sub):
    p = sub.add_parser('detail-plot-knn',
                        help='Bar chart comparing final kNN top-1 accuracy across N algorithms '
                             '(requires `run --knn_eval`)')
    p.add_argument('--entries', nargs='+', required=True,
                    help='RUN:ALG:LABEL triples, one per bar, in order')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_pico_oracle_parser(sub):
    p = sub.add_parser('detail-plot-pico-oracle',
                        help="PiCO-Oracle's graduated correction: natural (pre-correction) vs. "
                             "post-correction positive-pair precision, over epochs (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--hide_neg', action='store_true', help='Omit the negative-pair precision line')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    _add_run_parser(sub)
    _add_merge_parser(sub)
    _add_report_parser(sub)
    _add_plot_parser(sub)
    _add_detail_plot_parser(sub)
    _add_detail_plot_multi_parser(sub)
    _add_detail_plot_pico_parser(sub)
    _add_detail_plot_pico_multik_parser(sub)
    _add_detail_plot_pico_oracle_parser(sub)
    _add_detail_plot_concentration_parser(sub)
    _add_detail_plot_knn_parser(sub)
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
    elif args.command == 'report':
        import csv as csv_mod
        import statistics

        run_dirs = [os.path.join('results', r) for r in args.runs]
        by_seed = results_mod.load_results_by_seed(run_dirs)

        # Sort by k first, then by algorithm name (alphabetical) within each
        # k -- for the parametrized biased-sweep naming convention
        # (*-BiasedCand-W05/W06/W08/W10/W20, *-BiasedAll-W05/...), zero-padded
        # weight suffixes mean alphabetical order already groups each family
        # together in ascending true-class-weight order, so no separate
        # "family" key is needed.
        rows = []
        for C in sorted(by_seed):
            if args.c_values and C not in args.c_values:
                continue
            algs = sorted(a for a in by_seed[C] if not args.algorithms or a in args.algorithms)
            ks = sorted({k for alg in algs for k in by_seed[C][alg]})
            for k in ks:
                for alg in algs:
                    if k not in by_seed[C][alg]:
                        continue
                    accs = by_seed[C][alg][k]
                    mean = statistics.mean(accs)
                    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
                    rows.append({'C': C, 'k': k, 'algorithm': alg, 'n_seeds': len(accs),
                                 'accs': accs, 'mean': mean, 'std': std})

        if not rows:
            print(f'No results found in {run_dirs}' +
                  (f' matching --algorithms {args.algorithms}' if args.algorithms else '') +
                  (f' --c_values {args.c_values}' if args.c_values else ''))
        else:
            name_w = max(len(r['algorithm']) for r in rows)
            print(f"{'C':>4}{'k':>5}  {'algorithm':<{name_w}}  {'mean±std':>14}  {'n':>3}  per-seed")
            for r in rows:
                acc_str = ', '.join(f'{a:.2f}' for a in r['accs'])
                print(f"{r['C']:>4}{r['k']:>5}  {r['algorithm']:<{name_w}}  "
                      f"{r['mean']:>7.2f}±{r['std']:<5.2f}  {r['n_seeds']:>3}  [{acc_str}]")

        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
            with open(args.out, 'w', newline='') as f:
                w = csv_mod.DictWriter(f, fieldnames=['C', 'k', 'algorithm', 'n_seeds', 'mean', 'std', 'accs'])
                w.writeheader()
                for r in rows:
                    w.writerow({'C': r['C'], 'k': r['k'], 'algorithm': r['algorithm'],
                                'n_seeds': r['n_seeds'], 'mean': round(r['mean'], 4),
                                'std': round(r['std'], 4),
                                'accs': ';'.join(f'{a:.4f}' for a in r['accs'])})
            print(f'\nWrote -> {args.out}')
    elif args.command == 'plot':
        run_dirs = [os.path.join('results', r) for r in args.runs]
        rename = dict(pair.split('=', 1) for pair in args.rename) if args.rename else None
        plot_accuracy_vs_k(run_dirs, algorithms=args.algorithms, c_values=args.c_values,
                            out_path=args.out, group_by=args.group_by, rename=rename)
    elif args.command == 'detail-plot':
        class_names = None
        if args.show_class_names:
            from src.cifar100_subset import select_cifar100_classes
            from torchvision.datasets import CIFAR100
            indices = select_cifar100_classes(args.C, seed=args.seed)
            ds = CIFAR100(root=args.data_dir, train=True, download=False)
            class_names = [ds.classes[i] for i in indices]
        if args.side_by_side:
            if args.alg_r:
                parser.error('--side_by_side is single-algorithm only; drop --alg_r')
            from src.pipeline.detail import plot_heatmap_side_by_side
            plot_heatmap_side_by_side(os.path.join('results', args.run_name), args.alg_l, args.C, args.k, args.out,
                                       display_name=args.alg_l_display, class_names=class_names)
        else:
            from src.pipeline.detail import plot_heatmap
            plot_heatmap(os.path.join('results', args.run_name), args.alg_l, args.C, args.k, args.out,
                         alg_r=args.alg_r, acc_only=args.acc_only, class_names=class_names,
                         alg_l_display=args.alg_l_display, alg_r_display=args.alg_r_display)
    elif args.command == 'detail-plot-multi':
        class_names = None
        if args.show_class_names:
            from src.cifar100_subset import select_cifar100_classes
            from torchvision.datasets import CIFAR100
            indices = select_cifar100_classes(args.C, seed=args.seed)
            ds = CIFAR100(root=args.data_dir, train=True, download=False)
            class_names = [ds.classes[i] for i in indices]
        from src.pipeline.detail import plot_heatmap_multi
        entries = []
        for triple in args.entries:
            run_name, alg, label = triple.split(':', 2)
            entries.append((os.path.join('results', run_name), alg, label))
        plot_heatmap_multi(entries, args.C, args.k, args.out, acc_only=args.acc_only, class_names=class_names)
    elif args.command == 'detail-plot-pico':
        from src.pipeline.detail import plot_pico_selection_stats
        plot_pico_selection_stats(os.path.join('results', args.run_name), args.alg, args.C, args.k, args.out,
                                   display_name=args.display_name)
    elif args.command == 'detail-plot-pico-multik':
        from src.pipeline.detail import plot_pico_selection_stats_multi_k
        run_dirs = {}
        for pair in args.runs:
            k_str, run_name = pair.split('=', 1)
            run_dirs[int(k_str)] = os.path.join('results', run_name)
        plot_pico_selection_stats_multi_k(run_dirs, args.alg, args.C, args.out, display_name=args.display_name)
    elif args.command == 'detail-plot-pico-oracle':
        from src.pipeline.detail import plot_pico_oracle_correction_stats
        plot_pico_oracle_correction_stats(os.path.join('results', args.run_name), args.C, args.k, args.out,
                                           show_neg=not args.hide_neg)
    elif args.command == 'detail-plot-concentration':
        from src.pipeline.detail import plot_concentration_trend
        entries = []
        for triple in args.entries:
            run_name, alg, label = triple.split(':', 2)
            entries.append((os.path.join('results', run_name), alg, label))
        plot_concentration_trend(entries, args.C, args.k, args.out)
    elif args.command == 'detail-plot-knn':
        from src.pipeline.detail import plot_knn_eval_bar
        entries = []
        for triple in args.entries:
            run_name, alg, label = triple.split(':', 2)
            entries.append((os.path.join('results', run_name), alg, label))
        plot_knn_eval_bar(entries, args.C, args.k, args.out)


if __name__ == '__main__':
    main()
