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
from src.pipeline.plotting import plot_accuracy_vs_k, plot_accuracy_vs_weight
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
                         'pair-selection precision (plus the raw tp/fp/tn/fn pair counts behind it) '
                         'against ground truth to pico_selection_stats.csv '
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
    p.add_argument('--confusion', action='store_true',
                    help="Also report each cell's FINAL-epoch PiCO contrastive pair-selection confusion "
                         "matrix (tp/fp/tn/fn, summed over that epoch's batches, plus tp/(tp+fp) as "
                         "'precision_pct' -- requires `run --detail`). Unlike accuracy, this is NOT "
                         "averaged across seeds: --detail only logs for one seed per cell (the "
                         "diagnostics_seed), so these columns reflect that seed alone. Blank for any "
                         "(C, k, algorithm) with no pico_selection_stats.csv (non-PiCO algorithms, or a "
                         "log that never reached prot_start).")
    p.add_argument('--sort_by', default='default',
                    choices=['default', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision'],
                    help="Row order. 'default': C, then k, then algorithm name (alphabetical) -- the "
                         "original grouping. 'tp'/'fp'/'tn'/'fn': descending by that --confusion raw "
                         "count (requires --confusion; rows with no confusion data sort last). "
                         "'accuracy': descending by mean accuracy. 'precision': descending by "
                         "tp/(tp+fp) (requires --confusion).")


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


def _add_plot_weight_parser(sub):
    p = sub.add_parser('plot-weight',
                        help='Accuracy vs. true-class weight (W), for one fixed (C, k) -- the parametrized '
                             'biased-init sweep (PiCO-Fixed-Biased{Cand,All}-W* / PRODEN-Biased{Cand,All}-W*) '
                             'complement to `plot`\'s accuracy-vs-k view')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one plot')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--weights', nargs='+', type=float, required=True,
                    help='True-class weight values (percentages), x-axis category order, e.g. '
                         '--weights 5 6 8 10 20, or fractional --weights 4.5 5.2 6.6 8.3 10 20')
    p.add_argument('--series', nargs='+', required=True,
                    help="TEMPLATE:LABEL pairs, one line per entry, e.g. "
                         "--series 'PiCO-Fixed-BiasedCand-W{w}:PiCO-BiasedCand' "
                         "'PiCO-Fixed-BiasedAll-W{w}:PiCO-BiasedAll' -- TEMPLATE must contain literal '{w}', "
                         "filled with each --weights value as a 2-digit zero-padded tag.")
    p.add_argument('--baselines', nargs='+', default=None,
                    help='ALGORITHM:LABEL pairs drawn as horizontal dashed reference lines, e.g. '
                         '--baselines PiCO-Fixed:PiCO PRODEN:PRODEN ComCo-Fixed:ComCo -- pulled from the '
                         'same --runs, so pass a run that actually has them if the swept sweep\'s own '
                         'run_name doesn\'t (e.g. --runs biased_rand_sweep biased_weight_sweep)')
    p.add_argument('--no_zero_pad', action='store_true',
                    help="Format {w} as a plain int instead of a 2-digit zero-padded tag -- use this for "
                         "the BiasedRand family's Wf (e.g. --series "
                         "'PiCO-Fixed-BiasedRand-W10-Wf{w}:PiCO W10' --weights 5 8 10 12 15 --no_zero_pad), "
                         "since biased_rand_variant_name's 'Wf{wf}' suffix isn't zero-padded (unlike W).")
    p.add_argument('--x_prefix', default='W', help="Tick-label / axis-label prefix, e.g. 'Wf' for the "
                                                     "BiasedRand family's Wf sweep instead of W.")
    p.add_argument('--xlabel', default=None, help='Override the x-axis label text.')
    p.add_argument('--title', default=None)
    p.add_argument('--out', required=True)


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


def _add_detail_plot_pico_confusion_parser(sub):
    p = sub.add_parser('detail-plot-pico-confusion',
                        help="PiCO's contrastive pair-selection confusion matrix (TP/TN/FP/FN, as a "
                             "% of within-batch pairs, with raw counts annotated) vs. training step, "
                             "for one (algorithm, C, k) (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg', default='PiCO-Fixed', help="Which PiCO-family algorithm's log to read "
                                                         "(default: PiCO-Fixed)")
    p.add_argument('--display_name', default=None, help="Override --alg's title text only")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_pico_confusion_all_parser(sub):
    p = sub.add_parser('detail-plot-pico-confusion-all',
                        help='Batch version of detail-plot-pico-confusion: one PNG per (algorithm, C, k) '
                             'combination found under results/<run>/detail/, all written into one folder '
                             '(requires --detail)')
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--out_dir', required=True, help='Folder to write every PNG into (created if missing)')
    p.add_argument('--algorithms', nargs='+', default=None,
                    help='Restrict to these algorithms only (default: every algorithm found under detail/)')
    p.add_argument('--c_values', nargs='+', type=int, default=None,
                    help='Restrict to these C values only (default: every C found)')
    p.add_argument('--k_values', nargs='+', type=int, default=None,
                    help='Restrict to these k values only (default: every k found)')


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
    _add_plot_weight_parser(sub)
    _add_detail_plot_parser(sub)
    _add_detail_plot_multi_parser(sub)
    _add_detail_plot_pico_parser(sub)
    _add_detail_plot_pico_confusion_parser(sub)
    _add_detail_plot_pico_confusion_all_parser(sub)
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

        confusion_fn = None
        if args.confusion:
            from src.pipeline.detail import final_epoch_pico_confusion

            def confusion_fn(alg, C, k):
                # First run_dir that actually has a log wins -- --detail
                # output isn't seed-scoped, so at most one run_dir in a
                # --runs sweep would ever have this cell's log anyway.
                for rd in run_dirs:
                    conf = final_epoch_pico_confusion(rd, alg, C, k)
                    if conf is not None:
                        return conf
                return None

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
                    row = {'C': C, 'k': k, 'algorithm': alg, 'n_seeds': len(accs),
                           'accs': accs, 'mean': mean, 'std': std}
                    if confusion_fn is not None:
                        row['confusion'] = confusion_fn(alg, C, k)
                    rows.append(row)

        if args.sort_by != 'default':
            if args.sort_by in ('tp', 'fp', 'tn', 'fn', 'precision') and confusion_fn is None:
                parser.error(f"--sort_by {args.sort_by} requires --confusion")

            def _sort_key(r):
                if args.sort_by == 'accuracy':
                    return r['mean']
                c = r.get('confusion')
                field = 'pos_precision' if args.sort_by == 'precision' else args.sort_by
                v = (c or {}).get(field, -1)
                return -1 if v != v else v   # v != v catches NaN (e.g. tp+fp == 0)

            rows.sort(key=_sort_key, reverse=True)

        if not rows:
            print(f'No results found in {run_dirs}' +
                  (f' matching --algorithms {args.algorithms}' if args.algorithms else '') +
                  (f' --c_values {args.c_values}' if args.c_values else ''))
        else:
            name_w = max(len(r['algorithm']) for r in rows)
            header = f"{'C':>4}{'k':>5}  {'algorithm':<{name_w}}  {'mean±std':>14}  {'n':>3}"
            if confusion_fn is not None:
                header += f"  {'TP':>14}{'FP':>14}{'TN':>14}{'FN':>14}  {'TP/(TP+FP)':>10}  {'ep':>4}"
            header += '  per-seed'
            print(header)
            for r in rows:
                acc_str = ', '.join(f'{a:.2f}' for a in r['accs'])
                line = (f"{r['C']:>4}{r['k']:>5}  {r['algorithm']:<{name_w}}  "
                        f"{r['mean']:>7.2f}±{r['std']:<5.2f}  {r['n_seeds']:>3}")
                if confusion_fn is not None:
                    c = r['confusion']
                    if c is None:
                        line += f"  {'--':>14}{'--':>14}{'--':>14}{'--':>14}  {'--':>10}  {'--':>4}"
                    else:
                        def _cell(count, pct):
                            return f"{count:>6} ({pct:4.1f}%)"
                        prec = c['pos_precision']
                        prec_str = '--' if prec != prec else f"{prec * 100:.2f}%"
                        line += (f"  {_cell(c['tp'], c['tp_pct']):>14}{_cell(c['fp'], c['fp_pct']):>14}"
                                  f"{_cell(c['tn'], c['tn_pct']):>14}{_cell(c['fn'], c['fn_pct']):>14}"
                                  f"  {prec_str:>10}  {c['epoch']:>4}")
                line += f"  [{acc_str}]"
                print(line)

        if args.out:
            fields = ['C', 'k', 'algorithm', 'n_seeds', 'mean', 'std', 'accs']
            if confusion_fn is not None:
                fields += ['confusion_epoch', 'tp', 'tp_pct', 'fp', 'fp_pct',
                           'tn', 'tn_pct', 'fn', 'fn_pct', 'precision_pct']
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
            with open(args.out, 'w', newline='') as f:
                w = csv_mod.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    out_row = {'C': r['C'], 'k': r['k'], 'algorithm': r['algorithm'],
                               'n_seeds': r['n_seeds'], 'mean': round(r['mean'], 4),
                               'std': round(r['std'], 4),
                               'accs': ';'.join(f'{a:.4f}' for a in r['accs'])}
                    if confusion_fn is not None:
                        c = r['confusion']
                        prec = c['pos_precision'] if c else float('nan')
                        out_row.update({
                            'confusion_epoch': c['epoch'] if c else '',
                            'tp': c['tp'] if c else '', 'tp_pct': round(c['tp_pct'], 2) if c else '',
                            'fp': c['fp'] if c else '', 'fp_pct': round(c['fp_pct'], 2) if c else '',
                            'tn': c['tn'] if c else '', 'tn_pct': round(c['tn_pct'], 2) if c else '',
                            'fn': c['fn'] if c else '', 'fn_pct': round(c['fn_pct'], 2) if c else '',
                            'precision_pct': '' if prec != prec else round(prec * 100, 2),
                        })
                    w.writerow(out_row)
            print(f'\nWrote -> {args.out}')
    elif args.command == 'plot':
        run_dirs = [os.path.join('results', r) for r in args.runs]
        rename = dict(pair.split('=', 1) for pair in args.rename) if args.rename else None
        plot_accuracy_vs_k(run_dirs, algorithms=args.algorithms, c_values=args.c_values,
                            out_path=args.out, group_by=args.group_by, rename=rename)
    elif args.command == 'plot-weight':
        run_dirs = [os.path.join('results', r) for r in args.runs]
        series = [tuple(s.split(':', 1)) for s in args.series]
        baselines = [tuple(b.split(':', 1)) for b in args.baselines] if args.baselines else None
        plot_accuracy_vs_weight(run_dirs, args.C, args.k, series, args.weights,
                                 baselines=baselines, out_path=args.out, title=args.title,
                                 zero_pad=not args.no_zero_pad, x_prefix=args.x_prefix, xlabel=args.xlabel)
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
    elif args.command == 'detail-plot-pico-confusion':
        from src.pipeline.detail import plot_pico_confusion_stats
        plot_pico_confusion_stats(os.path.join('results', args.run_name), args.alg, args.C, args.k, args.out,
                                   display_name=args.display_name)
    elif args.command == 'detail-plot-pico-confusion-all':
        from src.pipeline.detail import plot_pico_confusion_stats_all
        written = plot_pico_confusion_stats_all(
            os.path.join('results', args.run_name), args.out_dir,
            algorithms=args.algorithms, c_values=args.c_values, k_values=args.k_values)
        print(f'\nWrote {len(written)} plot(s) -> {args.out_dir}')
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
