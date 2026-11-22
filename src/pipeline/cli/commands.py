"""Subcommand implementations for scripts/run_pipeline.py (split out of a 230-line if/elif main,
2026-11-22). COMMANDS maps the CLI name to its function(args)."""

import argparse  # noqa: F401
import os
import sys
from src.pipeline.algorithms import ALL_ALGORITHM_NAMES
from src.pipeline.config import load_config
from src.pipeline import results as results_mod
from src.pipeline.plotting import plot_accuracy_vs_k, plot_accuracy_vs_weight
from src.pipeline.datasets import ALL_DATASET_NAMES


def _error(msg):
    """argparse-style usage error (exit code 2)."""
    print(f'run_pipeline.py: error: {msg}', file=sys.stderr)
    raise SystemExit(2)


def cmd_run(args):
    if getattr(args, 'only_q', None) is not None and args.only_k is not None:
        _error('--only_q and --only_k are mutually exclusive')
    if getattr(args, 'only_q', None) is not None and not (0 <= args.only_q <= 1):
        _error('--only_q must be in [0, 1]')
    from src.pipeline.runner import run
    run(load_config(args))


def cmd_merge(args):
    out = results_mod.merge_shards(os.path.join('results', args.run_name))
    print(f'Merged -> {out}')


def cmd_report(args):
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
            _error(f"--sort_by {args.sort_by} requires --confusion")

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


def cmd_plot(args):
    run_dirs = [os.path.join('results', r) for r in args.runs]
    rename = dict(pair.split('=', 1) for pair in args.rename) if args.rename else None
    plot_accuracy_vs_k(run_dirs, algorithms=args.algorithms, c_values=args.c_values,
                        out_path=args.out, group_by=args.group_by, rename=rename)


def cmd_plot_weight(args):
    run_dirs = [os.path.join('results', r) for r in args.runs]
    series = [tuple(s.split(':', 1)) for s in args.series]
    baselines = [tuple(b.split(':', 1)) for b in args.baselines] if args.baselines else None
    plot_accuracy_vs_weight(run_dirs, args.C, args.k, series, args.weights,
                             baselines=baselines, out_path=args.out, title=args.title,
                             zero_pad=not args.no_zero_pad, x_prefix=args.x_prefix, xlabel=args.xlabel)


def cmd_detail_plot(args):
    class_names = None
    if args.show_class_names:
        from src.cifar100_subset import select_cifar100_classes
        from torchvision.datasets import CIFAR100
        indices = select_cifar100_classes(args.C, seed=args.seed)
        ds = CIFAR100(root=args.data_dir, train=True, download=False)
        class_names = [ds.classes[i] for i in indices]
    if args.side_by_side:
        if args.alg_r:
            _error('--side_by_side is single-algorithm only; drop --alg_r')
        from src.pipeline.detail import plot_heatmap_side_by_side
        plot_heatmap_side_by_side(os.path.join('results', args.run_name), args.alg_l, args.C, args.k, args.out,
                                   display_name=args.alg_l_display, class_names=class_names)
    else:
        from src.pipeline.detail import plot_heatmap
        plot_heatmap(os.path.join('results', args.run_name), args.alg_l, args.C, args.k, args.out,
                     alg_r=args.alg_r, acc_only=args.acc_only, class_names=class_names,
                     alg_l_display=args.alg_l_display, alg_r_display=args.alg_r_display)


def cmd_detail_plot_multi(args):
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


def cmd_detail_plot_pico(args):
    from src.pipeline.detail import plot_pico_selection_stats
    plot_pico_selection_stats(os.path.join('results', args.run_name), args.alg, args.C, args.k, args.out,
                               display_name=args.display_name)


def cmd_detail_plot_pico_confusion(args):
    from src.pipeline.detail import plot_pico_confusion_stats
    plot_pico_confusion_stats(os.path.join('results', args.run_name), args.alg, args.C, args.k, args.out,
                               display_name=args.display_name)


def cmd_detail_plot_pico_confusion_all(args):
    from src.pipeline.detail import plot_pico_confusion_stats_all
    written = plot_pico_confusion_stats_all(
        os.path.join('results', args.run_name), args.out_dir,
        algorithms=args.algorithms, c_values=args.c_values, k_values=args.k_values)
    print(f'\nWrote {len(written)} plot(s) -> {args.out_dir}')


def cmd_detail_plot_pico_multik(args):
    from src.pipeline.detail import plot_pico_selection_stats_multi_k
    run_dirs = {}
    for pair in args.runs:
        k_str, run_name = pair.split('=', 1)
        run_dirs[int(k_str)] = os.path.join('results', run_name)
    plot_pico_selection_stats_multi_k(run_dirs, args.alg, args.C, args.out, display_name=args.display_name)


def cmd_detail_plot_pico_oracle(args):
    from src.pipeline.detail import plot_pico_oracle_correction_stats
    plot_pico_oracle_correction_stats(os.path.join('results', args.run_name), args.C, args.k, args.out,
                                       show_neg=not args.hide_neg)


def cmd_detail_plot_concentration(args):
    from src.pipeline.detail import plot_concentration_trend
    entries = []
    for triple in args.entries:
        run_name, alg, label = triple.split(':', 2)
        entries.append((os.path.join('results', run_name), alg, label))
    plot_concentration_trend(entries, args.C, args.k, args.out)


def cmd_detail_plot_knn(args):
    from src.pipeline.detail import plot_knn_eval_bar
    entries = []
    for triple in args.entries:
        run_name, alg, label = triple.split(':', 2)
        entries.append((os.path.join('results', run_name), alg, label))
    plot_knn_eval_bar(entries, args.C, args.k, args.out)


COMMANDS = {
    'run': cmd_run,
    'merge': cmd_merge,
    'report': cmd_report,
    'plot': cmd_plot,
    'plot-weight': cmd_plot_weight,
    'detail-plot': cmd_detail_plot,
    'detail-plot-multi': cmd_detail_plot_multi,
    'detail-plot-pico': cmd_detail_plot_pico,
    'detail-plot-pico-confusion': cmd_detail_plot_pico_confusion,
    'detail-plot-pico-confusion-all': cmd_detail_plot_pico_confusion_all,
    'detail-plot-pico-multik': cmd_detail_plot_pico_multik,
    'detail-plot-pico-oracle': cmd_detail_plot_pico_oracle,
    'detail-plot-concentration': cmd_detail_plot_concentration,
    'detail-plot-knn': cmd_detail_plot_knn,
}
