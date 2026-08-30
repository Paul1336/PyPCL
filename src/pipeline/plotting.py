"""Single parameterized plotting tool — replaces the 23 bespoke plot_*.py scripts
that used to live in scripts/.

Reads merged results.csv from one or more run directories (results/<run_name>)
and draws accuracy-vs-k line charts, grouped by paradigm / algorithm / all-in-one.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.results import load_results, load_results_by_seed

STYLES = {
    'CLPL':     dict(color='#1f77b4', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'Wu2022':   dict(color='#17becf', marker='D', linestyle='--', linewidth=2, markersize=6),
    'PRODEN':   dict(color='#2ca02c', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'PiCO-Oracle': dict(color='#4b0082', marker='*', linestyle='-', linewidth=2, markersize=8),
    'PiCO-MCL': dict(color='#bcbd22', marker='p', linestyle=':',  linewidth=2, markersize=6),
    'PiCO-SC':  dict(color='#98df8a', marker='h', linestyle='--', linewidth=2, markersize=6),
    'PiCO-CLS': dict(color='#e377c2', marker='*', linestyle='-',  linewidth=2.5, markersize=8),
    'SoLar':    dict(color='#e8b13f', marker='*', linestyle='-',  linewidth=2.5, markersize=9),
    'MCL-LOG':  dict(color='#d62728', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'SCL-NL':   dict(color='#ff7f0e', marker='D', linestyle='--', linewidth=2, markersize=6),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'OP':       dict(color='#e377c2', marker='D', linestyle='-',  linewidth=2, markersize=6),
    'OP-W':     dict(color='#aa40fc', marker='P', linestyle='-',  linewidth=2, markersize=6),
    'CPE':      dict(color='#17becf', marker='s', linestyle='-',  linewidth=2, markersize=6),
}

PLL_ALGOS = ['CLPL', 'Wu2022', 'PRODEN', 'PiCO', 'PiCO-MCL', 'PiCO-SC', 'PiCO-CLS', 'SoLar']
CLL_ALGOS = ['MCL-LOG', 'SCL-NL', 'OP', 'OP-W', 'CPE', 'ComCo']


def _global_ymax(res: dict) -> int:
    vals = [acc for C_d in res.values() for alg_d in C_d.values() for acc in alg_d.values()]
    return 80 if not vals else int(np.ceil(max(vals) / 10) * 10)


def _draw(ax, alg: str, k_acc: dict, label: str = None):
    if not k_acc:
        return
    ks, accs = zip(*sorted(k_acc.items()))
    ax.plot(ks, accs, label=(label or alg), **STYLES.get(alg, {}))


def _setup_ax(ax, title: str, y_max: int, ylabel: bool = False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('k  (# candidate / complementary labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def plot_accuracy_vs_k(run_dirs: list, algorithms: list = None, c_values: list = None,
                        out_path: str = 'plots/comparison.png', group_by: str = 'paradigm',
                        rename: dict = None):
    """
    run_dirs:   list of results/<run_name> directories to merge.
    algorithms: algorithm names to plot; None = every algorithm found in the data.
    c_values:   C values to plot (one column per C); None = every C found in the data.
    group_by:   'paradigm'      -> two rows (PLL / CLL), one column per C
                'all-in-one'    -> one row, all algorithms together, one column per C
                'per-algorithm' -> one row per algorithm, one column per C
    rename:     optional {algorithm: display_label} overriding only the legend/title text
                (e.g. {'PiCO-Fixed': 'PiCO'} for a slide) -- data selection/lookup/line
                styling still use the real algorithm name, only the shown label changes.
    """
    rename = rename or {}
    res = load_results(run_dirs)
    if not res:
        raise ValueError(f'No results found in {run_dirs}')

    c_values = c_values or sorted(res.keys())
    ym = _global_ymax(res)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    if group_by == 'paradigm':
        pll = [a for a in (algorithms or PLL_ALGOS) if a in PLL_ALGOS]
        cll = [a for a in (algorithms or CLL_ALGOS) if a in CLL_ALGOS]
        rows = [(label, algos) for label, algos in [('PLL', pll), ('CLL', cll)] if algos]
        fig, axes = plt.subplots(len(rows), len(c_values),
                                  figsize=(8 * len(c_values), 5 * len(rows)), squeeze=False)
        fig.suptitle('PLL vs CLL', fontsize=13)
        for r, (label, algos) in enumerate(rows):
            for c_idx, C in enumerate(c_values):
                ax = axes[r][c_idx]
                _setup_ax(ax, f'{label}  —  C = {C}', ym, ylabel=(c_idx == 0))
                for alg in algos:
                    _draw(ax, alg, res.get(C, {}).get(alg, {}), label=rename.get(alg))
                ax.legend(fontsize=8, loc='best')

    elif group_by == 'per-algorithm':
        algos = algorithms or (PLL_ALGOS + CLL_ALGOS)
        fig, axes = plt.subplots(len(algos), len(c_values),
                                  figsize=(8 * len(c_values), 4 * len(algos)), squeeze=False)
        fig.suptitle('Per-algorithm accuracy vs k', fontsize=13)
        for r, alg in enumerate(algos):
            for c_idx, C in enumerate(c_values):
                ax = axes[r][c_idx]
                _setup_ax(ax, f'{rename.get(alg, alg)}  —  C = {C}', ym, ylabel=(c_idx == 0))
                _draw(ax, alg, res.get(C, {}).get(alg, {}), label=rename.get(alg))

    else:  # 'all-in-one'
        algos = algorithms or (PLL_ALGOS + CLL_ALGOS)
        fig, axes = plt.subplots(1, len(c_values), figsize=(8 * len(c_values), 5), squeeze=False)
        fig.suptitle('All methods', fontsize=13)
        for c_idx, C in enumerate(c_values):
            ax = axes[0][c_idx]
            _setup_ax(ax, f'C = {C}', ym, ylabel=(c_idx == 0))
            for alg in algos:
                _draw(ax, alg, res.get(C, {}).get(alg, {}), label=rename.get(alg))
            ax.legend(fontsize=8, loc='best', ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] -> {out_path}', flush=True)
    return out_path


def plot_accuracy_vs_weight(run_dirs: list, C: int, k: int, series: list, weights: list,
                             baselines: list = None, out_path: str = 'plots/comparison.png',
                             title: str = None, zero_pad: bool = True, x_prefix: str = 'W',
                             xlabel: str = None) -> str:
    """Accuracy vs. a swept init-parameter (true-class weight W, or the
    BiasedRand family's Wf), for one fixed (C, k) point -- the complementary
    view to plot_accuracy_vs_k's accuracy-vs-k-for-fixed-parameter. Plotted
    at CATEGORICAL, evenly-spaced x positions in the given `weights` order --
    not spaced proportionally to the numeric values -- since the sweep's own
    values (e.g. 5, 6, 8, 10, 20 for W; 5, 8, 10, 12, 15 for Wf) aren't meant
    to be read on a linear scale.

    series: list of (name_template, label) pairs. name_template must contain
    '{w}', filled in with each value from `weights` formatted per `zero_pad`
    (True: src.pll_init.weight_pct_str -- the exact same percentage-string
    logic biased_variant_name uses to build the real algorithm names, e.g.
    'PiCO-Fixed-BiasedCand-W{w}' -> '...-W05' for w=5, '...-W4.5' for
    w=4.5; False: plain int, matching biased_rand_variant_name's 'Wf{wf}'
    convention -- no padding -- e.g. 'PiCO-Fixed-BiasedRand-W10-Wf{w}' ->
    '...-Wf5' for w=5) -- one line per entry, plotted across every value in
    `weights` in order, with +-1 std error bars across seeds (see
    results.load_results_by_seed).

    weights: list of numbers (e.g. [5, 6, 8, 10] or [4.5, 5.2, 6.6, 8.3] for
    W, or [5, 8, 10, 12, 15] for Wf) -- x-axis category order; tick labels
    are '{x_prefix}{n}'. Fractional W values are only meaningful with
    zero_pad=True (see below) -- Wf is always a plain integer count.

    baselines: optional list of (algorithm, label) pairs, each drawn as a
    horizontal dashed reference line spanning the whole plot -- for
    algorithms that don't vary with the swept parameter (e.g. the unbiased
    PiCO-Fixed / PRODEN / ComCo-Fixed baselines). Pull these from a
    DIFFERENT run_dir than the swept series if the sweep's own run_name
    didn't include the baselines (run_dirs accepts more than one and merges
    them, same as plot_accuracy_vs_k) -- baseline accuracy doesn't depend on
    which sweep trained it, only on (C, k, algorithm, seed)."""
    import statistics

    from src.pll_init import weight_pct_str

    def _fmt_num(w):
        return str(int(w)) if float(w).is_integer() else str(w)

    by_seed = load_results_by_seed(run_dirs)
    xs = list(range(len(weights)))
    xticklabels = [f'{x_prefix}{_fmt_num(w)}' for w in weights]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap('tab10')

    for i, (name_template, label) in enumerate(series):
        means, stds = [], []
        for w in weights:
            tag = weight_pct_str(w / 100) if zero_pad else _fmt_num(w)
            alg = name_template.format(w=tag)
            accs = by_seed.get(C, {}).get(alg, {}).get(k, [])
            if accs:
                means.append(statistics.mean(accs))
                stds.append(statistics.stdev(accs) if len(accs) > 1 else 0.0)
            else:
                print(f'  [plot_accuracy_vs_weight] no data for {alg} at C={C} k={k}', flush=True)
                means.append(float('nan'))
                stds.append(0.0)
        ax.errorbar(xs, means, yerr=stds, label=label, color=cmap(i % 10), marker='o',
                    linewidth=2, capsize=4, markersize=6)

    if baselines:
        base_cmap = plt.get_cmap('Set2')
        for i, (alg, label) in enumerate(baselines):
            accs = by_seed.get(C, {}).get(alg, {}).get(k, [])
            if not accs:
                print(f'  [plot_accuracy_vs_weight] no data for baseline {alg} at C={C} k={k}', flush=True)
                continue
            mean = statistics.mean(accs)
            ax.axhline(mean, color=base_cmap(i), linestyle='--', linewidth=1.5,
                       label=f'{label} (baseline)')

    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel or f'{x_prefix} (swept parameter)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(title or f'Accuracy vs. {x_prefix}  —  C={C}  k={k}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] -> {out_path}', flush=True)
    return out_path
