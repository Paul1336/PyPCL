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

from src.pipeline.results import load_results

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
