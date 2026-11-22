"""Analysis report for experiments/final_1122.yaml (PiCO vs ComCo (MCL-NL) vs PRODEN, C=20/100 uniform).

    python scripts/report_final_1122.py            # -> docs/final_1122_report.md + plots/final_1122/*.png

Works on partial results (reports n per cell). Every number comes from src/analysis:
io (loading), stats (mean±std, paired tests), traces (epoch windows / final epoch).
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis import io, stats, traces  # noqa: E402

RUNS = os.environ.get('REPORT_RUNS', 'final_c20_uniform_fixk_1122,final_c100_uniform_fixk_1122').split(',')
ALGS = {'PCD-PiCO': 'PiCO', 'PCD-ComCoMCLNL': 'ComCo (MCL-NL)', 'PRODEN': 'PRODEN'}
ORDER = list(ALGS.values())
# reference categorical palette, slots 1-3 (validated all-pairs, light surface); text never in series color
COLORS = {'PiCO': '#2a78d6', 'ComCo (MCL-NL)': '#eb6834', 'PRODEN': '#1baf7a'}
MARKERS = {'PiCO': 'o', 'ComCo (MCL-NL)': 's', 'PRODEN': '^'}
SURFACE, GRID, INK, INK2 = '#fcfcfb', '#e1e0d9', '#0b0b0b', '#52514e'
OUT_MD = os.environ.get('REPORT_MD', 'docs/final_1122_report.md')
PLOTS = os.environ.get('REPORT_PLOTS', 'plots/final_1122')
LATE = 101          # ComCo positives start at warmup_pos = 100 -> "late training" window = epochs 101..200

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10, 'legend.frameon': False})


def name(df):
    if df.empty or 'algorithm' not in df:
        return pd.DataFrame(columns=['algorithm', 'method', 'run', 'C', 'k', 'seed', 'epoch'])
    df = df[df['algorithm'].isin(ALGS)].copy()
    df['method'] = df['algorithm'].map(ALGS)
    return df


def md_table(df, index, columns, value, digits=3, scale=1.0):
    """mean±std (n) table, methods as rows."""
    if df.empty:
        return '_no data yet_\n'
    d = df.copy()
    d[value] = d[value] * scale
    tab = stats.pivot_fmt(d, index=index, columns=columns, value=value, digits=digits, with_row_mean=True)
    tab = tab.reindex([m for m in ORDER if m in tab.index])
    means = stats.summarize(d, by=(index, columns), value=value).pivot(index=index, columns=columns, values='mean')
    return stats.to_markdown(tab, bold_best=means.reindex(tab.index)) + '\n'


def end_labels(ax, items):
    """Direct labels at line ends, nudged apart vertically so close values never overlap.
    items: [(x, y, text)]."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for x, y, t in items:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(t, (x, y), xytext=(x, yy), textcoords='data', fontsize=8, color=INK2, va='center',
                    ha='left', annotation_clip=False)
    # shift text right of the last point
    for txt in ax.texts[-len(items):]:
        txt.set_position((txt.get_position()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), txt.get_position()[1]))


def small_multiples(df, value, ylabel, fname, title, err=True, scale=1.0):
    """One panel per C, x = k, one line per method (mean, ±std band via error bars), direct labels."""
    Cs = sorted(df['C'].unique()) if not df.empty else []
    if not Cs:
        return None
    fig, axes = plt.subplots(1, len(Cs), figsize=(5.2 * len(Cs), 3.6), squeeze=False)
    s = stats.summarize(df, by=('method', 'C', 'k'), value=value)
    for ax, C in zip(axes[0], Cs):
        ends = []
        for m in ORDER:
            d = s[(s['method'] == m) & (s['C'] == C)].sort_values('k')
            if d.empty:
                continue
            y, e = d['mean'] * scale, d['std'].fillna(0) * scale
            ax.errorbar(d['k'], y, yerr=e if err else None, color=COLORS[m], marker=MARKERS[m], markersize=6,
                        linewidth=2, capsize=3, label=m, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends.append((d['k'].iloc[-1], y.iloc[-1], m))
        ax.set_title(f'C = {C}', color=INK, fontsize=10)
        ax.set_xlabel('candidate-set size k')
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(d_k for d_k in df[df['C'] == C]['k'].unique()))
        ax.margins(x=0.18)
        end_labels(ax, ends)
    axes[0][0].legend(loc='best', fontsize=8)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def curve_panels(df, value, ylabel, fname, title, ks):
    """Learning curves of one column at one k per C (mean over seeds)."""
    if df.empty:
        return None
    Cs = [C for C in ks if C in set(df['C'])]
    fig, axes = plt.subplots(1, len(Cs), figsize=(5.2 * len(Cs), 3.4), squeeze=False)
    for ax, C in zip(axes[0], Cs):
        d = df[(df['C'] == C) & (df['k'] == ks[C])].dropna(subset=[value])
        ends = []
        for m in ORDER:
            g = d[d['method'] == m].groupby('epoch')[value].mean()
            if g.empty:
                continue
            ax.plot(g.index, g.values, color=COLORS[m], linewidth=2, label=m)
            ends.append((g.index[-1], g.values[-1], m))
        ax.set_title(f'C = {C}, k = {ks[C]}', color=INK, fontsize=10)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.margins(x=0.22)
        end_labels(ax, ends)
    axes[0][0].legend(loc='best', fontsize=8)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    path = os.path.join(PLOTS, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def df_md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines)


def img(path, alt):
    if not path:
        return ''
    rel = os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(OUT_MD)))
    return f'![{alt}]({rel})\n'


def main():
    out = ['# Final comparison (post-refactor): PiCO vs ComCo (MCL-NL) vs PRODEN\n',
           'Generated by `scripts/report_final_1122.py` from `results/final_c{20,100}_uniform_fixk_1122/` '
           '(spec `experiments/final_1122.yaml`). Uniform CIFAR-100 subsets, fixed |Y| = k, 200 epochs, seeds 42-44. '
           'Cells show mean±std over seeds (n shown when < 3); bold = best method per column; `avg` = mean over k '
           '(`*` = some cell incomplete).\n',
           '| method | registry name | what it is |\n|---|---|---|\n'
           '| PiCO | PCD-PiCO | PiCO-Fixed (paper warm-up, candidate-masked init) + per-epoch diagnostics |\n'
           '| ComCo (MCL-NL) | PCD-ComCoMCLNL | ComCo framework with MCL-NL cls loss (ComCo Sec. 5.4.2) + diagnostics |\n'
           '| PRODEN | PRODEN | Lv et al. 2020, SGD 0.01, no contrastive branch |\n']

    # ── 1. accuracy ─────────────────────────────────────────────────────────
    res = name(io.load_results(RUNS))
    res['acc'] = res['final_accuracy']
    done = res.groupby(['method', 'C']).size().unstack(fill_value=0) if not res.empty else pd.DataFrame()
    out.append(f'\n**Progress:** {len(res)} / 90 cells finished.\n')
    out.append('\n## 1. Final test accuracy (%)\n')
    for C in (20, 100):
        out.append(f'\n**C = {C}**\n\n' + md_table(res[res['C'] == C], 'method', 'k', 'acc', digits=1))
    out.append(img(small_multiples(res, 'acc', 'final test accuracy (%)', 'acc_vs_k.png',
                                   'Final test accuracy vs k (mean ± std over seeds)'), 'accuracy vs k'))

    # paired comparisons (same C, k, seed)
    rows = []
    for C in (20, 100):
        sub = res[res['C'] == C]
        for a, b in (('PiCO', 'ComCo (MCL-NL)'), ('PiCO', 'PRODEN'), ('ComCo (MCL-NL)', 'PRODEN')):
            pc = stats.paired_compare(sub, a, b, on=('k', 'seed'), value='acc', alg_col='method')
            if pc['n_pairs'] == 0:
                continue
            rows.append({'C': C, 'A vs B': f'{a} vs {b}', 'pairs': pc['n_pairs'], 'mean(A-B)': round(pc['mean_diff'], 2),
                         '95% CI': ('–' if 'ci95' not in pc else f"[{pc['ci95'][0]:.2f}, {pc['ci95'][1]:.2f}]"),
                         'A wins': f"{pc['win_rate']:.2f}",
                         'p (paired t)': ('–' if 'p_ttest' not in pc else f"{pc['p_ttest']:.3g}")})
    if rows:
        out.append('\n**Paired comparison** (pairs = same k and seed; positive = A higher):\n\n'
                   + df_md(pd.DataFrame(rows)) + '\n')

    # ── 2. learning curves ──────────────────────────────────────────────────
    pcd = name(io.load_pcdiag(RUNS))
    per_class = name(io.load_detail_csv(RUNS, 'per_class_loss.csv', dedup_on=['epoch']))
    curves = []
    if not pcd.empty:
        curves.append(pcd[['method', 'C', 'k', 'seed', 'epoch', 'test_acc']])
    if not per_class.empty:
        pc_ = per_class[per_class['method'] == 'PRODEN'][['method', 'C', 'k', 'seed', 'epoch', 'overall_acc']]
        curves.append(pc_.rename(columns={'overall_acc': 'test_acc'}))
    cur = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    out.append('\n## 2. Learning curves\n')
    out.append(img(curve_panels(cur, 'test_acc', 'test accuracy (%)', 'curves_test_acc.png',
                                'Test accuracy during training (mean over seeds)', {20: 10, 100: 50}), 'curves'))

    # ── 3. label side: disambiguation / confidence / concentration (all 3 methods) ──
    lab = name(io.load_detail_csv(RUNS, 'label_diag.csv', dedup_on=['epoch']))
    out.append('\n## 3. Label-side diagnostics on the training set (final epoch)\n')
    out.append('From `label_diag.csv` (src/analysis/labels): `disamb_acc` = argmax of p restricted to Y is the true '
               'label; `cand_mass` = probability mass inside Y; `cand_entropy` = entropy of p within Y / log|Y|; '
               '`p_true`; `conf_acc` = argmax of the confidence buffer (PiCO / PRODEN only); `mean_max_prob` = '
               'prediction concentration.\n')
    if not lab.empty:
        last = lab.sort_values('epoch').groupby(['run', 'method', 'C', 'k', 'seed']).tail(1)
        for col, lbl, dg, sc in (('disamb_acc', 'disambiguation accuracy (%)', 1, 100), ('cand_mass', 'candidate mass', 3, 1),
                                 ('cand_entropy', 'normalised candidate entropy', 3, 1), ('p_true', 'p(true label)', 3, 1),
                                 ('conf_acc', 'confidence-buffer accuracy (%)', 1, 100), ('mean_max_prob', 'mean max prob', 3, 1)):
            d = last.dropna(subset=[col])
            if d.empty:
                continue
            out.append(f'\n**{col}**\n\n')
            for C in (20, 100):
                if (d['C'] == C).any():
                    out.append(f'C = {C}\n\n' + md_table(d[d['C'] == C], 'method', 'k', col, digits=dg, scale=sc) + '\n')
        out.append(img(small_multiples(last, 'disamb_acc', 'disambiguation accuracy', 'disamb_vs_k.png',
                                       'Train-set disambiguation accuracy at the final epoch'), 'disambiguation'))
        out.append(img(small_multiples(last, 'cand_entropy', 'entropy within Y / log|Y|', 'cand_entropy_vs_k.png',
                                       'How spread the prediction stays inside the candidate set (final epoch)'),
                       'candidate entropy'))
        out.append(img(curve_panels(lab, 'disamb_acc', 'disambiguation accuracy', 'curves_disamb.png',
                                    'Disambiguation accuracy during training', {20: 10, 100: 50}), 'disamb curves'))

    # ── 4. contrastive pair selection (PiCO, ComCo) ─────────────────────────
    out.append('\n## 4. Contrastive pair selection (PiCO, ComCo)\n')
    out.append(f'Mean over epochs {LATE}-200 (ComCo selects neighbour positives only from warmup_pos = 100), from '
               'the pcdiag CSV (src/analysis/pairs, full pool, self excluded). `pos_nonkey_prec` = precision of the '
               'positives other than the own augmented key; `pos_recall` = share of same-class pool entries selected; '
               '`neg_fn_rate` = negatives that are actually same class; `pos_viol_rate` = positives with disjoint '
               'candidate sets.\n')
    if not pcd.empty:
        cols = [c for c in ('pos_nonkey_prec', 'pos_prec', 'pos_recall', 'pos_per_anchor', 'neg_fn_rate',
                            'neg_certified_rate', 'pos_viol_rate', 'pl_acc', 'pl_in_comp', 'cstar_hit') if c in pcd]
        late = traces.window_mean(pcd, cols, start=LATE, by=('method', 'C', 'k'), per_seed=True)
        for col in cols:
            d = late.dropna(subset=[col])
            if d.empty:
                continue
            out.append(f'\n**{col}** (epochs {LATE}-200)\n\n')
            for C in (20, 100):
                if (d['C'] == C).any():
                    out.append(f'C = {C}\n\n' + md_table(d[d['C'] == C], 'method', 'k', col) + '\n')
        out.append(img(small_multiples(late, 'pos_nonkey_prec', 'neighbour-positive precision', 'pos_prec_vs_k.png',
                                       f'Precision of selected neighbour positives (epochs {LATE}-200)'), 'pos prec'))
        out.append(img(small_multiples(late, 'neg_fn_rate', 'false-negative rate', 'neg_fn_vs_k.png',
                                       f'Same-class pairs pushed apart as negatives (epochs {LATE}-200)'), 'neg fn'))

        # ── 5. gradients vs ground truth (PiCO, ComCo) ──────────────────────
        out.append('\n## 5. Gradient alignment with the ground-truth objective (PiCO, ComCo)\n')
        out.append('Cosine between the gradient (w.r.t. the backbone feature) of the loss actually used and of its '
                   'true-label counterpart (src/analysis/gradients.grad_alignment), mean over epochs '
                   f'{LATE}-200. `cls_vs_ce`: cls loss vs CE(true); `cont_vs_gt`: contrastive loss with the selected '
                   'pairs vs true-class pairs; `total_vs_gt`: the full objective; `cls_vs_cont_cos`: do the two '
                   'losses push the same way; `cont_over_cls_norm`: ||lambda g_cont|| / ||g_cls||.\n')
        gcols = [c for c in ('g_cls_vs_ce_cos', 'g_cont_vs_gt_cos', 'g_total_vs_gt_cos', 'g_cls_vs_cont_cos',
                             'g_cont_over_cls_norm') if c in pcd]
        glate = traces.window_mean(pcd, gcols, start=LATE, by=('method', 'C', 'k'), per_seed=True)
        for col in gcols:
            out.append(f'\n**{col}**\n\n')
            for C in (20, 100):
                if (glate['C'] == C).any():
                    out.append(f'C = {C}\n\n' + md_table(glate[glate['C'] == C], 'method', 'k', col) + '\n')
        out.append(img(small_multiples(glate, 'g_total_vs_gt_cos', 'cos(total grad, true-label grad)',
                                       'grad_total_vs_k.png', f'Total-objective gradient alignment (epochs {LATE}-200)'),
                       'grad alignment'))

    # ── 6. cls-loss gradient bias / variance (all 3) ────────────────────────
    bv = name(io.load_detail_csv(RUNS, 'biasvariance.csv', dedup_on=['epoch', 'loss_type']))
    out.append('\n## 6. Classification-loss gradient bias / variance vs CE (all methods)\n')
    out.append('50 resampled candidate sets on a fixed 256-sample batch (src/pipeline/biasvariance, reductions in '
               'src/analysis/gradients): `cos_sim` = cos(E[g], g_CE); `bias2`, `variance` per logit. Final epoch.\n')
    if not bv.empty:
        bv = bv[bv['loss_type'] == 'cls']
        bvl = bv.sort_values('epoch').groupby(['run', 'method', 'C', 'k', 'seed']).tail(1)
        for col, dg in (('cos_sim', 3), ('bias2', 8), ('variance', 8)):
            out.append(f'\n**{col}**\n\n')
            for C in (20, 100):
                if (bvl['C'] == C).any():
                    out.append(f'C = {C}\n\n' + md_table(bvl[bvl['C'] == C], 'method', 'k', col, digits=dg) + '\n')
        out.append(img(small_multiples(bvl, 'cos_sim', 'cos(E[g], g_CE)', 'bv_cos_vs_k.png',
                                       'Expected cls-loss gradient vs CE gradient (final epoch)'), 'bv cos'))

    # ── 7. per-class balance + representation (kNN) ─────────────────────────
    out.append('\n## 7. Per-class balance and representation quality\n')
    if not per_class.empty:
        lastpc = per_class.sort_values('epoch').groupby(['run', 'method', 'C', 'k', 'seed']).tail(1)
        accc = [c for c in lastpc.columns if c.startswith('acc_class_')]
        lastpc = lastpc.assign(class_acc_std=lastpc[accc].std(axis=1, ddof=0),
                               worst_class_acc=lastpc[accc].min(axis=1))
        for col in ('class_acc_std', 'worst_class_acc'):
            out.append(f'\n**{col}** (% , final checkpoint, test set)\n\n')
            for C in (20, 100):
                if (lastpc['C'] == C).any():
                    out.append(f'C = {C}\n\n' + md_table(lastpc[lastpc['C'] == C], 'method', 'k', col, digits=1) + '\n')
    knn = name(io.load_detail_csv(RUNS, 'knn_eval.csv'))
    if not knn.empty:
        out.append('\n**kNN probe on the contrastive features** (top-1 %, k=20, T=0.07; PRODEN has no projection head)\n\n')
        for C in (20, 100):
            if (knn['C'] == C).any():
                out.append(f'C = {C}\n\n' + md_table(knn[knn['C'] == C], 'method', 'k', 'knn_top1_acc', digits=1) + '\n')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(out))
    print(f'wrote {OUT_MD} ({len(res)} result rows)')


if __name__ == '__main__':
    main()
