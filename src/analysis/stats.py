"""Cross-seed / cross-method statistics on long DataFrames (from analysis.io).

Conventions (unified; older scripts differed):
* std is the sample std (ddof=1); with n = 1 it is NaN and fmt() prints "(n=1)".
* "mean over cells" is always a two-level average: seed-mean per cell first.
* Paired comparisons pair rows on identical (C, k, seed) [+ run if wanted].
"""

import numpy as np
import pandas as pd

# ─── summaries ─────────────────────────────────────────────────────────────


def summarize(df, by=('algorithm', 'C', 'k'), value='final_accuracy', seeds=None):
    """mean, std (ddof=1), n, sem, min, max per group; optional seed subset."""
    if seeds is not None:
        df = df[df['seed'].isin(seeds)]
    g = df.groupby(list(by))[value]
    out = g.agg(['mean', 'std', 'count', 'min', 'max']).rename(columns={'count': 'n'})
    out['sem'] = out['std'] / np.sqrt(out['n'])
    return out.reset_index()


def fmt(mean, std=float('nan'), n=None, digits=1, expected_n=3):
    """'72.3±1.2', '72.3±1.2 (n=2)' when incomplete, '72.3 (n=1)' for a single seed."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return '–'
    s = f'{mean:.{digits}f}'
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        s += f'±{std:.{digits}f}'
    if n is not None and n < expected_n:
        s += f' (n={n})'
    return s


def pivot_fmt(df, index='algorithm', columns='k', value='final_accuracy', digits=1, expected_n=3,
              with_row_mean=True, seeds=None):
    """Formatted mean±std table (strings) + optional row average over columns, flagged '*'
    when any cell is missing or has n < expected_n (picoplus_1108 convention)."""
    s = summarize(df, by=(index, columns), value=value, seeds=seeds)
    s['cell'] = [fmt(m, sd, n, digits, expected_n) for m, sd, n in zip(s['mean'], s['std'], s['n'])]
    tab = s.pivot(index=index, columns=columns, values='cell')
    if with_row_mean:
        means = s.pivot(index=index, columns=columns, values='mean')
        ns = s.pivot(index=index, columns=columns, values='n')
        flag = means.isna().any(axis=1) | (ns < expected_n).any(axis=1)
        tab['avg'] = [f'{m:.{digits}f}' + ('*' if f else '') for m, f in zip(means.mean(axis=1), flag)]
    return tab


def to_markdown(tab, bold_best=None):
    """Plain markdown table; bold_best = DataFrame of numeric means with the same shape to
    bold the column-wise maximum."""
    cols = [str(c) for c in tab.columns]
    lines = ['| ' + ' | '.join([tab.index.name or ''] + cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    best = bold_best.idxmax() if bold_best is not None else {}
    for idx, row in tab.iterrows():
        cells = []
        for c in tab.columns:
            v = str(row[c])
            if bold_best is not None and c in best and best[c] == idx:
                v = f'**{v}**'
            cells.append(v)
        lines.append('| ' + ' | '.join([str(idx)] + cells) + ' |')
    return '\n'.join(lines)


def two_level_mean(df, cell=('algorithm', 'C', 'k'), over=('C', 'k'), value='final_accuracy'):
    """Seed-mean per cell, then mean over cells -> one number per remaining group."""
    s = summarize(df, by=cell, value=value)
    keep = [c for c in cell if c not in over]
    return s.groupby(keep)['mean'].agg(['mean', 'count']).rename(columns={'count': 'n_cells'}).reset_index()


def completeness(df, expected, key=('algorithm', 'k', 'seed')):
    """expected: iterable of key tuples (the plan). Returns the missing ones."""
    have = set(map(tuple, df[list(key)].itertuples(index=False, name=None)))
    return sorted(set(map(tuple, expected)) - have)


# ─── comparisons ───────────────────────────────────────────────────────────


def diff_of_means(df, base, variants, by=('C', 'k'), value='final_accuracy', alg_col='algorithm'):
    """Seed-mean(variant) - seed-mean(base) per cell (oracle gain, method gap, '+X vs baseline')."""
    m = df.groupby([alg_col] + list(by))[value].mean()
    rows = []
    for v in variants:
        for idx in m.loc[v].index if v in m.index.get_level_values(0) else []:
            key = idx if isinstance(idx, tuple) else (idx,)
            if (base,) + key in m.index:
                rows.append(dict(zip(by, key), variant=v, base=base,
                                 diff=m[(v,) + key] - m[(base,) + key]))
    return pd.DataFrame(rows)


def paired_compare(df, a, b, on=('C', 'k', 'seed'), value='final_accuracy', alg_col='algorithm'):
    """Same-seed paired comparison of algorithm a vs b. Returns dict with n_pairs, mean diff
    (a-b), sd, se, 95% t-CI, win rate (a > b), tie rate, Pearson r, and the paired frame."""
    from scipy import stats as st
    x = df[df[alg_col] == a].set_index(list(on))[value]
    y = df[df[alg_col] == b].set_index(list(on))[value]
    j = pd.concat([x.rename('a'), y.rename('b')], axis=1, join='inner').dropna()
    d = j['a'] - j['b']
    n = len(d)
    out = dict(n_pairs=n, mean_diff=d.mean(), sd=d.std(), se=d.std() / np.sqrt(n) if n else np.nan,
               win_rate=(d > 0).mean(), tie_rate=(d == 0).mean(),
               pearson_r=j['a'].corr(j['b']) if n > 2 else np.nan)
    if n > 1:
        h = st.t.ppf(0.975, n - 1) * out['se']
        out['ci95'] = (out['mean_diff'] - h, out['mean_diff'] + h)
        out['p_ttest'] = st.ttest_rel(j['a'], j['b']).pvalue
        out['p_wilcoxon'] = st.wilcoxon(d).pvalue if (d != 0).any() else 1.0
    out['pairs'] = j
    return out


def reproducibility(df_a, df_b, on=('algorithm', 'C', 'k', 'seed'), value='final_accuracy'):
    """Same-seed rerun agreement: mean / max |delta| between two runs of the same cells."""
    j = df_a.set_index(list(on))[value].to_frame('a').join(df_b.set_index(list(on))[value].rename('b'),
                                                           how='inner')
    d = (j['a'] - j['b']).abs()
    return dict(n=len(d), mean_abs_delta=d.mean(), max_abs_delta=d.max(), pairs=j)


def best_per_column(df, index='algorithm', columns='k', value='final_accuracy', exclude=()):
    """Argmax of seed-means per column (best loss per k, best fixed rho per cell, ...)."""
    m = df[~df[index].isin(exclude)].groupby([index, columns])[value].mean().unstack(columns)
    return pd.DataFrame({'best': m.idxmax(), 'value': m.max()})


def select_best_config(scores: dict, default=None, digits=2):
    """hptune rule: highest mean validation score, ties (after rounding) broken toward default."""
    return max(scores, key=lambda c: (round(scores[c], digits), c == default))
