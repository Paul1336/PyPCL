"""Analyses of per-epoch series: pcdiag columns over training, and bandit / switch traces.

Input frames come from analysis.io (load_pcdiag, load_bandit_traces, load_prune_traces,
load_alpha_traces). Most of these were previously ad-hoc scripts outside the repo
(~/.claude/jobs/*/tmp/*.py) or inline snippets; definitions are kept identical.
"""

import math

import numpy as np
import pandas as pd

CELL = ['run', 'algorithm', 'C', 'k', 'seed']


def _cell(df):
    return [c for c in CELL if c in df.columns]


# ─── per-epoch diagnostics over training ───────────────────────────────────


def at_epochs(diag, cols, epochs=(50, 100, 200), by=('algorithm', 'C', 'k')):
    """Seed-mean (and n) of diagnostic columns at given epochs (tables.py P0 EPOCH, ep.py)."""
    x = diag[diag['epoch'].isin(epochs)]
    g = x.groupby(list(by) + ['epoch'])[list(cols)]
    out = g.mean()
    out['n'] = g.size()
    return out.reset_index()


def window_mean(diag, cols, start=None, end=None, by=('algorithm', 'C', 'k'), per_seed=False):
    """Mean of columns over an epoch window [start, end] (inclusive), e.g. ComCo positives only
    exist after warmup_pos=100 -> window_mean(d, cols, start=101). Seed-level first, then
    mean/std over seeds unless per_seed."""
    x = diag
    if start is not None:
        x = x[x['epoch'] >= start]
    if end is not None:
        x = x[x['epoch'] <= end]
    s = x.groupby(list(by) + ['seed'])[list(cols)].mean()
    if per_seed:
        return s.reset_index()
    return s.groupby(list(by)).agg(['mean', 'std', 'count']).reset_index()


def final_epoch(diag, cols, by=('algorithm', 'C', 'k')):
    """Last-epoch value per seed file, then mean/std over seeds (art/build_data.py)."""
    last = diag.sort_values('epoch').groupby(_cell(diag)).tail(1)
    return last.groupby(list(by))[list(cols)].agg(['mean', 'std', 'count']).reset_index()


def curves(diag, cols, every=5, by=('algorithm', 'C', 'k')):
    """Mean/std over seeds every `every` epochs (for plotting learning curves of any column).
    NaN entries (e.g. test_acc on non-eval epochs) are ignored."""
    x = diag[diag['epoch'] % every == 0]
    return x.groupby(list(by) + ['epoch'])[list(cols)].agg(['mean', 'std']).reset_index()


def first_epoch_reaching(diag, col, threshold, above=True, by=None):
    """First epoch at which a column crosses a threshold, per cell (NaN if never)."""
    rows = []
    for key, g in diag.groupby(by or _cell(diag)):
        g = g.sort_values('epoch')
        hit = g[g[col] >= threshold] if above else g[g[col] <= threshold]
        rows.append(dict(zip(by or _cell(diag), key if isinstance(key, tuple) else (key,)),
                         epoch=hit['epoch'].iloc[0] if len(hit) else np.nan))
    return pd.DataFrame(rows)


# ─── lookahead-estimate quality (bandit reward diagnostics) ────────────────


def estimate_quality(tr, est='est_soft', truth='true_soft', min_abs_dt=0.0):
    """Per-epoch quality of a label-free lookahead estimate vs the oracle, from expanded
    look_all columns (look0_<key>, look1_<key>). Adds:
      de = est(B) - est(A), dt = truth(B) - truth(A)
      agree = sign(de) == sign(dt)              (sign agreement)
      d_bias = de - dt                          (bias of the estimated advantage)
      biasA / biasB = est - truth per arm       (per-arm estimate bias)
    Rows with |dt| < min_abs_dt are dropped (ties carry no sign information)."""
    x = tr.dropna(subset=[f'look0_{est}', f'look1_{est}', f'look0_{truth}', f'look1_{truth}']).copy()
    x['de'] = x[f'look1_{est}'] - x[f'look0_{est}']
    x['dt'] = x[f'look1_{truth}'] - x[f'look0_{truth}']
    x = x[x['dt'].abs() >= min_abs_dt]
    x['agree'] = (np.sign(x['de']) == np.sign(x['dt'])).astype(float)
    x['d_bias'] = x['de'] - x['dt']
    x['biasA'] = x[f'look0_{est}'] - x[f'look0_{truth}']
    x['biasB'] = x[f'look1_{est}'] - x[f'look1_{truth}']
    return x


def estimate_quality_summary(tr, ests=('est_soft', 'rho_soft'), truth='true_soft', min_abs_dt=1e-3,
                             by=('algorithm', 'k')):
    """Seed-level means of agree / d_bias / per-arm bias for each estimator, then averaged over
    seeds (the biasreward_1020 / banditrho_1028 tables)."""
    out = []
    for est in ests:
        q = estimate_quality(tr, est, truth, min_abs_dt)
        s = q.groupby(list(by) + ['seed'])[['agree', 'd_bias', 'biasA', 'biasB']].mean()
        s = s.groupby(list(by)).mean()
        s.columns = [f'{est}:{c}' for c in s.columns]
        out.append(s)
    return pd.concat(out, axis=1).reset_index()


def wrong_direction_rates(tr, est='est_soft', truth='true_soft'):
    """How often the bandit is pushed toward arm 1 when arm 1 is truly worse:
    P(dt < 0 | de > 0), P(dt < 0 | mu > 0), and the Thompson-weighted rate
    sum_t pc_t 1[dt_t < 0] / sum_t pc_t with pc = P(choose arm 1) = Phi(mu/se).
    The runner logs mu/se right after PairedTS.update(), i.e. the posterior that epoch's
    choice was sampled from, so pc is exact (no re-simulation needed)."""
    q = estimate_quality(tr, est, truth)
    out = {'p_dt_neg_given_de_pos': (q.loc[q['de'] > 0, 'dt'] < 0).mean()}
    if {'mu', 'se'} <= set(q.columns):
        out['p_dt_neg_given_mu_pos'] = (q.loc[q['mu'] > 0, 'dt'] < 0).mean()
        z = q['mu'] / q['se'].clip(lower=1e-12)
        pc = 0.5 * (1 + z.map(lambda v: math.erf(v / math.sqrt(2))))
        out['ts_weighted_wrong'] = float((pc * (q['dt'] < 0)).sum() / pc.sum()) if pc.sum() > 0 else np.nan
    return out


def by_phase(df, cols, edges=(0, 40, 80, 120, 160, 200), by=('algorithm',)):
    """Mean of columns per training phase (pd.cut on epoch)."""
    x = df.copy()
    x['phase'] = pd.cut(x['epoch'], list(edges))
    return x.groupby(list(by) + ['phase'], observed=True)[list(cols)].mean().reset_index()


# ─── arm / switch behaviour ────────────────────────────────────────────────


def arm_summary(tr, value='value', block=40, settle_thr=0.1):
    """Per run-cell summary of the chosen arm/rho sequence:
    frac_arm1 (overall), frac per `block` epochs (b1, b2, ...), n_switches, last epoch
    with value 1 (last1), settle epoch = first e with mean(v[e:]) < settle_thr,
    switched_at (if logged), frac_d_pos (fraction of lookahead d = look1-look0 > 0)."""
    rows = []
    for key, g in tr.groupby(_cell(tr)):
        g = g.sort_values('epoch')
        v = g[value].astype(float).to_numpy()
        r = dict(zip(_cell(tr), key), n_epochs=len(v), frac_arm1=v.mean() if len(v) else np.nan,
                 n_switches=int((v[1:] != v[:-1]).sum()) if len(v) > 1 else 0,
                 last1=int(g['epoch'].to_numpy()[v == 1].max()) if (v == 1).any() else 0)
        for i in range(0, len(v), block):
            r[f'b{i // block + 1}'] = v[i:i + block].mean()
        settle = next((int(g['epoch'].iloc[e]) for e in range(len(v)) if v[e:].mean() < settle_thr), np.nan)
        r['settle_epoch'] = settle
        if 'switched_at' in g:
            sa = g['switched_at'].dropna()
            r['switched_at'] = sa.iloc[0] if len(sa) else np.nan
        if {'look0', 'look1'} <= set(g.columns):
            d = (g['look1'] - g['look0']).dropna()
            r['frac_d_pos'] = (d > 0).mean() if len(d) else np.nan
            r['d_mean_x100'] = 100 * d.mean() if len(d) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def lookahead_d_by_phase(tr, phases=((1, 20), (21, 100), (101, 200))):
    """Mean lookahead advantage d = look1 - look0 (x100) and frac d > 0 per epoch range."""
    x = tr.dropna(subset=['look0', 'look1']).copy()
    x['d'] = x['look1'] - x['look0']
    rows = []
    for (lo, hi) in phases:
        s = x[(x['epoch'] >= lo) & (x['epoch'] <= hi)]
        g = s.groupby([c for c in ('algorithm', 'k') if c in s])['d']
        part = pd.DataFrame({'d_x100': 100 * g.mean(), 'frac_d_pos': g.apply(lambda d: (d > 0).mean())})
        part['phase'] = f'{lo}-{hi}'
        rows.append(part.reset_index())
    return pd.concat(rows, ignore_index=True)


def alpha_summary(alpha_df):
    """Alpha trace: mean alpha and fraction of batches with alpha == 1, per cell."""
    g = alpha_df.groupby(_cell(alpha_df))['alpha']
    return pd.DataFrame({'mean_alpha': g.mean(), 'frac_alpha1': g.apply(lambda a: (a == 1).mean())}).reset_index()


def prune_summary(prune_df, at_epochs=None):
    """Retention / |S| / q_top1 at the last epoch (or chosen epochs), mean over seeds."""
    x = prune_df
    if at_epochs is None:
        x = x.sort_values('epoch').groupby(_cell(x)).tail(1)
    else:
        x = x[x['epoch'].isin(at_epochs)]
    keys = [c for c in ('algorithm', 'k') if c in x] + ([] if at_epochs is None else ['epoch'])
    return x.groupby(keys)[['S_size', 'retention', 'q_top1']].agg(['mean', 'std', 'count']).reset_index()
