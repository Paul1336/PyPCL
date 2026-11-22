"""Loaders for every result / diagnostic artifact the pipeline writes. All return pandas
DataFrames in long format with the run name attached, so analyses can concat runs freely.

Layout (results/<run>/...):
  results.csv, shards/worker*.csv          final accuracy per (C, k, algorithm, seed)
  pcdiag/<alg>/C{C}_k{k}_s{seed}.csv        per-epoch diagnostics (pcdiag.py / picoplus.py)
  bandit_trace/<alg>[_C{C}]_k{k}_seed{s}.json  per-epoch bandit records (+ switch_diag fields)
  prune_trace/<alg>_k{k}_seed{s}.json       {k, k_star, warm, every, ramp, trace: [...]}
  alpha_trace/<alg>_k{k}_seed{s}.json       [epoch][batch] realised alpha
  detail/<alg>/C{C}_k{k}/*.csv              --detail / --concentration / --knn_eval /
                                            --biasvariance outputs (+ Solo dynamics_s*.csv)
"""

import glob
import json
import os
import re

import pandas as pd

RESULTS_ROOT = 'results'
KEY = ['total_classes', 'k', 'algorithm', 'seed']


def _run_dir(run, root=RESULTS_ROOT):
    return run if os.path.isdir(run) else os.path.join(root, run)


# ─── final accuracy ────────────────────────────────────────────────────────


def load_results(runs, root=RESULTS_ROOT, include_shards=True, dedup=True):
    """Final-accuracy rows of one or more runs (results.csv + shards), deduplicated on
    (C, k, algorithm, seed) keeping the latest timestamp. Adds columns run, C, mean_k, varq."""
    runs = [runs] if isinstance(runs, str) else list(runs)
    frames = []
    for run in runs:
        d = _run_dir(run, root)
        files = [os.path.join(d, 'results.csv')]
        if include_shards:
            files += sorted(glob.glob(os.path.join(d, 'shards', '*.csv')))
        for f in files:
            if os.path.isfile(f) and os.path.getsize(f) > 0:
                x = pd.read_csv(f)
                x['run'] = os.path.basename(d.rstrip('/'))
                frames.append(x)
    if not frames:
        return pd.DataFrame(columns=KEY + ['final_accuracy', 'run'])
    df = pd.concat(frames, ignore_index=True)
    if dedup:
        if 'timestamp' in df:
            df = df.sort_values('timestamp', kind='stable')
        df = df.drop_duplicates(['run'] + KEY, keep='last')
    df['C'] = df['total_classes']
    df['varq'] = df['k'] < 0
    df['mean_k'] = [varq_to_mean_k(k, C) if k < 0 else k for k, C in zip(df['k'], df['C'])]
    return df.reset_index(drop=True)


def varq_to_mean_k(k_sentinel, C):
    """Variable-|Y| runs store k = -round(1000 q), q = (mean_k - 1)/(C - 1)."""
    return round(-k_sentinel / 1000 * (C - 1)) + 1


def mean_k_to_varq(mean_k, C):
    return -round((mean_k - 1) / (C - 1) * 1000)


# ─── per-epoch diagnostics (pcdiag) ────────────────────────────────────────

_PCDIAG_RE = re.compile(r'C(\d+)_k(-?\d+)_s(\d+)\.csv$')


def load_pcdiag(runs, algorithms=None, root=RESULTS_ROOT, min_epochs=None):
    """All pcdiag CSVs of the runs, one row per (run, algorithm, C, k, seed, epoch).
    min_epochs drops unfinished runs (e.g. 200)."""
    runs = [runs] if isinstance(runs, str) else list(runs)
    frames = []
    for run in runs:
        d = _run_dir(run, root)
        for f in sorted(glob.glob(os.path.join(d, 'pcdiag', '*', 'C*_k*_s*.csv'))):
            alg = os.path.basename(os.path.dirname(f))
            if algorithms and alg not in algorithms:
                continue
            m = _PCDIAG_RE.search(f)
            x = pd.read_csv(f)
            if min_epochs and len(x) < min_epochs:
                continue
            x.insert(0, 'seed', int(m.group(3)))
            x.insert(0, 'k', int(m.group(2)))
            x.insert(0, 'C', int(m.group(1)))
            x.insert(0, 'algorithm', alg)
            x.insert(0, 'run', os.path.basename(d.rstrip('/')))
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─── JSON traces ───────────────────────────────────────────────────────────

_TRACE_RE = re.compile(r'(.+?)(?:_C(\d+))?_k(-?\d+)_seed(\d+)\.json$')  # _C{C} since 2026-11-22


def _iter_traces(runs, sub, root):
    runs = [runs] if isinstance(runs, str) else list(runs)
    for run in runs:
        d = _run_dir(run, root)
        for f in sorted(glob.glob(os.path.join(d, sub, '*.json'))):
            m = _TRACE_RE.search(os.path.basename(f))
            if m:
                C = int(m.group(2)) if m.group(2) else None
                yield os.path.basename(d.rstrip('/')), m.group(1), C, int(m.group(3)), int(m.group(4)), f


def load_bandit_traces(runs, root=RESULTS_ROOT, expand_look_all=True):
    """Per-epoch bandit records (epoch >= 1). look=[A,B] -> look0/look1; with
    expand_look_all, look_all=[{...},{...}] -> look0_<key>/look1_<key>. Run-level header
    (epoch 0: reward, shrink_lambda, cv_loglik) is returned in df.attrs['headers']."""
    rows, headers = [], {}
    for run, alg, C, k, seed, f in _iter_traces(runs, 'bandit_trace', root):
        for rec in json.load(open(f)):
            base = dict(run=run, algorithm=alg, C=C, k=k, seed=seed)
            if rec.get('epoch', 1) == 0:
                headers[(run, alg, C, k, seed)] = rec
                continue
            r = dict(base)
            for key, v in rec.items():
                if key == 'look' and isinstance(v, list):
                    r['look0'], r['look1'] = v[0], v[1]
                elif key == 'probes' and isinstance(v, list):
                    r['probe0'], r['probe1'] = v[0], v[1]
                elif key == 'look_all' and isinstance(v, list):
                    if expand_look_all:
                        for i, dct in enumerate(v):
                            for kk, vv in dct.items():
                                r[f'look{i}_{kk}'] = vv
                else:
                    r[key] = v
            if 'value' not in r and 'arm' in r:
                r['value'] = r['arm']
            rows.append(r)
    df = pd.DataFrame(rows)
    df.attrs['headers'] = headers
    return df


def load_prune_traces(runs, root=RESULTS_ROOT):
    rows = []
    for run, alg, C, k, seed, f in _iter_traces(runs, 'prune_trace', root):
        d = json.load(open(f))
        hdr = {kk: v for kk, v in d.items() if kk != 'trace'}
        for rec in d['trace']:
            rows.append(dict(run=run, algorithm=alg, C=C, seed=seed, **{**hdr, 'k': k}, **rec))
    return pd.DataFrame(rows)


def load_alpha_traces(runs, root=RESULTS_ROOT):
    """One row per (epoch, batch) with the realised mean alpha."""
    rows = []
    for run, alg, C, k, seed, f in _iter_traces(runs, 'alpha_trace', root):
        for ep, batches in enumerate(json.load(open(f)), start=1):
            for b, a in enumerate(batches):
                rows.append(dict(run=run, algorithm=alg, C=C, k=k, seed=seed, epoch=ep, batch=b, alpha=a))
    return pd.DataFrame(rows)


# ─── detail/ CSVs ──────────────────────────────────────────────────────────

_CELL_RE = re.compile(r'C(\d+)_k(-?\d+)$')


def load_detail_csv(runs, filename, root=RESULTS_ROOT, dedup_on=None):
    """Any detail/<alg>/C{C}_k{k}/<filename> file (glob allowed, e.g. 'biasvariance*.csv',
    'dynamics_s*.csv', 'per_class_loss.csv', 'concentration_summary.csv', 'knn_eval.csv',
    'pico_selection_stats.csv', 'comco_selection_stats.csv',
    'pico_oracle_correction_stats.csv', 'pico_oracle_add_correction_stats.csv').
    Detail CSVs are append-only, so retried runs duplicate rows: pass dedup_on (e.g.
    ['epoch'] or ['epoch', 'batch']) to keep the last occurrence per cell."""
    runs = [runs] if isinstance(runs, str) else list(runs)
    frames = []
    for run in runs:
        d = _run_dir(run, root)
        for f in sorted(glob.glob(os.path.join(d, 'detail', '*', 'C*_k*', filename))):
            cell = os.path.dirname(f)
            m = _CELL_RE.search(os.path.basename(cell))
            x = pd.read_csv(f)
            x['run'] = os.path.basename(d.rstrip('/'))
            x['algorithm'] = x.get('algorithm', os.path.basename(os.path.dirname(cell)))
            x['C'], x['k'] = int(m.group(1)), int(m.group(2))
            sm = re.search(r'_s(\d+)\.csv$', f)
            if sm and 'seed' not in x:
                x['seed'] = int(sm.group(1))
            if dedup_on:
                keys = ['run', 'algorithm', 'C', 'k'] + (['seed'] if 'seed' in x else []) + list(dedup_on)
                x = x.drop_duplicates(keys, keep='last')
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def list_runs(pattern='*', root=RESULTS_ROOT):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, pattern)) if os.path.isdir(p))
