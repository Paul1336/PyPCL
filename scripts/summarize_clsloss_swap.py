#!/usr/bin/env python
"""Paste-friendly text summary of a cls-loss swap run (SwapPiCO-* / SwapComCo-*).

Reads every results.csv / shards/*.csv under the given run directories (shards
are used directly, so it works mid-run without a merge), and prints:
  1. provenance (rows, git commits, time range) and completion per plan phase,
  2. per framework: loss x k table of final accuracy mean±std (n seeds),
  3. best loss per k per framework,
  4. mean training hours per algorithm,
  5. with --raw: every row as compact CSV, for exact re-analysis.

    python scripts/summarize_clsloss_swap.py results/clsloss_swap_c100 --plan c100 --raw
"""

import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.clsloss_swap import CLS_LOSSES  # noqa: E402


def load(run_dirs):
    frames = []
    for d in run_dirs:
        shards = glob.glob(os.path.join(d, 'shards', '*.csv'))
        files = shards or glob.glob(os.path.join(d, 'results.csv')) or ([d] if d.endswith('.csv') else [])
        frames += [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.algorithm.str.startswith('Swap')]
    df = df.drop_duplicates(['total_classes', 'k', 'algorithm', 'seed'], keep='last')
    df['framework'] = df.algorithm.str.extract(r'^Swap(\w+?)-')[0]
    df['loss'] = df.algorithm.str.split('-', n=1).str[1]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='results/<run> directories (or results.csv files)')
    ap.add_argument('--plan', choices=['c20_mclnl', 'c100'], default=None,
                    help='also report completion against scripts/launch_clsloss_swap.py\'s plan')
    ap.add_argument('--raw', action='store_true')
    a = ap.parse_args()
    df = load(a.runs)
    pd.set_option('display.width', 250)

    print(f'=== SOURCE: {a.runs}')
    print(f'rows={len(df)}  C={sorted(df.total_classes.unique())}  seeds={sorted(df.seed.unique())}  '
          f'epochs={sorted(df.epochs.unique())}  git={sorted(df.git_commit.astype(str).unique())}  '
          f'time={df.timestamp.min()} .. {df.timestamp.max()}')

    if a.plan:
        from scripts.launch_clsloss_swap import build_plan
        _, C, jobs = build_plan(a.plan)
        have = set(zip(df.algorithm, df.k, df.seed))
        print('=== COMPLETION')
        for ph in sorted({j[0] for j in jobs}):
            cells = [j for j in jobs if j[0] == ph]
            missing = [j for j in cells if (j[1], j[2], j[3]) not in have]
            print(f'phase {ph}: {len(cells) - len(missing)}/{len(cells)}' +
                  (f'  missing e.g. {[(m[1], m[2], m[3]) for m in missing[:6]]}' if missing else ''))

    order = [l for l in CLS_LOSSES if l in set(df.loss)]
    for C in sorted(df.total_classes.unique()):
        dc = df[df.total_classes == C]
        for fw in sorted(dc.framework.unique()):
            d = dc[dc.framework == fw]
            g = d.groupby(['loss', 'k']).final_accuracy
            cell = (g.mean().round(1).astype(str) + '±' + g.std().fillna(0).round(1).astype(str) +
                    '(' + g.size().astype(str) + ')').unstack('k')
            print(f'\n=== C={C} Swap{fw}: final acc mean±std(n seeds)')
            print(cell.reindex([l for l in order if l in cell.index]).fillna('-').to_string())
            best = g.mean().unstack('k').idxmax()
            bestv = g.mean().unstack('k').max().round(1)
            print('best: ' + '  '.join(f'k={k}:{best[k]}({bestv[k]})' for k in best.index))

    print('\n=== MEAN TRAINING HOURS PER ALGORITHM (C, algorithm)')
    print((df.groupby(['total_classes', 'algorithm']).training_time_s.mean() / 3600).round(2).to_string())

    if a.raw:
        print('\n=== RAW CSV')
        print(df[['total_classes', 'algorithm', 'k', 'seed', 'final_accuracy', 'epochs', 'training_time_s',
                  'timestamp', 'git_commit']].sort_values(['total_classes', 'algorithm', 'k', 'seed'])
              .to_csv(index=False).strip())


if __name__ == '__main__':
    main()
