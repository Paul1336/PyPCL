#!/usr/bin/env python
"""Markdown tables for the smallk_1117 study (P0 / P1-1 / P1-2 / P2a), merging new cells with the existing
same-protocol baselines in switch-analysis. final_accuracy (last epoch, %), mean ± std (ddof=1) (n).
Usage: python logs/smallk_1117/tables.py [P0|P11|P12|P2|EPOCH|ALL]"""
import glob
import os
import sys

import pandas as pd

NW = '/nfs/home/Paul/PyPCL/.claude/worktrees/comco-smallk-1117/results'
SA = '/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis/results'


def load(paths):
    dfs = []
    for p in paths:
        f = os.path.join(p, 'results.csv')
        if os.path.exists(f):
            d = pd.read_csv(f)
            d['src'] = os.path.basename(p)
            dfs.append(d)
    if not dfs:
        return pd.DataFrame(columns=['algorithm', 'k', 'final_accuracy', 'total_classes', 'seed'])
    d = pd.concat(dfs, ignore_index=True)
    return d.drop_duplicates(['total_classes', 'k', 'algorithm', 'seed'], keep='last')


def fmt(s):
    s = s.dropna()
    if len(s) == 0:
        return '—'
    if len(s) == 1:
        return f'{s.iloc[0]:.2f} (n=1)'
    return f'{s.mean():.2f} ± {s.std(ddof=1):.2f} (n={len(s)})'


def table(df, algs, ks, C, labels=None, kname=None):
    d = df[df.total_classes == C]
    head = '| 方法 | ' + ' | '.join(f'k={kname(k) if kname else k}' for k in ks) + ' |'
    out = [head, '|' + '---|' * (len(ks) + 1)]
    for a in algs:
        row = [fmt(d[(d.algorithm == a) & (d.k == k)].final_accuracy) for k in ks]
        out.append(f'| {labels.get(a, a) if labels else a} | ' + ' | '.join(row) + ' |')
    return '\n'.join(out)


def p0():
    df = load([f'{SA}/pcdiag_c20_uniform_1103', f'{SA}/pcdiag_c20_uniform_1104mcl', f'{SA}/pcdiag_c20_uniform_1105plema',
               f'{NW}/smallk_p0_c20_uniform_1117'])
    ks = [1, 2, 5, 10, 15, 19]
    algs = [f'PCD-ComCoMCL{v}' for v in ('', '-OraclePL', '-OraclePos', '-OracleNeg', '-OracleSel')] + \
           ['PCD-ComCo-OracleCls', 'PCD-ComCoMCL-EMA99'] + \
           [f'PCD-PiCO{v}' for v in ('', '-OraclePL', '-OraclePos', '-OracleNeg', '-OracleSel', '-OracleProto',
                                      '-OracleCls')]
    print('### P0 final acc, C=20 uniform\n')
    print(table(df, algs, ks, 20))
    print('\n### P0 oracle gain (oracle mean − baseline mean)\n')
    d = df[df.total_classes == 20]
    m = d.groupby(['algorithm', 'k']).final_accuracy.mean()
    print('| baseline | oracle | ' + ' | '.join(f'k={k}' for k in ks) + ' |')
    print('|' + '---|' * (len(ks) + 2))
    for base, ors in (('PCD-ComCoMCL', ['-OraclePL', '-OracleSel', '-OraclePos', '-OracleNeg']),
                      ('PCD-PiCO', ['-OraclePL', '-OracleSel', '-OraclePos', '-OracleNeg', '-OracleProto'])):
        for o in ors:
            cells = []
            for k in ks:
                try:
                    cells.append(f'{m[(base + o, k)] - m[(base, k)]:+.2f}')
                except KeyError:
                    cells.append('—')
            print(f'| {base} | {o[1:]} | ' + ' | '.join(cells) + ' |')
    print('\n### P0 gap ComCoMCL − PiCO (and best ComCo oracle − PiCO)\n')
    for k in ks:
        try:
            g = m[('PCD-ComCoMCL', k)] - m[('PCD-PiCO', k)]
            best = max((m.get((f'PCD-ComCoMCL{o}', k), float('nan')), o) for o in
                       ('-OraclePL', '-OracleSel', '-OraclePos', '-OracleNeg'))
            print(f'- k={k}: ComCoMCL − PiCO = {g:+.2f}; best contrastive oracle {best[1][1:]} − PiCO = '
                  f'{best[0] - m[("PCD-PiCO", k)]:+.2f}')
        except KeyError:
            print(f'- k={k}: (missing)')


EP_COLS = ['pl_acc', 'pl_in_comp', 'pos_nonkey_prec', 'pos_recall', 'neg_fn_rate', 'neg_certified_rate', 'test_acc']


def epoch_tab():
    print('### P0 per-epoch metrics (mean over seeds), C=20 uniform, epochs 50 / 100 / 200\n')
    roots = {'PCD-ComCoMCL': [f'{SA}/pcdiag_c20_uniform_1104mcl', f'{NW}/smallk_p0_c20_uniform_1117'],
             'PCD-PiCO': [f'{SA}/pcdiag_c20_uniform_1103', f'{NW}/smallk_p0_c20_uniform_1117']}
    print('| 方法 | k | ' + ' | '.join(EP_COLS) + ' |')
    print('|' + '---|' * (len(EP_COLS) + 2))
    for a, rs in roots.items():
        for k in [1, 2, 5, 10, 15, 19]:
            fs = sorted({f for r in rs for f in glob.glob(f'{r}/pcdiag/{a}/C20_k{k}_s*.csv')})
            if not fs:
                continue
            ds = [pd.read_csv(f) for f in fs]
            cells = []
            for c in EP_COLS:
                vals = []
                for ep in (50, 100, 200):
                    v = [d.loc[d.epoch == ep, c].mean() for d in ds if c in d and (d.epoch == ep).any()]
                    vals.append(f'{sum(v) / len(v):.3f}' if v else '—')
                cells.append('/'.join(vals))
            print(f'| {a} (n={len(fs)}) | {k} | ' + ' | '.join(cells) + ' |')


def p11():
    df = load([f'{SA}/pcdiag_c20_uniform_1104mcl', f'{SA}/pcdiag_c20_uniform_1105plema',
               f'{SA}/pcdiag_c100_uniform_1105plema', f'{SA}/pcdiag_c20_uniform_1103', f'{SA}/pp_c100_uniform_fixk_1108',
               f'{NW}/smallk_p0_c20_uniform_1117'])
    algs = ['PCD-ComCoMCL'] + [f'PCD-ComCoMCL-EMA{t}' for t in ('50', '90', '99', '999')] + ['PCD-PiCO']
    print('### P1-1 EMA pseudo-label, C=20 uniform\n')
    print(table(df, algs, [1, 2, 5, 10, 15, 19], 20))
    print('\n### P1-1 EMA pseudo-label, C=100 uniform\n')
    print(table(df, algs, [5, 25, 50, 75, 99], 100))


def p12():
    df = load([f'{SA}/pp_c20_uniform_fixk_1108', f'{SA}/pp_c100_uniform_fixk_1108', f'{SA}/pcdiag_c20_uniform_1104mcl',
               f'{SA}/pcdiag_c100_uniform_1105plema', f'{SA}/pcdiag_c20_uniform_1103',
               f'{NW}/smallk_p12_c20_uniform_1117', f'{NW}/smallk_p12_c100_uniform_1117'])
    sc = ['0.25', '0.5', '2', '4']
    lab = {'PCD-ComCoMCLNL': 'MCL-NL ×1, λ=0.3 (baseline)', 'PCD-ComCoMCL': 'MCL-LOG = MCL-NL×(C−1)/k',
           'PCD-ComCoMCLFixed': 'MCL-LOG(Feng) = MCL-NL×2(C−1)/(C−k)', 'PCD-PiCO': 'PiCO (ref)'}
    lab.update({f'PCD-ComCoMCLNL-CLSx{t}': f'MCL-NL ×{t}' for t in sc})
    lab.update({f'PCD-ComCoMCLNL-LWx{t}': f'λ ×{t} (= {0.3 * float(t):g})' for t in sc})
    cls = ['PCD-ComCoMCLNL-CLSx0.25', 'PCD-ComCoMCLNL-CLSx0.5', 'PCD-ComCoMCLNL', 'PCD-ComCoMCLNL-CLSx2',
           'PCD-ComCoMCLNL-CLSx4', 'PCD-ComCoMCL', 'PCD-ComCoMCLFixed', 'PCD-PiCO']
    lw = ['PCD-ComCoMCLNL-LWx0.25', 'PCD-ComCoMCLNL-LWx0.5', 'PCD-ComCoMCLNL', 'PCD-ComCoMCLNL-LWx2',
          'PCD-ComCoMCLNL-LWx4']
    for C, ks in ((20, [5, 10, 15]), (100, [5, 50])):
        print(f'### P1-2 classification-loss coefficient, C={C}\n')
        print(table(df, cls, ks, C, lab))
        print(f'\n### P1-2 contrastive weight λ, C={C}\n')
        print(table(df, lw, ks, C, lab))
        print()


def p2():
    df = load([f'{SA}/varq_c20_uniform_1106', f'{NW}/smallk_p2_varq_c20_uniform_1117', f'{SA}/pp_c20_uniform_varq_1108'])
    ks = sorted(df[df.total_classes == 20].k.unique())
    print('### P2(a) variable-size candidate sets, C=20 uniform (column = mean |Y| = 1 + q·19)\n')
    print(table(df, ['PCD-PiCO', 'PCD-ComCoMCL', 'PCD-ComCoMCLNL', 'PCD-ComCoMCLFixed', 'PCD-PiCOPlus'], ks, 20,
                kname=lambda k: f'{1 + (-k / 1000) * 19:.0f}'))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    for name, fn in (('P0', p0), ('EPOCH', epoch_tab), ('P11', p11), ('P12', p12), ('P2', p2)):
        if what in (name, 'ALL'):
            fn()
            print()
