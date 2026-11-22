"""Golden regression harness for the refactor.

    python tests/golden/golden.py record [--only REGEX]   # fingerprint the CURRENT code
    python tests/golden/golden.py check  [--only REGEX]   # compare against tests/golden/data/
    python tests/golden/golden.py data-record | data-check  # candidate-set generation only (CPU)

Each cell = one algorithm trained for EPOCHS epochs on a small CIFAR-100 subset (C=10, k=3)
with warm-ups shortened (golden_config.yaml) so every branch (contrastive, positive/negative
selection, prototype updates, bandit lookahead, pruning, PiCO+ selection) is exercised.
Runs in subprocesses (GPUs 0-3, SLOTS per GPU), deterministic mode.

Fingerprint per cell: model checksums at every evaluation (params + buffers), returned
accuracy, final_accuracy row, and every artifact written under results/<run>/ (pcdiag CSVs,
bandit/prune/alpha traces, detail CSVs, concentration npz summaries). Timing columns ignored.
Comparison: relative tolerance RTOL (1e-6) on every number.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data')
STORE = os.path.join(HERE, 'data')
SCRATCH = os.environ.get('GOLDEN_SCRATCH', os.path.join('/tmp', f'golden_{os.getuid()}'))
PY = sys.executable
GPUS = [int(g) for g in os.environ.get('GOLDEN_GPUS', '0,1,2,3').split(',')]
SLOTS = int(os.environ.get('GOLDEN_SLOTS', '3'))
EPOCHS, C, K, BS, SEED = 4, 10, 3, 256, 42
RTOL, ATOL = 1e-6, 1e-9
IGNORE_COLS = {'epoch_time_s', 'training_time_s', 'timestamp', 'git_commit', 'elapsed', 'time'}

HO = ['--holdout_frac', '0.1']
DET = ['--detail', '--detail_log_every', '2']
CONC = ['--concentration', '--concentration_log_every', '2']
KNN = ['--knn_eval']
BV = ['--biasvariance', '--biasvariance_log_every', '2', '--biasvariance_m_resamples', '4',
      '--biasvariance_eval_size', '64']
PRUNE = ['src.pipeline.algorithms.runners:PRUNE_WARM=1', 'src.pipeline.algorithms.runners:PRUNE_EVERY=1',
         'src.pipeline.algorithms.runners:PRUNE_RAMP=2']
AC = ['src.pcdiag:AC_POS_START=2']

# (cell id, algorithm, extra run_pipeline args, module patches)
CELLS = [
    # simple shape
    *[(a, a, [], []) for a in ['CLPL', 'Wu2022', 'MCL-LOG', 'MCL-LOG-Fixed', 'SCL-NL', 'OP', 'OP-W']],
    ('CPE+detail', 'CPE', DET, []),
    # PRODEN family
    ('PRODEN+detail+conc', 'PRODEN', DET + CONC, []),
    *[(a, a, [], []) for a in ['PRODEN-UniformInit', 'PRODEN-BiasedInit', 'PRODEN-RandomCandInit',
                               'PRODEN-BiasedCand-W20', 'PRODEN-BiasedAll-W10', 'PRODEN-BiasedRand-W10-Wf5',
                               'PRODEN-BiasedRandAll-W20-N4', 'PRODEN-EMA050']],
    ('PRODEN@set1', 'PRODEN', ['--cl_bias', 'set1'], []),
    ('PRODEN@holdout', 'PRODEN', HO, []),
    # PiCO family
    ('PiCO+detail', 'PiCO', DET, []),
    ('PiCO-Oracle+detail', 'PiCO-Oracle', DET, []),
    ('PiCO-Oracle-Add+detail+conc+knn', 'PiCO-Oracle-Add', DET + CONC + KNN, []),
    ('PiCO-MOCO', 'PiCO-MOCO', [], []),
    ('PiCO-Fixed+detail+conc+knn+bv', 'PiCO-Fixed', DET + CONC + KNN + BV, []),
    *[(a, a, [], []) for a in ['PiCO-Fixed-UniformInit', 'PiCO-Fixed-BiasedInit', 'PiCO-Fixed-EMA050',
                               'PiCO-Fixed-BiasedCand-W20', 'PiCO-Fixed-BiasedRand-W10-Wf5',
                               'PiCO-Fixed-BiasedRandAll-W20-N4', 'PiCO-Fixed-SrcSoftmax-EMA050',
                               'PiCO-Fixed-SrcSoftmax-SoftUpdate-BiasedRandAll-W20-N19-EMA050',
                               'PiCO-MCL', 'PiCO-SC', 'PiCO-CLS']],
    ('PiCO-Weighted-A050+bv', 'PiCO-Weighted-A050', BV, []),
    ('PiCO-MCL-Fixed+bv', 'PiCO-MCL-Fixed', BV, []),
    # ComCo family
    ('ComCo+detail', 'ComCo', DET, []),
    ('ComCo-Fixed+detail+knn+bv', 'ComCo-Fixed', DET + KNN + BV + CONC, []),
    ('ComCo-Weighted-A050+bv', 'ComCo-Weighted-A050', BV, []),
    *[(a, a, [], []) for a in ['ComCo-Weighted-SampleCond', 'ComCo-Weighted-BatchCond',
                               'ComCo-Weighted-BatchRandom']],
    ('ComCo-Weighted-SampleCond@bimodal', 'ComCo-Weighted-SampleCond',
     ['--only_bimodal', '2', '6', '0.5', '--batching', 'bucketed'], []),
    *[(a, a, HO, []) for a in ['ComCo-Weighted-BanditPaired', 'ComCo-Weighted-BanditEpoch']],
    # generic bandit / switch (need held-out)
    ('PiCO-Uni-Bandit01+switchdiag', 'PiCO-Uni-Bandit01', HO + ['--switch_diag'], []),
    *[(a, a, HO, []) for a in ['PiCO-Uni-BanditCont', 'PiCO-Uni-BanditLong', 'PiCO-Uni-Fixed0',
                               'PiCO-Mix-BanditPaired', 'PiCO-Mix-Sched25',
                               'ComCo-Uni-Bandit01-RhoEM', 'ComCo-Mix-BanditPaired-RhoOracle',
                               'ComCo-Mix-BanditPaired-TrueAcc', 'ComCo-Uni-SFlip', 'ComCo-Uni-Plateau',
                               'Hybrid-Bandit01', 'Hybrid-Mix-Bandit01', 'Hybrid-FixedComCo',
                               'Solo-Uni-Bandit01', 'Solo-Mix-BanditPaired',
                               'SwapComCo-BanditPaired-PRODEN-MCLLOGFixed']],
    # swap / prune
    *[(a, a, [], []) for a in ['SwapPiCO-UCE', 'SwapPiCO-PiCOPartial', 'SwapPiCO-SoLar', 'SwapComCo-MCLNL',
                               'SwapComCo-PRODEN', 'SwapComCo-OPW', 'SwapComCo-Wu2022']],
    ('SwapComCo-SCLNL+bv', 'SwapComCo-SCLNL', BV, []),
    *[(a, a, [], PRUNE) for a in ['SwapComCo-Prune-PRODEN', 'SwapComCo-PruneOracle-PiCOPartial',
                                  'SwapPiCO-Prune-PiCOPartial']],
    ('SoLar', 'SoLar', [], []),
    # PCD diagnostics + PiCO+
    *[(a, a, [], []) for a in ['PCD-PiCO', 'PCD-PiCO-OraclePL', 'PCD-PiCO-OracleProto', 'PCD-PiCO-OracleSel',
                               'PCD-PiCO-CandFilter', 'PCD-PiCO-OracleCls', 'PCD-ComCo', 'PCD-ComCo-CandFilter',
                               'PCD-ComCoMCL-OraclePL', 'PCD-ComCoMCL-EMA90', 'PCD-ComCoMCLNL',
                               'PCD-ComCoMCLNL-CLSx2', 'PCD-ComCoMCLNL-LWx2', 'PCD-ComCoMCLNL-MixA',
                               'PCD-ComCoMCLFixed', 'PCD-PiCOPlus', 'PCD-PiCOPlus-AllClean']],
    ('PCD-ComCoMCLNL-PosNegAC', 'PCD-ComCoMCLNL-PosNegAC', [], AC),
    ('PCD-ComCoMCLNL@varq', 'PCD-ComCoMCLNL', ['--only_q', '0.3'], []),
    ('PCD-PiCO@set1-varq', 'PCD-PiCO', ['--only_q', '0.3', '--cl_bias', 'set1'], []),
    ('PCD-PiCOPlus-D60@noisy', 'PCD-PiCOPlus-D60', ['--noise_eta', '0.2'], []),
    ('PCD-ComCoMCLNL@noisy-set1', 'PCD-ComCoMCLNL', ['--noise_eta', '0.2', '--cl_bias', 'set1'], []),
]

DATA_CELLS = [  # (id, load_experiment_data kwargs) -- candidate-set generation fingerprints
    ('fixk', dict(C=10, k=3)), ('fixk-c20-k10', dict(C=20, k=10)), ('k1', dict(C=10, k=1)),
    ('varq', dict(C=10, k=-300, q=0.3)), ('set1', dict(C=10, k=3, cl_bias='set1')),
    ('set1-varq', dict(C=10, k=-300, q=0.3, cl_bias='set1')), ('bimodal', dict(C=10, k=0, bimodal=(2, 6, 0.5))),
    ('noisy', dict(C=10, k=3, noise_eta=0.2)), ('noisy-set1', dict(C=10, k=3, noise_eta=0.2, cl_bias='set1')),
    ('holdout', dict(C=10, k=3, holdout_frac=0.1)), ('bucket', dict(C=10, k=0, bimodal=(2, 6, 0.5), batching='bucketed')),
]


def safe(cid):
    return re.sub(r'[^A-Za-z0-9_.@+-]', '_', cid)


# ─── artifact fingerprints ──────────────────────────────────────────────────


def _num(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return v


def read_artifacts(run_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, '**', '*'), recursive=True)):
        rel = os.path.relpath(f, run_dir)
        if os.path.isdir(f) or rel == 'run_config.json' or rel.endswith('.tmp') or rel.endswith('.png'):
            continue
        if rel.endswith('.csv'):
            rows = list(csv.DictReader(open(f)))
            out[rel] = [{k: _num(v) for k, v in r.items() if k not in IGNORE_COLS} for r in rows]
        elif rel.endswith('.json'):
            out[rel] = json.load(open(f))
        elif rel.endswith('.npz'):
            import numpy as np
            z = np.load(f)
            out[rel] = {k: [list(z[k].shape), float(np.asarray(z[k], dtype=float).sum())] for k in z.files}
    return out


def compare(a, b, path=''):
    """Returns list of mismatch strings."""
    if isinstance(a, dict) and isinstance(b, dict):
        errs = []
        for k in sorted(set(a) | set(b)):
            if k in IGNORE_COLS:
                continue
            if k not in a or k not in b:
                errs.append(f'{path}/{k}: only in {"new" if k in b else "golden"}')
                continue
            errs += compare(a[k], b[k], f'{path}/{k}')
        return errs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} vs {len(b)}']
        return [e for i, (x, y) in enumerate(zip(a, b)) for e in compare(x, y, f'{path}[{i}]')]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return []
        if abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b)):
            return []
        return [f'{path}: {a!r} vs {b!r}']
    return [] if a == b else [f'{path}: {a!r} vs {b!r}']


# ─── running cells ──────────────────────────────────────────────────────────


def cell_cmd(cid, alg, extra, patches, workdir):
    fp = os.path.join(workdir, 'fp.json')
    cmd = [PY, os.path.join(HERE, 'run_cell.py'), '--out', fp]
    for p in patches:
        cmd += ['--patch', p]
    cmd += ['--', 'run', '--run_name', 'g', '--config', os.path.join(HERE, 'golden_config.yaml'),
            '--algorithms', alg, '--algo', alg, '--c_values', str(C), *([] if ('--only_q' in extra or '--only_bimodal' in extra) else ['--only_k', str(K)]),
            '--seeds', str(SEED), '--epochs', str(EPOCHS), '--batch_size', str(BS),
            '--data_dir', DATA_DIR, '--report_every', '1', *extra]
    return cmd, fp


def run_cells(cells):
    os.makedirs(SCRATCH, exist_ok=True)
    queue, running, done = list(cells), [], {}
    load = {g: 0 for g in GPUS}
    while queue or running:
        for item in list(running):
            p, g, cell, workdir, fp, t0 = item
            if p.poll() is None:
                continue
            running.remove(item)
            load[g] -= 1
            ok = p.returncode == 0 and os.path.isfile(fp)
            done[cell[0]] = (ok, workdir, fp, time.time() - t0)
            print(f"  {'ok ' if ok else 'ERR'} {cell[0]}  ({time.time() - t0:.0f}s)", flush=True)
        while queue and min(load.values()) < SLOTS:
            g = min(load, key=load.get)
            cell = queue.pop(0)
            workdir = os.path.join(SCRATCH, safe(cell[0]))
            shutil.rmtree(workdir, ignore_errors=True)
            os.makedirs(workdir)
            cmd, fp = cell_cmd(*cell, workdir)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(g), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0'}
            log = open(os.path.join(workdir, 'log.txt'), 'w')
            p = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            log.close()
            running.append((p, g, cell, workdir, fp, time.time()))
            load[g] += 1
        time.sleep(2)
    return done


def fingerprint(workdir, fp):
    d = json.load(open(fp))
    d['artifacts'] = read_artifacts(os.path.join(workdir, 'results', 'g'))
    return d


def select(cells, only):
    return [c for c in cells if not only or re.search(only, c[0])]


def cmd_record(only):
    os.makedirs(STORE, exist_ok=True)
    done = run_cells(select(CELLS, only))
    bad = [c for c, v in done.items() if not v[0]]
    for cid, (ok, workdir, fp, _) in done.items():
        if ok:
            json.dump(fingerprint(workdir, fp), open(os.path.join(STORE, safe(cid) + '.json'), 'w'))
    print(f'recorded {len(done) - len(bad)} cells; FAILED: {bad}')
    return 1 if bad else 0


def cmd_check(only):
    done = run_cells(select(CELLS, only))
    fails = {}
    for cid, (ok, workdir, fp, _) in done.items():
        gpath = os.path.join(STORE, safe(cid) + '.json')
        if not ok:
            fails[cid] = [f'run failed, see {workdir}/log.txt']
            continue
        if not os.path.isfile(gpath):
            fails[cid] = ['no golden recorded']
            continue
        errs = compare(json.load(open(gpath)), fingerprint(workdir, fp))
        if errs:
            fails[cid] = errs
    for cid, errs in fails.items():
        print(f'FAIL {cid}: {len(errs)} mismatches')
        for e in errs[:8]:
            print('     ', e)
    print(f'\n{len(done) - len(fails)}/{len(done)} cells match' + ('' if not fails else f'; FAILED: {list(fails)}'))
    return 1 if fails else 0


# ─── data generation fingerprints (CPU, in-process) ────────────────────────


def data_fingerprints():
    import hashlib
    import numpy as np
    sys.path.insert(0, ROOT)
    from src.pipeline import data
    from src.pipeline.runner import _set_seed
    out = {}
    for cid, kw in DATA_CELLS:
        kw = dict(kw)
        Cc, k = kw.pop('C'), kw.pop('k')
        _set_seed(SEED)
        loaders, pl_ds, orig = data.load_experiment_data(Cc, k, DATA_DIR, SEED, os.path.join(SCRATCH, 'datalogs'),
                                                         64, **kw)
        h = hashlib.sha256()
        for t in pl_ds.targets:
            h.update(np.asarray(t, dtype=np.int64).tobytes())
            h.update(b'|')
        h.update(np.asarray(orig, dtype=np.int64).tobytes())
        rec = {'targets_sha': h.hexdigest(), 'n': len(pl_ds.targets), 'loaders': sorted(loaders)}
        if 'train_idx' in loaders:
            rec['train_idx_sha'] = hashlib.sha256(np.asarray(loaders['train_idx'], dtype=np.int64).tobytes()).hexdigest()
        # first batch of the shuffled training loader (RNG order of loader construction)
        key = 'pico' if 'pico' in loaders else 'pl'
        b = next(iter(loaders[key]))
        rec['first_batch_sum'] = float(sum(float(x.double().sum()) for x in b if hasattr(x, 'double')))
        out[cid] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['record', 'check', 'list', 'data-record', 'data-check'])
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    if a.cmd == 'list':
        for c in select(CELLS, a.only):
            print(c[0], c[1], ' '.join(c[2]), c[3])
        return 0
    if a.cmd == 'record':
        return cmd_record(a.only)
    if a.cmd == 'check':
        return cmd_check(a.only)
    fps = data_fingerprints()
    path = os.path.join(STORE, '_data.json')
    if a.cmd == 'data-record':
        os.makedirs(STORE, exist_ok=True)
        json.dump(fps, open(path, 'w'), indent=1)
        print(f'recorded {len(fps)} data cells')
        return 0
    errs = compare(json.load(open(path)), fps)
    print('\n'.join(errs) if errs else f'data: {len(fps)}/{len(fps)} match')
    return 1 if errs else 0


if __name__ == '__main__':
    sys.exit(main())
