#!/usr/bin/env python
"""cls-loss solo experiment orchestrator (2026-10-12).

Every cls loss of the swap experiment trained ALONE (no contrastive branch):
11 losses x k in {1,3,5,10,15,19} x 3 seeds, C=20, Adam 3e-4, 200 epochs,
bs 512 -> 198 cells. See src/clsloss_solo.py for what each cell logs.

    python scripts/run_clsloss_solo.py run    --run_name clsloss_solo_1012 --gpus 0 1 2 3 --slots_per_gpu 4
    python scripts/run_clsloss_solo.py status --run_name clsloss_solo_1012
    python scripts/run_clsloss_solo.py report --run_name clsloss_solo_1012
    (--C 10 --k_values 1 2 3 5 7 9 for the C=10 run; --C defaults to 20)

Results: results/<run_name>/cells/<alg>__k<k>_s<seed>.json (one per finished
cell: final/best acc, best epoch, time), results.csv (merged), report.csv,
progress.json, detail/<alg>/C20_k<k>/{dynamics,biasvariance}_s<seed>.csv.
Per-cell logs: logs/<run_name>/<alg>__C20_k<k>_s<seed>.log
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.clsloss_solo import SOLO_LOSSES, solo_algorithm_name  # noqa: E402

DATASET = 'cifar100-subset'
C = 20
K_VALUES = [1, 3, 5, 10, 15, 19]
SEEDS = [42, 43, 44]
RESULT_FIELDS = ['dataset', 'total_classes', 'k', 'algorithm', 'loss', 'seed', 'final_accuracy',
                 'best_accuracy', 'best_epoch', 'epochs', 'training_time_s', 'timestamp', 'git_commit']


def cell_file(results_dir, loss, k, seed):
    return os.path.join(results_dir, 'cells', f'{solo_algorithm_name(loss)}__k{k}_s{seed}.json')


def build_cells(seeds, losses, k_values):
    return [dict(seed=s, loss=l, k=k, algorithm=solo_algorithm_name(l))
            for s in seeds for l in losses for k in k_values]


def cell_label(c):
    return f"{c['algorithm']} k={c['k']} seed={c['seed']}"


def is_done(results_dir, c):
    return os.path.isfile(cell_file(results_dir, c['loss'], c['k'], c['seed']))


def _fmt_dur(s):
    s = max(0.0, float(s))
    if s < 90:
        return f'{s:.0f}s'
    if s < 3600:
        return f'{s / 60:.1f}min'
    return f'{s / 3600:.2f}h'


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return ''


# ─── cell (one training run, in its own process) ───────────────────────────


def cmd_cell(args):
    import random

    import numpy as np
    import torch
    import yaml

    from src.clsloss_solo import run_solo
    from src.pipeline import data
    from src.pipeline.algorithms.hparams import _ADAM, make_optimizer

    def set_seed(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

    with open('config.yaml') as f:
        raw = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join('results', args.run_name)
    alg = solo_algorithm_name(args.loss)

    set_seed(args.seed)
    loaders, pl_ds, orig_targets = data.load_experiment_data(C, args.k, './data', args.seed,
                                                             'logs/cifar100_subset', args.batch_size)
    set_seed(args.seed)
    t0 = time.perf_counter()
    final_acc, best_acc, best_ep = run_solo(
        args.loss, pl_ds=pl_ds, orig_targets=orig_targets, test_loader=loaders['test'], C=C, k=args.k,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, raw_cfg=raw,
        make_opt=lambda m: make_optimizer(m, _ADAM),
        out_dir=os.path.join(results_dir, 'detail', alg, f'C{C}_k{args.k}'), device=device,
        bv_every=args.bv_every, bv_m=args.bv_m, bv_eval=args.bv_eval,
        log=lambda s: print(s, flush=True))
    rec = dict(dataset=DATASET, total_classes=C, k=args.k, algorithm=alg, loss=args.loss, seed=args.seed,
               final_accuracy=final_acc, best_accuracy=best_acc, best_epoch=best_ep, epochs=args.epochs,
               training_time_s=round(time.perf_counter() - t0, 1), timestamp=datetime.now().isoformat(),
               git_commit=_git_commit())
    path = cell_file(results_dir, args.loss, args.k, args.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(rec, f)
    os.replace(path + '.tmp', path)
    print(f'DONE {alg} k={args.k} seed={args.seed}  final {final_acc:.2f}  best {best_acc:.2f}@{best_ep}',
          flush=True)


# ─── run (scheduler) ───────────────────────────────────────────────────────


def merge(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, 'cells', '*.json'))):
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except (OSError, ValueError):
            continue
    out = os.path.join(results_dir, 'results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


def cmd_run(args):
    results_dir = os.path.join('results', args.run_name)
    log_dir = os.path.join('logs', args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'run_config.json'), 'w') as f:
        json.dump(dict(run_name=args.run_name, losses=args.losses, k_values=args.k_values, seeds=args.seeds,
                       C=C, epochs=args.epochs, batch_size=args.batch_size, optimizer='adam lr 3e-4 wd 1e-4',
                       bv_every=args.bv_every, bv_m=args.bv_m, bv_eval=args.bv_eval,
                       gpus=args.gpus, slots_per_gpu=args.slots_per_gpu, git_commit=_git_commit()), f, indent=2)

    cells = build_cells(args.seeds, args.losses, args.k_values)
    pending = deque(c for c in cells if not is_done(results_dir, c))
    if args.limit is not None:
        pending = deque(list(pending)[:args.limit])
    n_done0 = len(cells) - len(pending)
    print(f'run_name={args.run_name}  cells={len(cells)}  done={n_done0}  pending={len(pending)}\n'
          f'gpus={args.gpus} x{args.slots_per_gpu}  seeds={args.seeds}  k={args.k_values}  '
          f'epochs={args.epochs}', flush=True)
    if args.dry_run:
        for c in list(pending)[:10]:
            print('   ', cell_label(c))
        return

    slots = [dict(idx=i, gpu=g, proc=None, cell=None, t0=None, logf=None, log=None)
             for i, g in enumerate(g for g in args.gpus for _ in range(args.slots_per_gpu))]
    retries, failed, durations = {}, [], []
    session_t0, last_print = time.time(), 0.0

    def launch(slot, cell):
        log_path = os.path.join(log_dir, f"{cell['algorithm']}__C{C}_k{cell['k']}_s{cell['seed']}.log")
        cmd = [sys.executable, os.path.abspath(__file__), 'cell', '--run_name', args.run_name, '--C', str(C),
               '--loss', cell['loss'], '--k', str(cell['k']), '--seed', str(cell['seed']),
               '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
               '--bv_every', str(args.bv_every), '--bv_m', str(args.bv_m), '--bv_eval', str(args.bv_eval)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot['gpu']), PYTHONUNBUFFERED='1')
        logf = open(log_path, 'a')
        logf.write(f'\n===== {datetime.now().isoformat()}  {" ".join(cmd)}\n')
        logf.flush()
        slot.update(proc=subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT),
                    cell=cell, t0=time.time(), logf=logf, log=log_path)

    def finish(slot, rc):
        cell, dur = slot['cell'], time.time() - slot['t0']
        slot['logf'].close()
        if rc == 0 and is_done(results_dir, cell):
            durations.append(dur)
            print(f'  [done] {cell_label(cell)}  ({_fmt_dur(dur)})', flush=True)
        else:
            key = cell_label(cell)
            retries[key] = retries.get(key, 0) + 1
            if retries[key] <= args.retries:
                print(f'  [retry {retries[key]}] {key}  rc={rc}  log={slot["log"]}', flush=True)
                pending.appendleft(cell)
            else:
                print(f'  [FAILED] {key}  rc={rc}  log={slot["log"]}', flush=True)
                failed.append(dict(algorithm=cell['algorithm'], k=cell['k'], seed=cell['seed'],
                                   why=f'rc={rc}', log=slot['log']))
        slot.update(proc=None, cell=None, t0=None, logf=None, log=None)

    def write_progress():
        running = [dict(gpu=s['gpu'], slot=s['idx'], algorithm=s['cell']['algorithm'], k=s['cell']['k'],
                        seed=s['cell']['seed'], elapsed_s=round(time.time() - s['t0']), log=s['log'])
                   for s in slots if s['proc'] is not None]
        n_done = sum(is_done(results_dir, c) for c in cells)
        avg = statistics.mean(durations) if durations else None
        remaining = len(pending) + len(running)
        eta = avg * remaining / len(slots) if avg else None
        prog = dict(updated=datetime.now().isoformat(), run_name=args.run_name, pid=os.getpid(),
                    total=len(cells), done=n_done, pending=len(pending), running=running, failed=failed,
                    session_elapsed_s=round(time.time() - session_t0), eta_s=round(eta) if eta else None)
        with open(os.path.join(results_dir, 'progress.json'), 'w') as f:
            json.dump(prog, f, indent=2)
        return prog

    try:
        while pending or any(s['proc'] for s in slots):
            for s in slots:
                if s['proc'] is not None and (rc := s['proc'].poll()) is not None:
                    finish(s, rc)
                if s['proc'] is None and pending:
                    launch(s, pending.popleft())
            prog = write_progress()
            if time.time() - last_print >= args.poll:
                eta = _fmt_dur(prog['eta_s']) if prog['eta_s'] else '?'
                print(f"[{datetime.now():%H:%M:%S}] done {prog['done']}/{prog['total']}  pending "
                      f"{prog['pending']}  running {len(prog['running'])}  failed {len(failed)}  ETA ~{eta}",
                      flush=True)
                last_print = time.time()
                merge(results_dir)
            time.sleep(10)
    except KeyboardInterrupt:
        print('\nInterrupted -- terminating running cells (re-run the same command to resume)...', flush=True)
        for s in slots:
            if s['proc'] is not None:
                s['proc'].terminate()
        raise
    write_progress()
    print(f'All cells processed: failed {len(failed)}, session time {_fmt_dur(time.time() - session_t0)}',
          flush=True)
    print(f'Merged -> {merge(results_dir)}')
    cmd_report(args)


# ─── status / report ───────────────────────────────────────────────────────


def cmd_status(args):
    results_dir = os.path.join('results', args.run_name)
    cells = build_cells(args.seeds, args.losses, args.k_values)
    print(f"run_name={args.run_name}  done {sum(is_done(results_dir, c) for c in cells)}/{len(cells)}")
    path = os.path.join(results_dir, 'progress.json')
    if not os.path.isfile(path):
        print('(no progress.json yet)')
        return
    with open(path) as f:
        prog = json.load(f)
    age = time.time() - datetime.fromisoformat(prog['updated']).timestamp()
    print(f"orchestrator pid {prog['pid']}, heartbeat {_fmt_dur(age)} ago"
          f"{'  (STALE)' if age > 120 else ''}, ETA ~{_fmt_dur(prog['eta_s'] or 0)}")
    for r in prog['running']:
        print(f"  running gpu{r['gpu']}: {r['algorithm']} k={r['k']} s{r['seed']}  {_fmt_dur(r['elapsed_s'])}")
    for f in prog['failed']:
        print(f"  FAILED  {f['algorithm']} k={f['k']} s{f['seed']}: {f['why']}  ({f['log']})")


def cmd_report(args):
    """rows = loss, cols = k; final (last-epoch) and best-epoch acc, mean±std over seeds."""
    results_dir = os.path.join('results', args.run_name)
    merge(results_dir)
    with open(os.path.join(results_dir, 'results.csv')) as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault((r['loss'], int(r['k'])), []).append(r)

    def cell(rs, field):
        v = [float(r[field]) for r in rs]
        if not v:
            return ''
        s = f'{statistics.mean(v):.2f}' + (f'±{statistics.stdev(v):.2f}' if len(v) > 1 else '')
        return s + ('' if len(v) == len(args.seeds) else f' (n={len(v)})')

    table = []
    for metric, field in (('final', 'final_accuracy'), ('best', 'best_accuracy')):
        for loss in args.losses:
            rec = dict(metric=metric, loss=loss)
            for k in args.k_values:
                rec[f'k{k}'] = cell(by.get((loss, k), []), field)
            table.append(rec)
    cols = ['metric', 'loss'] + [f'k{k}' for k in args.k_values]
    widths = {c: max(len(c), *(len(str(r[c])) for r in table)) for c in cols}
    print('  '.join(c.ljust(widths[c]) for c in cols))
    for r in table:
        print('  '.join(str(r[c]).ljust(widths[c]) for c in cols))
    out = os.path.join(results_dir, 'report.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f'-> {out}')


def main():
    global C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'status', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--run_name', required=True)
        p.add_argument('--C', type=int, default=C)
        p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
        p.add_argument('--k_values', nargs='+', type=int, default=K_VALUES)
        p.add_argument('--losses', nargs='+', default=SOLO_LOSSES, choices=SOLO_LOSSES)
        if name == 'run':
            p.add_argument('--gpus', nargs='+', type=int, required=True)
            p.add_argument('--slots_per_gpu', type=int, default=4)
            p.add_argument('--epochs', type=int, default=200)
            p.add_argument('--batch_size', type=int, default=512)
            p.add_argument('--bv_every', type=int, default=20)
            p.add_argument('--bv_m', type=int, default=50)
            p.add_argument('--bv_eval', type=int, default=256)
            p.add_argument('--retries', type=int, default=1)
            p.add_argument('--poll', type=float, default=300)
            p.add_argument('--limit', type=int, default=None)
            p.add_argument('--dry_run', action='store_true')
    p = sub.add_parser('cell')
    p.add_argument('--run_name', required=True)
    p.add_argument('--C', type=int, default=C)
    p.add_argument('--loss', required=True, choices=SOLO_LOSSES)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--bv_every', type=int, default=20)
    p.add_argument('--bv_m', type=int, default=50)
    p.add_argument('--bv_eval', type=int, default=256)
    args = ap.parse_args()
    C = args.C
    {'run': cmd_run, 'status': cmd_status, 'report': cmd_report, 'cell': cmd_cell}[args.cmd](args)


if __name__ == '__main__':
    main()
