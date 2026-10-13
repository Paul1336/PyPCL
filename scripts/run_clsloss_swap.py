#!/usr/bin/env python
"""cls-loss swap experiment orchestrator (2026-10-06).

Every distinct cls loss used so far (src/clsloss_swap.py CLS_LOSSES, 12 of
them) plugged into the PiCO-Fixed and ComCo-Fixed architectures, only L_cls
changed:  2 frameworks x 12 losses x k in {1,3,5,10,15,19} x 3 seeds, C=20,
Adam 3e-4, 200 epochs, bs 512  ->  432 cells.

Cells are handed out dynamically (seed-major, so a complete seed lands first)
to --slots_per_gpu processes on each of --gpus; same scheduling / resume /
shard machinery as scripts/run_ab_sweep.py.

    python scripts/run_clsloss_swap.py run    --run_name clsloss_swap_1006 --gpus 0 1 2 3 --slots_per_gpu 4
    python scripts/run_clsloss_swap.py status --run_name clsloss_swap_1006
    python scripts/run_clsloss_swap.py report --run_name clsloss_swap_1006

Results: results/<run_name>/ (shards/, results.csv, progress.json,
report.csv). Per-cell logs: logs/<run_name>/<algorithm>__C20_k<k>_s<seed>.log
"""

import argparse
import csv
import glob
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.clsloss_swap import CLS_LOSSES, FRAMEWORKS, swap_algorithm_name  # noqa: E402
from src.pipeline import results as results_mod  # noqa: E402
from src.pipeline.algorithms import ALL_ALGORITHM_NAMES  # noqa: E402

DATASET = 'cifar100-subset'
C = 20
K_VALUES = [1, 3, 5, 10, 15, 19]
SEEDS = [42, 43, 44]
PROGRESS_FILE = 'progress.json'
REPORT_FILE = 'report.csv'


def build_cells(seeds, frameworks, losses, k_values):
    cells = []
    for seed in seeds:
        for fw in frameworks:
            for loss in losses:
                for k in k_values:
                    alg = swap_algorithm_name(fw, loss)
                    cells.append(dict(seed=seed, fw=fw, loss=loss, k=k, algorithm=alg,
                                      key=(DATASET, C, k, alg, seed)))
    unknown = sorted({c['algorithm'] for c in cells} - set(ALL_ALGORITHM_NAMES))
    if unknown:
        raise SystemExit(f'algorithm names missing from the registry: {unknown}')
    return cells


def cell_label(c):
    return f"{c['algorithm']} k={c['k']} seed={c['seed']}"


def load_done(results_dir):
    done = set()
    for path in sorted(glob.glob(os.path.join(results_dir, 'shards', 'worker*.csv'))):
        done |= results_mod.load_done(path)
    return done


def read_rows(results_dir):
    """key -> row (last write wins), tolerating a torn row mid-append."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'shards', 'worker*.csv'))):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if None in row or any(v is None for v in row.values()):
                    continue
                try:
                    key = (row.get('dataset') or DATASET, int(row['total_classes']), int(row['k']),
                           row['algorithm'], int(row['seed']))
                    row['final_accuracy'] = float(row['final_accuracy'])
                    row['training_time_s'] = float(row.get('training_time_s') or 0.0)
                except (ValueError, KeyError):
                    continue
                rows[key] = row
    return rows


def _fmt_dur(s):
    s = max(0.0, float(s))
    if s < 90:
        return f'{s:.0f}s'
    if s < 3600:
        return f'{s / 60:.1f}min'
    return f'{s / 3600:.2f}h'


def _cells_from_args(args):
    return build_cells(args.seeds, args.frameworks, args.losses, args.k_values)


# ─── run ───────────────────────────────────────────────────────────────────


def cmd_run(args):
    results_dir = os.path.join('results', args.run_name)
    log_dir = os.path.join('logs', args.run_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cells = _cells_from_args(args)
    done = load_done(results_dir)
    pending = deque(c for c in cells if c['key'] not in done)
    if args.limit is not None:
        pending = deque(list(pending)[:args.limit])
    n_slots = len(args.gpus) * args.slots_per_gpu
    print(f'run_name={args.run_name}  cells={len(cells)}  done={len(cells) - len(pending)}  '
          f'pending={len(pending)}\ngpus={args.gpus} x{args.slots_per_gpu}  seeds={args.seeds}  '
          f'k={args.k_values}  epochs={args.epochs}', flush=True)
    if args.dry_run:
        for c in list(pending)[:10]:
            print('   ', cell_label(c))
        return
    if not pending:
        _finish_up(results_dir, args)
        return

    slots = [dict(idx=i, gpu=g, proc=None, cell=None, t0=None, logf=None, log=None)
             for i, g in enumerate(g for g in args.gpus for _ in range(args.slots_per_gpu))]
    retries, failed, finished = {}, [], []
    session_t0, last_print = time.time(), 0.0

    def launch(slot, cell):
        alg, k, seed = cell['algorithm'], cell['k'], cell['seed']
        log_path = os.path.join(log_dir, f'{alg}__C{C}_k{k}_s{seed}.log')
        cmd = [sys.executable, os.path.join('scripts', 'run_pipeline.py'), 'run',
               '--run_name', args.run_name, '--algorithms', alg, '--algo', alg,
               '--c_values', str(C), '--only_k', str(k), '--seeds', str(seed),
               '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
               '--report_every', str(args.report_every),
               '--gpu_id', str(slot['idx']), '--num_gpus', str(n_slots)]
        if args.diag:
            cmd += ['--biasvariance', '--biasvariance_log_every', '20',
                    '--biasvariance_m_resamples', '50', '--biasvariance_eval_size', '256']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot['gpu']), PYTHONUNBUFFERED='1')
        logf = open(log_path, 'a')
        logf.write(f'\n===== {datetime.now().isoformat()}  {" ".join(cmd)}\n')
        logf.flush()
        slot.update(proc=subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT),
                    cell=cell, t0=time.time(), logf=logf, log=log_path)

    def finish(slot, rc):
        cell, dur = slot['cell'], time.time() - slot['t0']
        slot['logf'].close()
        recorded = cell['key'] in results_mod.load_done(results_mod.shard_path(results_dir, slot['idx']))
        if rc == 0 and recorded:
            done.add(cell['key'])
            finished.append((cell, dur))
            print(f'  [done] {cell_label(cell)}  ({_fmt_dur(dur)})', flush=True)
        else:
            n = retries.get(cell['key'], 0)
            why = f'exit code {rc}' if rc != 0 else 'exited 0 but no result row was written'
            if n < args.retries:
                retries[cell['key']] = n + 1
                pending.appendleft(cell)
                print(f"  [retry {n + 1}/{args.retries}] {cell_label(cell)}: {why} -- see {slot['log']}", flush=True)
            else:
                failed.append(dict(cell=cell, log=slot['log'], why=why))
                print(f"  [FAILED] {cell_label(cell)}: {why} -- see {slot['log']}", flush=True)
        slot.update(proc=None, cell=None, t0=None, logf=None, log=None)

    def est_seconds(cell):
        same = [d for c, d in finished if c['fw'] == cell['fw']] or [d for _, d in finished]
        return statistics.mean(same) if same else 3200.0 * args.epochs / 200.0

    def write_progress():
        running = [dict(gpu=s['gpu'], slot=s['idx'], algorithm=s['cell']['algorithm'], k=s['cell']['k'],
                        seed=s['cell']['seed'], elapsed_s=round(time.time() - s['t0']), log=s['log'])
                   for s in slots if s['proc'] is not None]
        run_left = max([max(0.0, est_seconds(s['cell']) - (time.time() - s['t0']))
                        for s in slots if s['proc'] is not None] or [0.0])
        eta = sum(est_seconds(c) for c in pending) / max(n_slots, 1) + run_left
        prog = dict(updated=datetime.now().isoformat(), run_name=args.run_name, pid=os.getpid(),
                    total=len(cells), done=len(done & {c['key'] for c in cells}), pending=len(pending),
                    running=running, failed=[dict(algorithm=f['cell']['algorithm'], k=f['cell']['k'],
                                                  seed=f['cell']['seed'], why=f['why'], log=f['log'])
                                             for f in failed],
                    session_elapsed_s=round(time.time() - session_t0), eta_s=round(eta))
        tmp = os.path.join(results_dir, PROGRESS_FILE + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(prog, f, indent=2)
        os.replace(tmp, os.path.join(results_dir, PROGRESS_FILE))
        return prog

    def _sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        while pending or any(s['proc'] is not None for s in slots):
            for s in slots:
                if s['proc'] is not None and s['proc'].poll() is not None:
                    finish(s, s['proc'].returncode)
                if s['proc'] is None and pending:
                    launch(s, pending.popleft())
            prog = write_progress()
            if time.time() - last_print >= args.poll:
                last_print = time.time()
                print(f"[{datetime.now():%H:%M:%S}] done {prog['done']}/{prog['total']}  pending "
                      f"{prog['pending']}  running {len(prog['running'])}  failed {len(failed)}  "
                      f"ETA ~{_fmt_dur(prog['eta_s'])}", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nInterrupted -- terminating running cells (re-run the same command to resume)...', flush=True)
        for s in slots:
            if s['proc'] is not None:
                s['proc'].terminate()
        deadline = time.time() + 15
        for s in slots:
            if s['proc'] is not None:
                try:
                    s['proc'].wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    s['proc'].kill()
                s['logf'].close()
        write_progress()
        sys.exit(130)

    write_progress()
    print(f'\nAll cells processed: failed {len(failed)}, session time {_fmt_dur(time.time() - session_t0)}')
    for f in failed:
        print(f"  FAILED {cell_label(f['cell'])}: {f['why']} -- {f['log']}")
    _finish_up(results_dir, args)


def _finish_up(results_dir, args):
    print(f'Merged -> {results_mod.merge_shards(results_dir)}')
    cmd_report(args)


# ─── status / report ───────────────────────────────────────────────────────


def cmd_status(args):
    results_dir = os.path.join('results', args.run_name)
    cells = _cells_from_args(args)
    done = load_done(results_dir)
    print(f"run_name={args.run_name}  done {sum(c['key'] in done for c in cells)}/{len(cells)}")
    for seed in args.seeds:
        sub = [c for c in cells if c['seed'] == seed]
        print(f"  seed {seed}: " + '  '.join(
            f"{fw} {sum(c['key'] in done for c in sub if c['fw'] == fw)}/{sum(c['fw'] == fw for c in sub)}"
            for fw in args.frameworks))
    path = os.path.join(results_dir, PROGRESS_FILE)
    if not os.path.isfile(path):
        print('(no progress.json yet)')
        return
    with open(path) as f:
        prog = json.load(f)
    age = time.time() - datetime.fromisoformat(prog['updated']).timestamp()
    print(f"orchestrator pid {prog['pid']}, heartbeat {_fmt_dur(age)} ago"
          f"{'  (STALE)' if age > 120 else ''}, ETA ~{_fmt_dur(prog['eta_s'])}")
    for r in prog['running']:
        print(f"  running gpu{r['gpu']}: {r['algorithm']} k={r['k']} s{r['seed']}  {_fmt_dur(r['elapsed_s'])}")
    for f in prog['failed']:
        print(f"  FAILED  {f['algorithm']} k={f['k']} s{f['seed']}: {f['why']}  ({f['log']})")


def cmd_report(args):
    """Pivot: rows = framework/loss, cols = k, cell = mean ± std over seeds (n)."""
    results_dir = os.path.join('results', args.run_name)
    rows = read_rows(results_dir)
    table = []
    for fw in args.frameworks:
        for loss in args.losses:
            alg = swap_algorithm_name(fw, loss)
            rec = dict(framework=fw, loss=loss)
            for k in args.k_values:
                accs = [rows[key]['final_accuracy'] for s in args.seeds
                        if (key := (DATASET, C, k, alg, s)) in rows]
                if not accs:
                    rec[f'k{k}'] = ''
                elif len(accs) == 1:
                    rec[f'k{k}'] = f'{accs[0]:.2f} (n=1)'
                else:
                    rec[f'k{k}'] = f'{statistics.mean(accs):.2f}±{statistics.stdev(accs):.2f}' + \
                                   ('' if len(accs) == len(args.seeds) else f' (n={len(accs)})')
            table.append(rec)
    cols = ['framework', 'loss'] + [f'k{k}' for k in args.k_values]
    widths = {c: max(len(c), *(len(str(r[c])) for r in table)) for c in cols}
    print('  '.join(c.ljust(widths[c]) for c in cols))
    for r in table:
        print('  '.join(str(r[c]).ljust(widths[c]) for c in cols))
    out = os.path.join(results_dir, REPORT_FILE)
    os.makedirs(results_dir, exist_ok=True)
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
        p.add_argument('--frameworks', nargs='+', default=FRAMEWORKS, choices=FRAMEWORKS)
        p.add_argument('--losses', nargs='+', default=CLS_LOSSES, choices=CLS_LOSSES)
        if name == 'run':
            p.add_argument('--gpus', nargs='+', type=int, required=True)
            p.add_argument('--slots_per_gpu', type=int, default=4)
            p.add_argument('--epochs', type=int, default=200)
            p.add_argument('--batch_size', type=int, default=512)
            p.add_argument('--report_every', type=int, default=10)
            p.add_argument('--retries', type=int, default=1)
            p.add_argument('--poll', type=float, default=300)
            p.add_argument('--limit', type=int, default=None)
            p.add_argument('--dry_run', action='store_true')
            p.add_argument('--diag', action='store_true',
                           help='also log the cls-loss dynamics/bias-variance CSVs (src/clsloss_solo.py) '
                                'under results/<run>/detail/')
    args = ap.parse_args()
    C = args.C
    {'run': cmd_run, 'status': cmd_status, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
