#!/usr/bin/env python
"""Confidence-update-MECHANISM sweep orchestrator (2026-09-16).

Follow-up to scripts/run_ema_sweep.py: that sweep matched the EMA momentum
*coefficient* between PiCO-Fixed and PRODEN, but PiCO-Fixed's W/N
init-sensitivity curves did not converge toward PRODEN's -- so this isolates
two further, EMA-coefficient-independent differences in PiCO-Fixed's
confidence update (see src/pico/utils_loss.py PartialLoss.confidence_update,
src/fixed_pico_engine.py train_pico_epoch_fixed, and src/pll_init.py's
AB_VARIANT_SOURCE_HARD docstring for the full rationale):

  Factor A (conf_source): 'prototype' (native) -- confidence driven by
      score_prot (embedding-vs-class-prototype similarity) -- vs
      'classifier' (A') -- confidence instead reuses the classifier's own
      candidate-masked softmax, the same kind of signal PRODEN's own update
      uses.
  Factor B (conf_hard): True (native) -- update target is one-hotted before
      EMA-blending -- vs False (B') -- the full masked/renormalized
      distribution is blended in instead (PRODEN-style soft update).

Re-runs the SAME two init-sensitivity experiments as run_ema_sweep.py
  exp 'w' (TC-PLS W sweep):   C=20, k in {10,15,19}, W in {baseline,5.2,10,20}%
  exp 'n' (TC-n-PLS n sweep): C=20, k=5,  W=20%, n in {4,9,14,19}
on the three NEW mechanism combinations (A'+B, A+B', A'+B' -- see
AB_SWEEP_BASES below), at three EMA levels only (highest/middle/lowest of
the original five: EMA100, EMA050, EMA000, per user request). The fourth
combination, A+B (native PiCO-Fixed), is NOT retrained here -- it's exactly
'PiCO-Fixed-...-EMA100/050/000' from run_ema_sweep.py's own run
(ema_sweep_0915), already fully recorded; pull it in as a fourth reference
column at report/plot time via `--reference_run ema_sweep_0915`.

Cells are handed out dynamically to the given GPUs, SEED-MAJOR (see
scripts/run_ema_sweep.py's docstring) -- identical scheduling/resume/shard
machinery, just against a different cell list.

    # launch (resumable: already-recorded cells are skipped)
    python scripts/run_ab_sweep.py run --run_name ab_sweep_0915 --gpus 0 1 2 3
    # progress, any time, from another terminal
    python scripts/run_ab_sweep.py status --run_name ab_sweep_0915
    # partial results so far (pivot: rows = exp/k/init, cols = AB-variant x EMA)
    python scripts/run_ab_sweep.py report --run_name ab_sweep_0915

Results: results/<run_name>/ (shards/, results.csv, ab_sweep_progress.json,
ab_sweep_report.csv). Per-cell logs: logs/<run_name>/<algorithm>__C20_k<k>_s<seed>.log
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

from src.pipeline import results as results_mod  # noqa: E402
from src.pipeline.algorithms import ALL_ALGORITHM_NAMES  # noqa: E402
from src.pll_init import (AB_SWEEP_BASES, AB_SWEEP_SCALES, biased_rand_all_variant_name,  # noqa: E402
                           biased_variant_name, ema_tag, ema_variant_name, weight_pct_str)

DATASET = 'cifar100-subset'
C = 20
EXP_W_K_VALUES = [10, 15, 19]
EXP_W_WEIGHTS = [0.052, 0.10, 0.20]
EXP_N_K_VALUES = [5, 10, 15, 19]  # extended 2026-09-17 to match exp 'w''s k grid, per user request
EXP_N_WEIGHT = 0.20
EXP_N_VALUES = [4, 9, 14, 19]
EXPERIMENTS = ('w', 'n')
REFERENCE_BASE = 'PiCO-Fixed'   # A+B, the native combo -- pulled from --reference_run, never trained here

PROGRESS_FILE = 'ab_sweep_progress.json'
REPORT_FILE = 'ab_sweep_report.csv'


# ─── cell enumeration ──────────────────────────────────────────────────────


def build_cells(seeds, experiments=EXPERIMENTS, bases=AB_SWEEP_BASES, scales=AB_SWEEP_SCALES,
                 include_baseline=True):
    """Seed-major list of cells; within a seed, exp 'w' (heavier, k up to 19)
    before exp 'n'."""
    cells = []
    for seed in seeds:
        for base in bases:
            if 'w' in experiments:
                for k in EXP_W_K_VALUES:
                    for w in ([None] if include_baseline else []) + EXP_W_WEIGHTS:
                        base_name = base if w is None else biased_variant_name(base, 'cand', w)
                        init = 'baseline' if w is None else f'W{weight_pct_str(w)}'
                        for scale in scales:
                            cells.append(_cell('w', seed, base, init, k, scale, base_name))
            if 'n' in experiments:
                for k in EXP_N_K_VALUES:
                    for n in EXP_N_VALUES:
                        base_name = biased_rand_all_variant_name(base, EXP_N_WEIGHT, n)
                        init = f'W{weight_pct_str(EXP_N_WEIGHT)}-N{n}'
                        for scale in scales:
                            cells.append(_cell('n', seed, base, init, k, scale, base_name))
    unknown = sorted({c['algorithm'] for c in cells} - set(ALL_ALGORITHM_NAMES))
    if unknown:
        raise SystemExit(f'algorithm names missing from the registry (src/pipeline/algorithms): {unknown}')
    return cells


def _cell(exp, seed, base, init, k, scale, base_name):
    algorithm = ema_variant_name(base_name, scale)
    return dict(exp=exp, seed=seed, base=base, init=init, k=k, scale=scale, ema=ema_tag(scale),
                algorithm=algorithm, key=(DATASET, C, k, algorithm, seed))


def cell_label(c):
    return f"{c['algorithm']} k={c['k']} seed={c['seed']}"


# ─── reading results (read-only, safe while workers are writing) ───────────


def results_dir_of(run_name):
    return os.path.join('results', run_name)


def load_done(results_dir):
    done = set()
    for path in sorted(glob.glob(os.path.join(results_dir, 'shards', 'worker*.csv'))):
        done |= results_mod.load_done(path)
    return done


def read_rows(results_dir):
    """key -> shard row (last write wins), tolerating a torn row a worker is
    mid-append on -- mirrors results.merge_shards without writing
    results.csv."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'shards', 'worker*.csv'))):
        if not os.path.isfile(path):
            continue
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


# ─── run ───────────────────────────────────────────────────────────────────


def _fmt_dur(s):
    s = max(0.0, float(s))
    if s < 90:
        return f'{s:.0f}s'
    if s < 3600:
        return f'{s / 60:.1f}min'
    return f'{s / 3600:.2f}h'


def cmd_run(args):
    run_name = args.run_name
    results_dir = results_dir_of(run_name)
    log_dir = os.path.join('logs', run_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cells = build_cells(args.seeds, args.experiments, args.bases, include_baseline=not args.no_baseline)
    done = load_done(results_dir)
    pending = deque(c for c in cells if c['key'] not in done)
    n_done, n_pending_total = len(cells) - len(pending), len(pending)
    if args.limit is not None:
        pending = deque(list(pending)[:args.limit])

    n_slots = len(args.gpus) * args.slots_per_gpu
    print(f'run_name={run_name}  cells={len(cells)}  already done={n_done}  pending={n_pending_total}'
          + (f'  (launching only the first {len(pending)}: --limit)' if args.limit is not None else '')
          + f'\ngpus={args.gpus} x{args.slots_per_gpu} slot(s)  seeds={args.seeds}  epochs={args.epochs}',
          flush=True)
    _print_breakdown(cells, done)

    if args.dry_run:
        print('\n--dry_run: first 10 pending cells in launch order:')
        for c in list(pending)[:10]:
            print('   ', cell_label(c))
        return

    if not pending:
        print('Nothing to do -- every cell is already recorded.')
        _finish_up(results_dir, args)
        return

    slots = [dict(idx=i, gpu=g, proc=None, cell=None, t0=None, logf=None, log=None)
             for i, g in enumerate(gpu for gpu in args.gpus for _ in range(args.slots_per_gpu))]
    retries = {}
    failed = []
    finished = []
    session_t0 = time.time()
    last_print = 0.0

    def launch(slot, cell):
        alg, k, seed = cell['algorithm'], cell['k'], cell['seed']
        log_path = os.path.join(log_dir, f"{alg}__C{C}_k{k}_s{seed}.log")
        cmd = [sys.executable, os.path.join('scripts', 'run_pipeline.py'), 'run',
               '--run_name', run_name, '--algorithms', alg, '--algo', alg,
               '--c_values', str(C), '--only_k', str(k), '--seeds', str(seed),
               '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
               '--report_every', str(args.report_every),
               '--gpu_id', str(slot['idx']), '--num_gpus', str(n_slots)]
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
            print(f"  [done] {cell_label(cell)}  ({_fmt_dur(dur)})", flush=True)
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

    def est_seconds(cell, shard_rows):
        same = [d for c, d in finished if c['base'] == cell['base']]
        if not same:
            same = [r['training_time_s'] + 45 for key, r in shard_rows.items()
                    if key[3].startswith(cell['base']) and r['training_time_s'] > 0]
        if same:
            return statistics.mean(same)
        return 900.0 * args.epochs / 200.0 + 45.0   # all bases here are PiCO-Fixed-derived (~15min/200ep)

    def write_progress():
        shard_rows = read_rows(results_dir)
        running = [dict(gpu=s['gpu'], slot=s['idx'], algorithm=s['cell']['algorithm'], k=s['cell']['k'],
                        seed=s['cell']['seed'], elapsed_s=round(time.time() - s['t0']), log=s['log'])
                   for s in slots if s['proc'] is not None]
        pend_secs = sum(est_seconds(c, shard_rows) for c in pending)
        run_left = max([max(0.0, est_seconds(s['cell'], shard_rows) - (time.time() - s['t0']))
                        for s in slots if s['proc'] is not None] or [0.0])
        eta = pend_secs / max(n_slots, 1) + run_left
        prog = dict(updated=datetime.now().isoformat(), run_name=run_name, pid=os.getpid(),
                    total=len(cells), done=len(done), pending=len(pending), running=running,
                    finished_this_session=len(finished), failed=[dict(algorithm=f['cell']['algorithm'],
                                                                      k=f['cell']['k'], seed=f['cell']['seed'],
                                                                      why=f['why'], log=f['log']) for f in failed],
                    session_elapsed_s=round(time.time() - session_t0), eta_s=round(eta),
                    seeds=args.seeds, gpus=args.gpus, epochs=args.epochs)
        tmp = os.path.join(results_dir, PROGRESS_FILE + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(prog, f, indent=2)
        os.replace(tmp, os.path.join(results_dir, PROGRESS_FILE))
        return prog

    import signal

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
                running = ', '.join(f"gpu{r['gpu']}: {r['algorithm']} k{r['k']} s{r['seed']} "
                                    f"({_fmt_dur(r['elapsed_s'])})" for r in prog['running']) or '-'
                print(f"[{datetime.now():%H:%M:%S}] done {prog['done']}/{prog['total']}  "
                      f"pending {prog['pending']}  failed {len(failed)}  ETA ~{_fmt_dur(prog['eta_s'])}\n"
                      f"    running: {running}", flush=True)
            time.sleep(min(args.poll, 5))
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
    print(f'\nAll cells processed: done {len(done)}/{len(cells)}, failed {len(failed)}, '
          f'session time {_fmt_dur(time.time() - session_t0)}')
    for f in failed:
        print(f"  FAILED {cell_label(f['cell'])}: {f['why']} -- {f['log']}")
    _finish_up(results_dir, args)


def _finish_up(results_dir, args):
    out = results_mod.merge_shards(results_dir)
    print(f'Merged -> {out}')
    _report(args.run_name, args.seeds, args.experiments, args.bases, not args.no_baseline,
            args.reference_run, out_path=None)


def _print_breakdown(cells, done):
    seeds = sorted({c['seed'] for c in cells})
    groups = sorted({(c['exp'], c['base']) for c in cells})
    head = f"{'seed':>6}  " + '  '.join(f"{exp}/{base:<28}" for exp, base in groups) + '     total'
    print(head)
    for seed in seeds:
        parts = []
        for exp, base in groups:
            sub = [c for c in cells if c['seed'] == seed and c['exp'] == exp and c['base'] == base]
            parts.append(f"{sum(c['key'] in done for c in sub):>4}/{len(sub):<25}")
        sub = [c for c in cells if c['seed'] == seed]
        print(f"{seed:>6}  " + '  '.join(parts) + f"   {sum(c['key'] in done for c in sub):>4}/{len(sub)}")


# ─── status ────────────────────────────────────────────────────────────────


def cmd_status(args):
    results_dir = results_dir_of(args.run_name)
    cells = build_cells(args.seeds, args.experiments, args.bases, include_baseline=not args.no_baseline)
    done = load_done(results_dir)
    print(f"run_name={args.run_name}  done {sum(c['key'] in done for c in cells)}/{len(cells)}")
    _print_breakdown(cells, done)

    path = os.path.join(results_dir, PROGRESS_FILE)
    if not os.path.isfile(path):
        print('\n(no ab_sweep_progress.json yet -- the `run` orchestrator has not started for this run_name)')
        return
    with open(path) as f:
        prog = json.load(f)
    age = time.time() - datetime.fromisoformat(prog['updated']).timestamp()
    print(f"\norchestrator pid {prog['pid']}, last heartbeat {_fmt_dur(age)} ago"
          f"{'  (STALE -- orchestrator probably not running)' if age > 120 else ''}, "
          f"ETA ~{_fmt_dur(prog['eta_s'])}, session elapsed {_fmt_dur(prog['session_elapsed_s'])}")
    for r in prog['running']:
        print(f"  running  gpu{r['gpu']}: {r['algorithm']} k={r['k']} seed={r['seed']}  "
              f"{_fmt_dur(r['elapsed_s'])}  ({r['log']})")
    for f in prog['failed']:
        print(f"  FAILED   {f['algorithm']} k={f['k']} seed={f['seed']}: {f['why']}  ({f['log']})")


# ─── report ────────────────────────────────────────────────────────────────


def _report(run_name, seeds, experiments, bases, include_baseline, reference_run, out_path):
    results_dir = results_dir_of(run_name)
    rows = read_rows(results_dir)
    cells = build_cells(seeds, experiments, bases, include_baseline=include_baseline)

    # Fold in the A+B (native PiCO-Fixed) reference cells from a separate
    # run (e.g. ema_sweep_0915), read the SAME way (shards, tolerant of
    # in-progress writes) -- as an extra pseudo-base column, EMA100/050/000
    # only (matching AB_SWEEP_SCALES).
    ref_rows = {}
    if reference_run:
        ref_rows = read_rows(results_dir_of(reference_run))
        for seed in seeds:
            for k in EXP_W_K_VALUES:
                for w in ([None] if include_baseline else []) + EXP_W_WEIGHTS:
                    base_name = REFERENCE_BASE if w is None else biased_variant_name(REFERENCE_BASE, 'cand', w)
                    init = 'baseline' if w is None else f'W{weight_pct_str(w)}'
                    for scale in AB_SWEEP_SCALES:
                        cells.append(_cell('w', seed, REFERENCE_BASE, init, k, scale, base_name))
            for k in EXP_N_K_VALUES:
                for n in EXP_N_VALUES:
                    base_name = biased_rand_all_variant_name(REFERENCE_BASE, EXP_N_WEIGHT, n)
                    init = f'W{weight_pct_str(EXP_N_WEIGHT)}-N{n}'
                    for scale in AB_SWEEP_SCALES:
                        cells.append(_cell('n', seed, REFERENCE_BASE, init, k, scale, base_name))

    accs = {}
    for c in cells:
        r = (ref_rows if c['base'] == REFERENCE_BASE else rows).get(c['key'])
        if r is not None:
            accs.setdefault((c['exp'], c['k'], c['base'], c['init'], c['ema']), []).append(r['final_accuracy'])

    emas = [ema_tag(s) for s in AB_SWEEP_SCALES]
    all_bases = list(bases) + ([REFERENCE_BASE] if reference_run else [])
    groups = sorted({(c['exp'], c['k'], c['base'], c['init']) for c in cells},
                    key=lambda g: (g[0], g[1], all_bases.index(g[2]) if g[2] in all_bases else 99,
                                   _init_sort_key(g[3])))

    def fmt(vals):
        if not vals:
            return '--'
        m = statistics.mean(vals)
        return f'{m:.2f}' + (f'±{statistics.stdev(vals):.2f}' if len(vals) > 1 else '') + f' ({len(vals)})'

    col_w = 16
    print(f"\n{'exp':<4}{'k':>3}  {'base':<28}{'init':<10}" + ''.join(f'{e:>{col_w}}' for e in emas))
    for exp, k, base, init in groups:
        print(f'{exp:<4}{k:>3}  {base:<28}{init:<10}' +
              ''.join(f"{fmt(accs.get((exp, k, base, init, e), [])):>{col_w}}" for e in emas))
    ref_note = f'; "{REFERENCE_BASE}" rows pulled from results/{reference_run}' if reference_run else ''
    print(f"\n(cell = mean±std over seeds (n); {sum(len(v) for v in accs.values())} of "
          f"{len(cells)} cells recorded{ref_note})")

    out_path = out_path or os.path.join(results_dir, REPORT_FILE)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['exp', 'k', 'base', 'init', 'ema', 'n_seeds', 'mean', 'std', 'accs'])
        w.writeheader()
        for exp, k, base, init in groups:
            for e in emas:
                vals = accs.get((exp, k, base, init, e), [])
                w.writerow(dict(exp=exp, k=k, base=base, init=init, ema=e, n_seeds=len(vals),
                                mean=round(statistics.mean(vals), 4) if vals else '',
                                std=round(statistics.stdev(vals), 4) if len(vals) > 1 else '',
                                accs=';'.join(f'{a:.2f}' for a in vals)))
    print(f'Wrote -> {out_path}')


def _init_sort_key(init):
    if init == 'baseline':
        return (0, 0.0)
    if '-N' in init:
        return (2, float(init.split('-N')[1]))
    return (1, float(init[1:]))


def cmd_report(args):
    _report(args.run_name, args.seeds, args.experiments, args.bases, not args.no_baseline,
            args.reference_run, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    def add_common(p):
        p.add_argument('--run_name', required=True)
        p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
        p.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
        p.add_argument('--bases', nargs='+', choices=list(AB_SWEEP_BASES), default=list(AB_SWEEP_BASES))
        p.add_argument('--no_baseline', action='store_true',
                       help="Drop the unbiased-init reference cells from exp 'w'")
        p.add_argument('--reference_run', default='ema_sweep_0915',
                       help="Run to pull the A+B (native PiCO-Fixed) reference column from at report time "
                            "(pass '' to disable)")

    p = sub.add_parser('run', help='Launch/resume the sweep across the given GPUs')
    add_common(p)
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--slots_per_gpu', type=int, default=1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--report_every', type=int, default=10)
    p.add_argument('--retries', type=int, default=1)
    p.add_argument('--poll', type=int, default=60)
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('status', help='Progress table + running/failed cells (read-only)')
    add_common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('report', help='Partial-results pivot table (rows exp/k/base/init, cols EMA level)')
    add_common(p)
    p.add_argument('--out', default=None)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    if getattr(args, 'reference_run', None) == '':
        args.reference_run = None
    args.fn(args)


if __name__ == '__main__':
    main()
