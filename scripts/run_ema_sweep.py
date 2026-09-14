#!/usr/bin/env python
"""conf_ema_m sweep orchestrator.

Re-runs the two init-sensitivity experiments from the 11/25 slides on BOTH
PiCO-Fixed and PRODEN, under five confidence-update EMA momentum levels
(config.yaml pico.conf_ema_range x {1, 0.75, 0.5, 0.25, 0} -- see
src/pll_init.py CONF_EMA_SCALES; scale 1 == original PiCO, scale 0 == hard
overwrite every update == original PRODEN):

  exp 'w' (slide 63, TC-PLS W sweep):   C=20, k in {5,10,12,15,19},
        init in {unbiased baseline, TC-PLS W in {4.5,5.2,6.6,8.3,10,20}%}
  exp 'n' (slide 64, TC-n-PLS n sweep): C=20, k=5, W=20%,
        n in {4,9,14,19} wrong classes drawn from ALL other classes

Every (exp, base, init, k, scale, seed) is one training cell, run as its own
`scripts/run_pipeline.py run --algo <name> ...` subprocess so the existing
resume / shard / merge / report machinery is reused unchanged. Cells are
handed out dynamically to the given GPUs (a free GPU always takes the next
pending cell), ordered SEED-MAJOR: every cell of the first seed finishes
before the second seed starts, so a complete single-seed picture is available
early -- heavier PiCO cells are queued before PRODEN cells within a seed.

    # launch (resumable: already-recorded cells are skipped)
    python scripts/run_ema_sweep.py run --run_name ema_sweep_0915 --gpus 0 1 2 3
    # progress, any time, from another terminal
    python scripts/run_ema_sweep.py status --run_name ema_sweep_0915
    # partial results so far (pivot: rows = exp/k/base/init, cols = EMA level)
    python scripts/run_ema_sweep.py report --run_name ema_sweep_0915

Results: results/<run_name>/ (shards/, results.csv, ema_sweep_progress.json,
ema_sweep_report.csv, detail/ for the first seed). Per-cell logs:
logs/<run_name>/<algorithm>__C20_k<k>_s<seed>.log
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
os.chdir(ROOT)  # results/ and logs/ are repo-relative, same as run_pipeline.py

from src.pipeline import results as results_mod  # noqa: E402
from src.pipeline.algorithms import ALL_ALGORITHM_NAMES  # noqa: E402
from src.pll_init import (BIAS_RAND_ALL_N_VALUES, CONF_EMA_SCALES, CONF_EMA_SWEEP_BASES,  # noqa: E402
                           biased_rand_all_variant_name, biased_variant_name, ema_tag,
                           ema_variant_name, weight_pct_str)

DATASET = 'cifar100-subset'
C = 20
EXP_W_K_VALUES = [5, 10, 12, 15, 19]
EXP_W_WEIGHTS = [0.045, 0.052, 0.066, 0.083, 0.10, 0.20]   # slide 63's TC-PLS sweep
EXP_N_K_VALUES = [5]
EXP_N_WEIGHT = 0.20
EXP_N_VALUES = list(BIAS_RAND_ALL_N_VALUES)                # [4, 9, 14, 19], slide 64
EXPERIMENTS = ('w', 'n')

PROGRESS_FILE = 'ema_sweep_progress.json'
REPORT_FILE = 'ema_sweep_report.csv'


# ─── cell enumeration ──────────────────────────────────────────────────────


def build_cells(seeds, experiments=EXPERIMENTS, bases=CONF_EMA_SWEEP_BASES, scales=CONF_EMA_SCALES):
    """Seed-major list of cells; within a seed, PiCO-Fixed (heavy) before
    PRODEN (light) so the longest jobs start first."""
    cells = []
    for seed in seeds:
        for base in bases:
            if 'w' in experiments:
                for k in EXP_W_K_VALUES:
                    for w in [None] + EXP_W_WEIGHTS:
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
    """key -> shard row (last write wins), tolerating a torn row that a
    worker is mid-append on -- mirrors results.merge_shards without writing
    results.csv, so this never races with the workers' own merges."""
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

    cells = build_cells(args.seeds, args.experiments, args.bases)
    done = load_done(results_dir)
    pending = deque(c for c in cells if c['key'] not in done)
    if args.limit is not None:
        pending = deque(list(pending)[:args.limit])

    n_slots = len(args.gpus) * args.slots_per_gpu
    print(f'run_name={run_name}  cells={len(cells)}  already done={len(cells) - len(pending)}  '
          f'pending={len(pending)}  gpus={args.gpus} x{args.slots_per_gpu} slot(s)  '
          f'seeds={args.seeds}  epochs={args.epochs}  detail={"first seed only" if args.detail else "off"}',
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
    finished = []   # (cell, duration_s)
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
        # --detail output paths aren't seed-scoped (see runner.py's
        # diagnostics_seed), so only the sweep's first seed gets it.
        if args.detail and seed == args.seeds[0]:
            cmd.append('--detail')
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
        """Mean duration of already-finished cells of the same base (this
        session first, else the training_time_s recorded in the shards)."""
        same = [d for c, d in finished if c['base'] == cell['base']]
        if not same:
            same = [r['training_time_s'] + 45 for key, r in shard_rows.items()
                    if key[3].startswith(cell['base']) and r['training_time_s'] > 0]
        return statistics.mean(same) if same else (900.0 if cell['base'] == 'PiCO-Fixed' else 180.0)

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

    # `kill <pid>` (SIGTERM, e.g. from another SSH session) should behave like
    # Ctrl-C: stop the children too, instead of orphaning the training runs.
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
    _report(args.run_name, args.seeds, args.experiments, args.bases, out_path=None)


def _print_breakdown(cells, done):
    seeds = sorted({c['seed'] for c in cells})
    groups = sorted({(c['exp'], c['base']) for c in cells})
    head = f"{'seed':>6}  " + '  '.join(f"{exp}/{base:<10}" for exp, base in groups) + '     total'
    print(head)
    for seed in seeds:
        parts = []
        for exp, base in groups:
            sub = [c for c in cells if c['seed'] == seed and c['exp'] == exp and c['base'] == base]
            parts.append(f"{sum(c['key'] in done for c in sub):>4}/{len(sub):<7}")
        sub = [c for c in cells if c['seed'] == seed]
        print(f"{seed:>6}  " + '  '.join(parts) + f"   {sum(c['key'] in done for c in sub):>4}/{len(sub)}")


# ─── status ────────────────────────────────────────────────────────────────


def cmd_status(args):
    results_dir = results_dir_of(args.run_name)
    cells = build_cells(args.seeds, args.experiments, args.bases)
    done = load_done(results_dir)
    print(f"run_name={args.run_name}  done {sum(c['key'] in done for c in cells)}/{len(cells)}")
    _print_breakdown(cells, done)

    path = os.path.join(results_dir, PROGRESS_FILE)
    if not os.path.isfile(path):
        print('\n(no ema_sweep_progress.json yet -- the `run` orchestrator has not started for this run_name)')
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


def _report(run_name, seeds, experiments, bases, out_path):
    results_dir = results_dir_of(run_name)
    rows = read_rows(results_dir)
    cells = build_cells(seeds, experiments, bases)

    # (exp, k, base, init, ema) -> list of accuracies over seeds
    accs = {}
    for c in cells:
        r = rows.get(c['key'])
        if r is not None:
            accs.setdefault((c['exp'], c['k'], c['base'], c['init'], c['ema']), []).append(r['final_accuracy'])

    emas = [ema_tag(s) for s in CONF_EMA_SCALES]
    groups = sorted({(c['exp'], c['k'], c['base'], c['init']) for c in cells},
                    key=lambda g: (g[0], g[1], g[2], _init_sort_key(g[3])))

    def fmt(vals):
        if not vals:
            return '--'
        m = statistics.mean(vals)
        return f'{m:.2f}' + (f'±{statistics.stdev(vals):.2f}' if len(vals) > 1 else '') + f' ({len(vals)})'

    col_w = 16
    print(f"\n{'exp':<4}{'k':>3}  {'base':<11}{'init':<10}" + ''.join(f'{e:>{col_w}}' for e in emas))
    for exp, k, base, init in groups:
        print(f'{exp:<4}{k:>3}  {base:<11}{init:<10}' +
              ''.join(f"{fmt(accs.get((exp, k, base, init, e), [])):>{col_w}}" for e in emas))
    print(f"\n(cell = mean±std over seeds (n); {sum(len(v) for v in accs.values())} of "
          f"{len(cells)} cells recorded; EMA100 = original PiCO schedule, EMA000 = hard overwrite)")

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
    # baseline first, then W ascending, then N ascending
    if init == 'baseline':
        return (0, 0.0)
    if '-N' in init:
        return (2, float(init.split('-N')[1]))
    return (1, float(init[1:]))


def cmd_report(args):
    _report(args.run_name, args.seeds, args.experiments, args.bases, args.out)


# ─── CLI ───────────────────────────────────────────────────────────────────


def _add_common(p):
    p.add_argument('--run_name', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                   help='Run order is seed-major: all cells of the first seed, then the second, ...')
    p.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS),
                   help="'w' = slide-63 TC-PLS W sweep, 'n' = slide-64 TC-n-PLS n sweep")
    p.add_argument('--bases', nargs='+', choices=list(CONF_EMA_SWEEP_BASES), default=list(CONF_EMA_SWEEP_BASES))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('run', help='Launch/resume the sweep across the given GPUs')
    _add_common(p)
    p.add_argument('--gpus', nargs='+', type=int, required=True, help='Physical GPU ids, e.g. --gpus 0 1 2 3')
    p.add_argument('--slots_per_gpu', type=int, default=1,
                   help='Concurrent cells per GPU (PRODEN cells are light; PiCO cells are not)')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--report_every', type=int, default=10)
    p.add_argument('--detail', action='store_true',
                   help="Pass `run --detail` (per-batch TP/FP/TN/FN etc.) for the FIRST seed's cells")
    p.add_argument('--retries', type=int, default=1, help='Re-queue a failed cell this many times')
    p.add_argument('--poll', type=int, default=60, help='Seconds between progress lines')
    p.add_argument('--dry_run', action='store_true', help='List the plan and exit without launching')
    p.add_argument('--limit', type=int, default=None,
                   help='Only launch the first N pending cells (smoke test / partial batch), then stop')
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('status', help='Progress table + running/failed cells (read-only)')
    _add_common(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('report', help='Partial-results pivot table (rows exp/k/base/init, cols EMA level)')
    _add_common(p)
    p.add_argument('--out', default=None, help=f'CSV path (default results/<run_name>/{REPORT_FILE})')
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
