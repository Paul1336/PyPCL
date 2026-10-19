#!/usr/bin/env python
"""cls-loss swap sweep launcher ({PiCO-Fixed, ComCo-Fixed} x every cls loss), resumable.

Same protocol as clsloss_swap_1006 (C=20): 200 epochs, batch 512, seeds 42-44,
no held-out split, uniform constant-k candidate sets. Each (algorithm, k, seed)
cell is its own `run_pipeline.py run` process pinned to one GPU; its worker id
is the cell's index in the full job list, so results land in
results/<run>/shards/worker<id>.csv and a relaunch skips every finished cell.

Plans:
  c20_mclnl  C=20,  k in {1,5,10,15,19}: SwapPiCO-MCLNL, SwapComCo-MCLNL           30 cells
             (every other loss at these cells is already in clsloss_swap_1006)
  c100       C=100, k in {1,5,10,25,50,75,99}, all 13 losses x 2 frameworks        546 cells
             phase 1: priority losses (UCE CLPL PRODEN PiCOPartial PiCOCLS
                      MCLLOGFixed MCLNL SCLNL) x both frameworks, seed 42        112
             phase 2: same, seeds 43 and 44                                      224
             phase 3: remaining losses (Wu2022 SoLar MCLLOG OP OPW), seeds 42-44 210

    python scripts/launch_clsloss_swap.py --plan c100 --gpus 0 1 2 3 --slots 4
    python scripts/launch_clsloss_swap.py --plan c100 --status      # progress, no launch
    python scripts/launch_clsloss_swap.py --plan c100 --dry_run     # print the queue
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.clsloss_swap import CLS_LOSSES, FRAMEWORKS, swap_algorithm_name  # noqa: E402
from src.pipeline import results  # noqa: E402

SEEDS = (42, 43, 44)
PRIORITY_LOSSES = ['UCE', 'CLPL', 'PRODEN', 'PiCOPartial', 'PiCOCLS', 'MCLLOGFixed', 'MCLNL', 'SCLNL']
OTHER_LOSSES = [l for l in CLS_LOSSES if l not in PRIORITY_LOSSES]


def _algos(losses):
    return [swap_algorithm_name(fw, l) for l in losses for fw in FRAMEWORKS]


def build_plan(plan):
    """-> (run_name, C, [(phase, algorithm, k, seed)]); order = launch order."""
    if plan == 'c20_mclnl':
        ks = [1, 5, 10, 15, 19]
        return 'clsloss_swap_c20_mclnl', 20, [(1, a, k, s) for s in SEEDS for a in _algos(['MCLNL']) for k in ks]
    if plan == 'c100':
        ks = [1, 5, 10, 25, 50, 75, 99]
        jobs = [(1, a, k, 42) for a in _algos(PRIORITY_LOSSES) for k in ks]
        jobs += [(2, a, k, s) for s in (43, 44) for a in _algos(PRIORITY_LOSSES) for k in ks]
        jobs += [(3, a, k, s) for s in SEEDS for a in _algos(OTHER_LOSSES) for k in ks]
        return 'clsloss_swap_c100', 100, jobs
    raise ValueError(plan)


def finished(run, C):
    done = set()
    shard_dir = os.path.join(ROOT, 'results', run, 'shards')
    if os.path.isdir(shard_dir):
        for f in os.listdir(shard_dir):
            if f.endswith('.csv'):
                done |= {(a, k, s) for (_, c, k, a, s) in results.load_done(os.path.join(shard_dir, f)) if c == C}
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', choices=['c20_mclnl', 'c100'], required=True)
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--slots', type=int, default=4, help='concurrent runs per GPU')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    os.chdir(ROOT)
    run, C, jobs = build_plan(a.plan)
    algos = sorted({j[1] for j in jobs})
    log_dir = os.path.join('logs', run)
    os.makedirs(log_dir, exist_ok=True)

    def log(msg):
        line = f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}'
        print(line, flush=True)
        with open(os.path.join(log_dir, 'launcher.log'), 'a') as f:
            f.write(line + '\n')

    done = finished(run, C)
    if a.status or a.dry_run:
        for ph in sorted({j[0] for j in jobs}):
            cells = [j for j in jobs if j[0] == ph]
            print(f'phase {ph}: {sum((j[1], j[2], j[3]) in done for j in cells)}/{len(cells)} done')
        print(f'total: {len(done & {(j[1], j[2], j[3]) for j in jobs})}/{len(jobs)} done  ({run}, C={C})')
        if a.dry_run:
            for wid, (ph, alg, k, seed) in enumerate(jobs):
                if (alg, k, seed) not in done:
                    print(f'  worker{wid:3d} phase{ph} {alg} k={k} seed={seed}')
        return

    queue = [(wid, j) for wid, j in enumerate(jobs) if (j[1], j[2], j[3]) not in done]
    log(f'QUEUED {len(queue)}/{len(jobs)} cells of {run} (C={C}) on gpus {a.gpus} x {a.slots} slots')
    running = {}
    while queue or running:
        for p in [p for p in running if p.poll() is not None]:
            gpu, (ph, alg, k, seed) = running.pop(p)
            log(f'END rc={p.returncode} gpu{gpu} phase{ph} {alg} k={k} seed={seed}')
        for gpu in a.gpus:
            while queue and sum(1 for g, _ in running.values() if g == gpu) < a.slots:
                wid, (ph, alg, k, seed) = queue.pop(0)
                cmd = [a.python, 'scripts/run_pipeline.py', 'run', '--run_name', run, '--algorithms', *algos,
                       '--algo', alg, '--gpu_id', str(wid), '--num_gpus', str(len(jobs)), '--c_values', str(C),
                       '--only_k', str(k), '--seeds', str(seed), '--epochs', str(a.epochs),
                       '--data_dir', a.data_dir]
                out = open(os.path.join(log_dir, f'{alg}_k{k}_s{seed}.log'), 'w')
                p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
                                          'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
                running[p] = (gpu, (ph, alg, k, seed))
                log(f'START gpu{gpu} phase{ph} {alg} k={k} seed={seed}')
        time.sleep(20)
    subprocess.run([a.python, 'scripts/run_pipeline.py', 'merge', '--run', run])
    log('ALL DONE')


if __name__ == '__main__':
    main()
