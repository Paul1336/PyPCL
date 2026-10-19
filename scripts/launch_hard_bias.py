#!/usr/bin/env python
"""Harder-setting sweep (2026-10-17): BanditLookahead vs paper-faithful baselines.

Algorithms (all 10% held-out, so every method trains on the same 90% split):
  ComCo-Mix-BanditPaired   BanditLookahead over {PartialLoss, SCL-NL} on ComCo
  SwapComCo-SCLNL          ComCo-Fixed + SCL-NL (its native loss)
  SwapComCo-PiCOPartial    ComCo-Fixed + PartialLoss
  SwapPiCO-PiCOPartial     PiCO-Fixed + PartialLoss (its native loss)

Cells, seeds 42-44, 200 epochs:
  hard_c100_1017            C=100, k in {1,2,5,10,25,50,75,99}, uniform              96
  bias_c20_<level>_1017     C=20,  k in {2,5,10,15,19}, --cl_bias level in
                            {uniform, set3, set2, set1} (Gao & Zhang 2021 Sec 4.3)  240

Runs a 2-epoch smoke test of every algorithm at the edge cells first, then
queues the 336 cells (slowest first) on --gpus with --slots runs per GPU.

    python scripts/launch_hard_bias.py --gpus 0 1 2 3
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/Paul/miniconda/envs/PyPCL/bin/python'
DATA = '/nfs/home/Paul/PyPCL/data'
RUN = 'hard_bias_1017'   # launcher log dir; cells go to the run names in build_jobs()
ALGOS = ['ComCo-Mix-BanditPaired', 'SwapComCo-SCLNL', 'SwapComCo-PiCOPartial', 'SwapPiCO-PiCOPartial']
SEEDS = (42, 43, 44)
C100_K = [1, 2, 5, 10, 25, 50, 75, 99]
C20_K = [2, 5, 10, 15, 19]
BIAS_LEVELS = ['uniform', 'set3', 'set2', 'set1']


def build_jobs():
    """(run_name, C, bias, algorithm, k, seed), longest first."""
    jobs = [('hard_c100_1017', 100, 'uniform', a, k, s) for a in ALGOS for s in SEEDS for k in C100_K]
    jobs += [(f'bias_c20_{b}_1017', 20, b, a, k, s)
             for a in ALGOS for s in SEEDS for b in BIAS_LEVELS for k in C20_K]
    return jobs


def log(msg):
    with open(os.path.join(ROOT, 'logs', RUN, 'launcher.log'), 'a') as f:
        f.write(f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}\n')


def cmd_for(run, C, bias, alg, wid, nworkers, k, seed, epochs):
    return [PY, 'scripts/run_pipeline.py', 'run', '--run_name', run, '--algorithms', *ALGOS, '--algo', alg,
            '--gpu_id', str(wid), '--num_gpus', str(nworkers), '--c_values', str(C), '--only_k', str(k),
            '--seeds', str(seed), '--epochs', str(epochs), '--holdout_frac', '0.1', '--cl_bias', bias,
            '--data_dir', DATA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--skip_smoke', action='store_true')
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join('logs', RUN), exist_ok=True)
    jobs = build_jobs()
    log(f'QUEUED {len(jobs)} jobs on gpus {a.gpus} x {a.slots} slots')

    if not a.skip_smoke:
        log('SMOKE start')
        for C, k, bias in ((100, 1, 'uniform'), (100, 99, 'uniform'), (20, 2, 'set1'), (20, 19, 'set1')):
            smoke = subprocess.run(
                [PY, 'scripts/run_pipeline.py', 'run', '--run_name', 'smoke_hard_bias_tmp', '--algorithms', *ALGOS,
                 '--c_values', str(C), '--only_k', str(k), '--seeds', '42', '--epochs', '2',
                 '--holdout_frac', '0.1', '--cl_bias', bias, '--data_dir', DATA],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(a.gpus[0])},
                stdout=open(os.path.join('logs', RUN, f'smoke_C{C}_k{k}_{bias}.log'), 'w'), stderr=subprocess.STDOUT)
            if smoke.returncode != 0:
                log(f'SMOKE FAILED C={C} k={k} {bias} rc={smoke.returncode}; aborting')
                return
        subprocess.run(['rm', '-rf', 'results/smoke_hard_bias_tmp'])
        log('SMOKE ok')

    queue = list(enumerate(jobs))
    running = {}
    while queue or running:
        for p in [p for p in running if p.poll() is not None]:
            gpu, (run, C, bias, alg, k, seed) = running.pop(p)
            log(f'END rc={p.returncode} gpu{gpu} {run} {alg} k={k} seed={seed}')
        for gpu in a.gpus:
            while queue and sum(1 for g, _ in running.values() if g == gpu) < a.slots:
                wid, (run, C, bias, alg, k, seed) = queue.pop(0)
                out = open(os.path.join('logs', RUN, f'{run}__{alg}_k{k}_s{seed}.log'), 'w')
                p = subprocess.Popen(cmd_for(run, C, bias, alg, wid, len(jobs), k, seed, 200), stdout=out,
                                     stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
                running[p] = (gpu, (run, C, bias, alg, k, seed))
                log(f'START gpu{gpu} {run} {alg} k={k} seed={seed}')
        time.sleep(20)
    log('ALL DONE')


if __name__ == '__main__':
    main()
