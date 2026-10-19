#!/usr/bin/env python
"""Bias-aware bandit reward sweep (2026-10-20), resumable.

ComCo-Mix-BanditPaired with the lookahead score behind d replaced
(src/bias_aware_reward.py):
  ComCo-Mix-BanditPaired-RhoEM      rho correction, Pi estimated label-free (EM + CV shrink to uniform)
  ComCo-Mix-BanditPaired-RhoOracle  rho correction, Pi from true labels (upper bound of the correction)
  ComCo-Mix-BanditPaired-TrueAcc    held-out true-class probability (upper bound of the bandit)
The original ComCo-Mix-BanditPaired at these cells is reused from bias_c20_<level>_1017
(same protocol: C=20, 10% held-out, 200 epochs, seeds 42-44).

Cells: --cl_bias in {set1, set2, uniform} x k in {5, 10, 15} x seeds 42-44 x 3 algorithms = 81,
written to results/biasreward_c20_<level>_1020/. Seed 42 first; within a seed the most
informative cells (strong bias, small k) first.

    python scripts/launch_bias_reward.py --gpus 0 1 2 3 --wait_for_file logs/clsloss_swap_c20_mclnl/launcher.log
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import results  # noqa: E402

RUN = 'biasreward_1020'  # launcher log dir
ALGOS = ['ComCo-Mix-BanditPaired-RhoEM', 'ComCo-Mix-BanditPaired-RhoOracle', 'ComCo-Mix-BanditPaired-TrueAcc']
BIAS = ['set1', 'set2', 'uniform']
KS = [5, 10, 15]
SEEDS = [42, 43, 44]


def run_name(bias):
    return f'biasreward_c20_{bias}_1020'


def build_jobs():
    return [(run_name(b), b, a, k, s) for s in SEEDS for k in KS for b in BIAS for a in ALGOS]


def is_done(run, alg, k, seed):
    d = os.path.join(ROOT, 'results', run, 'shards')
    if not os.path.isdir(d):
        return False
    return any((alg, k, seed) in {(a, kk, ss) for (_, _, kk, a, ss) in results.load_done(os.path.join(d, f))}
               for f in os.listdir(d) if f.endswith('.csv'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--data_dir', default='/nfs/home/Paul/PyPCL/data')
    ap.add_argument('--wait_for_file', default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join('logs', RUN), exist_ok=True)

    def log(msg):
        with open(os.path.join('logs', RUN, 'launcher.log'), 'a') as f:
            f.write(f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}\n')

    jobs = build_jobs()
    log(f'QUEUED {len(jobs)} jobs; waiting for {a.wait_for_file}')
    while a.wait_for_file:
        if os.path.exists(a.wait_for_file) and 'ALL DONE' in open(a.wait_for_file).read():
            break
        time.sleep(60)

    log('SMOKE start')
    smoke = subprocess.run([a.python, 'scripts/run_pipeline.py', 'run', '--run_name', 'smoke_biasreward_tmp',
                            '--algorithms', *ALGOS, '--c_values', '20', '--only_k', '5', '--seeds', '42',
                            '--epochs', '2', '--holdout_frac', '0.1', '--cl_bias', 'set1', '--data_dir', a.data_dir],
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(a.gpus[0])},
                           stdout=open(os.path.join('logs', RUN, 'smoke.log'), 'w'), stderr=subprocess.STDOUT)
    if smoke.returncode != 0:
        log(f'SMOKE FAILED rc={smoke.returncode}; aborting (see smoke.log)')
        return
    subprocess.run(['rm', '-rf', 'results/smoke_biasreward_tmp'])
    log('SMOKE ok')

    queue = [(wid, j) for wid, j in enumerate(jobs) if not is_done(j[0], j[2], j[3], j[4])]
    running = {}
    while queue or running:
        for p in [p for p in running if p.poll() is not None]:
            gpu, (run, bias, alg, k, seed) = running.pop(p)
            log(f'END rc={p.returncode} gpu{gpu} {run} {alg} k={k} seed={seed}')
        for gpu in a.gpus:
            while queue and sum(1 for g, _ in running.values() if g == gpu) < a.slots:
                wid, (run, bias, alg, k, seed) = queue.pop(0)
                cmd = [a.python, 'scripts/run_pipeline.py', 'run', '--run_name', run, '--algorithms', *ALGOS,
                       '--algo', alg, '--gpu_id', str(wid), '--num_gpus', str(len(jobs)), '--c_values', '20',
                       '--only_k', str(k), '--seeds', str(seed), '--epochs', '200', '--holdout_frac', '0.1',
                       '--cl_bias', bias, '--data_dir', a.data_dir]
                out = open(os.path.join('logs', RUN, f'{run}__{alg}_k{k}_s{seed}.log'), 'w')
                p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
                running[p] = (gpu, (run, bias, alg, k, seed))
                log(f'START gpu{gpu} {run} {alg} k={k} seed={seed}')
        time.sleep(20)
    log('ALL DONE')


if __name__ == '__main__':
    main()
