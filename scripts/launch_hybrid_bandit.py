#!/usr/bin/env python
"""Hybrid framework-switching bandit (2026-10-14).

Bandit chooses, every epoch, between two whole frameworks sharing one encoder:
  arm 0 = PiCO (SupCon contrastive) + PartialLoss, arm 1 = ComCo (ComCo contrastive) + MCL-LOG
(src/hybrid_switch.py). Paired 4-batch lookahead + label-free held-out reward, as Uni-Bandit01.
  Hybrid-Bandit01: k in {5,10,15} x seeds 42-44                                         9
  Hybrid-FixedPiCO / Hybrid-FixedComCo (implementation check vs PiCO-Uni-Fixed0 /
    ComCo-Uni-Fixed1): k in {5,10,15} x seed 42                                          6
C=20, 10% held-out, 200 epochs, --switch_diag. Waits for --wait_for_file 'ALL DONE'.
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/Paul/miniconda/envs/PyPCL/bin/python'
DATA = '/nfs/home/Paul/PyPCL/data'
RUN = 'hybrid_bandit_1014'
ALGOS = ['Hybrid-Bandit01', 'Hybrid-FixedPiCO', 'Hybrid-FixedComCo']


def log(msg):
    with open(os.path.join(ROOT, 'logs', RUN, 'launcher.log'), 'a') as f:
        f.write(f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}\n')


def cmd(run, algos, alg, wid, n, k, seed, epochs):
    return [PY, 'scripts/run_pipeline.py', 'run', '--run_name', run, '--algorithms', *algos, '--algo', alg,
            '--gpu_id', str(wid), '--num_gpus', str(n), '--c_values', '20', '--only_k', str(k), '--seeds', str(seed),
            '--epochs', str(epochs), '--holdout_frac', '0.1', '--switch_diag', '--data_dir', DATA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--wait_for_file', default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join('logs', RUN), exist_ok=True)
    jobs = [('Hybrid-Bandit01', k, seed) for seed in (42, 43, 44) for k in (15, 10, 5)]
    jobs += [(alg, k, 42) for alg in ('Hybrid-FixedPiCO', 'Hybrid-FixedComCo') for k in (15, 10, 5)]
    log(f'QUEUED {len(jobs)} jobs; waiting for {a.wait_for_file}')
    while a.wait_for_file:
        if os.path.exists(a.wait_for_file) and 'ALL DONE' in open(a.wait_for_file).read():
            break
        time.sleep(60)

    log('SMOKE start')
    smoke = subprocess.run([PY, 'scripts/run_pipeline.py', 'run', '--run_name', 'smoke_hybrid_tmp',
                            '--algorithms', 'Hybrid-Bandit01', '--c_values', '20',
                            '--only_k', '15', '--seeds', '43', '--epochs', '2', '--holdout_frac', '0.1',
                            '--switch_diag', '--data_dir', DATA],
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(a.gpus[0])},
                           stdout=open(os.path.join('logs', RUN, 'smoke.log'), 'w'), stderr=subprocess.STDOUT)
    if smoke.returncode != 0:
        log(f'SMOKE FAILED rc={smoke.returncode}; aborting (see smoke.log)')
        return
    subprocess.run(['rm', '-rf', 'results/smoke_hybrid_tmp'])
    log('SMOKE ok')

    queue = list(enumerate(jobs))
    running = {}
    while queue or running:
        for p in [p for p in running if p.poll() is not None]:
            gpu, (alg, k, seed) = running.pop(p)
            log(f'END rc={p.returncode} gpu{gpu} {alg} k={k} seed={seed}')
        for gpu in a.gpus:
            while queue and sum(1 for g, _ in running.values() if g == gpu) < a.slots:
                wid, (alg, k, seed) = queue.pop(0)
                out = open(os.path.join('logs', RUN, f'{alg}_k{k}_s{seed}.log'), 'w')
                p = subprocess.Popen(cmd(RUN, ALGOS, alg, wid, len(jobs), k, seed, 200), stdout=out,
                                     stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
                running[p] = (gpu, (alg, k, seed))
                log(f'START gpu{gpu} {alg} k={k} seed={seed}')
        time.sleep(20)
    log('ALL DONE')


if __name__ == '__main__':
    main()
