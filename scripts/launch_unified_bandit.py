#!/usr/bin/env python
"""Unified-loss bandit sweep (2026-10-13), PiCO + ComCo.

C=20, fixed k in {5,10,15}, 10% held-out, seeds 42-44, 200 epochs. One
pipeline process per (algorithm, k, seed) cell, dynamically queued onto
--gpus with --slots per GPU. GPUs listed in --late_gpus only join once
--wait_for_file contains 'ALL DONE' (lets the queue grow onto a GPU another
sweep is still finishing on, without ever exceeding --slots there).

ComCo's rho=0 baseline is not rerun: ComCo-Weighted-A000 in
results/bandit_fixedk_1013 is the same loss, data split and settings.

    python scripts/launch_unified_bandit.py --gpus 0 1 2 --late_gpus 3 \
        --wait_for_file logs/bimodal_1013/launcher.log
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/Paul/miniconda/envs/PyPCL/bin/python'
DATA = '/nfs/home/Paul/PyPCL/data'
RUN = 'unified_bandit_1013'
ALGOS = [  # slowest (paired lookahead) first
    'PiCO-Uni-Bandit01', 'PiCO-Uni-BanditCont', 'PiCO-Mix-BanditPaired',
    'ComCo-Uni-Bandit01', 'ComCo-Uni-BanditCont',
    'PiCO-Uni-Fixed0', 'PiCO-Uni-Fixed1', 'PiCO-Mix-A100', 'ComCo-Uni-Fixed1',
]
K_VALUES = [15, 10, 5]
SEEDS = [42, 43, 44]


def log(msg):
    with open(os.path.join(ROOT, 'logs', RUN, 'launcher.log'), 'a') as f:
        f.write(f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--late_gpus', nargs='*', type=int, default=[])
    ap.add_argument('--wait_for_file', default=None)
    ap.add_argument('--slots', type=int, default=4)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join('logs', RUN), exist_ok=True)

    jobs = [(alg, k, seed) for seed in SEEDS for alg in ALGOS for k in K_VALUES]
    queue = list(enumerate(jobs))
    running = {}  # Popen -> (gpu, job)
    gpus = list(a.gpus)
    log(f'START {len(jobs)} jobs on gpus {gpus} (+{a.late_gpus} later), {a.slots} slots each')
    while queue or running:
        if a.late_gpus and a.wait_for_file and os.path.exists(a.wait_for_file):
            with open(a.wait_for_file) as f:
                if 'ALL DONE' in f.read():
                    gpus += a.late_gpus
                    log(f'late gpus {a.late_gpus} joined')
                    a.late_gpus = []
        for p in [p for p in running if p.poll() is not None]:
            gpu, (wid, (alg, k, seed)) = running.pop(p)
            log(f'END rc={p.returncode} gpu{gpu} {alg} k={k} seed={seed}')
        for gpu in gpus:
            while queue and sum(1 for g, _ in running.values() if g == gpu) < a.slots:
                wid, (alg, k, seed) = queue.pop(0)
                cmd = [PY, 'scripts/run_pipeline.py', 'run', '--run_name', RUN, '--algorithms', *ALGOS,
                       '--algo', alg, '--gpu_id', str(wid), '--num_gpus', str(len(jobs)),
                       '--c_values', '20', '--only_k', str(k), '--seeds', str(seed), '--epochs', '200',
                       '--holdout_frac', '0.1', '--data_dir', DATA]
                out = open(os.path.join('logs', RUN, f'{alg}_k{k}_s{seed}.log'), 'w')
                p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
                running[p] = (gpu, (wid, (alg, k, seed)))
                log(f'START gpu{gpu} {alg} k={k} seed={seed}')
        time.sleep(20)
    log('ALL DONE')


if __name__ == '__main__':
    main()
