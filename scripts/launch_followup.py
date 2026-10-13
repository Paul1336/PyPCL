#!/usr/bin/env python
"""Follow-up sweep (2026-10-14): no-contrastive bandits + original-loss-pair switch analysis.

Waits until --wait_for_file contains 'ALL DONE' (switch_analysis_1014), runs a
2-epoch smoke test, then queues 138 cells (C=20, k in {5,10,15}, 10% held-out,
200 epochs, --switch_diag on every cell), under two run names:

  nocontrast_1014 (no contrastive branch; s updated from the classifier), seeds 42-44:
    Solo-Uni-{Bandit01, BanditCont, Fixed0, Fixed1}, Solo-Mix-{BanditPaired, A100}             54
  mix_analysis_1014 (original PartialLoss <-> ComCo-Fixed/SCL-NL pair), PiCO + ComCo:
    A  diagnostics: {PiCO,ComCo}-Mix-{BanditPaired, A100}, seed 42                            12
    B1 forced switch: {PiCO,ComCo}-Mix-Sched{25,50,100,150}, seeds 42,43                      48
    B2 frozen s: {PiCO,ComCo}-Mix-Sched{50,100}-Frozen, seeds 42,43                           24
  (rho=0 / PartialLoss diagnostics are shared with switch_analysis_1014's *-Uni-Fixed0.)

    python scripts/launch_followup.py --gpus 0 1 2 3 --wait_for_file logs/switch_analysis_1014/launcher.log
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/home/Paul/miniconda/envs/PyPCL/bin/python'
DATA = '/nfs/home/Paul/PyPCL/data'
RUN = 'followup_1014'   # launcher log dir; cells go to the two run names in build_jobs()
K_VALUES = [15, 10, 5]
FWS = ['PiCO', 'ComCo']


def build_jobs():
    nc, mx = 'nocontrast_1014', 'mix_analysis_1014'
    jobs = []
    # slowest (paired lookahead) first
    jobs += [(nc, a, k, s) for s in (42, 43, 44)
             for a in ('Solo-Uni-Bandit01', 'Solo-Uni-BanditCont', 'Solo-Mix-BanditPaired') for k in K_VALUES]
    jobs += [(mx, f'{fw}-Mix-BanditPaired', k, 42) for fw in FWS for k in K_VALUES]
    for seed in (42, 43):
        for fw in FWS:
            for suf in ('Sched25', 'Sched50', 'Sched100', 'Sched150', 'Sched50-Frozen', 'Sched100-Frozen'):
                jobs += [(mx, f'{fw}-Mix-{suf}', k, seed) for k in K_VALUES]
    jobs += [(mx, f'{fw}-Mix-A100', k, 42) for fw in FWS for k in K_VALUES]
    jobs += [(nc, a, k, s) for s in (42, 43, 44)
             for a in ('Solo-Uni-Fixed0', 'Solo-Uni-Fixed1', 'Solo-Mix-A100') for k in K_VALUES]
    return jobs


def log(msg):
    with open(os.path.join(ROOT, 'logs', RUN, 'launcher.log'), 'a') as f:
        f.write(f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}\n')


def cmd_for(run, algos, alg, wid, nworkers, k, seed, epochs):
    return [PY, 'scripts/run_pipeline.py', 'run', '--run_name', run, '--algorithms', *algos, '--algo', alg,
            '--gpu_id', str(wid), '--num_gpus', str(nworkers), '--c_values', '20', '--only_k', str(k),
            '--seeds', str(seed), '--epochs', str(epochs), '--holdout_frac', '0.1', '--switch_diag',
            '--data_dir', DATA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--slots', type=int, default=4)
    ap.add_argument('--wait_for_file', default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join('logs', RUN), exist_ok=True)
    jobs = build_jobs()
    algos = {run: sorted({j[1] for j in jobs if j[0] == run}) for run in {j[0] for j in jobs}}
    log(f'QUEUED {len(jobs)} jobs; waiting for {a.wait_for_file}')

    while a.wait_for_file:
        if os.path.exists(a.wait_for_file) and 'ALL DONE' in open(a.wait_for_file).read():
            break
        time.sleep(60)

    log('SMOKE start')
    smoke = subprocess.run([PY, 'scripts/run_pipeline.py', 'run', '--run_name', 'smoke_followup_tmp',
                            '--algorithms', 'Solo-Uni-Bandit01', 'ComCo-Mix-Sched50-Frozen', '--c_values', '20',
                            '--only_k', '15', '--seeds', '42', '--epochs', '2', '--holdout_frac', '0.1',
                            '--switch_diag', '--data_dir', DATA],
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(a.gpus[0])},
                           stdout=open(os.path.join('logs', RUN, 'smoke.log'), 'w'), stderr=subprocess.STDOUT)
    if smoke.returncode != 0:
        log(f'SMOKE FAILED rc={smoke.returncode}; aborting (see smoke.log)')
        return
    subprocess.run(['rm', '-rf', 'results/smoke_followup_tmp'])
    log('SMOKE ok')

    queue = list(enumerate(jobs))
    running = {}
    while queue or running:
        for p in [p for p in running if p.poll() is not None]:
            gpu, (run, alg, k, seed) = running.pop(p)
            log(f'END rc={p.returncode} gpu{gpu} {run} {alg} k={k} seed={seed}')
        for gpu in a.gpus:
            while queue and sum(1 for g, _ in running.values() if g == gpu) < a.slots:
                wid, (run, alg, k, seed) = queue.pop(0)
                out = open(os.path.join('logs', RUN, f'{run}__{alg}_k{k}_s{seed}.log'), 'w')
                p = subprocess.Popen(cmd_for(run, algos[run], alg, wid, len(jobs), k, seed, 200), stdout=out,
                                     stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
                running[p] = (gpu, (run, alg, k, seed))
                log(f'START gpu{gpu} {run} {alg} k={k} seed={seed}')
        time.sleep(20)
    log('ALL DONE')


if __name__ == '__main__':
    main()
