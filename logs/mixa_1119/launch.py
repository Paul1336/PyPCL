#!/usr/bin/env python
"""PCD-ComCoMCLNL-MixA (ComCo-Fixed + MCL-NL + loss-level mixup) in three more settings, 2026-11-19. Resumable.

User order (11-19): C=20 set1 -> C=20 uniform noisy eta=0.3 -> C=100 uniform; inside each setting seed 42 first, then
seeds 43, 44. Fixed |Y| = k, 200 epochs, no held-out. Noise = src/cifar100_subset.inject_fixed_k_noise (same code
and data as noisy_1117; ported from switch-analysis on 11-19).
Baselines: C=20 set1 / C=100 -> PCD-ComCoMCLNL in switch-analysis results/pp_c{C}_{b}_fixk_1108 (n=3);
noisy eta=0.3 -> results/noisy_c20_uniform_eta30_1117 (seed 42 only).
GPUs 0-3, counting every run_pipeline process; MixA peaks ~6.4 GB -> max 3 per A5000 (MEM 7000 / 24000 budget).
"""
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = '/nfs/home/Paul/PyPCL/.claude/worktrees/comco-smallk-1117'
sys.path.insert(0, ROOT)
from src.pipeline import results  # noqa: E402

GPUS, SLOTS, BUDGET_MB, MEM = [0, 1, 2, 3], 4, 24000, 7000
MAX_RETRY = 2
DATA_DIR = '/nfs/home/Paul/PyPCL/data'
LOG_DIR = os.path.join(ROOT, 'logs', 'mixa_1119')
ALG, EPOCHS = 'PCD-ComCoMCLNL-MixA', 200
SETTINGS = [  # (run, C, bias, eta, ks) in the user's order
    ('mixa_c20_set1_fixk_1119', 20, 'set1', 0.0, [2, 5, 10, 15, 19]),
    ('mixa_c20_uniform_eta30_1119', 20, 'uniform', 0.3, [2, 5, 10, 15, 19]),
    ('mixa_c100_uniform_fixk_1119', 100, 'uniform', 0.0, [5, 25, 50, 75, 99]),
]
os.chdir(ROOT)

order = []
for run, C, b, eta, ks in SETTINGS:
    cells = [(k, s) for s in (42, 43, 44) for k in ks]
    for wid, (k, s) in enumerate(cells):
        order.append(dict(run=run, C=C, b=b, eta=eta, k=k, s=s, wid=wid, n=len(cells)))
# seed 42 of a setting before its seeds 43/44, settings in the user's order (cells list is already seed-major)


def gpu_procs():
    load, mem, live = {g: 0 for g in GPUS}, {g: 0 for g in GPUS}, set()
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            if open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[0] == 'Z':
                continue
            argv = open(f'/proc/{pid}/cmdline', 'rb').read().decode().split('\0')
            env = open(f'/proc/{pid}/environ', 'rb').read().decode().split('\0')
        except OSError:
            continue
        if not any(a.endswith('scripts/run_pipeline.py') for a in argv):
            continue
        try:
            gpu = next((e.split('=', 1)[1] for e in env if e.startswith('CUDA_VISIBLE_DEVICES=')), None)
            gpu = int(gpu) if gpu is not None and gpu.isdigit() else None
            if gpu in load:
                load[gpu] += 1
                mem[gpu] += MEM
            if '--run_name' in argv:
                live.add((argv[argv.index('--run_name') + 1], int(argv[argv.index('--only_k') + 1]),
                          int(argv[argv.index('--seeds') + 1])))
        except (ValueError, IndexError, StopIteration):
            pass
    return load, mem, live


def done_set(run):
    d = os.path.join('results', run, 'shards')
    out = set()
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.csv'):
                out |= {(kk, ss) for (_, _, kk, a, ss) in results.load_done(os.path.join(d, f)) if a == ALG}
    return out


def log(msg):
    line = f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}'
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, 'launcher.log'), 'a') as f:
        f.write(line + '\n')


def cmd_for(j):
    c = [sys.executable, 'scripts/run_pipeline.py', 'run', '--run_name', j['run'], '--algorithms', ALG, f'--algo={ALG}',
         '--gpu_id', str(j['wid']), '--num_gpus', str(j['n']), '--c_values', str(j['C']), '--only_k', str(j['k']),
         '--seeds', str(j['s']), '--epochs', str(EPOCHS), '--cl_bias', j['b'], '--data_dir', DATA_DIR]
    if j['eta'] > 0:
        c += ['--noise_eta', str(j['eta'])]
    return c


_, _, live0 = gpu_procs()
dones = {r: done_set(r) for r, *_ in SETTINGS}
queue = [j for j in order if (j['k'], j['s']) not in dones[j['run']] and (j['run'], j['k'], j['s']) not in live0]
if '--dry_run' in sys.argv:
    for j in queue:
        print(j['run'], 'k=%d' % j['k'], 'seed=%d' % j['s'], 'eta=%s' % j['eta'])
    print(f'{len(queue)} queued')
    sys.exit()

log(f'QUEUED {len(queue)} cells on gpus {GPUS} (max {BUDGET_MB // MEM} MixA runs per GPU)')
running = {}
while queue or running:
    for p in [p for p in running if p.poll() is not None]:
        gpu, j = running.pop(p)
        log(f"END rc={p.returncode} gpu{gpu} {j['run']} {ALG} k={j['k']} seed={j['s']}")
        if p.returncode != 0 and j.get('retry', 0) < MAX_RETRY:
            j['retry'] = j.get('retry', 0) + 1
            queue.insert(0, j)
            log(f"RETRY {j['retry']} {j['run']} k={j['k']} seed={j['s']}")
    load, mem, _ = gpu_procs()
    for gpu in GPUS:
        while queue and load[gpu] < SLOTS and mem[gpu] + MEM <= BUDGET_MB:
            j = queue.pop(0)
            out = open(os.path.join(LOG_DIR, f"{j['run']}__{ALG}_k{j['k']}_s{j['s']}.log"), 'w')
            p = subprocess.Popen(cmd_for(j), stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
            running[p] = (gpu, j)
            load[gpu] += 1
            mem[gpu] += MEM
            log(f"START gpu{gpu} {j['run']} {ALG} k={j['k']} seed={j['s']}")
    time.sleep(60)  # retry after a failure only after a minute (OOM on 11-18 burned both retries within 60 s)
log('ALL DONE')
