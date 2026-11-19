#!/usr/bin/env python
"""PCD-ComCoMCLNL-MixA gap fill, 2026-11-19 10:00. Resumable. Runs next to logs/mixa_1119/launch.py (shares slots).

User request: run MixA in every earlier setting. Already covered: C=20 uniform fixk (mixa_1118), C=20 set1 / C=20
uniform eta=0.3 / C=100 uniform (mixa_1119). This launcher adds, in the user's order (uniform noisy before
variable-k), seed 42 first then seeds 43, 44 within each setting:
  1. C=20 uniform noisy eta=0.1, fixed k {2,5,10,15,19}   (noise as noisy_1117 / --noise_eta)
  2. C=20 uniform noisy eta=0.2, fixed k
  3. C=20 uniform, variable-size candidate sets, mean k {2,5,10,15,19}: --only_q (k-1)/(C-1) exactly as varq_1106
Baselines: noisy_c20_uniform_eta{10,20}_1117 (PCD-ComCoMCLNL, seed 42 only); varq_c20_uniform_1106 (n=3).
GPUs 0-3, counting every run_pipeline process; MixA peaks ~6.4 GB -> max 3 per A5000.
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
ALG, EPOCHS, C, KS = 'PCD-ComCoMCLNL-MixA', 200, 20, [2, 5, 10, 15, 19]
SETTINGS = [  # (run, eta, mode)
    ('mixa_c20_uniform_eta10_1119', 0.1, 'fixk'),
    ('mixa_c20_uniform_eta20_1119', 0.2, 'fixk'),
    ('mixa_c20_uniform_varq_1119', 0.0, 'varq'),
]
os.chdir(ROOT)


def qrepr(k):
    return repr((k - 1) / (C - 1))


def res_k(j):
    return j['k'] if j['mode'] == 'fixk' else -round(float(qrepr(j['k'])) * 1000)


order = []
for run, eta, mode in SETTINGS:
    cells = [(k, s) for s in (42, 43, 44) for k in KS]
    for wid, (k, s) in enumerate(cells):
        order.append(dict(run=run, eta=eta, mode=mode, k=k, s=s, wid=wid, n=len(cells)))


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
                r, s = argv[argv.index('--run_name') + 1], int(argv[argv.index('--seeds') + 1])
                if '--only_k' in argv:
                    live.add((r, int(argv[argv.index('--only_k') + 1]), s))
                else:
                    q = next(a.split('=', 1)[1] for a in argv if a.startswith('--only_q='))
                    live.add((r, round(float(q) * (C - 1)) + 1, s))
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
    with open(os.path.join(LOG_DIR, 'launcher_fill.log'), 'a') as f:
        f.write(line + '\n')


def cmd_for(j):
    data = ['--only_k', str(j['k'])] if j['mode'] == 'fixk' else [f"--only_q={qrepr(j['k'])}"]
    c = [sys.executable, 'scripts/run_pipeline.py', 'run', '--run_name', j['run'], '--algorithms', ALG, f'--algo={ALG}',
         '--gpu_id', str(j['wid']), '--num_gpus', str(j['n']), '--c_values', str(C), *data,
         '--seeds', str(j['s']), '--epochs', str(EPOCHS), '--cl_bias', 'uniform', '--data_dir', DATA_DIR]
    if j['eta'] > 0:
        c += ['--noise_eta', str(j['eta'])]
    return c


_, _, live0 = gpu_procs()
dones = {r: done_set(r) for r, *_ in SETTINGS}
queue = [j for j in order if (res_k(j), j['s']) not in dones[j['run']] and (j['run'], j['k'], j['s']) not in live0]
if '--dry_run' in sys.argv:
    for j in queue:
        print(j['run'], 'k=%d' % j['k'], 'seed=%d' % j['s'], ' '.join(cmd_for(j)[11:13]), 'eta=%s' % j['eta'])
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
    time.sleep(60)
log('ALL DONE')
