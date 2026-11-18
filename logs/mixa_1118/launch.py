#!/usr/bin/env python
"""ComCo-Fixed + MCL-NL + loss-level mixup ("option A", PCD-ComCoMCLNL-MixA), 2026-11-18. Resumable.

Mixed image x~ = lam*x_a + (1-lam)*x_b (weak views, in-batch pairing), lam ~ Beta(4,4), lam <- max(lam, 1-lam);
L_mix = lam*MCLNL(f(x~), S_a) + (1-lam)*MCLNL(f(x~), S_b), added with weight 1 (src/pcdiag.py, flag mix_weight).
Baseline = PCD-ComCoMCLNL in results/pp_c20_uniform_fixk_1108 (switch-analysis worktree; same code path, flag off).
Setting: C=20 uniform, fixed |Y| = k in {2,5,10,15,19}, seeds 42-44 (seed 42 first), 200 epochs, no held-out.
GPUs 0-3, <= 4 run_pipeline processes per GPU counting EVERY process (shares slots with other launchers, never
pauses or kills them). Children start_new_session=True. Run dir: results/mixa_c20_uniform_fixk_1118/.
"""
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = '/nfs/home/Paul/PyPCL/.claude/worktrees/comco-smallk-1117'
sys.path.insert(0, ROOT)
from src.pipeline import results  # noqa: E402

GPUS, SLOTS, BUDGET_MB, MEM = [0, 1, 2, 3], 4, 24000, 7000  # mixup run peaks ~6 GB; 4 per A5000 OOMed on 11-18 -> max 3 per GPU
MAX_RETRY = 2
DATA_DIR = '/nfs/home/Paul/PyPCL/data'
LOG_DIR = os.path.join(ROOT, 'logs', 'mixa_1118')
RUN, ALG, C, KS, SEEDS, EPOCHS = 'mixa_c20_uniform_fixk_1118', 'PCD-ComCoMCLNL-MixA', 20, [2, 5, 10, 15, 19], [42, 43, 44], 200
os.chdir(ROOT)

order = [dict(k=k, s=s) for s in SEEDS for k in KS]
for i, j in enumerate(order):
    j['wid'] = i


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
            if '--run_name' in argv and argv[argv.index('--run_name') + 1] == RUN:
                live.add((int(argv[argv.index('--only_k') + 1]), int(argv[argv.index('--seeds') + 1])))
        except (ValueError, IndexError, StopIteration):
            pass
    return load, mem, live


def done_set():
    d = os.path.join('results', RUN, 'shards')
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
    return [sys.executable, 'scripts/run_pipeline.py', 'run', '--run_name', RUN, '--algorithms', ALG, f'--algo={ALG}',
            '--gpu_id', str(j['wid']), '--num_gpus', str(len(order)), '--c_values', str(C), '--only_k', str(j['k']),
            '--seeds', str(j['s']), '--epochs', str(EPOCHS), '--cl_bias', 'uniform', '--data_dir', DATA_DIR]


_, _, live0 = gpu_procs()
done = done_set()
queue = [j for j in order if (j['k'], j['s']) not in done and (j['k'], j['s']) not in live0]
if '--dry_run' in sys.argv:
    for j in queue:
        print(' '.join(cmd_for(j)[1:]))
    print(f'{len(queue)} queued; done {len(done)}; live {len(live0)}')
    sys.exit()

log(f'QUEUED {len(queue)} cells on gpus {GPUS} x {SLOTS} slots')
running = {}
while queue or running:
    for p in [p for p in running if p.poll() is not None]:
        gpu, j = running.pop(p)
        log(f"END rc={p.returncode} gpu{gpu} {RUN} {ALG} k={j['k']} seed={j['s']}")
        if p.returncode != 0 and j.get('retry', 0) < MAX_RETRY:
            j['retry'] = j.get('retry', 0) + 1
            queue.insert(0, j)
            log(f"RETRY {j['retry']} k={j['k']} seed={j['s']}")
    load, mem, _ = gpu_procs()
    for gpu in GPUS:
        while queue and load[gpu] < SLOTS and mem[gpu] + MEM <= BUDGET_MB:
            j = queue.pop(0)
            out = open(os.path.join(LOG_DIR, f"{RUN}__{ALG}_k{j['k']}_s{j['s']}.log"), 'w')
            p = subprocess.Popen(cmd_for(j), stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
            running[p] = (gpu, j)
            load[gpu] += 1
            mem[gpu] += MEM
            log(f"START gpu{gpu} {RUN} {ALG} k={j['k']} seed={j['s']}")
    time.sleep(20)
log('ALL DONE')
