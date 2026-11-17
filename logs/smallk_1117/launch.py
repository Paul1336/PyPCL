#!/usr/bin/env python
"""ComCo small-k study for the 11/20 advisor meeting (2026-11-17). Resumable.

Protocol for every cell: ComCo-Fixed / PiCO-Fixed frameworks of src/pcdiag.py, C=20 or C=100 (CIFAR-100 subset),
uniform candidates, no held-out split, seeds 42-44, 200 epochs, Adam 3e-4 / wd 1e-4 / batch 512 (hparams._ADAM),
config.yaml comco/pico blocks unchanged. Same code path as pcdiag_1103 / pcdiag_1104mcl / plema_1105 / pp_1108,
so existing cells there are reused instead of re-run.

Tiers (queue order):
  P0   C=20 k=1: PCD-ComCoMCL{,-OraclePL,-OraclePos,-OracleNeg,-OracleSel}, PCD-PiCO{same}, PCD-ComCoMCL-EMA99
       C=20 k in {5,10,15,19}: PCD-ComCoMCL-OracleSel, PCD-PiCO-OracleSel (other oracles exist in pcdiag_1103/1104mcl)
  P12  PCD-ComCoMCLNL-CLSx{0.25,0.5,2,4} and -LWx{0.25,0.5,2,4} (x1 = PCD-ComCoMCLNL in pp_c*_uniform_fixk_1108)
       C=100 k {5,50} seed 42 first, then C=20 k {5,10,15} all seeds, then C=100 seeds 43, 44
  P2a  PCD-ComCoMCL with variable-size candidate sets, C=20, mean k {2,5,10,15,19} (q=(k-1)/(C-1), as varq_1106)

GPUs 0-3, <= 4 run_pipeline processes per GPU counting every user process (shared with acsel_1116's launcher).
Once all acsel_1116 fixed-k cells have started, acsel's launcher is paused with SIGSTOP (its running cells continue;
resume with `kill -CONT <pid>`) so the remaining slots go to this study.
"""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

ROOT = '/nfs/home/Paul/PyPCL/.claude/worktrees/comco-smallk-1117'
sys.path.insert(0, ROOT)
from src.pipeline import results  # noqa: E402

GPUS, SLOTS, BUDGET_MB, MEM = [0, 1, 2, 3], 4, 24000, 4500
MAX_RETRY = 2
DATA_DIR = '/nfs/home/Paul/PyPCL/data'
LOG_DIR = os.path.join(ROOT, 'logs', 'smallk_1117')
EPOCHS, SEEDS = 200, [42, 43, 44]
ACSEL_DIR = '/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis/logs/acsel_1116'
ACSEL_PID = 67410
os.chdir(ROOT)

ORA = ['', '-OraclePL', '-OraclePos', '-OracleNeg', '-OracleSel']
SC = ['0.25', '0.5', '2', '4']
P12 = [f'PCD-ComCoMCLNL-CLSx{t}' for t in SC] + [f'PCD-ComCoMCLNL-LWx{t}' for t in SC]

order = []


def add(run, algs, C, ks, seeds, mode='fixk'):
    for s in seeds:
        for k in ks:
            for a in algs:
                order.append(dict(run=run, a=a, k=k, s=s, C=C, mode=mode))


P0R = 'smallk_p0_c20_uniform_1117'
add(P0R, [f'PCD-ComCoMCL{v}' for v in ORA] + [f'PCD-PiCO{v}' for v in ORA] + ['PCD-ComCoMCL-EMA99'], 20, [1], SEEDS)
add(P0R, ['PCD-ComCoMCL-OracleSel', 'PCD-PiCO-OracleSel'], 20, [5, 10, 15, 19], SEEDS)
add('smallk_p12_c100_uniform_1117', P12, 100, [5, 50], [42])
add('smallk_p12_c20_uniform_1117', P12, 20, [5, 10, 15], SEEDS)
add('smallk_p12_c100_uniform_1117', P12, 100, [5, 50], [43, 44])
add('smallk_p2_varq_c20_uniform_1117', ['PCD-ComCoMCL'], 20, [2, 5, 10, 15, 19], SEEDS, mode='varq')

# worker ids unique per run dir (one shard file per cell -> no concurrent writes)
by_run = {}
for j in order:
    by_run.setdefault(j['run'], []).append(j)
for r, js in by_run.items():
    for i, j in enumerate(js):
        j['wid'], j['n'] = i, len(js)


def qrepr(k, C):
    return repr((k - 1) / (C - 1))


def res_k(j):
    return j['k'] if j['mode'] == 'fixk' else -round(float(qrepr(j['k'], j['C'])) * 1000)


def gpu_procs():
    load, live = {g: 0 for g in GPUS}, set()
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
        gpu = next((e.split('=', 1)[1] for e in env if e.startswith('CUDA_VISIBLE_DEVICES=')), None)
        if gpu is not None and gpu.isdigit() and int(gpu) in load:
            load[int(gpu)] += 1
        try:
            alg = next(a.split('=', 1)[1] for a in argv if a.startswith('--algo='))
            live.add((argv[argv.index('--run_name') + 1], alg, argv[argv.index('--seeds') + 1],
                      argv[argv.index('--only_k') + 1] if '--only_k' in argv else
                      next(a for a in argv if a.startswith('--only_q='))))
        except (StopIteration, ValueError, IndexError):
            pass
    return load, live


def live_key(j):
    return (j['run'], j['a'], str(j['s']), str(j['k']) if j['mode'] == 'fixk' else f"--only_q={qrepr(j['k'], j['C'])}")


def done_set(run):
    d = os.path.join('results', run, 'shards')
    out = set()
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith('.csv'):
                out |= {(a, kk, ss) for (_, _, kk, a, ss) in results.load_done(os.path.join(d, f))}
    return out


def log(msg):
    line = f'{datetime.now():%Y-%m-%d_%H:%M:%S} {msg}'
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, 'launcher.log'), 'a') as f:
        f.write(line + '\n')


def acsel_fixk_all_started():
    want = {(f'acsel_c{C}_{b}_fixk_1116', a, k, s) for C, b in ((100, 'uniform'), (20, 'uniform'), (20, 'set1'))
            for a in ('PCD-ComCoMCLNL-PosAC', 'PCD-ComCoMCLNL-NegAC', 'PCD-ComCoMCLNL-PosNegAC')
            for k in ({100: [5, 25, 50, 75, 99], 20: [2, 5, 10, 15, 19]}[C]) for s in SEEDS}
    got = set()
    try:
        for line in open(os.path.join(ACSEL_DIR, 'launcher.out')):
            p = line.split()
            if len(p) >= 7 and p[1] == 'START':
                got.add((p[3], p[4], int(p[5][2:]), int(p[6][5:])))
    except OSError:
        return False
    return want <= got


def acsel_alive():
    try:
        return 'launch.py' in open(f'/proc/{ACSEL_PID}/cmdline').read() and \
            open(f'/proc/{ACSEL_PID}/stat').read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


_, live0 = gpu_procs()
dones, queue = {}, []
for j in order:
    if j['run'] not in dones:
        dones[j['run']] = done_set(j['run'])
    if (j['a'], res_k(j), j['s']) in dones[j['run']] or live_key(j) in live0:
        continue
    queue.append(j)
if '--dry_run' in sys.argv:
    for j in queue:
        print(j['run'], j['a'], j['k'], j['s'], j['mode'])
    print(f'{len(queue)} queued; {len(live0)} run_pipeline processes live')
    sys.exit()


def cmd_for(j):
    data = ['--only_k', str(j['k'])] if j['mode'] == 'fixk' else [f"--only_q={qrepr(j['k'], j['C'])}"]
    return [sys.executable, 'scripts/run_pipeline.py', 'run', '--run_name', j['run'], '--algorithms', j['a'],
            f"--algo={j['a']}", '--gpu_id', str(j['wid']), '--num_gpus', str(j['n']), '--c_values', str(j['C']),
            *data, '--seeds', str(j['s']), '--epochs', str(EPOCHS), '--cl_bias', 'uniform', '--data_dir', DATA_DIR]


log(f'QUEUED {len(queue)} cells on gpus {GPUS} x {SLOTS} slots')
paused = '--no_pause_acsel' in sys.argv
running = {}
while queue or running:
    if not paused and acsel_alive() and acsel_fixk_all_started():
        os.kill(ACSEL_PID, signal.SIGSTOP)
        paused = True
        log(f'PAUSED acsel_1116 launcher pid {ACSEL_PID} (all fixed-k cells started); resume: kill -CONT {ACSEL_PID}')
    for p in [p for p in running if p.poll() is not None]:
        gpu, j = running.pop(p)
        log(f"END rc={p.returncode} gpu{gpu} {j['run']} {j['a']} k={j['k']} seed={j['s']}")
        if p.returncode != 0 and j.get('retry', 0) < MAX_RETRY:
            j['retry'] = j.get('retry', 0) + 1
            queue.insert(0, j)
            log(f"RETRY {j['retry']} {j['run']} {j['a']} k={j['k']} seed={j['s']}")
    load, _ = gpu_procs()
    for gpu in GPUS:
        while load[gpu] < SLOTS and queue and load[gpu] * MEM + MEM <= BUDGET_MB:
            j = queue.pop(0)
            out = open(os.path.join(LOG_DIR, f"{j['run']}__{j['a']}_k{j['k']}_s{j['s']}.log"), 'w')
            p = subprocess.Popen(cmd_for(j), stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
            running[p] = (gpu, j)
            load[gpu] += 1
            log(f"START gpu{gpu} {j['run']} {j['a']} k={j['k']} seed={j['s']}")
    time.sleep(20)
log('ALL DONE')
