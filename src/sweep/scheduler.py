"""Sweep scheduler: packs cells onto GPUs, resumes, retries, never kills its children.

Behaviour (lessons from the old launchers, see docs/refactor_plan.md 4.9):
  * done = the cell's row is in its shard (results/<run>/shards/worker<wid>.csv); a child that
    exits 0 without writing its row counts as a failure (run_ema_sweep.py's check).
  * worker ids (= shard ids) are assigned once per run_name and persisted in
    logs/<id>/wid.json, so relaunching never renumbers shards.
  * GPU occupancy counts every run_pipeline.py process on the machine (other sweeps / manual
    runs), read from /proc with a parser that never raises; our own children are identified by
    the SWEEP_CELL environment variable, not by parsing their argv.
  * children run in their own session (start_new_session): a launcher crash or tmux pane close
    never SIGHUPs training runs. Per-cell logs are appended, never truncated.
  * logs/<id>/control: 'pause' (start nothing new), 'drain' (start nothing, exit when our
    children finish), 'resume'. logs/<id>/launcher.jsonl: one JSON event per line.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from src.sweep import spec as S

ROOT = S.ROOT


class Sweep:
    def __init__(self, spec_path, python=None):
        self.spec = S.load_spec(spec_path)
        self.id = self.spec['id']
        self.python = python or self.spec.get('python') or sys.executable
        self.log_dir = os.path.join(ROOT, 'logs', self.id)
        os.makedirs(self.log_dir, exist_ok=True)
        self._prepare_config()
        self.cells = S.expand(self.spec)
        self._assign_wids()   # in expansion (grid) order -- matches the old launchers' shard ids
        self.cells = S.order(self.cells, self.spec.get('order', ['seed', 'C', '-est', 'algorithm', 'k']))
        res = self.spec.get('resources', {})
        env_gpus = os.environ.get('SWEEP_GPUS')
        self.gpus = [int(g) for g in env_gpus.split(',')] if env_gpus else res.get('gpus', [0, 1, 2, 3])
        self.slots = res.get('slots_per_gpu', 4)
        self.budget = res.get('mem_budget_mb', 24000)
        self.foreign_mem = res.get('foreign_mem_mb', 4500)
        pol = self.spec.get('policy', {})
        self.retries, self.backoff = pol.get('retries', 2), pol.get('retry_backoff_s', 60)

    # ── setup ────────────────────────────────────────────────────────────────

    def _prepare_config(self):
        """Optional yaml overlay -> logs/<id>/config.yaml (deep-merged onto spec.config / config.yaml)."""
        self.spec['_config_path'] = None
        base = self.spec.get('config')
        overlay = self.spec.get('overlay')
        if not base and not overlay:
            return
        import yaml
        with open(os.path.join(ROOT, base or 'config.yaml')) as f:
            cfg = yaml.safe_load(f)

        def merge(a, b):
            for kk, v in b.items():
                a[kk] = merge(a.get(kk, {}), v) if isinstance(v, dict) and isinstance(a.get(kk), dict) else v
            return a
        cfg = merge(cfg, overlay or {})
        path = os.path.join(self.log_dir, 'config.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.spec['_config_path'] = path

    def _assign_wids(self):
        path = os.path.join(self.log_dir, 'wid.json')
        wid = json.load(open(path)) if os.path.isfile(path) else {}
        for c in self.cells:
            run = wid.setdefault(c.run_name, {})
            if c.key not in run:
                run[c.key] = len(run)
            c.wid = run[c.key]
        with open(path, 'w') as f:
            json.dump(wid, f, indent=1)
        self.spec['_n_shards'] = {r: max(1, len(v)) for r, v in wid.items()}

    # ── state ────────────────────────────────────────────────────────────────

    def is_done(self, c):
        shard = os.path.join(ROOT, 'results', c.run_name, 'shards', f'worker{c.wid}.csv')
        if not os.path.isfile(shard):
            return False
        # read-only (no schema migration): rows have 9 (pre-'dataset'), 10 or 11 (config_hash) columns
        import csv
        try:
            with open(shard, newline='') as f:
                rows = list(csv.reader(f))
        except OSError:
            return False
        want = (str(c.C), str(c.result_k), c.algorithm, str(c.seed))
        for r in rows[1:]:
            vals = r[1:5] if len(r) >= 10 else r[0:4]
            if tuple(vals) == want:
                return True
        return False

    def event(self, kind, **kw):
        rec = {'t': datetime.now().isoformat(timespec='seconds'), 'event': kind, **kw}
        with open(os.path.join(self.log_dir, 'launcher.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"{rec['t']} {kind} " + ' '.join(f'{k}={v}' for k, v in kw.items()), flush=True)

    def control(self):
        p = os.path.join(self.log_dir, 'control')
        return open(p).read().strip() if os.path.isfile(p) else 'run'

    def gpu_usage(self, ours):
        """{gpu: (n_processes, mem_mb)} over every run_pipeline process; ours: {pid: cell}."""
        use = {g: [0, 0] for g in self.gpus}
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                argv = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
                if not any(a.endswith(b'run_pipeline.py') for a in argv):
                    continue
                env = open(f'/proc/{pid}/environ', 'rb').read().split(b'\0')
                vis = next((e.split(b'=', 1)[1].decode() for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES=')), '')
                g = int(vis.split(',')[0]) if vis.split(',')[0].isdigit() else None
            except Exception:
                continue
            if g in use:
                use[g][0] += 1
                c = ours.get(int(pid))
                use[g][1] += c.mem if c else self.foreign_mem
        return use

    # ── commands ─────────────────────────────────────────────────────────────

    def status(self):
        done = [c for c in self.cells if self.is_done(c)]
        print(f'{self.id}: {len(done)}/{len(self.cells)} cells done')
        by_run = {}
        for c in self.cells:
            by_run.setdefault(c.run_name, [0, 0])
            by_run[c.run_name][1] += 1
            by_run[c.run_name][0] += c in done
        for r, (d, n) in by_run.items():
            print(f'  {r}: {d}/{n}')
        return len(done), len(self.cells)

    def dry_run(self):
        for c in self.cells:
            print(('DONE ' if self.is_done(c) else '     ') + ' '.join(S.build_argv(self.spec, c, self.python)))

    def run(self):
        queue = [c for c in self.cells if not self.is_done(c)]
        self.event('start', cells=len(self.cells), todo=len(queue), gpus=self.gpus, slots=self.slots)
        running = {}   # pid -> (Popen, cell, gpu, t0)
        tries = {}
        next_ok = {}
        stop = {'flag': False}
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
        while (queue or running) and not stop['flag']:
            for pid, (p, c, g, t0) in list(running.items()):
                if p.poll() is None:
                    continue
                del running[pid]
                ok = p.returncode == 0 and self.is_done(c)
                self.event('end', cell=c.key, gpu=g, rc=p.returncode, ok=ok, minutes=round((time.time() - t0) / 60, 1))
                if not ok:
                    tries[c.key] = tries.get(c.key, 0) + 1
                    if tries[c.key] <= self.retries:
                        next_ok[c.key] = time.time() + self.backoff * tries[c.key]
                        queue.insert(0, c)
                    else:
                        self.event('give_up', cell=c.key)
            ctl = self.control()
            if ctl == 'drain' and not running:
                break
            if ctl in ('pause', 'drain'):
                time.sleep(20)
                continue
            use = self.gpu_usage({pid: v[1] for pid, v in running.items()})
            for g in sorted(self.gpus, key=lambda x: use[x][0]):
                n, mem = use[g]
                if n >= self.slots:
                    continue
                pick = next((c for c in queue if mem + c.mem <= self.budget and next_ok.get(c.key, 0) <= time.time()),
                            None)
                if pick is None:
                    continue
                queue.remove(pick)
                if self.is_done(pick):
                    continue
                argv = S.build_argv(self.spec, pick, self.python)
                log_path = os.path.join(self.log_dir, 'cells', pick.run_name, f'{pick.algorithm}_C{pick.C}_k{pick.k}_s{pick.seed}.log')
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a') as log:
                    log.write(f'\n===== {datetime.now().isoformat(timespec="seconds")} {" ".join(argv)}\n')
                    log.flush()
                    p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(g), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                                              'SWEEP_CELL': pick.key, 'SWEEP_ID': self.id})
                running[p.pid] = (p, pick, g, time.time())
                use[g][0] += 1
                use[g][1] += pick.mem
                self.event('start_cell', cell=pick.key, gpu=g, wid=pick.wid)
                time.sleep(5)
            time.sleep(20)
        self.event('stop' if stop['flag'] else 'all_done', running=len(running), left=len(queue))
