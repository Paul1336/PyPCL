"""Declarative experiment specs -> training cells (2026-11-22; replaces ~35 hand-written launchers).

A spec (experiments/<id>.yaml) lists families; each family is a cartesian grid that expands
into Cells. Example:

    id: final_1122
    epochs: 200
    data_dir: data                        # relative to the repo root, or absolute
    config: config.yaml                   # optional yaml overlay: overlay: {pico: {...}} merges into it
    families:
      - name: c20
        run_name: "final_c{C}_{cl_bias}_fixk_1122"
        grid: {C: [20], cl_bias: [uniform], algorithm: [PCD-PiCO, PCD-ComCoMCLNL, PRODEN],
               seed: [42, 43, 44], k: [2, 5, 10, 15, 19]}
                                          # grid KEY ORDER matters: it is the expansion order, which
                                          # fixes the shard ids (wid) -- persisted in logs/<id>/wid.json
        k_mode: fixk                      # fixk (--only_k) | varq (--only_q (k-1)/(C-1))
        args: {detail: true, concentration: true}   # extra run_pipeline flags (bool -> flag)
        exclude: [{algorithm: PRODEN, k: [2]}]      # optional filters (lists = any of)
    order: [seed, C, -est, algorithm, k]  # sort keys; '-est' = longest first (resources.est_minutes)
    resources:
      gpus: [0, 1, 2, 3]                  # env SWEEP_GPUS overrides
      slots_per_gpu: 4
      mem_budget_mb: 24000
      mem_mb: {default: 4500, 'PCD-PiCOPlus*': 6450}
      est_minutes: {default: 60, 'C=100': 330}
    policy: {retries: 2, retry_backoff_s: 60}
"""

import fnmatch
import itertools
import os
from dataclasses import dataclass, field

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class Cell:
    family: str
    run_name: str
    algorithm: str
    C: int
    k: int                      # mean candidate-set size (fixk: exact k)
    seed: int
    k_mode: str = 'fixk'
    params: dict = field(default_factory=dict)   # every grid value (cl_bias, eta, ...)
    args: dict = field(default_factory=dict)     # extra run_pipeline flags
    wid: int = -1                                # shard id inside run_name (persisted)
    est: float = 0.0
    mem: int = 0

    @property
    def key(self):
        return f'{self.run_name}|{self.algorithm}|C{self.C}|k{self.k}|s{self.seed}'

    @property
    def result_k(self):
        """The k stored in results.csv (negative sentinel for variable-|Y| runs)."""
        if self.k_mode == 'varq':
            return -round((self.k - 1) / (self.C - 1) * 1000)
        return self.k

    @property
    def q(self):
        return repr((self.k - 1) / (self.C - 1))


def load_spec(path):
    with open(path) as f:
        spec = yaml.safe_load(f)
    spec.setdefault('id', os.path.splitext(os.path.basename(path))[0])
    spec['_path'] = os.path.abspath(path)
    return spec


def _matches(d, cond):
    for kk, v in cond.items():
        vals = v if isinstance(v, list) else [v]
        if kk not in d:
            return False
        if not any(fnmatch.fnmatch(str(d[kk]), str(x)) for x in vals):
            return False
    return True


def _lookup(table, cell_d, default):
    """table keys: 'default', an algorithm glob, or 'name=value' conditions joined by ','."""
    best = table.get('default', default)
    for key, val in table.items():
        if key == 'default':
            continue
        conds = [c.strip() for c in key.split(',')]
        ok = True
        for c in conds:
            if '=' in c:
                kk, vv = c.split('=', 1)
                ok &= fnmatch.fnmatch(str(cell_d.get(kk.strip())), vv.strip())
            else:
                ok &= fnmatch.fnmatch(cell_d['algorithm'], c)
        if ok:
            best = val
    return best


def expand(spec):
    res = spec.get('resources', {})
    cells = []
    for fam in spec['families']:
        grid = dict(fam['grid'])
        per_c = grid.pop('k_per_C', None)
        keys = list(grid)
        for combo in itertools.product(*[grid[kk] if isinstance(grid[kk], list) else [grid[kk]] for kk in keys]):
            d = dict(zip(keys, combo))
            if per_c and 'k' not in d:
                raise ValueError('k_per_C needs no k in grid')
            ks = per_c[d['C']] if per_c else [d['k']]
            for k in ks:
                dd = dict(d, k=k)
                if any(_matches(dd, ex) for ex in fam.get('exclude', [])):
                    continue
                if fam.get('include') and not any(_matches(dd, inc) for inc in fam['include']):
                    continue
                params = {kk: v for kk, v in dd.items() if kk not in ('C', 'k', 'seed', 'algorithm')}
                c = Cell(family=fam['name'], run_name=fam['run_name'].format(**dd), algorithm=dd['algorithm'],
                         C=int(dd['C']), k=int(k), seed=int(dd['seed']), k_mode=fam.get('k_mode', 'fixk'),
                         params=params, args={kk: (v.format(**dd) if isinstance(v, str) else v)
                                              for kk, v in fam.get('args', {}).items()})
                c.est = float(_lookup(res.get('est_minutes', {}), dd, 60))
                c.mem = int(_lookup(res.get('mem_mb', {}), dd, 4500))
                cells.append(c)
    seen = {}
    for c in cells:
        if c.key in seen:
            raise ValueError(f'duplicate cell {c.key}')
        seen[c.key] = c
    return cells


def order(cells, keys):
    def sort_key(c):
        out = []
        for kk in keys:
            if kk == '-est':
                out.append(-c.est)
            elif kk.startswith('-'):
                out.append(-getattr(c, kk[1:]))
            elif kk in ('seed', 'C', 'k', 'algorithm', 'family', 'run_name'):
                out.append(getattr(c, kk))
            else:
                out.append(str(c.params.get(kk, '')))
        return out
    return sorted(cells, key=sort_key)


def build_argv(spec, cell, python):
    """argv for one cell (run from the repo root)."""
    data_dir = spec.get('data_dir', 'data')
    data_dir = data_dir if os.path.isabs(data_dir) else os.path.join(ROOT, data_dir)
    n_shards = spec['_n_shards'][cell.run_name]
    argv = [python, os.path.join(ROOT, 'scripts', 'run_pipeline.py'), 'run',
            '--run_name', cell.run_name, '--algorithms', cell.algorithm, '--algo', cell.algorithm,
            '--gpu_id', str(cell.wid), '--num_gpus', str(n_shards), '--c_values', str(cell.C),
            '--seeds', str(cell.seed), '--epochs', str(spec.get('epochs', 200)), '--data_dir', data_dir]
    argv += ['--only_k', str(cell.k)] if cell.k_mode == 'fixk' else [f'--only_q={cell.q}']
    if 'batch_size' in spec:
        argv += ['--batch_size', str(spec['batch_size'])]
    if spec.get('_config_path'):
        argv += ['--config', spec['_config_path']]
    for flag in ('cl_bias', 'noise_eta', 'holdout_frac'):
        if flag in cell.params and cell.params[flag] not in (None, 'uniform', 0, 0.0):
            argv += [f'--{flag}', str(cell.params[flag])]
    for kk, v in cell.args.items():
        if v is True:
            argv.append(f'--{kk}')
        elif v not in (False, None):
            argv += [f'--{kk}', str(v)]
    return argv
