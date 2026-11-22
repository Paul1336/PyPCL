"""Refactor safety net -- run every regression check (no pytest dependency).

    python tests/run_all.py            # fast checks + registry + data + full golden (GPU, ~10 min)
    python tests/run_all.py --fast     # skip the GPU golden cells
    python tests/run_all.py --only REGEX   # golden cells subset

Exit code 0 only if everything passes.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = sys.executable
sys.path.insert(0, ROOT)

CHECKS = [  # (name, argv) -- existing equivalence scripts, run from repo root
    ('unit_checks', ['tests/unit_checks.py']),
    ('step_equiv', ['tests/step_equiv.py']),
    ('check_analysis', ['verify_scripts/check_analysis.py']),
    ('check_clsloss_swap', ['verify_scripts/check_clsloss_swap.py']),
    ('check_clsloss_solo_grads', ['verify_scripts/check_clsloss_solo_grads.py']),
    ('check_pcdiag', ['verify_scripts/check_pcdiag.py']),
    ('check_pcdiag_plema', ['verify_scripts/check_pcdiag_plema.py']),
]


def run(name, argv):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0')}
    p = subprocess.run([PY] + argv, cwd=ROOT, env=env, capture_output=True, text=True)
    ok = p.returncode == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)
    if not ok:
        print((p.stdout + p.stderr)[-3000:])
    return ok


def registry_check():
    from src.pipeline.algorithms import ALGORITHMS as A
    from src.pipeline.algorithms.hparams import ALGO_HPARAMS as H
    gold = json.load(open(os.path.join(ROOT, 'tests/golden/data/_registry.json')))
    errs = []
    if set(A) != set(gold):
        errs.append(f'names differ: missing {sorted(set(gold) - set(A))[:10]}, new {sorted(set(A) - set(gold))[:10]}')
    for n in set(A) & set(gold):
        if A[n].paradigm != gold[n]['paradigm']:
            errs.append(f'{n}: paradigm {A[n].paradigm} vs {gold[n]["paradigm"]}')
        if H.get(n) != gold[n]['hparams']:
            errs.append(f'{n}: hparams {H.get(n)} vs {gold[n]["hparams"]}')
    print(f"{'PASS' if not errs else 'FAIL'}  registry ({len(A)} names)")
    for e in errs[:20]:
        print('     ', e)
    return not errs


def main():
    fast = '--fast' in sys.argv
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
    ok = all([run(n, a) for n, a in CHECKS])
    ok = registry_check() and ok
    ok = run('golden data', ['tests/golden/golden.py', 'data-check']) and ok
    if not fast:
        ok = run('golden cells', ['tests/golden/golden.py', 'check'] + (['--only', only] if only else [])) and ok
    print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
