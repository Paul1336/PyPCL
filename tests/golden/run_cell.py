"""Run ONE golden cell deterministically and record a fingerprint.

Called by tests/golden/golden.py in a fresh subprocess whose cwd is a scratch dir
(results/ and logs/ land there). Usage:
    run_cell.py --out fp.json [--patch module:ATTR=pyexpr ...] -- <run_pipeline.py run args>

Fingerprint = every evaluation call's model checksum (sum / sum of squares / abs-sum of all
parameters AND buffers in float64, i.e. weights + queues + prototypes) and returned accuracy.
Evaluation functions are found by name in every loaded src.* module, so the harness keeps
working when the code moves (refactor): any function named in EVAL_NAMES gets wrapped.
"""

import ast
import importlib
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '0')

import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

EVAL_NAMES = ('evaluate_model', '_evaluate_argmin')
RECORD = {'evals': []}


@torch.no_grad()
def checksum(model):
    s = s2 = sa = 0.0
    n = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if not t.is_floating_point():
            s += float(t.double().sum())
            continue
        x = t.detach().double()
        s += float(x.sum())
        s2 += float((x * x).sum())
        sa += float(x.abs().sum())
        n += x.numel()
    return [s, s2, sa, n]


def wrap(fn):
    if getattr(fn, '_golden', False):
        return fn

    def w(model, *a, **k):
        cs = checksum(model)
        out = fn(model, *a, **k)
        RECORD['evals'].append({'checksum': cs, 'acc': float(out) if isinstance(out, (int, float)) else None})
        return out
    w._golden = True
    return w


def patch_eval_functions():
    originals = {}
    for name, mod in list(sys.modules.items()):
        if not (name == 'src' or name.startswith('src.')) or mod is None:
            continue
        for en in EVAL_NAMES:
            f = getattr(mod, en, None)
            if callable(f) and not getattr(f, '_golden', False):
                originals.setdefault(id(f), wrap(f))
                setattr(mod, en, originals[id(f)])
            elif callable(f):
                pass


def apply_patches(patches):
    for p in patches:
        target, expr = p.split('=', 1)
        modname, attr = target.split(':')
        setattr(importlib.import_module(modname), attr, ast.literal_eval(expr))


def main():
    argv = sys.argv[1:]
    sep = argv.index('--')
    own, rp_args = argv[:sep], argv[sep + 1:]
    out = own[own.index('--out') + 1]
    patches = [own[i + 1] for i, a in enumerate(own) if a == '--patch']

    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import run_pipeline  # noqa: E402  (imports the whole registry)
    apply_patches(patches)
    patch_eval_functions()
    sys.argv = ['run_pipeline.py'] + rp_args
    run_pipeline.main()
    json.dump(RECORD, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
