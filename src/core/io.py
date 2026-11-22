"""Artifact writers shared by every runner / diagnostic.

write_trace: results/<run>/<subdir>/<algorithm>_C{C}_k{k}_seed{seed}.json (bandit_trace,
prune_trace, alpha_trace). atomic_write_text: tmp + os.replace, safe against readers on NFS.
"""

import json
import os


def trace_path(results_dir, subdir, algorithm, C, k, seed):
    return os.path.join(results_dir, subdir, f'{algorithm}_C{C}_k{k}_seed{seed}.json')


def write_trace(raw_cfg, subdir, algorithm, C, obj):
    """Uses the cell context published in raw_cfg (see src.core.context)."""
    path = trace_path(raw_cfg.get('_results_dir', 'results'), subdir, algorithm, C,
                      raw_cfg.get('_current_k'), raw_cfg.get('_current_seed'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_write_text(path, json.dumps(obj))
    return path


def atomic_write_text(path, text):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
