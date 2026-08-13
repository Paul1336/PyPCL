"""Unified results schema for the comparison pipeline.

Layout:
    results/<run_name>/
        run_config.json            snapshot of the resolved config for this run
        shards/worker<gpu_id>.csv  one CSV per worker process (avoids write races
                                    when multiple GPU processes train in parallel)
        results.csv                merged canonical table, (re)written by merge_shards()

CSV columns (one row per completed (C, k, algorithm) cell):
    total_classes, k, algorithm, seed, final_accuracy, epochs, training_time_s,
    timestamp, git_commit
"""

import csv
import glob
import json
import os
import subprocess
from datetime import datetime

FIELDS = ['total_classes', 'k', 'algorithm', 'seed', 'final_accuracy',
          'epochs', 'training_time_s', 'timestamp', 'git_commit']


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def shard_path(results_dir: str, gpu_id: int) -> str:
    return os.path.join(results_dir, 'shards', f'worker{gpu_id}.csv')


def merged_path(results_dir: str) -> str:
    return os.path.join(results_dir, 'results.csv')


def load_done(shard: str) -> set:
    """Set of (total_classes, k, algorithm) already recorded in this shard."""
    done = set()
    if not os.path.isfile(shard):
        return done
    with open(shard, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['total_classes']), int(row['k']), row['algorithm']))
    return done


def append_result(shard: str, C: int, k: int, algorithm: str, seed: int,
                   final_accuracy: float, epochs: int, training_time_s: float):
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    new_file = not os.path.isfile(shard)
    with open(shard, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'total_classes': C,
            'k': k,
            'algorithm': algorithm,
            'seed': seed,
            'final_accuracy': round(final_accuracy, 4),
            'epochs': epochs,
            'training_time_s': round(training_time_s, 1),
            'timestamp': datetime.now().isoformat(),
            'git_commit': _git_commit(),
        })


def merge_shards(results_dir: str) -> str:
    """Merge all shards/worker*.csv into a single results.csv, deduping by
    (total_classes, k, algorithm, seed) — last write wins."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'shards', 'worker*.csv'))):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['total_classes'], row['k'], row['algorithm'], row['seed'])
                rows[key] = row

    out_path = merged_path(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows.values():
            w.writerow(row)
    return out_path


def load_results(run_dirs: list) -> dict:
    """Merge one or more run dirs' results.csv into res[C][algorithm][k] = accuracy."""
    res = {}
    for run_dir in run_dirs:
        path = merged_path(run_dir)
        if not os.path.isfile(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                C = int(row['total_classes'])
                k = int(row['k'])
                acc = float(row['final_accuracy'])
                res.setdefault(C, {}).setdefault(row['algorithm'], {})[k] = acc
    return res


def write_run_config(results_dir: str, config: dict):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
