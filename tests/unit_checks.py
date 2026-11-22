"""Fast CPU unit checks for refactored infrastructure (no GPU, no dataset).

    python tests/unit_checks.py
"""

import csv
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

FAILS = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def test_results_schema():
    from src.pipeline import results as R
    with tempfile.TemporaryDirectory() as d:
        shard = os.path.join(d, 'shards', 'worker0.csv')
        os.makedirs(os.path.dirname(shard))
        # a pre-2026-11-22 shard (10 columns, no config_hash) + one even older 9-column row
        with open(shard, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(R._PREV_FIELDS)
            w.writerow(['cifar100-subset', 20, 5, 'PRODEN', 42, 61.5, 200, 10.0, 't', 'abc'])
            w.writerow([20, 10, 'PRODEN', 42, 50.0, 200, 10.0, 't', 'abc'])
        h = R.load_done_hashes(shard)
        check('schema: legacy rows migrated with empty hash',
              h == {('cifar100-subset', 20, 5, 'PRODEN', 42): '', ('cifar100-subset', 20, 10, 'PRODEN', 42): ''})
        R.append_result(shard, 'cifar100-subset', 20, 15, 'PRODEN', 42, 30.0, 200, 1.0, config_hash='h1')
        check('schema: new row keeps hash', R.load_done_hashes(shard)[('cifar100-subset', 20, 15, 'PRODEN', 42)] == 'h1')
        out = R.merge_shards(d)
        rows = list(csv.DictReader(open(out)))
        check('merge: 3 rows, full header, no tmp left',
              len(rows) == 3 and list(rows[0]) == R.FIELDS and not [x for x in os.listdir(d) if '.tmp' in x])
        p1 = R.write_cell_config(d, 'h1', {'a': 1})
        check('run_configs content-addressed', json.load(open(p1)) == {'a': 1})


def test_config_hash():
    from types import SimpleNamespace
    from src.core.context import config_hash, training_config
    base = dict(dataset='cifar100-subset', epochs=200, batch_size=256, only_q=None, only_bimodal=None,
                batching='random', holdout_frac=0.0, cl_bias='uniform', noise_eta=0.0,
                raw={'pico': {'a': 1}, 'comco': {'b': 2}, 'training': {'ignored': 1}, '_detail': {'x': 1}})
    h0 = config_hash(training_config(SimpleNamespace(**base)))
    h1 = config_hash(training_config(SimpleNamespace(**{**base, 'cl_bias': 'set1'})))
    h2 = config_hash(training_config(SimpleNamespace(**{**base, 'raw': {**base['raw'], '_detail': {'x': 2}}})))
    check('config_hash: setting change -> new hash; diagnostics change -> same hash', h0 != h1 and h0 == h2)


def test_trace_io():
    from src.core.io import write_trace
    from src.analysis import io as aio
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, 'r')
        raw = {'_results_dir': run, '_current_k': 5, '_current_seed': 42}
        p = write_trace(raw, 'bandit_trace', 'X-Alg', 20, [{'epoch': 1, 'value': 1.0, 'look': [0.1, 0.2]}])
        os.makedirs(os.path.join(run, 'bandit_trace'), exist_ok=True)
        json.dump([{'epoch': 1, 'value': 0.0}], open(os.path.join(run, 'bandit_trace', 'Old-Alg_k7_seed43.json'), 'w'))
        df = aio.load_bandit_traces(run)
        got = sorted(zip(df['algorithm'], df['C'].fillna(-1).astype(int), df['k'], df['seed']))
        check('trace: new _C{C} and legacy names both parsed',
              os.path.basename(p) == 'X-Alg_C20_k5_seed42.json'
              and got == [('Old-Alg', -1, 7, 43), ('X-Alg', 20, 5, 42)])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nALL PASS' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
