"""Epoch accumulation + CSV writing shared by every training-time diagnostic.

Two aggregation conventions are used across the project and both are kept here:

* ratio-of-sums  (``add`` / ``ratio``): numerator and denominator are summed over
  every batch of the epoch, then divided once. Used for pair precision/recall,
  PiCO+ reliable-set stats. Batches with more pairs weigh more.
* mean-of-batch-means (``mean`` / ``avg``): each batch contributes one already
  averaged value, every batch weighs the same. Used for pseudo-label accuracy,
  gradient cosines, losses.

Origin: ``_Acc`` / ``_CsvLog`` in src/pcdiag.py (also used by src/picoplus.py).
"""

import csv
import math
import os


def _f(v):
    try:
        return float(v)
    except TypeError:  # 0-d tensor on GPU etc.
        return float(v.item())


class EpochAccumulator:
    """Drop-in equivalent of pcdiag._Acc.

    ``ratio(a, b)`` returns NaN when the denominator is <= 0 and 0.0 when only the
    numerator key is missing (same as the original).
    """

    def __init__(self):
        self.s, self.m, self.n = {}, {}, {}

    def add(self, k, v):
        self.s[k] = self.s.get(k, 0.0) + _f(v)

    def add_many(self, sums: dict):
        for k, v in sums.items():
            self.add(k, v)

    def mean(self, k, v):
        self.m[k] = self.m.get(k, 0.0) + _f(v)
        self.n[k] = self.n.get(k, 0) + 1

    def mean_many(self, vals: dict):
        for k, v in vals.items():
            self.mean(k, v)

    def ratio(self, a, b):
        d = self.s.get(b, 0.0)
        return self.s.get(a, 0.0) / d if d > 0 else float('nan')

    def avg(self, k):
        return self.m[k] / self.n[k] if self.n.get(k) else float('nan')

    def means(self) -> dict:
        """All mean-of-batch-means columns, sorted by name (pcdiag column order)."""
        return {k: self.avg(k) for k in sorted(self.m)}

    def ratios(self, spec: dict) -> dict:
        """spec = {column: (numerator_key, denominator_key)}."""
        return {col: self.ratio(a, b) for col, (a, b) in spec.items()}


class RewritingCsvLog:
    """One row per call; the whole file is rewritten (tmp + os.replace) so columns
    that first appear in later epochs are kept. Equivalent of pcdiag._CsvLog but
    takes an explicit path."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.path = path
        self.fields, self.rows = [], []

    def write(self, row: dict):
        self.rows.append(row)
        self.fields += [k for k in row if k not in self.fields]
        tmp = self.path + '.tmp'
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, self.fields)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)


def pcdiag_csv_path(results_dir, algorithm, C, k, seed):
    """results/<run>/pcdiag/<alg>/C{C}_k{k}_s{seed}.csv (pcdiag / picoplus layout)."""
    return os.path.join(results_dir, 'pcdiag', algorithm, f'C{C}_k{k}_s{seed}.csv')


def append_csv(path, fields, rows, nan_as_blank=True):
    """Append rows (dicts) to a CSV, writing the header if the file is new.
    NaN -> '' like the detail.py per-batch loggers."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        for r in rows:
            if nan_as_blank:
                r = {k: ('' if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            w.writerow(r)
