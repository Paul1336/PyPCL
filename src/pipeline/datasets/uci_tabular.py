"""UCI tabular datasets used as real-world benchmarks by PRODEN (Lv et al.
2020) and MCL-LOG (Feng et al. 2020): Dermatology, Ecoli, Abalone, Yeast,
Synthetic Control Chart Time Series.

Requires `ucimlrepo` (registered in src/pipeline/datasets/__init__.py behind
a try/except ImportError, so a bare install without it just doesn't offer
these datasets rather than failing to import the whole registry).

All five feed into the MLP backbone (DatasetSpec.backbone='mlp') -- a CNN's
spatial-convolution assumption doesn't apply to a flat feature vector.
None support the PiCO/ComCo/SoLar family (no images to augment).
"""

import numpy as np

from src.pipeline.datasets.generic_loaders import build_tabular_loaders
from src.pipeline.datasets.specs import DatasetSpec

_CACHE: dict = {}


def _impute_and_standardize(X: np.ndarray) -> np.ndarray:
    """Some UCI exports have missing values (e.g. dermatology's 'Age' column
    has 8 samples marked '?' in the original data, which ucimlrepo parses as
    NaN) -- mean-impute per column before standardizing, since NaN would
    otherwise propagate through every downstream matmul and silently turn
    into a constant-output, dead-gradient model (found via a real NaN-loss
    smoke run on dermatology, not assumed)."""
    col_mean = np.nanmean(X, axis=0, keepdims=True)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = np.where(nan_mask, np.broadcast_to(col_mean, X.shape), X)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def _encode_labels(y) -> np.ndarray:
    """Maps arbitrary (string or non-contiguous int) labels to 0..C-1."""
    classes = sorted(set(y))
    remap = {c: i for i, c in enumerate(classes)}
    return np.array([remap[v] for v in y], dtype=np.int64), len(classes)


def _load_ucirepo(name: str, uci_id: int, target_col: int = 0) -> dict:
    if name not in _CACHE:
        from ucimlrepo import fetch_ucirepo
        print(f"Fetching UCI dataset '{name}' (id={uci_id}) via ucimlrepo...", flush=True)
        d = fetch_ucirepo(id=uci_id)
        features = d.data.features
        # Some UCI exports mix categorical columns (e.g. abalone's 'Sex':
        # M/F/I) in with numeric ones -- one-hot encode any non-numeric
        # column instead of assuming every column is already a float.
        import pandas as pd
        numeric_cols = features.select_dtypes(include=[np.number])
        categorical_cols = features.select_dtypes(exclude=[np.number])
        if not categorical_cols.empty:
            categorical_encoded = pd.get_dummies(categorical_cols, dtype=np.float64)
            features = pd.concat([numeric_cols, categorical_encoded], axis=1)
        else:
            features = numeric_cols
        X = features.to_numpy(dtype=np.float64)
        y_raw = d.data.targets.iloc[:, target_col].to_numpy()
        y, n_classes = _encode_labels(y_raw)
        _CACHE[name] = {'X': _impute_and_standardize(X).astype(np.float32), 'y': y, 'n_classes': n_classes}
        print(f"'{name}': {X.shape[0]} samples, {X.shape[1]} features (after encoding), "
              f"{n_classes} classes.", flush=True)
    return _CACHE[name]


def _load_synthetic_control(data_dir: str) -> dict:
    """No ucimlrepo Python export available for this dataset (confirmed via
    a live fetch_ucirepo(id=139) call, which raises DatasetNotFoundError) --
    falls back to the raw whitespace-delimited data file, which IS directly
    downloadable. Labels aren't in the file; per the dataset's own
    documentation the 600 rows are 6 contiguous blocks of 100 (class order:
    normal, cyclic, increasing trend, decreasing trend, upward shift,
    downward shift)."""
    name = 'synthetic-control'
    if name not in _CACHE:
        import os
        import requests
        cache_path = os.path.join(data_dir, 'synthetic_control.data')
        if not os.path.isfile(cache_path):
            os.makedirs(data_dir, exist_ok=True)
            url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'synthetic_control-mld/synthetic_control.data')
            print(f"Downloading Synthetic Control Chart data from {url} ...", flush=True)
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(cache_path, 'w') as f:
                f.write(r.text)
        X = np.loadtxt(cache_path)  # [600, 60]
        y = np.repeat(np.arange(6), 100)  # 6 blocks of 100, per UCI docs
        _CACHE[name] = {'X': _impute_and_standardize(X).astype(np.float32), 'y': y.astype(np.int64), 'n_classes': 6}
        print(f"'{name}': {X.shape[0]} samples, {X.shape[1]} features, 6 classes.", flush=True)
    return _CACHE[name]


_UCI_IDS = {
    'dermatology': 33,
    'ecoli': 39,
    'abalone': 1,
    'yeast': 110,
}


def _make_loader(name: str):
    def loader(C, k, data_dir, seed, log_dir, batch_size):
        if name == 'synthetic-control':
            data = _load_synthetic_control(data_dir)
        else:
            data = _load_ucirepo(name, _UCI_IDS[name])
        spec = DATASETS_BY_NAME[name]
        if C != data['n_classes']:
            raise ValueError(f"'{name}' has {data['n_classes']} classes, got C={C}")
        return build_tabular_loaders(data['X'], data['y'], spec, k, batch_size, seed=seed, log_dir=log_dir)
    return loader


def _build_specs() -> dict:
    # Verified 2026-08-14 via live fetch_ucirepo() calls (not just the papers'
    # cited dimensions, which can disagree with what the live export actually
    # contains -- e.g. abalone's categorical 'Sex' column one-hot-encodes to
    # 3 extra columns, and its real class count is 28, not 29 as naively
    # assumed from "ring count 1-29" -- 28 is missing from the data).
    dims = {
        'dermatology': (34, 6),
        'ecoli': (7, 8),
        'abalone': (10, 28),  # 7 numeric + one-hot 'Sex' (M/F/I); target = Rings, treated as classification
        'yeast': (8, 10),
        'synthetic-control': (60, 6),
    }
    specs = {}
    for name, (input_dim, n_classes) in dims.items():
        specs[name] = DatasetSpec(
            name=name, modality='tabular', backbone='mlp', fixed_num_classes=n_classes,
            supports_pico_family=False, loader=_make_loader(name), input_dim=input_dim,
            notes='Tabular UCI benchmark; MLP backbone; PiCO/ComCo/SoLar unsupported (no images).',
        )
    return specs


DATASETS_BY_NAME = _build_specs()
