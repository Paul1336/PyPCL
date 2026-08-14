"""20 Newsgroups text classification, used as a real-world benchmark by
MCL-LOG (Feng et al. 2020). Requires scikit-learn (registered behind a
try/except ImportError in src/pipeline/datasets/__init__.py).

TF-IDF vectorizes each document into a fixed-size dense feature vector (2000
dims), then reuses the same tabular/MLP path as uci_tabular.py -- there's no
"raw text augmentation" story for PiCO/ComCo/SoLar to plug into, so this
dataset never supports that family either.
"""

import numpy as np

from src.pipeline.datasets.generic_loaders import build_tabular_loaders
from src.pipeline.datasets.specs import DatasetSpec

_MAX_FEATURES = 2000
_CACHE: dict = {}


def _load(data_dir: str) -> dict:
    if 'data' not in _CACHE:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("Fetching 20 Newsgroups via sklearn (one-time download)...", flush=True)
        bunch = fetch_20newsgroups(data_home=data_dir, subset='all',
                                    remove=('headers', 'footers', 'quotes'))
        vectorizer = TfidfVectorizer(max_features=_MAX_FEATURES, stop_words='english')
        X = vectorizer.fit_transform(bunch.data).toarray().astype(np.float32)
        y = np.asarray(bunch.target, dtype=np.int64)
        n_classes = len(bunch.target_names)
        _CACHE['data'] = {'X': X, 'y': y, 'n_classes': n_classes}
        print(f"'20newsgroups': {X.shape[0]} samples, {X.shape[1]} TF-IDF features, "
              f"{n_classes} classes.", flush=True)
    return _CACHE['data']


def _loader(C, k, data_dir, seed, log_dir, batch_size):
    data = _load(data_dir)
    spec = DATASETS_BY_NAME['20newsgroups']
    if C != data['n_classes']:
        raise ValueError(f"'20newsgroups' has {data['n_classes']} classes, got C={C}")
    return build_tabular_loaders(data['X'], data['y'], spec, k, batch_size, seed=seed, log_dir=log_dir)


def _build_specs() -> dict:
    return {
        '20newsgroups': DatasetSpec(
            name='20newsgroups', modality='tabular', backbone='mlp', fixed_num_classes=20,
            supports_pico_family=False, loader=_loader, input_dim=_MAX_FEATURES,
            notes='TF-IDF text features; MLP backbone; PiCO/ComCo/SoLar unsupported (no images).',
        ),
    }


DATASETS_BY_NAME = _build_specs()
