"""The 5 classic real-world PLL benchmarks used by PRODEN (Lv et al. 2020)
and MCL-LOG (Feng et al. 2020): Lost, MSRCv2, BirdSong, Soccer Player,
Yahoo!News. Hosted as .mat files at Min-Ling Zhang's PALM lab
(palm.seu.edu.cn) -- confirmed alive by scripts/probe_dataset_availability.py
(2026-08-14); exact filenames (Soccer Player.rar / Yahoo! News.rar have
spaces, not CamelCase) confirmed via mikigom/DNPL-PyTorch's README, which
cites the same source.

.mat schema (verified 2026-08-14 by downloading+inspecting lost.rar,
MSRCv2.rar, BirdSong.rar directly -- not assumed): each file has exactly
three keys:
    data:           [N, D] float64 dense feature matrix
    target:         [C, N] sparse one-hot true-label matrix
    partial_target: [C, N] sparse multi-hot REAL candidate-label matrix
                     (always includes the true label; this IS genuine
                     annotator/task ambiguity, not synthetically generated)

These are pre-ambiguous datasets (DatasetSpec.is_preambiguous=True): the
candidate sets are the real data, not something ComparisonDataGenerator
should invent. Complementary labels (needed for CLL algorithms) are NOT
part of the original data -- these are PLL-only benchmarks -- so CL is
synthetically derived as the complement of the real PL set, loudly flagged
by build_preambiguous_loaders(). No PiCO/ComCo/SoLar support: these are
pre-extracted feature vectors, not raw images.

Extraction requires an external unrar tool (scipy/Python have no built-in
RAR decompressor for compressed archives). This module shells out to `unrar`
(assumed on PATH) or, on this development machine, WinRAR's bundled
UnRAR.exe (not on PATH by default -- see docs/dataset_availability_report.md's
manual follow-up notes). If neither is found, raises DatasetUnavailableError
with instructions rather than failing with an opaque error.
"""

import os
import shutil
import subprocess

import numpy as np

from src.pipeline.datasets.generic_loaders import build_preambiguous_loaders
from src.pipeline.datasets.specs import DatasetSpec, DatasetUnavailableError

_BASE_URL = 'http://palm.seu.edu.cn/zhangml/files/'
_FILENAMES = {
    'lost': 'lost.rar',
    'msrcv2': 'MSRCv2.rar',
    'birdsong': 'BirdSong.rar',
    'soccer-player': 'Soccer%20Player.rar',
    'yahoo-news': 'Yahoo!%20News.rar',
}
_MAT_NAMES = {
    'lost': 'lost.mat',
    'msrcv2': 'MSRCv2.mat',
    'birdsong': 'BirdSong.mat',
    'soccer-player': 'Soccer Player.mat',
    'yahoo-news': 'Yahoo! News.mat',
}
# Known dims -- verified 2026-08-14 for all 5 datasets by direct
# download+extraction+load (not just paper-cited): every entry below was
# confirmed to match its real file's (data.shape[0], data.shape[1],
# target.shape[0]) exactly, including a full end-to-end training smoke run.
_DIMS = {
    'lost': (1122, 108, 16),
    'msrcv2': (1758, 48, 23),
    'birdsong': (4998, 38, 13),
    'soccer-player': (17472, 279, 171),
    'yahoo-news': (22991, 163, 219),
}

_CACHE: dict = {}


def _find_unrar() -> str:
    for candidate in ('unrar', 'bsdtar'):
        path = shutil.which(candidate)
        if path:
            return path
    win_default = r'C:\Program Files\WinRAR\UnRAR.exe'
    if os.path.isfile(win_default):
        return win_default
    raise DatasetUnavailableError(
        "No 'unrar' (or 'bsdtar') found on PATH, and the Windows default "
        f"WinRAR install path ({win_default}) doesn't exist either. Install "
        "unrar (Linux: `apt install unrar`; Windows: WinRAR) to extract the "
        "real-world PLL .rar archives. See docs/dataset_availability_report.md.")


def _download_and_extract(name: str, data_dir: str) -> str:
    """Returns the path to the extracted .mat file, downloading+extracting
    only if not already cached under data_dir."""
    out_dir = os.path.join(data_dir, 'real_pll', f'{name}_extracted')
    mat_path = os.path.join(out_dir, _MAT_NAMES[name])
    if os.path.isfile(mat_path):
        return mat_path

    os.makedirs(out_dir, exist_ok=True)
    rar_path = os.path.join(data_dir, 'real_pll', _FILENAMES[name].replace('%20', ' '))
    if not os.path.isfile(rar_path):
        import requests
        url = _BASE_URL + _FILENAMES[name]
        print(f"Downloading real-world PLL dataset '{name}' from {url} ...", flush=True)
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; PyPCL-dataset-loader/1.0)'}
        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
        except Exception as e:
            raise DatasetUnavailableError(
                f"Failed to download '{name}' from {url}: {e}. See "
                f"docs/dataset_availability_report.md for the last known status.") from e
        with open(rar_path, 'wb') as f:
            f.write(r.content)

    unrar = _find_unrar()
    print(f"Extracting {rar_path} with {unrar} ...", flush=True)
    subprocess.run([unrar, 'x', '-o+', rar_path, out_dir + os.sep],
                    check=True, capture_output=True)
    if not os.path.isfile(mat_path):
        raise DatasetUnavailableError(
            f"Extraction of {rar_path} did not produce the expected "
            f"{_MAT_NAMES[name]} at {mat_path} -- archive layout may have changed.")
    return mat_path


def _load(name: str, data_dir: str) -> dict:
    if name not in _CACHE:
        try:
            import scipy.io as sio
        except ImportError as e:
            raise DatasetUnavailableError(
                "scipy is required to read the real-world PLL .mat files "
                "(pip install scipy).") from e

        mat_path = _download_and_extract(name, data_dir)
        d = sio.loadmat(mat_path)
        data = np.asarray(d['data'], dtype=np.float64)
        target = d['target']
        partial_target = d['partial_target']
        target = target.toarray() if hasattr(target, 'toarray') else np.asarray(target)
        partial_target = partial_target.toarray() if hasattr(partial_target, 'toarray') else np.asarray(partial_target)

        # [C, N] -> per-sample true label index and candidate-label index list
        true_labels = target.argmax(axis=0)
        candidate_sets = [np.nonzero(partial_target[:, i])[0] for i in range(partial_target.shape[1])]

        # Standardize features (same convention as uci_tabular.py).
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        data = ((data - mean) / std).astype(np.float32)

        n_classes = target.shape[0]
        _CACHE[name] = {
            'data': data, 'true_labels': true_labels,
            'candidate_sets': candidate_sets, 'n_classes': n_classes,
        }
        print(f"'{name}': {data.shape[0]} samples, {data.shape[1]} features, "
              f"{n_classes} classes (real candidate label sets, avg size "
              f"{np.mean([len(c) for c in candidate_sets]):.2f}).", flush=True)
    return _CACHE[name]


def _make_loader(name: str):
    def loader(C, k, data_dir, seed, log_dir, batch_size):
        data = _load(name, data_dir)
        spec = DATASETS_BY_NAME[name]
        if C != data['n_classes']:
            raise ValueError(f"'{name}' has {data['n_classes']} classes, got C={C}")

        rng = np.random.RandomState(seed)
        N = len(data['true_labels'])
        idx = rng.permutation(N)
        n_test = max(1, int(N * 0.2))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        train_data = data['data'][train_idx]
        train_candidates = [data['candidate_sets'][i] for i in train_idx]
        test_data = data['data'][test_idx]
        test_targets = data['true_labels'][test_idx].tolist()

        return build_preambiguous_loaders(
            train_data, train_candidates, test_data, test_targets, spec, batch_size,
            seed=seed, log_dir=log_dir)
    return loader


def _build_specs() -> dict:
    specs = {}
    for name, (n_samples, n_features, n_classes) in _DIMS.items():
        specs[name] = DatasetSpec(
            name=name, modality='tabular', backbone='mlp', fixed_num_classes=n_classes,
            supports_pico_family=False, loader=_make_loader(name), input_dim=n_features,
            is_preambiguous=True,
            batch_size_override=min(64, max(8, n_samples // 20)),
            notes=('Real-world PLL benchmark with genuine candidate label sets (not '
                   'synthetically generated); CL is synthetically derived as the PL '
                   'complement -- see build_preambiguous_loaders warning at load time.'),
        )
    return specs


DATASETS_BY_NAME = _build_specs()
