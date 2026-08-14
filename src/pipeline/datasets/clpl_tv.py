"""CLPL's (Cour, Sapp & Taskar, JMLR 2011) own original benchmark data --
see docs/cour2011_explanation.md's Step 3 benchmark section. Two datasets
from the same tv_data.tar.gz archive (hosted at the first author's page,
confirmed alive by scripts/probe_dataset_availability.py 2026-08-14):

- 'clpl-lost': `lost_with_screenplay_supervision.mat` -- 1122 raw 90x90x3
  RGB face crops from the TV show LOST, with REAL screenplay-derived
  candidate label sets (pre-ambiguous, like pll_mat.py's datasets).
  Verified 2026-08-14 by direct inspection: this is the SAME underlying
  1122-sample, 16-class dataset as pll_mat.py's 'lost' (identical candidate
  set size distribution: sizes {1,2,3} with counts {67,728,327}) -- 'lost'
  there is a 108-dim hand-engineered feature extraction of these same raw
  images, not a different dataset. Having both lets you compare a
  feature-vector (MLP) run against a raw-image (CNN) run of literally the
  same ground truth.
- 'clpl-fiw': `fiw_data.mat` -- 500 grayscale face crops (resized to 48x48 square) (top-10 most
  frequent identities from LFW), CLEAN labels (no ambiguity in the source
  data) -- goes through the normal ComparisonDataGenerator synthetic
  candidate-label path like any other clean-label image dataset.

Both require scipy (`pip install scipy`, already needed by pll_mat.py).
"""

import os

import numpy as np

from src.pipeline.datasets.generic_loaders import build_image_loaders_full, build_preambiguous_loaders
from src.pipeline.datasets.specs import DatasetSpec, DatasetUnavailableError

_URL = 'http://www.timotheecour.com/tv_data/tv_data.tar.gz'
_CACHE: dict = {}


def _ensure_extracted(data_dir: str) -> str:
    out_dir = os.path.join(data_dir, 'clpl_tv', 'tv_data')
    if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, 'fiw_data.mat')):
        return out_dir

    os.makedirs(os.path.join(data_dir, 'clpl_tv'), exist_ok=True)
    tar_path = os.path.join(data_dir, 'clpl_tv', 'tv_data.tar.gz')
    if not os.path.isfile(tar_path):
        import requests
        print(f"Downloading CLPL's original benchmark data from {_URL} (~117MB)...", flush=True)
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; PyPCL-dataset-loader/1.0)'}
        try:
            r = requests.get(_URL, headers=headers, timeout=300, stream=True)
            r.raise_for_status()
        except Exception as e:
            raise DatasetUnavailableError(
                f"Failed to download {_URL}: {e}. See docs/dataset_availability_report.md.") from e
        with open(tar_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    import tarfile
    print(f"Extracting {tar_path} ...", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(os.path.join(data_dir, 'clpl_tv'))
    if not os.path.isfile(os.path.join(out_dir, 'fiw_data.mat')):
        raise DatasetUnavailableError(f"Extraction of {tar_path} didn't produce the expected "
                                       f"fiw_data.mat under {out_dir}.")
    return out_dir


def _load_lost(data_dir: str) -> dict:
    if 'lost' not in _CACHE:
        import scipy.io as sio
        tv_dir = _ensure_extracted(data_dir)
        d = sio.loadmat(os.path.join(tv_dir, 'lost_with_screenplay_supervision.mat'))
        n = d['labels'].shape[0]
        # Face crops are close to but not exactly 90x90 (a few pixels off near
        # frame edges) -- found via a real shape-mismatch crash on np.stack,
        # not assumed from the readme. Resize each to a uniform 90x90 first.
        from PIL import Image
        raw_images = [d['faces'][i, 0]['image'] for i in range(n)]
        images = np.stack([
            np.asarray(Image.fromarray(im).resize((90, 90))) if im.shape[:2] != (90, 90) else im
            for im in raw_images
        ]).astype(np.uint8)  # [N,90,90,3]
        true_labels = (d['labels'].flatten() - 1).astype(np.int64)  # 1-indexed -> 0-indexed
        candidate_sets = [np.nonzero(d['labels_bag'][i])[0] for i in range(n)]
        n_classes = d['labels_bag'].shape[1]
        _CACHE['lost'] = {
            'images': images, 'true_labels': true_labels,
            'candidate_sets': candidate_sets, 'n_classes': n_classes,
        }
        print(f"'clpl-lost': {n} samples, {images.shape[1:]} images, {n_classes} classes "
              f"(real screenplay-derived candidate labels).", flush=True)
    return _CACHE['lost']


def _load_fiw(data_dir: str) -> dict:
    if 'fiw' not in _CACHE:
        import scipy.io as sio
        tv_dir = _ensure_extracted(data_dir)
        d = sio.loadmat(os.path.join(tv_dir, 'fiw_data.mat'))
        n = d['labels'].shape[0]
        # Native images are 55x45 (non-square) -- DatasetSpec.image_size is a
        # single int everywhere else in this registry (every other dataset is
        # square), and RandomCrop(image_size) with a bare int produces a
        # square crop target; found via a real "crop size larger than input"
        # crash, not assumed. Resize to a 48x48 square instead of threading a
        # (h, w) tuple through the whole DatasetSpec/generic_loaders interface
        # for this one dataset.
        from PIL import Image
        images = np.stack([
            np.asarray(Image.fromarray(d['images'][0, i]).resize((48, 48)))
            for i in range(n)
        ]).astype(np.uint8)  # [N,48,48]
        labels = (d['labels'].flatten() - 1).astype(np.int64)  # 1-indexed -> 0-indexed
        n_classes = int(labels.max()) + 1
        _CACHE['fiw'] = {'images': images, 'labels': labels, 'n_classes': n_classes}
        print(f"'clpl-fiw': {n} samples, {images.shape[1:]} images, {n_classes} classes "
              f"(clean labels; candidate sets synthetically generated).", flush=True)
    return _CACHE['fiw']


def _loader_lost(C, k, data_dir, seed, log_dir, batch_size):
    data = _load_lost(data_dir)
    spec = DATASETS_BY_NAME['clpl-lost']
    if C != data['n_classes']:
        raise ValueError(f"'clpl-lost' has {data['n_classes']} classes, got C={C}")

    rng = np.random.RandomState(seed)
    N = len(data['true_labels'])
    idx = rng.permutation(N)
    n_test = max(1, int(N * 0.2))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_images = data['images'][train_idx]
    train_candidates = [data['candidate_sets'][i] for i in train_idx]
    test_images = data['images'][test_idx]
    test_targets = data['true_labels'][test_idx].tolist()

    return build_preambiguous_loaders(
        train_images, train_candidates, test_images, test_targets, spec, batch_size,
        seed=seed, log_dir=log_dir)


def _loader_fiw(C, k, data_dir, seed, log_dir, batch_size):
    data = _load_fiw(data_dir)
    spec = DATASETS_BY_NAME['clpl-fiw']
    if C != data['n_classes']:
        raise ValueError(f"'clpl-fiw' has {data['n_classes']} classes, got C={C}")

    rng = np.random.RandomState(seed)
    N = len(data['labels'])
    idx = rng.permutation(N)
    n_test = max(1, int(N * 0.2))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    return build_image_loaders_full(
        data['images'][train_idx], data['labels'][train_idx],
        data['images'][test_idx], data['labels'][test_idx],
        spec, k, batch_size, seed=seed, log_dir=log_dir)


def _build_specs() -> dict:
    return {
        'clpl-lost': DatasetSpec(
            name='clpl-lost', modality='image', backbone='cnn', fixed_num_classes=16,
            supports_pico_family=True, loader=_loader_lost,
            in_channels=3, image_size=90, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616),
            is_preambiguous=True, batch_size_override=32,
            notes=('CLPL paper\'s own raw-image "Lost" benchmark (Section 7-8). Real screenplay-'
                   'derived candidate labels -- same underlying data as pll_mat.py\'s "lost" '
                   '(feature-vector form). CL synthetically derived as PL complement.'),
        ),
        'clpl-fiw': DatasetSpec(
            name='clpl-fiw', modality='image', backbone='cnn', fixed_num_classes=10,
            supports_pico_family=False, loader=_loader_fiw,
            in_channels=1, image_size=48, mean=(0.5,), std=(0.5,),
            notes=('CLPL paper\'s LFW-derived "Faces in the Wild" subset (top-10 identities, 50 '
                   'faces each). Clean labels -- candidate sets synthetically generated. '
                   'Grayscale -> PiCO/ComCo/SoLar unsupported.'),
        ),
    }


DATASETS_BY_NAME = _build_specs()
