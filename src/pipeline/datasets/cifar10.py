"""Real CIFAR-10 (full 10-class dataset), as opposed to 'cifar100-subset'
(a C-class subset drawn from CIFAR-100's 100 classes). Needed because
several papers' original benchmarks (e.g. SCL-NL / Chou et al. 2020's
Table 1) report results on real CIFAR-10 specifically, not a CIFAR-100
subset -- same image format (32x32 RGB) as CIFAR-100, so this reuses
build_image_loaders_full exactly like image_builtin.py's MNIST family,
just RGB instead of grayscale.

Uses the same normalization constants as src/cifar100_subset.py's CIFAR-100
loader (_MEAN/_STD) for consistency across the project.
"""

import numpy as np

from src.pipeline.datasets.generic_loaders import build_image_loaders_full
from src.pipeline.datasets.specs import DatasetSpec

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.247, 0.2435, 0.2616)

_CACHE: dict = {}  # keyed by data_dir


def _get_raw(data_dir: str) -> dict:
    if data_dir not in _CACHE:
        from torchvision.datasets import CIFAR10
        print("Loading cifar10 from disk (one-time)...", flush=True)
        train = CIFAR10(root=data_dir, train=True, download=True)
        test = CIFAR10(root=data_dir, train=False, download=True)
        _CACHE[data_dir] = {
            'train_data': train.data,                   # uint8 [50000,32,32,3]
            'train_targets': np.array(train.targets),    # int64 [50000]
            'test_data': test.data,                      # uint8 [10000,32,32,3]
            'test_targets': np.array(test.targets),       # int64 [10000]
        }
        print(f"cifar10 cached in memory ({len(train)} train / {len(test)} test samples).", flush=True)
    return _CACHE[data_dir]


def _loader(C, k, data_dir, seed, log_dir, batch_size):
    raw = _get_raw(data_dir)
    spec = DATASETS_BY_NAME['cifar10']
    if C != spec.fixed_num_classes:
        raise ValueError(f"'cifar10' has a fixed {spec.fixed_num_classes} classes, got C={C}")
    return build_image_loaders_full(
        raw['train_data'], raw['train_targets'], raw['test_data'], raw['test_targets'],
        spec, k, batch_size, seed=seed, log_dir=log_dir)


DATASETS_BY_NAME = {
    'cifar10': DatasetSpec(
        name='cifar10', modality='image', backbone='cnn', fixed_num_classes=10,
        supports_pico_family=True, loader=_loader,
        in_channels=3, image_size=32, mean=_CIFAR_MEAN, std=_CIFAR_STD,
        notes='Real CIFAR-10 (not a CIFAR-100 subset). RGB, supports PiCO/ComCo/SoLar.',
    ),
}
