"""torchvision-built-in image datasets: MNIST, FashionMNIST, KMNIST
(Kuzushiji-MNIST). All three are 28x28 grayscale, 10 classes, small enough to
load fully into memory (like CIFAR-100_subset.py does).

SUN397 was originally planned for this tier too, but torchvision's SUN397 is
path-based (108,754 JPEGs, ~36GB total) rather than array-based, and that
full download is impractical for this pass -- deferred to the lazy-path
loader alongside CUB-200 (see cub200.py) rather than attempted here. Not
silently dropped: this is documented in docs/00_paper_alignment_guide.md.

None of these three datasets support the PiCO/ComCo/SoLar family: they are
grayscale, and PicoDataset/ComCoDataset's augmentation stack (ColorJitter,
RandomGrayscale) is an RGB-only operation.
"""

import numpy as np

from src.pipeline.datasets.generic_loaders import build_image_loaders_full
from src.pipeline.datasets.specs import DatasetSpec

_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)

_CACHE: dict = {}  # keyed by (dataset_name, data_dir)


def _get_raw(dataset_name: str, data_dir: str) -> dict:
    key = (dataset_name, data_dir)
    if key not in _CACHE:
        from torchvision.datasets import MNIST, FashionMNIST, KMNIST
        cls = {'mnist': MNIST, 'fashion-mnist': FashionMNIST, 'kmnist': KMNIST}[dataset_name]
        print(f"Loading {dataset_name} from disk (one-time)...", flush=True)
        train = cls(root=data_dir, train=True, download=True)
        test = cls(root=data_dir, train=False, download=True)
        _CACHE[key] = {
            'train_data': train.data.numpy(),       # uint8 [N,28,28]
            'train_targets': train.targets.numpy(),  # int64 [N]
            'test_data': test.data.numpy(),
            'test_targets': test.targets.numpy(),
        }
        print(f"{dataset_name} cached in memory "
              f"({len(train)} train / {len(test)} test samples).", flush=True)
    return _CACHE[key]


def _make_loader(dataset_name: str):
    def loader(C, k, data_dir, seed, log_dir, batch_size):
        raw = _get_raw(dataset_name, data_dir)
        spec = DATASETS_BY_NAME[dataset_name]
        if C != spec.fixed_num_classes:
            raise ValueError(f"'{dataset_name}' has a fixed {spec.fixed_num_classes} classes, got C={C}")
        return build_image_loaders_full(
            raw['train_data'], raw['train_targets'], raw['test_data'], raw['test_targets'],
            spec, k, batch_size, seed=seed, log_dir=log_dir)
    return loader


def _build_specs() -> dict:
    specs = {}
    for name in ('mnist', 'fashion-mnist', 'kmnist'):
        specs[name] = DatasetSpec(
            name=name, modality='image', backbone='cnn', fixed_num_classes=10,
            supports_pico_family=False, loader=_make_loader(name),
            in_channels=1, image_size=28, mean=_MNIST_MEAN, std=_MNIST_STD,
            notes='Grayscale; PiCO/ComCo/SoLar unsupported (RGB-only augmentation ops).',
        )
    return specs


DATASETS_BY_NAME = _build_specs()
