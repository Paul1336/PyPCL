"""
CIFAR-100 class-subset utilities.

Selects a fixed subset of classes from CIFAR-100, generates partial-label (PL)
and complementary-label (CL) datasets that are consistent with each other
(CL = complement of PL for every sample), and builds DataLoaders for use with
the Cour 2011 and MCL-LOG training pipelines.
"""

import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.collate import collate_fn
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset

# Reuse the same normalisation as the rest of the codebase.
_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.247,  0.2435, 0.2616]

# ---------------------------------------------------------------------------
# In-process CIFAR-100 cache
# Avoids re-reading the ~170 MB dataset from disk on every (C, k) call.
# ---------------------------------------------------------------------------
_CIFAR100_CACHE: dict = {}   # keyed by data_dir


def _get_cifar100_raw(data_dir: str) -> dict:
    """
    Loads CIFAR-100 from disk exactly once per process and caches the numpy
    arrays in memory.  Subsequent calls with the same data_dir are O(1).
    """
    if data_dir not in _CIFAR100_CACHE:
        from torchvision.datasets import CIFAR100
        print("Loading CIFAR-100 from disk (one-time) …", flush=True)
        train = CIFAR100(root=data_dir, train=True,  download=True)
        test  = CIFAR100(root=data_dir, train=False, download=True)
        _CIFAR100_CACHE[data_dir] = {
            'train_data':    train.data,                  # uint8 [50000,32,32,3]
            'train_targets': np.array(train.targets),     # int64 [50000]
            'test_data':     test.data,                   # uint8 [10000,32,32,3]
            'test_targets':  np.array(test.targets),      # int64 [10000]
            'classes':       train.classes,               # list[str], length 100
        }
        n_tr, n_te = len(train), len(test)
        print(f"CIFAR-100 cached in memory ({n_tr} train / {n_te} test samples).")
    return _CIFAR100_CACHE[data_dir]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SubsetDataset:
    """Minimal CIFAR-like object accepted by ComparisonDataGenerator."""

    def __init__(self, data: np.ndarray, targets: list, classes: list):
        self.data = data
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        for i in range(len(self)):
            yield self.data[i], self.targets[i]


class _ArrayTestDataset(Dataset):
    """Wraps numpy image arrays + integer labels for evaluation."""

    def __init__(self, data: np.ndarray, targets: list, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_cifar100_classes(total_classes: int, seed: int = 42) -> list:
    """
    Returns a sorted list of `total_classes` class indices (0–99) chosen from
    CIFAR-100 using the given seed.  The same (total_classes, seed) pair always
    yields the same selection, so different k values within a sweep share the
    same class set.
    """
    rng = np.random.RandomState(seed)
    selected = sorted(rng.choice(100, size=total_classes, replace=False).tolist())
    return selected


def prepare_cifar100_subset(
    total_classes: int,
    n_partial_labels: int,
    data_dir: str,
    seed: int = 42,
    log_dir: str = "logs/cifar100_subset",
):
    """
    Loads CIFAR-100, selects `total_classes` classes, generates PL labels
    (k = n_partial_labels candidates per sample, always including the true label)
    and derives consistent CL labels as the complement of each PL set.

    Args:
        total_classes:    Number of classes to use (2–100).
        n_partial_labels: k, number of candidate labels per sample (1 ≤ k ≤ C-1).
        data_dir:         Directory where CIFAR-100 is / will be downloaded.
        seed:             RNG seed for class selection (fixed per total_classes).
        log_dir:          Directory for JSON selection logs.

    Returns:
        pl_dataset_raw:   WeaklySupervisedDataset with PL targets.
        cl_dataset_raw:   WeaklySupervisedDataset with CL targets (complement of PL).
        original_targets: torch.Tensor of remapped true labels (0..C-1).
        test_info:        (test_data: np.ndarray, test_targets: list[int]).
        log_info:         dict with selection metadata.
    """
    if not (1 <= n_partial_labels <= total_classes - 1):
        raise ValueError(
            f"n_partial_labels must be in [1, {total_classes - 1}], got {n_partial_labels}"
        )

    # --- Load CIFAR-100 (cached after first call) ---
    raw = _get_cifar100_raw(data_dir)

    # --- Select classes (deterministic per total_classes + seed) ---
    selected_indices = select_cifar100_classes(total_classes, seed=seed)
    selected_class_names = [raw['classes'][i] for i in selected_indices]
    label_remap = {orig: new for new, orig in enumerate(selected_indices)}

    # --- Filter & remap train ---
    train_mask    = np.isin(raw['train_targets'], selected_indices)
    train_data    = raw['train_data'][train_mask]
    train_targets = [label_remap[t] for t in raw['train_targets'][train_mask]]

    # --- Filter & remap test ---
    test_mask    = np.isin(raw['test_targets'], selected_indices)
    test_data    = raw['test_data'][test_mask]
    test_targets = [label_remap[t] for t in raw['test_targets'][test_mask]]

    # --- Build PL dataset ---
    subset_ds = _SubsetDataset(
        data=train_data,
        targets=train_targets,
        classes=[str(c) for c in range(total_classes)],
    )
    generator = ComparisonDataGenerator(subset_ds, noise_type='clean', eta=0.0)

    if n_partial_labels == 1:
        # Special case: candidate set = {true_label} only.
        pl_targets = [torch.tensor([t], dtype=torch.long) for t in train_targets]
        pl_dataset_raw = WeaklySupervisedDataset(train_data, pl_targets)
    else:
        pl_dataset_raw = generator.generate_pl_dataset(k=n_partial_labels)

    # --- Derive CL as complement of PL (consistent partition) ---
    all_class_set = set(range(total_classes))
    cl_targets = []
    for pl_tensor in pl_dataset_raw.targets:
        cl_set = sorted(all_class_set - set(pl_tensor.tolist()))
        cl_targets.append(torch.tensor(cl_set, dtype=torch.long))
    cl_dataset_raw = WeaklySupervisedDataset(train_data, cl_targets)

    original_targets = generator.original_targets  # torch.Tensor, remapped

    # --- Log selection ---
    log_info = {
        "selected_class_indices":  selected_indices,
        "selected_class_names":    selected_class_names,
        "total_classes":           total_classes,
        "n_partial_labels":        n_partial_labels,
        "n_complementary_labels":  total_classes - n_partial_labels,
        "seed":                    seed,
        "n_train":                 len(train_targets),
        "n_test":                  len(test_targets),
        "timestamp":               datetime.now().isoformat(),
    }
    os.makedirs(log_dir, exist_ok=True)
    log_fname = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{total_classes}classes_{n_partial_labels}k.json"
    )
    with open(os.path.join(log_dir, log_fname), "w") as f:
        json.dump(log_info, f, indent=2)
    print(
        f"  [log] {total_classes} classes ({selected_class_names[:3]}...), "
        f"k={n_partial_labels}, m={total_classes - n_partial_labels}, "
        f"train={len(train_targets)}, test={len(test_targets)}"
    )

    return pl_dataset_raw, cl_dataset_raw, original_targets, (test_data, test_targets), log_info


def get_subset_dataloaders(
    pl_dataset_raw,
    cl_dataset_raw,
    original_targets,
    test_info,
    batch_size: int,
):
    """
    Builds 'pl', 'cl', and 'test' DataLoaders from raw subset datasets.
    Only these three loaders are needed for Cour 2011 and MCL-LOG.
    """
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    pl_dataset = WeaklySupervisedDataset(
        pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform
    )
    cl_dataset = WeaklySupervisedDataset(
        cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform
    )

    pl_loader = DataLoader(pl_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)

    test_data, test_targets = test_info
    test_dataset = _ArrayTestDataset(test_data, test_targets, transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'pl': pl_loader, 'cl': cl_loader, 'test': test_loader}


def get_subset_dataloaders_full(
    pl_dataset_raw,
    cl_dataset_raw,
    original_targets,
    test_info,
    batch_size: int,
):
    """
    Builds all 5 DataLoaders needed for PiCO and ComCo:
    'pl', 'cl', 'pico', 'comco', 'test'.

    PiCO / ComCo loaders use drop_last=True so the MoCo queue assertion
    (queue_size % batch_size == 0) is always satisfied.
    """
    from torchvision import transforms
    from src.data_utils import PicoDataset, ComCoDataset
    from src.collate import pico_collate_fn, comco_collate_fn

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    pl_dataset = WeaklySupervisedDataset(
        pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform
    )
    cl_dataset = WeaklySupervisedDataset(
        cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform
    )
    pl_loader = DataLoader(pl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    pico_dataset  = PicoDataset(pl_dataset_raw, original_targets)
    pico_loader   = DataLoader(pico_dataset,  batch_size=batch_size, shuffle=True,
                               drop_last=True, collate_fn=pico_collate_fn, pin_memory=True)

    comco_dataset = ComCoDataset(cl_dataset_raw, original_targets)
    comco_loader  = DataLoader(comco_dataset, batch_size=batch_size, shuffle=True,
                               drop_last=True, collate_fn=comco_collate_fn, pin_memory=True)

    test_data, test_targets = test_info
    test_dataset = _ArrayTestDataset(test_data, test_targets, transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'pl': pl_loader, 'cl': cl_loader,
        'pico': pico_loader, 'comco': comco_loader,
        'test': test_loader,
    }
