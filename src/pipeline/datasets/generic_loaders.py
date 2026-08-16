"""Dataset-agnostic loader-building helpers shared by every non-CIFAR-100
dataset spec. Mirrors src/cifar100_subset.py's prepare_cifar100_subset +
get_subset_dataloaders_full, but parameterized by DatasetSpec instead of
being hardcoded to CIFAR, and without the "select C classes out of a bigger
pool" step (that's specific to the CIFAR-100 class-subset design; every
other dataset here uses its full native class set).
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.collate import collate_fn, comco_collate_fn, pico_collate_fn
from src.data_utils import ComCoDataset, ComparisonDataGenerator, PicoDataset, WeaklySupervisedDataset


class _ArrayDataset:
    """Minimal (.data/.targets/.classes/__iter__) shim accepted by
    ComparisonDataGenerator -- same contract as cifar100_subset.py's
    _SubsetDataset, reusable for any array-backed (non-lazy) dataset."""

    def __init__(self, data, targets, classes):
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

    def __init__(self, data, targets, transform=None):
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


class LazyImagePathDataset:
    """Like _ArrayDataset, but .data holds file path strings instead of
    preloaded arrays -- for datasets too large / too variable-resolution to
    load into memory up front (CUB-200, SUN397). __getitem__ opens the file
    lazily. Accepted by ComparisonDataGenerator (same .data/.targets/.classes
    contract) and by WeaklySupervisedDataset (Image.open instead of
    Image.fromarray)."""

    def __init__(self, paths: list, targets: list, classes: list):
        self.data = paths
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        from PIL import Image
        for i in range(len(self)):
            yield Image.open(self.data[i]).convert('RGB'), self.targets[i]


class LazyWeaklySupervisedDataset(Dataset):
    """Path-based counterpart to src.data_utils.WeaklySupervisedDataset."""

    def __init__(self, paths, targets, transform=None):
        self.data = paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]


class _LazyArrayTestDataset(Dataset):
    def __init__(self, paths, targets, transform=None):
        self.data = paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def _make_pl_cl(train_data, train_targets, C, k, wrap_cls, q=None):
    """Shared PL/CL generation, identical logic to cifar100_subset.py's
    prepare_cifar100_subset, factored out so both build_image_loaders_full
    and (lazy-path) callers can reuse it.

    q, if given, overrides k entirely: each false label is independently
    included w.p. q (generate_variable_pl_cl_datasets) instead of a fixed-size
    k-candidate set. Mutually exclusive with k in practice (callers only ever
    set one), enforced by the caller (--only_q vs --only_k in runner.py)."""
    classes = [str(c) for c in range(C)]
    subset_ds = _ArrayDataset(train_data, train_targets, classes)
    generator = ComparisonDataGenerator(subset_ds, noise_type='clean', eta=0.0)

    if q is not None:
        pl_raw, cl_raw = generator.generate_variable_pl_cl_datasets(q=q, num_classes=C)
        pl_dataset_raw = wrap_cls(train_data, pl_raw.targets)
        cl_dataset_raw = wrap_cls(train_data, cl_raw.targets)
        return pl_dataset_raw, cl_dataset_raw, generator.original_targets

    if k == 1:
        pl_targets = [torch.tensor([t], dtype=torch.long) for t in train_targets]
        pl_dataset_raw = wrap_cls(train_data, pl_targets)
    else:
        raw = generator.generate_pl_dataset(k=k)
        pl_dataset_raw = wrap_cls(train_data, raw.targets)

    all_class_set = set(range(C))
    cl_targets = []
    for pl_tensor in pl_dataset_raw.targets:
        cl_set = sorted(all_class_set - set(pl_tensor.tolist()))
        cl_targets.append(torch.tensor(cl_set, dtype=torch.long))
    cl_dataset_raw = wrap_cls(train_data, cl_targets)

    return pl_dataset_raw, cl_dataset_raw, generator.original_targets


def build_image_loaders_full(train_data, train_targets, test_data, test_targets, spec, k,
                              batch_size, seed=42, log_dir='logs/dataset_subset', q=None):
    """Generic version of cifar100_subset.py's prepare_cifar100_subset +
    get_subset_dataloaders_full, parameterized by DatasetSpec (image_size,
    in_channels, mean, std, supports_pico_family). train_data/test_data are
    numpy arrays (already the dataset's full native class set --
    fixed_num_classes classes, no subset selection).

    q, if given, switches to variable (--only_q) candidate-set generation and
    k is ignored entirely (no [1, C-1] validation either, since q doesn't
    have that constraint).
    """
    C = spec.fixed_num_classes
    if q is None and not (1 <= k <= C - 1):
        raise ValueError(f"k must be in [1, {C - 1}] for dataset '{spec.name}', got {k}")

    pl_dataset_raw, cl_dataset_raw, original_targets = _make_pl_cl(
        train_data, list(train_targets), C, k, WeaklySupervisedDataset, q=q)

    os.makedirs(log_dir, exist_ok=True)
    log_info = {
        'dataset': spec.name, 'total_classes': C,
        'n_partial_labels': k if q is None else None,
        'n_complementary_labels': (C - k) if q is None else None,
        'q': q, 'seed': seed,
        'n_train': len(train_targets), 'n_test': len(test_targets),
        'timestamp': datetime.now().isoformat(),
    }
    log_suffix = f"{k}k" if q is None else f"q{q}"
    log_fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.name}_{C}classes_{log_suffix}.json"
    with open(os.path.join(log_dir, log_fname), 'w') as f:
        json.dump(log_info, f, indent=2)
    if q is None:
        print(f"  [log] dataset={spec.name} C={C} k={k} m={C - k} "
              f"train={len(train_targets)} test={len(test_targets)}", flush=True)
    else:
        print(f"  [log] dataset={spec.name} C={C} q={q} "
              f"train={len(train_targets)} test={len(test_targets)}", flush=True)

    mean, std, size = spec.mean, spec.std, spec.image_size
    train_transform = transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    pl_dataset = WeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
    cl_dataset = WeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)
    pl_loader = DataLoader(pl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = _ArrayTestDataset(test_data, list(test_targets), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loaders = {'pl': pl_loader, 'cl': cl_loader, 'test': test_loader}

    if spec.supports_pico_family:
        pico_dataset = PicoDataset(pl_dataset_raw, original_targets, image_size=size, mean=mean, std=std)
        loaders['pico'] = DataLoader(pico_dataset, batch_size=batch_size, shuffle=True,
                                      drop_last=True, collate_fn=pico_collate_fn, pin_memory=True)
        comco_dataset = ComCoDataset(cl_dataset_raw, original_targets, image_size=size, mean=mean, std=std)
        loaders['comco'] = DataLoader(comco_dataset, batch_size=batch_size, shuffle=True,
                                       drop_last=True, collate_fn=comco_collate_fn, pin_memory=True)

    return loaders, pl_dataset_raw, original_targets


def build_lazy_image_loaders_full(train_paths, train_targets, test_paths, test_targets, spec, k,
                                   batch_size, seed=42, log_dir='logs/dataset_subset'):
    """Same as build_image_loaders_full, but for path-based (lazy-loaded)
    image datasets (CUB-200, SUN397) where preloading every image into a
    numpy array up front isn't practical (variable resolution, large total
    size). Uses LazyWeaklySupervisedDataset instead of WeaklySupervisedDataset.
    """
    C = spec.fixed_num_classes
    if not (1 <= k <= C - 1):
        raise ValueError(f"k must be in [1, {C - 1}] for dataset '{spec.name}', got {k}")

    pl_dataset_raw, cl_dataset_raw, original_targets = _make_pl_cl(
        train_paths, list(train_targets), C, k, LazyWeaklySupervisedDataset)

    os.makedirs(log_dir, exist_ok=True)
    log_info = {
        'dataset': spec.name, 'total_classes': C, 'n_partial_labels': k,
        'n_complementary_labels': C - k, 'seed': seed,
        'n_train': len(train_targets), 'n_test': len(test_targets),
        'timestamp': datetime.now().isoformat(),
    }
    log_fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.name}_{C}classes_{k}k.json"
    with open(os.path.join(log_dir, log_fname), 'w') as f:
        json.dump(log_info, f, indent=2)
    print(f"  [log] dataset={spec.name} C={C} k={k} m={C - k} "
          f"train={len(train_targets)} test={len(test_targets)}", flush=True)

    mean, std, size = spec.mean, spec.std, spec.image_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    pl_dataset = LazyWeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
    cl_dataset = LazyWeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)
    pl_loader = DataLoader(pl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = _LazyArrayTestDataset(test_paths, list(test_targets), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # PiCO/ComCo need a raw-array-backed PicoDataset/ComCoDataset (ToPILImage
    # first op) -- not compatible with lazy path-based loading without further
    # plumbing, so lazy-path datasets never support the PiCO family regardless
    # of spec.supports_pico_family (enforced by the caller / IMAGE_ONLY_ALGORITHMS
    # filter reading spec.supports_pico_family, which loader authors must set
    # to False for any lazy_paths dataset).
    return {'pl': pl_loader, 'cl': cl_loader, 'test': test_loader}, pl_dataset_raw, original_targets


def build_tabular_loaders(features, targets, spec, k, batch_size, seed=42, log_dir='logs/dataset_subset',
                           test_split=0.2):
    """Builds pl/cl/test loaders for a tabular (feature-vector) dataset.
    No pico/comco keys -- IMAGE_ONLY_ALGORITHMS filtering in runner.py keeps
    PiCO*/ComCo*/SoLar off any spec with supports_pico_family=False, which
    every tabular DatasetSpec must set.

    features: np.ndarray [N, D] float32, already standardized by the caller.
    targets: np.ndarray [N] int, 0..C-1.
    """
    rng = np.random.RandomState(seed)
    N = len(targets)
    idx = rng.permutation(N)
    n_test = max(1, int(N * test_split))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_features, train_targets = features[train_idx], targets[train_idx].tolist()
    test_features, test_targets = features[test_idx], targets[test_idx].tolist()

    C = spec.fixed_num_classes
    if not (1 <= k <= C - 1):
        raise ValueError(f"k must be in [1, {C - 1}] for dataset '{spec.name}', got {k}")

    pl_dataset_raw, cl_dataset_raw, original_targets = _make_pl_cl(
        train_features, train_targets, C, k, _FeatureDataset)

    os.makedirs(log_dir, exist_ok=True)
    log_info = {
        'dataset': spec.name, 'total_classes': C, 'n_partial_labels': k,
        'n_complementary_labels': C - k, 'seed': seed,
        'n_train': len(train_targets), 'n_test': len(test_targets),
        'timestamp': datetime.now().isoformat(),
    }
    log_fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.name}_{C}classes_{k}k.json"
    with open(os.path.join(log_dir, log_fname), 'w') as f:
        json.dump(log_info, f, indent=2)
    print(f"  [log] dataset={spec.name} C={C} k={k} m={C - k} "
          f"train={len(train_targets)} test={len(test_targets)}", flush=True)

    pl_dataset = _FeatureDataset(pl_dataset_raw.data, pl_dataset_raw.targets)
    cl_dataset = _FeatureDataset(cl_dataset_raw.data, cl_dataset_raw.targets)
    pl_loader = DataLoader(pl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = _FeatureTestDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'pl': pl_loader, 'cl': cl_loader, 'test': test_loader}, pl_dataset_raw, original_targets


class _FeatureDataset(Dataset):
    """Tabular counterpart to WeaklySupervisedDataset: returns a raw feature
    vector (already standardized) instead of a PIL-image-derived tensor."""

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.as_tensor(self.data[idx], dtype=torch.float32), self.targets[idx]


class _FeatureTestDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.as_tensor(self.data[idx], dtype=torch.float32), self.targets[idx]


def build_preambiguous_loaders(pl_data, pl_candidate_sets, test_data, test_targets, spec, batch_size,
                                seed=42, log_dir='logs/dataset_subset'):
    """For datasets that ship REAL candidate label sets already (the 5
    classic real-world PLL datasets, CLPL's original data) -- skips
    ComparisonDataGenerator entirely. `k` doesn't apply (candidate-set size
    varies per real sample, it isn't a sweep parameter).

    CL is synthetically derived as the complement of the real PL set, since
    none of these are native CLL benchmarks -- this is announced loudly, not
    silently, per docs/00_paper_alignment_guide.md's Phase 4 design.

    pl_data: list/array of feature vectors OR image arrays (modality decides
             which Dataset wrapper is used -- image if spec.modality=='image',
             else tabular).
    pl_candidate_sets: list of 1-D arrays/lists, one real candidate set per sample.
    """
    print(f"  [WARNING] dataset='{spec.name}' is pre-ambiguous (real candidate label sets from "
          f"the original paper). Complementary labels below are SYNTHETICALLY DERIVED as the "
          f"complement of the real PL set -- this dataset has no native CLL ground truth. "
          f"See {spec.name}'s entry in the relevant docs/*_explanation.md.", flush=True)

    C = spec.fixed_num_classes
    pl_targets = [torch.as_tensor(sorted(set(int(c) for c in s)), dtype=torch.long) for s in pl_candidate_sets]
    all_class_set = set(range(C))
    cl_targets = [torch.as_tensor(sorted(all_class_set - set(t.tolist())), dtype=torch.long) for t in pl_targets]

    is_image = spec.modality == 'image'
    wrap_cls = WeaklySupervisedDataset if is_image else _FeatureDataset
    pl_dataset_raw = wrap_cls(pl_data, pl_targets)
    cl_dataset_raw = wrap_cls(pl_data, cl_targets)
    # No independent "true label" signal beyond the (possibly ambiguous) PL
    # set itself for some of these sources; callers that have real ground
    # truth pass it through test_targets/via the caller-specific loader.
    original_targets = None

    os.makedirs(log_dir, exist_ok=True)
    bs = spec.batch_size_override or batch_size
    log_info = {
        'dataset': spec.name, 'total_classes': C, 'mode': 'preambiguous',
        'n_train': len(pl_targets), 'n_test': len(test_targets),
        'seed': seed, 'timestamp': datetime.now().isoformat(),
    }
    log_fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.name}_preambiguous.json"
    with open(os.path.join(log_dir, log_fname), 'w') as f:
        json.dump(log_info, f, indent=2)

    if is_image:
        mean, std, size = spec.mean, spec.std, spec.image_size
        train_transform = transforms.Compose([
            transforms.RandomCrop(size, padding=4) if isinstance(pl_data[0], np.ndarray) and
            pl_data[0].shape[0] == size else transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        pl_dataset = WeaklySupervisedDataset(pl_dataset_raw.data, pl_dataset_raw.targets, transform=train_transform)
        cl_dataset = WeaklySupervisedDataset(cl_dataset_raw.data, cl_dataset_raw.targets, transform=train_transform)
        test_dataset = _ArrayTestDataset(test_data, list(test_targets), transform=test_transform)
    else:
        pl_dataset = _FeatureDataset(pl_dataset_raw.data, pl_dataset_raw.targets)
        cl_dataset = _FeatureDataset(cl_dataset_raw.data, cl_dataset_raw.targets)
        test_dataset = _FeatureTestDataset(test_data, list(test_targets))

    pl_loader = DataLoader(pl_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(cl_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return {'pl': pl_loader, 'cl': cl_loader, 'test': test_loader}, pl_dataset_raw, original_targets
