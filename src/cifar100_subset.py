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
from torch.utils.data import DataLoader, Dataset, Subset

from src.collate import collate_fn
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset

# Reuse the same normalisation as the rest of the codebase.
from src.data.transforms import CIFAR_MEAN as _MEAN, CIFAR_STD as _STD  # noqa: E402  (single source, 2026-11-22)

# Which C classes get selected is deliberately independent of --seeds: a
# seed sweep (run_pipeline.py's --seeds) is meant to vary training-side
# randomness (model init, dataloader shuffling, augmentation, PL/CL
# candidate-label sampling -- see runner.py's _set_seed), while every seed in
# the sweep still trains/evaluates on the exact same C-class subset. Fixed at
# the CLI's original default (42) rather than threaded through from the
# caller's `seed` argument -- mirrors src/pll_init.py's DEFAULT_BIASED_INIT_SEED.
CLASS_SELECTION_SEED = 42

# CIFAR-100's fine-label (0-99) -> coarse-superclass (0-19) mapping, extracted
# directly from the dataset's own pickled 'coarse_labels' field (see
# docs/00_paper_alignment_guide.md's Phase 3 notes) -- not reconstructed from
# memory, to avoid a transcription error across 100 entries. Used by the
# 'cifar100-h' DatasetSpec (src/pipeline/datasets/cifar100_h.py) for
# hierarchical (same-superclass) partial-label generation, per PiCO's
# "CIFAR-100-H" setting.
_CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

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

# Biased complementary-label settings of Gao & Zhang (ICML 2021, Sec 4.3), also
# used by ComCo: total selection probability of each of the three label groups.
CL_BIAS_SETS = {
    'set1': (0.75, 0.24, 0.01),
    'set2': (0.66, 0.24, 0.10),
    'set3': (0.45, 0.30, 0.25),
}


def inject_fixed_k_noise(pl_targets, true_labels, C, eta, seed, cl_bias='uniform', partition=None):
    """Noisy PLL for FIXED candidate-set size (2026-11-17).

    PiCO+ (Wang et al., TPAMI 2023, Sec. 5.1) defines the noise rate eta = 1 - P(y in Y | y): the
    probability that the ground-truth label is NOT in the candidate set. Its generator is the
    variable-size q-flip one (true label kept w.p. 1 - eta, each false label w.p. q). Here |Y| = k is
    kept fixed: each sample is independently made noisy w.p. eta, and a noisy sample gets a candidate
    set of k FALSE labels, i.e. its complementary set becomes {y} + (m-1) other labels (m = C - k):
      * uniform: Y = k labels drawn uniformly from the C-1 false labels;
      * set1/2/3: the m-1 non-true complementary labels are drawn without replacement with the same
        per-class biased weights as generate_pl_dataset_biased_cl (the partition passed in), Y = rest.
    Clean samples keep exactly the candidate set produced by the clean generator (the noise uses its own
    RNG seeded from (seed, eta)), so a noisy run differs from the eta=0 run of the same seed only on the
    noisy rows. Returns (new_targets list, noisy bool np.ndarray)."""
    rng = np.random.RandomState([int(seed), int(round(eta * 1000)), 20261117])
    all_labels = np.arange(C)
    m = C - len(pl_targets[0])
    weights = None
    if cl_bias != 'uniform':
        weights = []
        for y in range(C):
            w = np.zeros(C)
            for g, pg in zip(partition[y], CL_BIAS_SETS[cl_bias]):
                w[g] = pg / len(g)
            weights.append(w / w.sum())
    noisy = rng.rand(len(pl_targets)) < eta
    out = list(pl_targets)
    for i in np.flatnonzero(noisy):
        y = int(true_labels[i])
        k = len(pl_targets[i])
        others = np.delete(all_labels, y)
        if weights is None:
            cand = np.sort(rng.choice(others, size=k, replace=False))
        else:
            comp = rng.choice(all_labels, size=C - k - 1, replace=False, p=weights[y]) if C - k - 1 > 0 else []
            cand = np.setdiff1d(others, comp)
        assert len(cand) == k and y not in cand
        out[i] = torch.tensor(cand, dtype=pl_targets[i].dtype)
    return out, noisy


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


def _validate_generation_args(total_classes, n_partial_labels, hierarchical, hierarchical_q, q, bimodal,
                              cl_bias, noise_eta):
    if cl_bias != 'uniform' and (cl_bias not in CL_BIAS_SETS or hierarchical_q is not None
                                 or bimodal is not None):
        raise ValueError(f"cl_bias={cl_bias!r} needs fixed-k or q generation and one of {sorted(CL_BIAS_SETS)}")
    if noise_eta > 0 and (hierarchical or hierarchical_q is not None or q is not None or bimodal is not None
                          or not 2 <= n_partial_labels <= total_classes - 1):
        raise ValueError("noise_eta is only wired up for fixed-k (uniform / biased-CL) generation, 2 <= k <= C-1")
    if hierarchical_q is None and q is None and bimodal is None and not (1 <= n_partial_labels <= total_classes - 1):
        raise ValueError(
            f"n_partial_labels must be in [1, {total_classes - 1}], got {n_partial_labels}"
        )


def load_class_subset(total_classes, data_dir):
    """CIFAR-100 restricted to the fixed C-class subset (CLASS_SELECTION_SEED, independent of the
    training seed), labels remapped to 0..C-1. Returns a dict."""
    raw = _get_cifar100_raw(data_dir)
    selected_indices = select_cifar100_classes(total_classes, seed=CLASS_SELECTION_SEED)
    label_remap = {orig: new for new, orig in enumerate(selected_indices)}
    train_mask = np.isin(raw['train_targets'], selected_indices)
    test_mask = np.isin(raw['test_targets'], selected_indices)
    return {
        'selected_indices': selected_indices,
        'selected_class_names': [raw['classes'][i] for i in selected_indices],
        'train_data': raw['train_data'][train_mask],
        'train_targets': [label_remap[t] for t in raw['train_targets'][train_mask]],
        'test_data': raw['test_data'][test_mask],
        'test_targets': [label_remap[t] for t in raw['test_targets'][test_mask]],
    }


def _generate_candidates(sub, total_classes, n_partial_labels, hierarchical, hierarchical_q, q, bimodal, cl_bias):
    """Dispatch to the ComparisonDataGenerator method for the requested setting.
    Draws from the GLOBAL np.random state (seeded by runner._set_seed) -- order matters.
    Returns (pl_dataset_raw, cl_bias_partition, generator)."""
    subset_ds = _SubsetDataset(data=sub['train_data'], targets=sub['train_targets'],
                               classes=[str(c) for c in range(total_classes)])
    generator = ComparisonDataGenerator(subset_ds, noise_type='clean', eta=0.0)
    partition = None
    coarse = lambda: [_CIFAR100_FINE_TO_COARSE[orig] for orig in sub['selected_indices']]  # noqa: E731
    if bimodal is not None:
        pl = generator.generate_pl_dataset_bimodal(*bimodal)                      # (k_lo, k_hi, p_lo)
    elif hierarchical_q is not None:
        pl = generator.generate_pl_dataset_hierarchical_variable(q=hierarchical_q, class_coarse=coarse())
    elif q is not None and cl_bias != 'uniform':
        if hierarchical:
            raise ValueError("cl_bias and hierarchical are mutually exclusive")
        # variable-size biased CL: E|Y| = 1 + q(C-1), same mean as the uniform q branch
        pl, partition, _ = generator.generate_pl_dataset_biased_cl_variable(
            mean_k=1 + q * (total_classes - 1), group_probs=CL_BIAS_SETS[cl_bias],
            partition_seed=CLASS_SELECTION_SEED)
    elif q is not None:
        pl, _ = generator.generate_variable_pl_cl_datasets(q=q, num_classes=total_classes)
    elif n_partial_labels == 1:  # candidate set = {true label}
        pl = WeaklySupervisedDataset(sub['train_data'],
                                     [torch.tensor([t], dtype=torch.long) for t in sub['train_targets']])
    elif cl_bias != 'uniform':
        if hierarchical:
            raise ValueError("cl_bias and hierarchical are mutually exclusive")
        pl, partition = generator.generate_pl_dataset_biased_cl(
            k=n_partial_labels, group_probs=CL_BIAS_SETS[cl_bias], partition_seed=CLASS_SELECTION_SEED)
    elif hierarchical:
        pl = generator.generate_pl_dataset_hierarchical(k=n_partial_labels, class_coarse=coarse())
    else:
        pl = generator.generate_pl_dataset(k=n_partial_labels)
    return pl, partition, generator


def complement_targets(pl_targets, total_classes):
    """CL target lists = sorted complement of each PL candidate set (consistent partition)."""
    all_class_set = set(range(total_classes))
    return [torch.tensor(sorted(all_class_set - set(t.tolist())), dtype=torch.long) for t in pl_targets]


def _write_selection_log(log_dir, sub, total_classes, n_partial_labels, hierarchical_q, q, bimodal, cl_bias,
                         partition, noise_info, seed):
    is_probabilistic = hierarchical_q is not None or q is not None or bimodal is not None
    log_info = {
        "selected_class_indices":  sub['selected_indices'],
        "selected_class_names":    sub['selected_class_names'],
        "total_classes":           total_classes,
        "n_partial_labels":        n_partial_labels if not is_probabilistic else None,
        "hierarchical_q":          hierarchical_q,
        "q":                       q,
        "bimodal":                 list(bimodal) if bimodal is not None else None,
        "cl_bias":                 cl_bias,
        "cl_bias_partition":       partition,
        "noise":                   noise_info,
        "n_complementary_labels":  (total_classes - n_partial_labels) if not is_probabilistic else None,
        "seed":                    seed,
        "n_train":                 len(sub['train_targets']),
        "n_test":                  len(sub['test_targets']),
        "timestamp":               datetime.now().isoformat(),
    }
    os.makedirs(log_dir, exist_ok=True)
    if bimodal is not None:
        log_suffix = f"bimodal{bimodal[0]}-{bimodal[1]}-p{bimodal[2]}"
    elif hierarchical_q is not None:
        log_suffix = f"hq{hierarchical_q}"
    elif q is not None:
        log_suffix = f"q{q}" + (f"_bias-{cl_bias}" if cl_bias != 'uniform' else '')
    else:
        log_suffix = f"{n_partial_labels}k" + (f"_bias-{cl_bias}" if cl_bias != 'uniform' else '')
    log_fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{total_classes}classes_{log_suffix}.json"
    with open(os.path.join(log_dir, log_fname), "w") as f:
        json.dump(log_info, f, indent=2)
    names, ntr, nte = sub['selected_class_names'][:3], len(sub['train_targets']), len(sub['test_targets'])
    if hierarchical_q is not None:
        print(f"  [log] {total_classes} classes ({names}...), hierarchical q={hierarchical_q}, train={ntr}, test={nte}")
    elif q is not None:
        print(f"  [log] {total_classes} classes ({names}...), q={q}, train={ntr}, test={nte}")
    else:
        print(f"  [log] {total_classes} classes ({names}...), k={n_partial_labels}, "
              f"m={total_classes - n_partial_labels}, train={ntr}, test={nte}")
    return log_info


def prepare_cifar100_subset(
    total_classes: int,
    n_partial_labels: int,
    data_dir: str,
    seed: int = 42,
    log_dir: str = "logs/cifar100_subset",
    hierarchical: bool = False,
    hierarchical_q: float = None,
    q: float = None,
    bimodal: tuple = None,
    cl_bias: str = 'uniform',
    noise_eta: float = 0.0,
):
    """
    Loads CIFAR-100, selects `total_classes` classes, generates PL labels
    (k = n_partial_labels candidates per sample, always including the true label)
    and derives consistent CL labels as the complement of each PL set.
    
    Args:
        total_classes:    Number of classes to use (2–100).
        n_partial_labels: k, number of candidate labels per sample (1 ≤ k ≤ C-1).
                           Ignored when hierarchical_q or q is set (both are
                           probabilistic, with no fixed candidate-set size).
        data_dir:         Directory where CIFAR-100 is / will be downloaded.
        seed:             Does NOT affect which classes get selected (see
                           CLASS_SELECTION_SEED) -- caller (runner.py's
                           _set_seed) is expected to have already seeded the
                           global random/numpy/torch RNGs with this value
                           before calling, so PL/CL candidate-label sampling
                           (ComparisonDataGenerator, which draws from the
                           global np.random state) varies with it. Still
                           recorded in log_info for provenance.
        log_dir:          Directory for JSON selection logs.
        hierarchical:     If True, candidate labels are drawn preferentially from
                           the same CIFAR-100 coarse superclass as the true label,
                           with a FIXED candidate-set size k (an adaptation of
                           PiCO's "CIFAR-100-H" setting to fit this repo's k-swept
                           CLI -- see ComparisonDataGenerator.generate_pl_dataset_hierarchical).
                           Mutually exclusive with hierarchical_q.
        hierarchical_q:   If set (0-1), uses the PAPER-EXACT CIFAR-100-H generation
                           (Wang et al., ICLR 2022, Sec 4.4): each same-superclass
                           false label independently included with probability q
                           (see ComparisonDataGenerator.generate_pl_dataset_hierarchical_variable).
                           Takes priority over `hierarchical` and `n_partial_labels`
                           if both are given.
        q:                If set (0-1), each false label (no superclass
                           restriction -- unlike hierarchical_q) is independently
                           included w.p. q (see generate_variable_pl_cl_datasets).
                           This is --only_q's general (non-CIFAR-100-H-specific)
                           path. Takes priority over `hierarchical`/`n_partial_labels`
                           if given; mutually exclusive with hierarchical_q (caller's
                           responsibility -- runner.py never sets both).
        cl_bias:          'uniform' (default, unchanged behaviour) or one of
                           CL_BIAS_SETS ('set1'/'set2'/'set3'): biased
                           complementary-label generation of ComCo / Gao & Zhang
                           2021 (see ComparisonDataGenerator.generate_pl_dataset_biased_cl).
                           Only for the fixed-k path with 2 <= k <= C-1.
    
    Returns:
        pl_dataset_raw:   WeaklySupervisedDataset with PL targets.
        cl_dataset_raw:   WeaklySupervisedDataset with CL targets (complement of PL).
        original_targets: torch.Tensor of remapped true labels (0..C-1).
        test_info:        (test_data: np.ndarray, test_targets: list[int]).
        log_info:         dict with selection metadata.
    """
    _validate_generation_args(total_classes, n_partial_labels, hierarchical, hierarchical_q, q, bimodal,
                              cl_bias, noise_eta)
    sub = load_class_subset(total_classes, data_dir)
    pl_dataset_raw, cl_bias_partition, generator = _generate_candidates(
        sub, total_classes, n_partial_labels, hierarchical, hierarchical_q, q, bimodal, cl_bias)

    noise_info = None
    if noise_eta > 0:
        new_t, noisy = inject_fixed_k_noise(pl_dataset_raw.targets, sub['train_targets'], total_classes, noise_eta,
                                            seed, cl_bias=cl_bias, partition=cl_bias_partition)
        pl_dataset_raw = WeaklySupervisedDataset(pl_dataset_raw.data, new_t)
        noise_info = {'eta': noise_eta, 'n_noisy': int(noisy.sum()), 'frac_noisy': float(noisy.mean())}
        print(f"  [noise] eta={noise_eta}: {noise_info['n_noisy']}/{len(noisy)} "
              f"({noise_info['frac_noisy']:.4f}) samples have y not in Y (|Y|={n_partial_labels} kept)")

    cl_dataset_raw = WeaklySupervisedDataset(sub['train_data'], complement_targets(pl_dataset_raw.targets,
                                                                                  total_classes))
    log_info = _write_selection_log(log_dir, sub, total_classes, n_partial_labels, hierarchical_q, q, bimodal,
                                    cl_bias, cl_bias_partition, noise_info, seed)
    return (pl_dataset_raw, cl_dataset_raw, generator.original_targets,
            (sub['test_data'], sub['test_targets']), log_info)


class SizeBucketBatchSampler:
    """Batches that each contain samples of a single candidate-set size.

    Every epoch: shuffle within each size group, cut it into full batches
    (drop_last per group, so MoCo's queue_size % batch_size still holds),
    then shuffle the order of all batches. Under a bimodal-k dataset this
    makes a batch's average set size equal to one of the two modes instead
    of concentrating at the overall mean -- what the BatchCond alpha switch
    needs in order to actually switch."""

    def __init__(self, sizes, batch_size: int):
        self.batch_size = batch_size
        groups = {}
        for i, s in enumerate(sizes):
            groups.setdefault(s, []).append(i)
        self.groups = [np.array(g) for g in groups.values()]

    def __iter__(self):
        batches = []
        for g in self.groups:
            perm = g[np.random.permutation(len(g))]
            n_full = len(perm) // self.batch_size
            batches += [perm[i * self.batch_size:(i + 1) * self.batch_size].tolist() for i in range(n_full)]
        for j in np.random.permutation(len(batches)):
            yield batches[j]

    def __len__(self):
        return sum(len(g) // self.batch_size for g in self.groups)


def get_subset_dataloaders_full(
    pl_dataset_raw,
    cl_dataset_raw,
    original_targets,
    test_info,
    batch_size: int,
    batching: str = 'random',
    holdout_frac: float = 0.0,
):
    """
    holdout_frac > 0 removes that fraction of the training set from every
    training loader and returns it as loaders['holdout'] = {'x', 'partial',
    'y'} (test-transformed images, candidate-set mask, true labels -- the
    true labels are for diagnostics only). Used by the bandit runners, whose
    reward is an unbiased label-free accuracy estimate on this split.

    batching='bucketed' builds the PiCO/ComCo loaders from a
    SizeBucketBatchSampler (every batch holds samples of one candidate-set
    size only) instead of a plain shuffle -- see that class.

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
    n_train = len(pl_dataset_raw.targets)
    holdout_idx = None
    train_idx = np.arange(n_train)
    if holdout_frac > 0:
        perm = np.random.permutation(n_train)
        n_hold = int(round(holdout_frac * n_train))
        holdout_idx, train_idx = np.sort(perm[:n_hold]), np.sort(perm[n_hold:])

    def _train_view(ds):
        # Subset keeps each sample's GLOBAL index (ComCo/PiCO datasets
        # return it), so per-sample confidence buffers stay N-sized.
        return ds if holdout_idx is None else Subset(ds, train_idx.tolist())

    pl_loader = DataLoader(_train_view(pl_dataset), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    cl_loader = DataLoader(_train_view(cl_dataset), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    pico_dataset  = _train_view(PicoDataset(pl_dataset_raw, original_targets))
    comco_dataset = _train_view(ComCoDataset(cl_dataset_raw, original_targets))
    if batching == 'bucketed':
        sizes = [len(pl_dataset_raw.targets[i]) for i in train_idx]
        pico_loader  = DataLoader(pico_dataset, batch_sampler=SizeBucketBatchSampler(sizes, batch_size),
                                  collate_fn=pico_collate_fn, pin_memory=True)
        comco_loader = DataLoader(comco_dataset, batch_sampler=SizeBucketBatchSampler(sizes, batch_size),
                                  collate_fn=comco_collate_fn, pin_memory=True)
    elif batching == 'random':
        pico_loader  = DataLoader(pico_dataset,  batch_size=batch_size, shuffle=True,
                                  drop_last=True, collate_fn=pico_collate_fn, pin_memory=True)
        comco_loader = DataLoader(comco_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, collate_fn=comco_collate_fn, pin_memory=True)
    else:
        raise ValueError(f"unknown batching '{batching}' (random | bucketed)")

    test_data, test_targets = test_info
    test_dataset = _ArrayTestDataset(test_data, test_targets, transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loaders = {
        'pl': pl_loader, 'cl': cl_loader,
        'pico': pico_loader, 'comco': comco_loader,
        'test': test_loader,
    }
    if holdout_idx is not None:
        C = len(set(original_targets.tolist()))
        x = torch.tensor(pl_dataset_raw.data[holdout_idx]).permute(0, 3, 1, 2).float() / 255.0
        x = (x - torch.tensor(_MEAN).view(1, 3, 1, 1)) / torch.tensor(_STD).view(1, 3, 1, 1)
        partial = torch.zeros(len(holdout_idx), C)
        for row, i in enumerate(holdout_idx):
            partial[row, pl_dataset_raw.targets[i]] = 1.0
        loaders['holdout'] = {'x': x, 'partial': partial, 'y': original_targets[holdout_idx].clone()}
        loaders['train_idx'] = train_idx
    return loaders
