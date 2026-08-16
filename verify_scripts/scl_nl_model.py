"""SCL-NL-verification-scoped model + complementary-label-generation code.

Kept in an SCL-NL-prefixed filename (not a generic `models.py`) because other
papers' verification scripts live alongside this one in `verify_scripts/` and
may define their own, possibly conflicting, architectures / generators (e.g.
`verify_scripts/mcl_log_model.py`, built independently in parallel for a
different paper -- some duplication between the two files is expected and
fine).

Two independent pieces live here:

1. `ResNet34CIFAR` -- torchvision ResNet-34, conv1 swapped for a CIFAR-sized
   stem, mirroring `src/models.py::ResNet18`'s own precedent exactly (see
   class docstring for why maxpool is left untouched). This is the SAME
   judgment call made independently in `verify_scripts/mcl_log_model.py`'s
   own `ResNet34CIFAR`, so the two verification scripts stay architecturally
   consistent with each other.
2. `generate_single_complementary_labels` -- wraps
   `src.data_utils.ComparisonDataGenerator.generate_cl_dataset(m=1)` (already
   implemented and directly usable) to draw exactly ONE uniformly-random
   complementary label per sample, matching Chou et al. (ICML 2020)'s
   "Uniform Assumption": Ybar ~ Uniform([K] \\ {y}). `generate_cl_dataset`
   does exactly this: it deletes the true label from the class list and
   draws `m` labels from what remains via `np.random.choice(..., size=m,
   replace=False)` -- for m=1 that is precisely a single uniform draw from
   the K-1 wrong classes, no further adaptation needed. This is simpler and
   more direct than the alternative "k=9 candidate-label-set-complement"
   trick also available in this repo (`--only_k 9` on the C=10 pipeline),
   since it generates the single complementary label directly instead of via
   a detour through partial-label generation + set complement.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ResNet34CIFAR(nn.Module):
    """ResNet-34 adapted for CIFAR-sized (32x32) inputs.

    The SCL-NL paper (Chou et al., ICML 2020) cites ResNet-34 generically
    (He et al. 2016) with no in-text statement of whether the ImageNet stem
    (7x7/stride-2 conv1 + maxpool) is kept or swapped for a CIFAR-friendly
    stem.

    JUDGMENT CALL: this repo already has an established, working precedent
    for exactly this situation in `src/models.py::ResNet18`: replace ONLY
    `conv1` with a 3x3/stride-1/padding-1 conv (re-initialized with Kaiming
    normal) and leave `maxpool` as torchvision's stock
    `nn.MaxPool2d(3, stride=2, padding=1)`. We mirror that pattern exactly
    here for ResNet-34, for consistency with the rest of the codebase (and
    with `verify_scripts/mcl_log_model.py`'s independently-made identical
    choice) rather than as an independent claim about what the paper did.
    """

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.resnet = models.resnet34(num_classes=num_classes, weights=None)

        # Modify the first convolutional layer for small (32x32) images,
        # same as src/models.py::ResNet18 -- maxpool is intentionally left
        # untouched (see class docstring).
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        return self.resnet(x)


def create_model(num_classes, in_channels=3):
    """Creates the CIFAR-adapted ResNet-34 used by this verification script."""
    return ResNet34CIFAR(num_classes=num_classes, in_channels=in_channels)


# ---------------------------------------------------------------------------
# Complementary-label generation (paper's Uniform Assumption, m=1)
# ---------------------------------------------------------------------------

class _ArrayDatasetShim:
    """Minimal (.data/.targets/.classes/__iter__) shim accepted by
    ComparisonDataGenerator -- same contract as
    src/pipeline/datasets/generic_loaders.py's private _ArrayDataset, kept
    local here rather than imported since that class is a private helper of
    the pipeline module, not part of this repo's reuse surface."""

    def __init__(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        for i in range(len(self)):
            yield self.data[i], self.targets[i]


def generate_single_complementary_labels(train_data, train_targets, num_classes: int, seed: int):
    """Draws exactly ONE uniformly-random complementary label per sample.

    Directly reuses `ComparisonDataGenerator.generate_cl_dataset(m=1)`
    (src/data_utils.py) unmodified -- confirmed to implement the paper's
    Uniform Assumption exactly (see module docstring). Returns a
    WeaklySupervisedDataset whose .targets are length-1 LongTensors (each
    holding the single complementary label for that sample), and the
    original (true) targets as a LongTensor, for later accuracy bookkeeping.

    seed controls both this draw and any other numpy global RNG use inside
    ComparisonDataGenerator (which uses np.random.* directly, not a local
    Generator instance).
    """
    np.random.seed(seed)
    classes = [str(c) for c in range(num_classes)]
    shim = _ArrayDatasetShim(train_data, list(train_targets), classes)
    generator = ComparisonDataGenerator(shim, noise_type='clean', eta=0.0)
    cl_dataset_raw = generator.generate_cl_dataset(m=1)
    return cl_dataset_raw, generator.original_targets
