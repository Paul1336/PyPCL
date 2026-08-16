"""MCL-LOG-verification-scoped model + complementary-label-generation code.

Kept in an MCL-LOG-prefixed filename (not a generic `models.py`) because
other papers' verification scripts live alongside this one in
`verify_scripts/` and may define their own, possibly conflicting,
architectures / generators.

Two independent pieces live here:

1. `ResNet34CIFAR` -- torchvision ResNet-34, conv1 swapped for a CIFAR-sized
   stem, mirroring `src/models.py::ResNet18`'s own precedent exactly (see
   module docstring below for why maxpool is left untouched).
2. `sample_complementary_labels` / `generate_complementary_labels` --
   Feng et al. (ICML 2020)'s own combinatorial complementary-label
   generation scheme:

       p(s) = C(k, s) / (2^k - 2),   s = 1, ..., k-1

   i.e. first sample the complementary-set SIZE s from this distribution,
   then sample the actual s-sized complementary set uniformly at random
   from the C(k-1, s) subsets of the k-1 wrong classes. This is NOT the
   same mechanism as `src/data_utils.py::ComparisonDataGenerator`'s
   independent-per-label-inclusion-probability `q` scheme, so it is
   reimplemented fresh here rather than reused.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ResNet34CIFAR(nn.Module):
    """ResNet-34 adapted for CIFAR-sized (32x32) inputs.

    The MCL-LOG paper (Feng et al., ICML 2020) cites ResNet-34 generically
    (He et al. 2016) with no in-text statement of whether the ImageNet stem
    (7x7/stride-2 conv1 + maxpool) is kept or swapped for a CIFAR-friendly
    stem; that level of detail is Appendix-E.1-only and was not present in
    the (10-page, appendix-less) PDF available for this verification pass.

    JUDGMENT CALL: this repo already has an established, working precedent
    for exactly this situation in `src/models.py::ResNet18`: replace ONLY
    `conv1` with a 3x3/stride-1/padding-1 conv (re-initialized with Kaiming
    normal) and leave `maxpool` as torchvision's stock
    `nn.MaxPool2d(3, stride=2, padding=1)`. We mirror that pattern exactly
    here for ResNet-34, for consistency with the rest of the codebase
    rather than as an independent claim about what the paper did.
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
# Complementary-label generation (paper's own p(s) combinatorial scheme)
# ---------------------------------------------------------------------------

def ps_distribution(k: int):
    """Returns (s_values, probs) for p(s) = C(k,s) / (2^k - 2), s=1..k-1.

    sum_{s=1}^{k-1} C(k,s) = 2^k - 2 (all subsets of a k-set except the
    empty set and the full set), so `probs` sums to 1 exactly.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 to have any valid complementary-label size, got k={k}")
    s_values = np.arange(1, k)  # 1 .. k-1 inclusive
    weights = np.array([math.comb(k, int(s)) for s in s_values], dtype=np.float64)
    denom = float(2 ** k - 2)
    probs = weights / denom
    return s_values, probs


def sample_complementary_labels(true_label: int, k: int, rng: np.random.Generator):
    """Draws one complementary label set for a single example.

    1. Sample the set SIZE s ~ p(s) = C(k,s) / (2^k - 2).
    2. Sample the actual s-sized complementary set uniformly at random from
       the C(k-1, s) subsets of the (k-1) wrong classes (i.e.
       p(Ybar | s) = 1 / C(k-1, s)), via `random.sample`.
    """
    s_values, probs = ps_distribution(k)
    s = int(rng.choice(s_values, p=probs))
    wrong_classes = [c for c in range(k) if c != true_label]
    comp_set = random.sample(wrong_classes, s)
    return comp_set


def generate_complementary_labels(true_labels: np.ndarray, k: int, seed: int):
    """Vectorized-over-the-dataset wrapper around `sample_complementary_labels`.

    Returns a torch.LongTensor of shape [N, k-1], padded with -1 up to the
    maximum possible complementary-set size (k-1), compatible with
    `src/mcl_losses.py::MCL_LOG`'s `(complementary_labels != -1)` masking
    convention.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n = len(true_labels)
    max_width = k - 1
    padded = np.full((n, max_width), -1, dtype=np.int64)
    for i in range(n):
        comp_set = sample_complementary_labels(int(true_labels[i]), k, rng)
        padded[i, :len(comp_set)] = comp_set
    return torch.from_numpy(padded)
