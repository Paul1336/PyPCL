"""Small PiCO-scoped helpers for verify_scripts/pico_verify.py.

Kept separate from a generic verify_scripts/utils.py or models.py on purpose:
other papers' standalone verification scripts live in this same directory
and build their own small helper modules independently (see
verify_scripts/pico_verify.py's module docstring / the launcher contract).
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ArrayDatasetShim:
    """Minimal (.data/.targets/.classes/__len__/__iter__) shim accepted by
    src.data_utils.ComparisonDataGenerator. Same contract as
    src/pipeline/datasets/generic_loaders.py's _ArrayDataset, replicated
    here (rather than imported) since that module is pipeline-internal and
    this script is meant to stand alone."""

    def __init__(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        for i in range(len(self)):
            yield self.data[i], self.targets[i]


class ArrayTestDataset(Dataset):
    """Wraps raw uint8 CIFAR-10 test images + integer labels for
    evaluation: ToTensor + Normalize only, no augmentation (matches
    src/pipeline/datasets/generic_loaders.py's test_transform)."""

    def __init__(self, data, targets, mean, std):
        self.data = data
        self.targets = targets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return self.transform(img), self.targets[idx]


def candidate_masked_init_conf(pl_dataset_raw, C: int, device) -> torch.Tensor:
    """Paper-faithful pseudo-target initialization (PiCO Eq. 6, "Pseudo
    Target Updating"): s_j = 1/|Y| * I(j in Y) -- uniform WITHIN the
    candidate set Y only, zero outside it.

    Same construction as
    src/pipeline/algorithms/runners.py::_candidate_masked_init_conf
    (replicated here rather than imported -- that module pulls in
    pipeline-specific dependencies this standalone script doesn't need).
    This is the paper-faithful counterpart to the original pipeline's
    (known-deviating) `torch.ones(N, C) / C` init, which is uniform over
    ALL C classes regardless of the candidate set -- see
    docs/pico_explanation.md, Step 2 item 3."""
    conf = torch.zeros(len(pl_dataset_raw), C)
    for i, cands in enumerate(pl_dataset_raw.targets):
        conf[i, cands] = 1.0 / len(cands)
    return conf.to(device)
