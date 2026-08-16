"""ComCo-specific helpers for verify_scripts/comco_verify.py.

Builds the CIFAR-10 unbiased single-complementary-label dataloaders (paper
Table 1 setting: Ishida et al. 2017 uniform single-CL scheme), the ComCo
model + losses + optimizer at paper-exact hyperparameters, and the CSV/
summary output helpers.

Kept separate from comco_verify.py's CLI/orchestration so a synthetic-data
smoke test can import build_model_and_losses / train_one_epoch / evaluate /
write_result_row directly without pulling in the real CIFAR-10 download --
see the bottom of comco_verify.py's module docstring for how the smoke test
substitutes build_dataloaders() with a fake loader while reusing everything
else unchanged.

Reused as-is from src/ (see CLAUDE.md's "Reuse from the existing repo"
guidance for this verification pass):
  - src/pipeline/datasets/cifar10.py's _get_raw() -- raw CIFAR-10 arrays.
  - src/data_utils.py's ComparisonDataGenerator.generate_cl_dataset(m=1) --
    uniform single-complementary-label generation (Ishida et al. 2017).
  - src/data_utils.py's ComCoDataset -- weak/strong augmentation pair +
    dense complementary-mask wrapper ComCo's dataloader needs.
  - src/comco/model.py's ComCoModel -- dual-encoder + MoCo queue architecture.
  - src/comco/utils_loss.py's ComCoContrastiveLoss -- already paper-verified.
  - src/comco/fixed_utils_loss.py's FixedComCoCLSLoss -- paper-faithful
    (unscaled) SCL-NL classification loss; the ORIGINAL ComCoCLSLoss in
    src/comco/utils_loss.py has a confirmed (C-1)/(C-m) scaling bug for the
    multi-CL case (see that file's docstring) -- not used here.
  - src/engine.py's train_comco_epoch / evaluate_model -- the exact training-
    loop orchestration already validated in
    src/pipeline/algorithms/runners.py::run_comco_fixed.
"""

import csv
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.collate import comco_collate_fn
from src.comco.fixed_utils_loss import FixedComCoCLSLoss
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoContrastiveLoss
from src.data_utils import ComCoDataset, ComparisonDataGenerator, WeaklySupervisedDataset
from src.engine import evaluate_model, train_comco_epoch
from src.pipeline.datasets.cifar10 import _CIFAR_MEAN, _CIFAR_STD, _get_raw

NUM_CLASSES = 10
PAPER_TARGET_ACCURACY = 96.70
CONFIG_LABEL = "unbiased_single_cl,lr=0.02,tau=0.17,lambda=0.3"

# Paper-exact ComCo hyperparameters (Jiang, Sun & Tian, Neural Networks 2024),
# CIFAR-10/100 column -- confirmed via direct PDF extraction (see task brief):
# tau=0.17, lambda=0.3 (CIFAR value; MNIST-family uses 0.5), K=1, moco_queue=
# 8192 (CIFAR value), warmup_pos=100, warmup_neg=1. Also matches this repo's
# config.yaml `comco:` section defaults.
COMCO_HPARAMS = dict(
    low_dim=128,
    moco_queue=8192,
    moco_m=0.999,
    loss_weight=0.3,
    temperature=0.17,
    top_k=1,
    warmup_neg=1,
    warmup_pos=100,
)
LR = 0.02              # paper: grid-searched over {1e-1,...,1e-5}, best for CIFAR-10/100 = 0.02
WEIGHT_DECAY = 1e-4
BATCH_SIZE_DEFAULT = 256
EPOCHS_DEFAULT = 1000  # CIFAR-10/100 specifically use 1000 (paper uses 800 for other datasets)


class _ArrayShim:
    """Minimal (.data/.targets/.classes/__iter__) contract expected by
    ComparisonDataGenerator. Reproduced here (rather than imported) because
    the equivalent shim in src/pipeline/datasets/generic_loaders.py
    (_ArrayDataset) is a module-private helper of a sibling module not meant
    for cross-module reuse."""

    def __init__(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        for i in range(len(self)):
            yield self.data[i], self.targets[i]


def build_dataloaders(data_dir: str, batch_size: int, seed: int):
    """Real CIFAR-10, unbiased single-complementary-label (m=1) generation --
    paper Table 1 setting (uniform random draw of exactly 1 complementary
    label from the C-1 non-true classes per sample, standard Ishida et al.
    2017 scheme). Returns (train_loader, test_loader).
    """
    raw = _get_raw(data_dir)
    train_data, train_targets = raw['train_data'], raw['train_targets']
    test_data, test_targets = raw['test_data'], raw['test_targets']

    classes = [str(c) for c in range(NUM_CLASSES)]
    shim = _ArrayShim(train_data, list(train_targets), classes)
    generator = ComparisonDataGenerator(shim, noise_type='clean', eta=0.0)

    cl_raw = generator.generate_cl_dataset(m=1)
    original_targets = generator.original_targets

    train_dataset = ComCoDataset(cl_raw, original_targets, image_size=32,
                                  mean=_CIFAR_MEAN, std=_CIFAR_STD)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=comco_collate_fn, pin_memory=torch.cuda.is_available())

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    test_dataset = WeaklySupervisedDataset(test_data, list(test_targets), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def build_model_and_losses(epochs: int, device):
    """ComCoModel + FixedComCoCLSLoss (paper-faithful classification loss --
    see module docstring) + ComCoContrastiveLoss (reused as-is) + Adam
    optimizer, all at paper-exact hyperparameters."""
    comco_args = {
        'num_class': NUM_CLASSES,
        'epochs': epochs,
        'low_dim': COMCO_HPARAMS['low_dim'],
        'moco_queue': COMCO_HPARAMS['moco_queue'],
        'moco_m': COMCO_HPARAMS['moco_m'],
        'loss_weight': COMCO_HPARAMS['loss_weight'],
        'temperature': COMCO_HPARAMS['temperature'],
        'top_k': COMCO_HPARAMS['top_k'],
        'warmup_neg': COMCO_HPARAMS['warmup_neg'],
        'warmup_pos': COMCO_HPARAMS['warmup_pos'],
    }
    model = ComCoModel(comco_args).to(device)
    cls_loss = FixedComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'], top_k=comco_args['top_k'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return model, cls_loss, cont_loss, optimizer, comco_args


def train_one_epoch(comco_args, model, loader, cls_loss, cont_loss, optimizer, epoch, device):
    """Thin named wrapper around src.engine.train_comco_epoch (the exact
    training-loop orchestration already validated in
    src/pipeline/algorithms/runners.py::run_comco_fixed), so this module's
    public surface names the single call site the smoke test also exercises."""
    return train_comco_epoch(comco_args, model, loader, cls_loss, cont_loss, optimizer, epoch, device)


def evaluate(model, test_loader, device) -> float:
    return evaluate_model(model, test_loader, device)


def write_result_row(csv_path: str, seed: int, epochs: int, final_accuracy: float,
                      training_time_s: float, notes: str = '') -> None:
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['dataset', 'config', 'seed', 'epochs', 'final_accuracy',
                              'paper_target_accuracy', 'training_time_s', 'timestamp', 'notes'])
        writer.writerow(['cifar10', CONFIG_LABEL, seed, epochs, f'{final_accuracy:.4f}',
                          PAPER_TARGET_ACCURACY, f'{training_time_s:.2f}',
                          datetime.now().isoformat(), notes])


def print_summary(final_accuracy: float) -> None:
    print(f'[ComCo] dataset=cifar10 setting=unbiased_single_cl '
          f'final_accuracy={final_accuracy:.2f}%  paper_target={PAPER_TARGET_ACCURACY:.2f}%', flush=True)
