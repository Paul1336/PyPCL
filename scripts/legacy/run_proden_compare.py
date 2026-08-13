"""
比較 PRODEN 在三種資料設定下的表現：

  Setting A: CIFAR-100 子集  C=10, k=7  (constant-k，與 run_adam_comparison.py 相同)
  Setting B: CIFAR-10        C=10, k=7  (constant-k)
  Setting C: CIFAR-10        C=10, q=0.666 (binomial flipping，可變 k)

三種設定均使用 ProdenLoss（跨 epoch confidence 累積版）
Optimizer: SGD  lr=0.01  momentum=0.9  bs=256  wd=1e-4

Usage:
    python scripts/run_proden_compare.py
    python scripts/run_proden_compare.py --epochs 500 --data_dir data/
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import prepare_cifar100_subset, get_subset_dataloaders
from src.collate import collate_fn
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset
from src.models import create_model
from src.proden_loss import ProdenLoss

C      = 10
BS     = 256
_MEAN  = [0.4914, 0.4822, 0.4465]
_STD   = [0.247,  0.2435, 0.2616]
CHUNK  = 10

_TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
_TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ── Dataset that returns (img, index) for ProdenLoss ─────────────────────────

class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data          # numpy uint8 [N, 32, 32, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return _TRAIN_TF(img), idx


# ── Data builders ─────────────────────────────────────────────────────────────

def build_cifar100_subset(data_dir, seed=42):
    """Setting A: CIFAR-100 子集，C=10，k=7 (constant)"""
    pl_raw, cl_raw, orig_targets, test_info, log_info = prepare_cifar100_subset(
        total_classes=C, n_partial_labels=7,
        data_dir=data_dir, seed=seed,
    )
    loaders = get_subset_dataloaders(pl_raw, cl_raw, orig_targets, test_info, BS)
    print(f"[A] CIFAR-100 子集  classes={log_info['selected_class_names']}")
    print(f"    k=7 (constant)  train={log_info['n_train']}  test={log_info['n_test']}")
    return pl_raw, loaders['test']


def build_cifar10_constant(data_dir):
    """Setting B: CIFAR-10，k=7 (constant)"""
    train_raw = CIFAR10(root=data_dir, train=True,  download=True)
    test_ds   = CIFAR10(root=data_dir, train=False, download=True, transform=_TEST_TF)
    gen    = ComparisonDataGenerator(train_raw)
    pl_raw = gen.generate_pl_dataset(k=7)
    avg_k  = sum(len(t) for t in pl_raw.targets) / len(pl_raw.targets)
    print(f"[B] CIFAR-10  k=7 (constant)  avg_k={avg_k:.2f}  train={len(pl_raw)}")
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)
    return pl_raw, test_loader


def build_cifar10_binomial(data_dir, q=0.666):
    """Setting C: CIFAR-10，binomial flipping，q=0.666"""
    train_raw = CIFAR10(root=data_dir, train=True,  download=True)
    test_ds   = CIFAR10(root=data_dir, train=False, download=True, transform=_TEST_TF)
    gen = ComparisonDataGenerator(train_raw)
    pl_raw, _ = gen.generate_variable_pl_cl_datasets(q=q, num_classes=C)
    avg_k = sum(len(t) for t in pl_raw.targets) / len(pl_raw.targets)
    print(f"[C] CIFAR-10  q={q} (binomial)  avg_k={avg_k:.2f}  train={len(pl_raw)}")
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)
    return pl_raw, test_loader


# ── Training ──────────────────────────────────────────────────────────────────

def _fmt_eta(s):
    if s < 90:   return f'{s:.0f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def run_proden(label, pl_raw, test_loader, epochs, lr, momentum, wd, device):
    model   = create_model(C).to(device)
    loss_fn = ProdenLoss(pl_raw.targets, C).to(device)
    opt     = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    idx_loader = DataLoader(IndexedDataset(pl_raw.data),
                            batch_size=BS, shuffle=True, num_workers=2)

    t_start   = time.perf_counter()
    final_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        for imgs, indices in idx_loader:
            imgs, indices = imgs.to(device), indices.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), indices).backward()
            opt.step()

        if ep % CHUNK == 0 or ep == epochs:
            final_acc = _evaluate(model, test_loader, device)
            avg_s     = (time.perf_counter() - t_start) / ep
            eta_s     = avg_s * (epochs - ep)
            print(f'  [{label}]  ep {ep:>3}/{epochs}  '
                  f'{avg_s:.1f}s/ep  acc={final_acc:.2f}%  ETA {_fmt_eta(eta_s)}',
                  flush=True)

    total_s = time.perf_counter() - t_start
    print(f'  [{label}] DONE  final={final_acc:.2f}%  ({total_s/60:.1f} min)\n')

    del model, loss_fn, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',  type=int,   default=200)
    parser.add_argument('--lr',      type=float, default=0.01)
    parser.add_argument('--mom',     type=float, default=0.9)
    parser.add_argument('--wd',      type=float, default=1e-4)
    parser.add_argument('--q',       type=float, default=0.666)
    parser.add_argument('--seed',    type=int,   default=42)
    parser.add_argument('--data_dir',type=str,   default='data/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Epochs : {args.epochs}  lr={args.lr}  momentum={args.mom}  '
          f'bs={BS}  wd={args.wd}\n', flush=True)

    # ── 準備資料 ──────────────────────────────────────────────────────────────
    pl_A, test_A = build_cifar100_subset(args.data_dir, seed=args.seed)
    pl_B, test_B = build_cifar10_constant(args.data_dir)
    pl_C, test_C = build_cifar10_binomial(args.data_dir, q=args.q)
    print()

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    acc_A = run_proden('A: CIFAR100-subset k=7', pl_A, test_A,
                       args.epochs, args.lr, args.mom, args.wd, device)

    acc_B = run_proden('B: CIFAR10 k=7 const', pl_B, test_B,
                       args.epochs, args.lr, args.mom, args.wd, device)

    acc_C = run_proden(f'C: CIFAR10 q={args.q} binom', pl_C, test_C,
                       args.epochs, args.lr, args.mom, args.wd, device)

    # ── 結果 ──────────────────────────────────────────────────────────────────
    print('=' * 55)
    print(f'  PRODEN  SGD lr={args.lr}  momentum={args.mom}  epochs={args.epochs}')
    print(f'  A  CIFAR-100 子集 C=10 k=7 (constant) : {acc_A:.2f}%')
    print(f'  B  CIFAR-10      C=10 k=7 (constant)  : {acc_B:.2f}%')
    print(f'  C  CIFAR-10      C=10 q={args.q} (binom) : {acc_C:.2f}%')
    print('=' * 55)


if __name__ == '__main__':
    main()
