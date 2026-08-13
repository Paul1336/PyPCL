"""
PRODEN 超參數消融實驗：比較四種設定

  Setting A      : CIFAR-100 子集 C=10, k=7  — 我們的超參（SGD wd=1e-4, 無 scheduler）
  Setting A_paper: CIFAR-100 子集 C=10, k=7  — 原論文超參（SGD wd=1e-3, cosine annealing）
  Setting B      : CIFAR-10        C=10, k=7  — 我們的超參
  Setting C      : CIFAR-10        C=10, q=0.666 — 我們的超參

原論文參考: Lv et al., "Progressive Identification of True Labels
            for Partial-Label Learning", NeurIPS 2020.
論文超參: lr=0.01, momentum=0.9, wd=1e-3, cosine annealing, epochs=500
          (原文用 ResNet-32；此處維持 ResNet-18 以公平比較架構以外的差異)

Usage:
    python scripts/run_proden_ablation.py
    python scripts/run_proden_ablation.py --epochs 500 --paper_epochs 500
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
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset
from src.models import create_model
from src.proden_loss import ProdenLoss

# ── Constants ─────────────────────────────────────────────────────────────────
C     = 10
BS    = 256
K     = 7
Q     = 0.666
CHUNK = 10

_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.247,  0.2435, 0.2616]

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


# ── Dataset: returns (img, index) for ProdenLoss ──────────────────────────────

class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data  # numpy uint8 [N, 32, 32, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        return _TRAIN_TF(img), idx


# ── Data builders ─────────────────────────────────────────────────────────────

def build_cifar100_subset(data_dir, seed=42):
    """Setting A / A_paper: 共用同一份資料"""
    pl_raw, cl_raw, orig_targets, test_info, log_info = prepare_cifar100_subset(
        total_classes=C, n_partial_labels=K,
        data_dir=data_dir, seed=seed,
    )
    loaders = get_subset_dataloaders(pl_raw, cl_raw, orig_targets, test_info, BS)
    print(f"[A/A_paper] CIFAR-100 子集  classes={log_info['selected_class_names']}")
    print(f"            k={K} (constant)  train={log_info['n_train']}  test={log_info['n_test']}")
    return pl_raw, loaders['test']


def build_cifar10_constant(data_dir):
    """Setting B: CIFAR-10，k=7 (constant)"""
    train_raw = CIFAR10(root=data_dir, train=True,  download=True)
    test_ds   = CIFAR10(root=data_dir, train=False, download=True, transform=_TEST_TF)
    gen    = ComparisonDataGenerator(train_raw)
    pl_raw = gen.generate_pl_dataset(k=K)
    avg_k  = sum(len(t) for t in pl_raw.targets) / len(pl_raw.targets)
    print(f"[B] CIFAR-10  k={K} (constant)  avg_k={avg_k:.2f}  train={len(pl_raw)}")
    return pl_raw, DataLoader(test_ds, batch_size=BS, shuffle=False)


def build_cifar10_binomial(data_dir, q=Q):
    """Setting C: CIFAR-10，binomial flipping"""
    train_raw = CIFAR10(root=data_dir, train=True,  download=True)
    test_ds   = CIFAR10(root=data_dir, train=False, download=True, transform=_TEST_TF)
    gen = ComparisonDataGenerator(train_raw)
    pl_raw, _ = gen.generate_variable_pl_cl_datasets(q=q, num_classes=C)
    avg_k = sum(len(t) for t in pl_raw.targets) / len(pl_raw.targets)
    print(f"[C] CIFAR-10  q={q} (binomial)  avg_k={avg_k:.2f}  train={len(pl_raw)}")
    return pl_raw, DataLoader(test_ds, batch_size=BS, shuffle=False)


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


def run_proden(label, pl_raw, test_loader, epochs, lr, momentum, wd, device,
               use_cosine=False):
    """
    use_cosine=True → CosineAnnealingLR (lr: 0.01 → 0 over `epochs`)  [原論文設定]
    use_cosine=False → 固定 lr                                          [我們的設定]
    """
    model   = create_model(C).to(device)
    loss_fn = ProdenLoss(pl_raw.targets, C).to(device)
    opt     = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
        if use_cosine else None
    )

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
        if scheduler:
            scheduler.step()

        if ep % CHUNK == 0 or ep == epochs:
            final_acc = _evaluate(model, test_loader, device)
            avg_s     = (time.perf_counter() - t_start) / ep
            eta_s     = avg_s * (epochs - ep)
            cur_lr    = opt.param_groups[0]['lr']
            print(f'  [{label}]  ep {ep:>3}/{epochs}  lr={cur_lr:.5f}  '
                  f'{avg_s:.1f}s/ep  acc={final_acc:.2f}%  ETA {_fmt_eta(eta_s)}',
                  flush=True)

    total_s = time.perf_counter() - t_start
    print(f'  [{label}] DONE  final={final_acc:.2f}%  ({total_s/60:.1f} min)\n')

    del model, loss_fn, opt, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=200,
                        help='epochs for settings A/B/C (our hyperparams)')
    parser.add_argument('--paper_epochs', type=int,   default=500,
                        help='epochs for setting A_paper (original paper)')
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--mom',          type=float, default=0.9)
    parser.add_argument('--wd',           type=float, default=1e-4,
                        help='weight decay for A/B/C (our setting)')
    parser.add_argument('--paper_wd',     type=float, default=1e-3,
                        help='weight decay for A_paper (original paper)')
    parser.add_argument('--q',            type=float, default=Q)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--data_dir',     type=str,   default='data/')
    parser.add_argument('--only',         type=str,   default=None,
                        help='只跑指定設定，逗號分隔。可選: A,Ap,B,C  (Ap = A*)')
    args = parser.parse_args()

    only = set(args.only.split(',')) if args.only else {'A', 'Ap', 'B', 'C'}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if only & {'A', 'B', 'C'}:
        print(f'[A/B/C] SGD lr={args.lr}  mom={args.mom}  wd={args.wd}  '
              f'epochs={args.epochs}  scheduler=None')
    if 'Ap' in only:
        print(f'[A*]    SGD lr={args.lr}  mom={args.mom}  wd={args.paper_wd}  '
              f'epochs={args.paper_epochs}  scheduler=CosineAnnealing')
    print()

    # ── 準備資料（只載入需要的） ───────────────────────────────────────────────
    need_cifar100 = only & {'A', 'Ap'}
    need_cifar10  = only & {'B', 'C'}

    if need_cifar100:
        pl_A, test_A = build_cifar100_subset(args.data_dir, seed=args.seed)
    if 'B' in only:
        pl_B, test_B = build_cifar10_constant(args.data_dir)
    if 'C' in only:
        pl_C, test_C = build_cifar10_binomial(args.data_dir, q=args.q)
    print()

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    results = {}

    if 'A' in only:
        results['A'] = run_proden(
            f'A  CIFAR100-sub k={K} (ours)',
            pl_A, test_A,
            epochs=args.epochs, lr=args.lr, momentum=args.mom, wd=args.wd,
            device=device, use_cosine=False,
        )

    if 'Ap' in only:
        results['Ap'] = run_proden(
            f'A* CIFAR100-sub k={K} (paper)',
            pl_A, test_A,
            epochs=args.paper_epochs, lr=args.lr, momentum=args.mom, wd=args.paper_wd,
            device=device, use_cosine=True,
        )

    if 'B' in only:
        results['B'] = run_proden(
            f'B  CIFAR10 k={K} const (ours)',
            pl_B, test_B,
            epochs=args.epochs, lr=args.lr, momentum=args.mom, wd=args.wd,
            device=device, use_cosine=False,
        )

    if 'C' in only:
        results['C'] = run_proden(
            f'C  CIFAR10 q={args.q} binom (ours)',
            pl_C, test_C,
            epochs=args.epochs, lr=args.lr, momentum=args.mom, wd=args.wd,
            device=device, use_cosine=False,
        )

    # ── 結果 ──────────────────────────────────────────────────────────────────
    print('=' * 65)
    print(f'  PRODEN  C={C}  k={K}  lr={args.lr}  momentum={args.mom}')
    if 'A'  in results:
        print(f'  A   CIFAR-100 子集 k={K}  wd={args.wd}  ep={args.epochs}            : {results["A"]:.2f}%')
    if 'Ap' in results:
        print(f'  A*  CIFAR-100 子集 k={K}  wd={args.paper_wd}  ep={args.paper_epochs}  cosine : {results["Ap"]:.2f}%')
    if 'B'  in results:
        print(f'  B   CIFAR-10      k={K}  wd={args.wd}  ep={args.epochs}            : {results["B"]:.2f}%')
    if 'C'  in results:
        print(f'  C   CIFAR-10      q={args.q} wd={args.wd}  ep={args.epochs}          : {results["C"]:.2f}%')
    if 'A' in results and 'Ap' in results:
        print()
        print(f'  A vs A* 差距: {results["Ap"] - results["A"]:+.2f}%  (正 = 原論文超參較好)')
    print('=' * 65)


if __name__ == '__main__':
    main()
