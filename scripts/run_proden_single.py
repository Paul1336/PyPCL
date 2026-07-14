"""
PRODEN + Cour2011 on CIFAR-10 with binomial flipping (variable PL).

PL generation: true label always in set; each wrong label added with prob q.
Default: q=0.7, C=10.

PRODEN : SGD  momentum=0.9  lr=0.01  bs=256  wd=1e-4  (cross-epoch confidence)
Cour   : Adam              lr=3e-4  bs=256  wd=1e-4

Usage:
    python scripts/run_proden_single.py
    python scripts/run_proden_single.py --q 0.1 --epochs 500 --data_dir data/
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

from src.clpl_loss import CLPLSquaredHingeLoss
from src.collate import collate_fn
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset
from src.engine import train_algorithm
from src.models import create_model
from src.proden_loss import ProdenLoss

CHUNK  = 10
C      = 10
_MEAN  = [0.4914, 0.4822, 0.4465]
_STD   = [0.247,  0.2435, 0.2616]


# ── Dataset that also returns sample index ────────────────────────────────────

class IndexedDataset(Dataset):
    """Wraps a raw dataset (data + partial_targets); returns (img, index)."""

    def __init__(self, data, transform=None):
        self.data      = data       # numpy uint8 [N, 32, 32, 3]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, idx


# ── Data preparation ──────────────────────────────────────────────────────────

def build_loaders(data_dir, q, batch_size):
    train_raw = CIFAR10(root=data_dir, train=True,  download=True)
    test_tf   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_raw  = CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    gen = ComparisonDataGenerator(train_raw)
    pl_raw, _ = gen.generate_variable_pl_cl_datasets(q=q, num_classes=C)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    # Loader for Cour: returns (img, partial_labels) with -1 padding
    pl_dataset  = WeaklySupervisedDataset(pl_raw.data, pl_raw.targets, transform=train_tf)
    cour_loader = DataLoader(pl_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)

    # Loader for PRODEN: returns (img, index)
    idx_dataset   = IndexedDataset(pl_raw.data, transform=train_tf)
    proden_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_raw, batch_size=batch_size, shuffle=False)

    avg_k = sum(len(t) for t in pl_raw.targets) / len(pl_raw.targets)
    print(f'PL generated: q={q}, avg candidates/sample = {avg_k:.2f}')
    return {
        'pl_raw':  pl_raw,
        'proden':  proden_loader,
        'cour':    cour_loader,
        'test':    test_loader,
    }


# ── Training helpers ──────────────────────────────────────────────────────────

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


def _train_proden_epoch(model, loss_fn, loader, optimizer, device):
    """One epoch: uses stored confidence for loss, updates confidence in-place."""
    model.train()
    for imgs, indices in loader:
        imgs, indices = imgs.to(device), indices.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = loss_fn(outputs, indices)   # updates loss_fn.conf in-place
        loss.backward()
        optimizer.step()


def run_proden(loaders, partial_targets, epochs, lr, momentum, wd, device):
    model   = create_model(C).to(device)
    loss_fn = ProdenLoss(partial_targets, C).to(device)
    opt     = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    t_start   = time.perf_counter()
    final_acc = 0.0

    for ep in range(1, epochs + 1):
        _train_proden_epoch(model, loss_fn, loaders['proden'], opt, device)

        if ep % CHUNK == 0 or ep == epochs:
            acc       = _evaluate(model, loaders['test'], device)
            final_acc = acc
            avg_s     = (time.perf_counter() - t_start) / ep
            eta_s     = avg_s * (epochs - ep)
            print(f'  [PRODEN]  ep {ep:>3}/{epochs}  '
                  f'{avg_s:.1f}s/ep  acc={acc:.2f}%  ETA {_fmt_eta(eta_s)}',
                  flush=True)

    total_s = time.perf_counter() - t_start
    print(f'  [PRODEN] DONE  final={final_acc:.2f}%  ({total_s/60:.1f} min)\n')
    del model, loss_fn, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


def run_cour(loaders, epochs, lr, wd, device):
    model = create_model(C).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    t_start   = time.perf_counter()
    final_acc = 0.0
    loss_fn   = CLPLSquaredHingeLoss()

    for ep_start in range(0, epochs, CHUNK):
        chunk     = min(CHUNK, epochs - ep_start)
        t0        = time.perf_counter()
        accs      = train_algorithm(model, loaders['cour'], loaders['test'],
                                    loss_fn, opt, chunk, device)
        elapsed   = time.perf_counter() - t0
        final_acc = accs[-1]

        ep_done = ep_start + chunk
        avg_s   = elapsed / chunk
        eta_s   = avg_s * (epochs - ep_done)
        print(f'  [Cour2011]  ep {ep_done:>3}/{epochs}  '
              f'{avg_s:.1f}s/ep  acc={final_acc:.2f}%  ETA {_fmt_eta(eta_s)}',
              flush=True)

    total_s = time.perf_counter() - t_start
    print(f'  [Cour2011] DONE  final={final_acc:.2f}%  ({total_s/60:.1f} min)\n')
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q',          type=float, default=0.7,  help='false-label inclusion prob')
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--bs',         type=int,   default=256)
    parser.add_argument('--wd',         type=float, default=1e-4)
    parser.add_argument('--proden_lr',  type=float, default=0.01, help='SGD lr for PRODEN')
    parser.add_argument('--proden_mom', type=float, default=0.9,  help='SGD momentum for PRODEN')
    parser.add_argument('--cour_lr',    type=float, default=3e-4, help='Adam lr for Cour2011')
    parser.add_argument('--data_dir',   type=str,   default='data/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device  : {device}')
    print(f'Dataset : CIFAR-10  C={C}  q={args.q}  epochs={args.epochs}  bs={args.bs}')
    print(f'PRODEN  : SGD  lr={args.proden_lr}  momentum={args.proden_mom}')
    print(f'Cour    : Adam lr={args.cour_lr}\n', flush=True)

    loaders = build_loaders(args.data_dir, args.q, args.bs)

    acc_proden = run_proden(
        loaders, loaders['pl_raw'].targets,
        args.epochs, args.proden_lr, args.proden_mom, args.wd, device,
    )

    acc_cour = run_cour(
        loaders, args.epochs, args.cour_lr, args.wd, device,
    )

    print('=' * 50)
    print(f'  CIFAR-10  q={args.q}  epochs={args.epochs}  bs={args.bs}')
    print(f'  PRODEN  (SGD  lr={args.proden_lr}): {acc_proden:.2f}%')
    print(f'  Cour2011(Adam lr={args.cour_lr}  ): {acc_cour:.2f}%')
    print('=' * 50)


if __name__ == '__main__':
    main()
