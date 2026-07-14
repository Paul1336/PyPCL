"""
PRODEN + Cour2011 on CIFAR-10 with binomial flipping (variable PL).

PL generation: true label always in set; each wrong label added with prob q.
Default: q=0.7, C=10.

PRODEN : SGD  momentum=0.9  lr=0.01  bs=256  wd=1e-4
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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clpl_loss import CLPLSquaredHingeLoss
from src.collate import collate_fn
from src.data_utils import ComparisonDataGenerator, WeaklySupervisedDataset
from src.engine import train_algorithm
from src.models import create_model
from src.proden_loss import proden

CHUNK   = 10
C       = 10
_MEAN   = [0.4914, 0.4822, 0.4465]
_STD    = [0.247,  0.2435, 0.2616]


def _fmt_eta(s):
    if s < 90:   return f'{s:.0f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'


def build_loaders(data_dir, q, batch_size):
    train_raw = CIFAR10(root=data_dir, train=True,  download=True)
    test_raw  = CIFAR10(root=data_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(_MEAN, _STD),
                        ]))

    gen = ComparisonDataGenerator(train_raw)
    pl_raw, _ = gen.generate_variable_pl_cl_datasets(q=q, num_classes=C)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    pl_dataset = WeaklySupervisedDataset(pl_raw.data, pl_raw.targets, transform=train_tf)
    pl_loader  = DataLoader(pl_dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_raw,  batch_size=batch_size, shuffle=False)

    avg_k = sum(len(t) for t in pl_raw.targets) / len(pl_raw.targets)
    print(f'PL generated: q={q}, avg candidates per sample = {avg_k:.2f}')
    return {'pl': pl_loader, 'test': test_loader}


def run_method(name, loss_fn, opt_factory, loaders, epochs, device):
    model = create_model(C).to(device)
    opt   = opt_factory(model.parameters())

    t_start   = time.perf_counter()
    final_acc = 0.0

    for ep_start in range(0, epochs, CHUNK):
        chunk = min(CHUNK, epochs - ep_start)
        t0    = time.perf_counter()
        accs  = train_algorithm(model, loaders['pl'], loaders['test'],
                                loss_fn, opt, chunk, device)
        elapsed   = time.perf_counter() - t0
        final_acc = accs[-1]

        ep_done = ep_start + chunk
        avg_s   = elapsed / chunk
        eta_s   = avg_s * (epochs - ep_done)
        print(f'  [{name}]  ep {ep_done:>3}/{epochs}  '
              f'{avg_s:.1f}s/ep  acc={final_acc:.2f}%  ETA {_fmt_eta(eta_s)}',
              flush=True)

    total_s = time.perf_counter() - t_start
    print(f'  [{name}] DONE  final={final_acc:.2f}%  ({total_s/60:.1f} min)\n')

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


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

    acc_proden = run_method(
        'PRODEN', proden(),
        lambda p: optim.SGD(p, lr=args.proden_lr,
                            momentum=args.proden_mom, weight_decay=args.wd),
        loaders, args.epochs, device,
    )

    acc_cour = run_method(
        'Cour2011', CLPLSquaredHingeLoss(),
        lambda p: optim.Adam(p, lr=args.cour_lr, weight_decay=args.wd),
        loaders, args.epochs, device,
    )

    print('=' * 50)
    print(f'  CIFAR-10  q={args.q}  epochs={args.epochs}  bs={args.bs}')
    print(f'  PRODEN  (SGD  lr={args.proden_lr}): {acc_proden:.2f}%')
    print(f'  Cour2011(Adam lr={args.cour_lr}  ): {acc_cour:.2f}%')
    print('=' * 50)


if __name__ == '__main__':
    main()
