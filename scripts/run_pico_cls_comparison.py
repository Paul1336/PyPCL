"""
Runs standalone PiCO-CLS (PiCOCLSLoss, no contrastive) on the CIFAR-100 subset sweep,
then after each (C, k) draws a 2-subplot comparison plot with MCL-LOG, PiCO, ComCo
loaded from the existing run_adam_comparison CSVs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pico_cls_comparison.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pico_cls_comparison.py --only_c 20 --only_k 7
"""

import argparse
import csv
import gc
import glob
import os
import sys
import time
from datetime import datetime
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import prepare_cifar100_subset, get_subset_dataloaders
from src.models import create_model
from src.pico_cls_loss import PiCOCLSLoss
from src.engine import evaluate_model
from torchvision import transforms

# ─── Constants ────────────────────────────────────────────────────────────────

C_VALUES     = [5, 20]
LR           = 3e-4
BS           = 512
WD           = 1e-4
REPORT_EVERY = 10
ALG_NAME     = 'PiCO-CLS'

_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.247,  0.2435, 0.2616]

STYLES = {
    'PiCO-CLS': dict(color='#e377c2', marker='*', linestyle='-',  linewidth=2, markersize=7),
    'MCL-LOG':  dict(color='#d62728', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
}
PLOT_ALGOS = ['PiCO-CLS', 'MCL-LOG', 'PiCO', 'ComCo']

# ─── IndexedDataset ───────────────────────────────────────────────────────────

_TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

class _IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _TRAIN_TF(Image.fromarray(self.data[idx])), idx

# ─── ETA helpers ──────────────────────────────────────────────────────────────

def _fmt_eta(s):
    if s < 90:   return f'{s:.0f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'

def _print_eta(tag, ep_done, ep_total, t_chunk, chunk_size):
    avg_s = t_chunk / chunk_size
    eta   = avg_s * (ep_total - ep_done)
    print(f'  [{tag}]  ep {ep_done:>3}/{ep_total}  '
          f'{avg_s:.1f}s/ep  ETA {_fmt_eta(eta)}', flush=True)

# ─── CSV helpers ──────────────────────────────────────────────────────────────

_CSV_FIELDS = ['total_classes', 'n_partial_labels', 'algorithm',
               'final_accuracy', 'epochs', 'training_time_s', 'timestamp']


def _load_done(csv_path):
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['total_classes']),
                      int(row['n_partial_labels']),
                      row['algorithm']))
    return done


def _append_result(csv_path, C, k, alg, acc, epochs, elapsed_s):
    new_file = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'total_classes':    C,
            'n_partial_labels': k,
            'algorithm':        alg,
            'final_accuracy':   round(acc, 4),
            'epochs':           epochs,
            'training_time_s':  round(elapsed_s, 1),
            'timestamp':        datetime.now().isoformat(),
        })


def _load_all_results(base_dirs):
    """Merge CSVs from multiple directories → res[C][alg][k] = accuracy."""
    res  = {}
    seen = set()
    for base in base_dirs:
        for pat in [os.path.join(base, 'results.csv'),
                    os.path.join(base, 'gpu*', 'results.csv')]:
            for path in sorted(glob.glob(pat)):
                with open(path, newline='') as f:
                    for row in csv.DictReader(f):
                        key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                        if key in seen:
                            continue
                        seen.add(key)
                        C_   = int(row['total_classes'])
                        k_   = int(row['n_partial_labels'])
                        alg  = row['algorithm']
                        acc  = float(row['final_accuracy'])
                        res.setdefault(C_, {}).setdefault(alg, {})[k_] = acc
    return res

# ─── Training ─────────────────────────────────────────────────────────────────

def _train_pico_cls(pl_ds, test_loader, C, epochs, device, tag):
    model   = create_model(C).to(device)
    loss_fn = PiCOCLSLoss(pl_ds.targets, C, epochs=epochs).to(device)
    opt     = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    idx_loader = DataLoader(_IndexedDataset(pl_ds.data),
                            batch_size=BS, shuffle=True, num_workers=2,
                            drop_last=True)

    chunk_t0  = time.perf_counter()
    final_acc = 0.0

    for ep in range(epochs):
        loss_fn.set_conf_ema_m(ep)
        model.train()
        for imgs, indices in idx_loader:
            imgs, indices = imgs.to(device), indices.to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = loss_fn(out, indices)
            loss.backward()
            opt.step()
            loss_fn.update_confidence(out.detach(), indices)

        if (ep + 1) % REPORT_EVERY == 0 or ep + 1 == epochs:
            final_acc = evaluate_model(model, test_loader, device)
            elapsed   = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0  = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    del model, loss_fn, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc

# ─── Plotting ─────────────────────────────────────────────────────────────────

def _draw(ax, alg, k_acc):
    if not k_acc:
        return
    ks, accs = zip(*sorted(k_acc.items()))
    ax.plot(ks, accs, label=alg, **STYLES[alg])


def _setup_ax(ax, title, y_max, ylabel=False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def make_plot(res, out_dir, c_values):
    os.makedirs(out_dir, exist_ok=True)
    vals = [acc for C_d in res.values()
            for alg_d in C_d.values()
            for acc in alg_d.values()]
    ym = 80 if not vals else int(np.ceil(max(vals) / 10) * 10)

    fig, axes = plt.subplots(1, len(c_values), figsize=(8 * len(c_values), 5))
    if len(c_values) == 1:
        axes = [axes]
    fig.suptitle('PiCO-CLS vs MCL-LOG vs PiCO vs ComCo', fontsize=13)

    for col, C in enumerate(c_values):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in PLOT_ALGOS:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')

    fig.tight_layout()
    path = os.path.join(out_dir, 'pico_cls_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] → {path}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./data')
    parser.add_argument('--out_dir',    default='results/pico_cls/')
    parser.add_argument('--adam_dir',   default='results/adam_comparison/',
                        help='Directory with existing run_adam_comparison CSVs')
    parser.add_argument('--plots_dir',  default='plots/pico_cls/')
    parser.add_argument('--log_dir',    default='logs/cifar100_subset')
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--only_c',     type=int, default=None)
    parser.add_argument('--only_k',     type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = os.path.join(args.out_dir, 'results.csv')
    os.makedirs(args.out_dir,    exist_ok=True)
    os.makedirs(args.plots_dir,  exist_ok=True)

    done = _load_done(csv_path)
    c_values = [args.only_c] if args.only_c else C_VALUES

    print(f'Device={device}  epochs={args.epochs}  lr={LR}  bs={BS}')
    print(f'Resume: {len(done)} entries in {csv_path}\n', flush=True)

    for C in c_values:
        k_vals = [args.only_k] if args.only_k else get_k_values(C)
        for k in k_vals:
            if (C, k, ALG_NAME) in done:
                print(f'  [skip] C={C} k={k}', flush=True)
                continue

            print(f'\n--- C={C}  k={k} ---', flush=True)
            pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                total_classes=C, n_partial_labels=k,
                data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
            )

            loaders     = get_subset_dataloaders(pl_ds, cl_ds, orig_targets, test_info, BS)
            test_loader = loaders['test']

            tag = f'{ALG_NAME} C={C} k={k}'
            t0  = time.perf_counter()
            acc = _train_pico_cls(pl_ds, test_loader, C, args.epochs, device, tag)
            elapsed = time.perf_counter() - t0

            _append_result(csv_path, C, k, ALG_NAME, acc, args.epochs, elapsed)
            print(f'  [done] C={C} k={k}  acc={acc:.2f}%  ({elapsed/60:.1f} min)',
                  flush=True)

            # Merge own results with adam_comparison results and plot
            res = _load_all_results([args.out_dir, args.adam_dir])
            make_plot(res, args.plots_dir, c_values)

    print('\nAll done.')


def get_k_values(C):
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [C - 1]))


if __name__ == '__main__':
    main()
