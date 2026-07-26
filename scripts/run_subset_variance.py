"""
Variance experiment: PRODEN vs ComCo across different CIFAR-100 10-class subsets.

Each seed selects a unique set of 10 classes. Runs start from seed=1 and increase.
After every seed, results are saved and the accuracy distribution bar chart is updated.

Settings:
  C=10, k=7 (constant-k)
  PRODEN : SGD  lr=0.01  momentum=0.9  wd=1e-4  bs=256
  ComCo  : Adam lr=3e-4               wd=1e-4  bs=512

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_subset_variance.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_subset_variance.py --max_seeds 50 --epochs 200
"""

import argparse
import csv
import gc
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
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import prepare_cifar100_subset, get_subset_dataloaders_full
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss
from src.engine import evaluate_model, train_comco_epoch
from src.models import create_model
from src.proden_loss import ProdenLoss

# ─── Constants ────────────────────────────────────────────────────────────────

C            = 10
K            = 7
MAX_SEEDS    = 1000
REPORT_EVERY = 10

PRODEN_LR  = 0.01
PRODEN_MOM = 0.9
PRODEN_WD  = 1e-4
PRODEN_BS  = 256

COMCO_LR = 3e-4
COMCO_WD = 1e-4
COMCO_BS = 512

_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.247,  0.2435, 0.2616]

_TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ─── IndexedDataset (for ProdenLoss) ─────────────────────────────────────────

class _IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _TRAIN_TF(Image.fromarray(self.data[idx])), idx

# ─── ETA helper ───────────────────────────────────────────────────────────────

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

_CSV_FIELDS = ['seed', 'algorithm', 'final_accuracy',
               'epochs', 'training_time_s', 'timestamp']


def _load_done(csv_path):
    """Return set of (seed, alg) already recorded."""
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['seed']), row['algorithm']))
    return done


def _append_result(csv_path, seed, alg, acc, epochs, elapsed_s):
    new_file = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'seed':            seed,
            'algorithm':       alg,
            'final_accuracy':  round(acc, 4),
            'epochs':          epochs,
            'training_time_s': round(elapsed_s, 1),
            'timestamp':       datetime.now().isoformat(),
        })


def _load_results(base_dir):
    """Merge gpu*/results.csv → {alg: [acc, ...]}."""
    import glob as _glob
    res  = {}
    seen = set()
    for path in sorted(_glob.glob(os.path.join(base_dir, 'gpu*', 'results.csv'))):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['seed'], row['algorithm'])
                if key in seen:
                    continue
                seen.add(key)
                res.setdefault(row['algorithm'], []).append(float(row['final_accuracy']))
    return res

# ─── Training ─────────────────────────────────────────────────────────────────

def _train_proden(pl_ds, test_loader, epochs, device, tag):
    model   = create_model(C).to(device)
    loss_fn = ProdenLoss(pl_ds.targets, C).to(device)
    opt     = optim.SGD(model.parameters(), lr=PRODEN_LR,
                        momentum=PRODEN_MOM, weight_decay=PRODEN_WD)
    idx_loader = DataLoader(_IndexedDataset(pl_ds.data),
                            batch_size=PRODEN_BS, shuffle=True, num_workers=2)

    chunk_t0 = time.perf_counter()
    final_acc = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, indices in idx_loader:
            imgs, indices = imgs.to(device), indices.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), indices).backward()
            opt.step()
        if (ep + 1) % REPORT_EVERY == 0 or ep + 1 == epochs:
            final_acc = evaluate_model(model, test_loader, device)
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect(); torch.cuda.empty_cache()

    del model, loss_fn, opt
    gc.collect(); torch.cuda.empty_cache()
    return final_acc


def _train_comco(loaders, epochs, comco_config, device, tag):
    comco_args = {
        'num_class':   C,
        'epochs':      epochs,
        'low_dim':     comco_config['low_dim'],
        'moco_queue':  comco_config['moco_queue'],
        'moco_m':      comco_config['moco_m'],
        'loss_weight': comco_config['loss_weight'],
        'temperature': comco_config['temperature'],
        'top_k':       comco_config['top_k'],
        'warmup_neg':  comco_config['warmup_neg'],
        'warmup_pos':  comco_config['warmup_pos'],
    }
    model     = ComCoModel(comco_args).to(device)
    cls_loss  = ComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'],
                                     top_k=comco_args['top_k'])
    opt = optim.Adam(model.parameters(), lr=COMCO_LR, weight_decay=COMCO_WD)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_comco_epoch(comco_args, model, loaders['comco'],
                          cls_loss, cont_loss, opt, ep, device)
        if (ep + 1) % REPORT_EVERY == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect(); torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect(); torch.cuda.empty_cache()
    return acc

# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {'PRODEN': '#2ca02c', 'ComCo': '#8c564b'}
BINS   = list(range(0, 101, 5))   # [0, 5, 10, ..., 100]

def make_bar_chart(base_dir, plots_dir, n_done):
    res = _load_results(base_dir)
    if not res:
        return

    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f'Accuracy Distribution across CIFAR-100 10-class Subsets\n'
        f'C=10, k=7  —  {n_done} seeds completed',
        fontsize=12,
    )

    for col, alg in enumerate(['PRODEN', 'ComCo']):
        ax   = axes[col]
        accs = res.get(alg, [])
        if not accs:
            ax.set_title(f'{alg}  (no data yet)')
            continue

        counts, edges = np.histogram(accs, bins=BINS)
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
        widths  = [edges[i+1] - edges[i] for i in range(len(counts))]

        ax.bar(centers, counts, width=[w * 0.8 for w in widths],
               color=COLORS[alg], alpha=0.8, edgecolor='white', linewidth=0.5)

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        ax.axvline(mean_acc, color='black', linestyle='--', linewidth=1.5,
                   label=f'mean={mean_acc:.1f}%')
        ax.set_title(f'{alg}  —  n={len(accs)}\n'
                     f'mean={mean_acc:.1f}%  std={std_acc:.1f}%', fontsize=11)
        ax.set_xlabel('Test Accuracy (%)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(plots_dir, 'accuracy_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] → {path}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='./data')
    parser.add_argument('--out_dir',   default='results/subset_variance/')
    parser.add_argument('--plots_dir', default='plots/subset_variance/')
    parser.add_argument('--log_dir',   default='logs/cifar100_subset')
    parser.add_argument('--config',    default='config.yaml')
    parser.add_argument('--epochs',    type=int, default=200)
    parser.add_argument('--max_seeds', type=int, default=MAX_SEEDS)
    parser.add_argument('--start_seed',type=int, default=1)
    parser.add_argument('--gpu_id',    type=int, default=0,
                        help='This GPU index (0-based). Handles seeds where (seed-start)%%num_gpus==gpu_id.')
    parser.add_argument('--num_gpus',  type=int, default=1,
                        help='Total number of parallel GPU workers.')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Each GPU writes to its own subdir to avoid CSV conflicts
    gpu_dir  = os.path.join(args.out_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir,       exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    with open(args.config) as f:
        comco_config = yaml.safe_load(f)['comco']

    done = _load_done(csv_path)

    # Round-robin seed assignment: GPU i handles seeds where offset % num_gpus == gpu_id
    my_seeds = [s for s in range(args.start_seed, args.start_seed + args.max_seeds)
                if (s - args.start_seed) % args.num_gpus == args.gpu_id]

    print(f'Device={device}  GPU {args.gpu_id}/{args.num_gpus}  C={C}  k={K}  epochs={args.epochs}')
    print(f'My seeds: {my_seeds[0]} … {my_seeds[-1]}  (total {len(my_seeds)})')
    print(f'Resume: {len(done)//2} seeds already in {csv_path}\n', flush=True)

    for seed in my_seeds:
        need_proden = (seed, 'PRODEN') not in done
        need_comco  = (seed, 'ComCo')  not in done

        if not need_proden and not need_comco:
            print(f'[skip] seed={seed}', flush=True)
            continue

        print(f'\n{"="*55}')
        print(f'seed={seed}', flush=True)

        pl_ds, cl_ds, orig_targets, test_info, log_info = prepare_cifar100_subset(
            total_classes=C, n_partial_labels=K,
            data_dir=args.data_dir, seed=seed, log_dir=args.log_dir,
        )
        print(f'  classes: {log_info["selected_class_names"]}', flush=True)

        loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, COMCO_BS)
        test_loader = loaders['test']

        if need_proden:
            t0  = time.perf_counter()
            acc = _train_proden(pl_ds, test_loader, args.epochs, device,
                                f'PRODEN seed={seed}')
            elapsed = time.perf_counter() - t0
            _append_result(csv_path, seed, 'PRODEN', acc, args.epochs, elapsed)
            print(f'  PRODEN  acc={acc:.2f}%  ({elapsed/60:.1f} min)', flush=True)

        if need_comco:
            t0  = time.perf_counter()
            acc = _train_comco(loaders, args.epochs, comco_config, device,
                               f'ComCo seed={seed}')
            elapsed = time.perf_counter() - t0
            _append_result(csv_path, seed, 'ComCo', acc, args.epochs, elapsed)
            print(f'  ComCo   acc={acc:.2f}%  ({elapsed/60:.1f} min)', flush=True)

        done.add((seed, 'PRODEN'))
        done.add((seed, 'ComCo'))
        n_seeds_done = sum(1 for s, a in done if a == 'PRODEN')
        make_bar_chart(args.out_dir, args.plots_dir, n_seeds_done)

    print('\nAll done.')


if __name__ == '__main__':
    main()
