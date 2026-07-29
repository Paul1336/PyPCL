"""
PL-size-variance experiment: PRODEN and ComCo across 3 PL-size-distribution variances.

C=10, mean PL size = 7. Distribution support fixed at {5, 7, 9}:
  Var=1: P(5)=1/8,  P(7)=3/4,  P(9)=1/8
  Var=2: P(5)=1/4,  P(7)=1/2,  P(9)=1/4
  Var=3: P(5)=3/8,  P(7)=1/4,  P(9)=3/8

For each seed the SAME 10-class CIFAR-100 subset is reused across all 3 variance levels
so results are directly comparable.  4 GPUs split seeds round-robin.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pl_size_variance.py --gpu_id 0 --num_gpus 4
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pl_size_variance.py --gpu_id 0 --num_gpus 1 --epochs 5
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
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import prepare_cifar100_subset
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss
from src.data_utils import ComCoDataset, WeaklySupervisedDataset
from src.engine import evaluate_model, train_comco_epoch
from src.models import create_model
from src.proden_loss import ProdenLoss

# ─── Constants ────────────────────────────────────────────────────────────────

C          = 10
VAR_LEVELS = [1, 2, 3]
MAX_SEEDS  = 1000
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

# ─── PL-size distributions ────────────────────────────────────────────────────
# Support {5,7,9}, all have E[k]=7.  Var = 8*p where P(5)=P(9)=p, P(7)=1-2p.
#   Var=1 → p=1/8,  Var=2 → p=1/4,  Var=3 → p=3/8

VARIANCE_DISTS = {
    1: (np.array([5, 7, 9]), np.array([1/8, 3/4, 1/8])),
    2: (np.array([5, 7, 9]), np.array([1/4, 1/2, 1/4])),
    3: (np.array([5, 7, 9]), np.array([3/8, 1/4, 3/8])),
}

VAR_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#d62728'}

# ─── Transforms ───────────────────────────────────────────────────────────────

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

# ─── Datasets ─────────────────────────────────────────────────────────────────

class _IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _TRAIN_TF(Image.fromarray(self.data[idx])), idx


class _TestDataset(Dataset):
    def __init__(self, data, targets):
        self.data    = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _TEST_TF(Image.fromarray(self.data[idx])), self.targets[idx]

# ─── PL/CL generation ─────────────────────────────────────────────────────────

def _generate_pl_cl(train_data, true_labels_np, var_level, seed):
    """Sample per-sample PL size from VARIANCE_DISTS[var_level], then pick random labels."""
    np.random.seed(seed * 100 + var_level)   # reproducible per (seed, var_level)
    sizes, probs = VARIANCE_DISTS[var_level]
    all_classes  = np.arange(C)

    pl_targets = []
    cl_targets = []
    for true_label in true_labels_np:
        k            = np.random.choice(sizes, p=probs)
        false_labels = np.delete(all_classes, true_label)
        chosen_false = np.random.choice(false_labels, size=k - 1, replace=False)
        pl_set = np.sort(np.concatenate([[true_label], chosen_false]))
        cl_set = np.array(sorted(set(range(C)) - set(pl_set.tolist())), dtype=np.int64)
        pl_targets.append(torch.tensor(pl_set, dtype=torch.long))
        cl_targets.append(torch.tensor(cl_set, dtype=torch.long))

    pl_ds = WeaklySupervisedDataset(train_data, pl_targets)
    cl_ds = WeaklySupervisedDataset(train_data, cl_targets)
    return pl_ds, cl_ds

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

_CSV_FIELDS = ['seed', 'var_level', 'algorithm',
               'final_accuracy', 'epochs', 'training_time_s', 'timestamp']


def _load_done(csv_path):
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['seed']), int(row['var_level']), row['algorithm']))
    return done


def _append_result(csv_path, seed, var_level, alg, acc, epochs, elapsed_s):
    new_file = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'seed':            seed,
            'var_level':       var_level,
            'algorithm':       alg,
            'final_accuracy':  round(acc, 4),
            'epochs':          epochs,
            'training_time_s': round(elapsed_s, 1),
            'timestamp':       datetime.now().isoformat(),
        })


def _load_results(base_dir):
    """Merge gpu*/results.csv → {alg: {var_level: [acc, ...]}}."""
    res  = {}
    seen = set()
    for path in sorted(glob.glob(os.path.join(base_dir, 'gpu*', 'results.csv'))):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['seed'], row['var_level'], row['algorithm'])
                if key in seen:
                    continue
                seen.add(key)
                alg = row['algorithm']
                vl  = int(row['var_level'])
                acc = float(row['final_accuracy'])
                res.setdefault(alg, {}).setdefault(vl, []).append(acc)
    return res

# ─── Training ─────────────────────────────────────────────────────────────────

def _train_proden(pl_ds, test_loader, epochs, device, tag):
    model   = create_model(C).to(device)
    loss_fn = ProdenLoss(pl_ds.targets, C).to(device)
    opt     = optim.SGD(model.parameters(), lr=PRODEN_LR,
                        momentum=PRODEN_MOM, weight_decay=PRODEN_WD)
    idx_loader = DataLoader(_IndexedDataset(pl_ds.data),
                            batch_size=PRODEN_BS, shuffle=True, num_workers=2)

    chunk_t0  = time.perf_counter()
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
            elapsed   = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0  = time.perf_counter()
            gc.collect(); torch.cuda.empty_cache()

    del model, loss_fn, opt
    gc.collect(); torch.cuda.empty_cache()
    return final_acc


def _train_comco(cl_ds, orig_targets, test_loader, epochs, comco_config, device, tag):
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
    comco_ds     = ComCoDataset(cl_ds, orig_targets)
    comco_loader = DataLoader(comco_ds, batch_size=COMCO_BS, shuffle=True,
                              num_workers=2, drop_last=True)

    model     = ComCoModel(comco_args).to(device)
    cls_loss  = ComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'],
                                     top_k=comco_args['top_k'])
    opt = optim.Adam(model.parameters(), lr=COMCO_LR, weight_decay=COMCO_WD)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_comco_epoch(comco_args, model, comco_loader,
                          cls_loss, cont_loss, opt, ep, device)
        if (ep + 1) % REPORT_EVERY == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect(); torch.cuda.empty_cache()

    acc = evaluate_model(model, test_loader, device)
    del model, cls_loss, cont_loss, opt, comco_loader, comco_ds
    gc.collect(); torch.cuda.empty_cache()
    return acc

# ─── Plotting ─────────────────────────────────────────────────────────────────

BINS = list(range(0, 101, 5))


def make_plot(base_dir, plots_dir, n_seeds_done):
    res = _load_results(base_dir)
    if not res:
        return

    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f'Accuracy Distribution — PL size variance effect\n'
        f'C=10, mean PL size=7, {n_seeds_done} seeds completed',
        fontsize=12,
    )

    for col, alg in enumerate(['PRODEN', 'ComCo']):
        ax = axes[col]
        ax.set_title(alg, fontsize=11)
        ax.set_xlabel('Test Accuracy (%)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9) if col == 0 else None
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        alg_data = res.get(alg, {})
        all_accs = [a for v in alg_data.values() for a in v]
        if not all_accs:
            ax.set_title(f'{alg}  (no data yet)')
            continue

        ym = max(np.histogram(a, bins=BINS)[0].max() for a in alg_data.values()
                 if a) if alg_data else 10
        ax.set_ylim(0, ym * 1.15)

        for vl in VAR_LEVELS:
            accs = alg_data.get(vl, [])
            if not accs:
                continue
            counts, edges = np.histogram(accs, bins=BINS)
            centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
            widths  = [(edges[i+1] - edges[i]) * 0.25 for i in range(len(counts))]
            offset  = (vl - 2) * 1.4   # slight horizontal offset to avoid overlap
            ax.bar([c + offset for c in centers], counts, width=widths,
                   color=VAR_COLORS[vl], alpha=0.75,
                   label=f'Var={vl}  n={len(accs)}  μ={np.mean(accs):.1f}%  σ={np.std(accs):.1f}%')
            ax.axvline(np.mean(accs), color=VAR_COLORS[vl],
                       linestyle='--', linewidth=1.2, alpha=0.8)

        ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    path = os.path.join(plots_dir, 'pl_size_variance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] → {path}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./data')
    parser.add_argument('--out_dir',    default='results/pl_size_variance/')
    parser.add_argument('--plots_dir',  default='plots/pl_size_variance/')
    parser.add_argument('--log_dir',    default='logs/cifar100_subset')
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--max_seeds',  type=int, default=MAX_SEEDS)
    parser.add_argument('--start_seed', type=int, default=1)
    parser.add_argument('--gpu_id',     type=int, default=0)
    parser.add_argument('--num_gpus',   type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_dir  = os.path.join(args.out_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir,        exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    with open(args.config) as f:
        comco_config = yaml.safe_load(f)['comco']

    done = _load_done(csv_path)

    my_seeds = [s for s in range(args.start_seed, args.start_seed + args.max_seeds)
                if (s - args.start_seed) % args.num_gpus == args.gpu_id]

    print(f'Device={device}  GPU {args.gpu_id}/{args.num_gpus}  C={C}  epochs={args.epochs}')
    print(f'Seeds: {my_seeds[0]} … {my_seeds[-1]}  (total {len(my_seeds)})')
    print(f'Var distributions: {VARIANCE_DISTS}')
    print(f'Resume: {len(done)} entries in {csv_path}\n', flush=True)

    for seed in my_seeds:
        need_any = any(
            (seed, vl, alg) not in done
            for vl in VAR_LEVELS for alg in ['PRODEN', 'ComCo']
        )
        if not need_any:
            print(f'[skip] seed={seed}', flush=True)
            continue

        print(f'\n{"="*60}')
        print(f'seed={seed}', flush=True)

        # Load 10-class subset (k=7 used only for class selection; PL labels regenerated below)
        pl_ds_base, _, orig_targets, test_info, log_info = prepare_cifar100_subset(
            total_classes=C, n_partial_labels=7,
            data_dir=args.data_dir, seed=seed, log_dir=args.log_dir,
        )
        print(f'  classes: {log_info["selected_class_names"]}', flush=True)

        train_data      = pl_ds_base.data
        true_labels_np  = orig_targets.numpy()
        test_loader     = DataLoader(
            _TestDataset(test_info[0], test_info[1]),
            batch_size=COMCO_BS, shuffle=False, num_workers=2,
        )

        for var_level in VAR_LEVELS:
            need_proden = (seed, var_level, 'PRODEN') not in done
            need_comco  = (seed, var_level, 'ComCo')  not in done
            if not need_proden and not need_comco:
                print(f'  [skip] var={var_level}', flush=True)
                continue

            print(f'\n  --- var_level={var_level} ---', flush=True)
            pl_ds, cl_ds = _generate_pl_cl(train_data, true_labels_np, var_level, seed)

            if need_proden:
                t0  = time.perf_counter()
                acc = _train_proden(pl_ds, test_loader, args.epochs, device,
                                    f'PRODEN v{var_level} seed={seed}')
                elapsed = time.perf_counter() - t0
                _append_result(csv_path, seed, var_level, 'PRODEN', acc, args.epochs, elapsed)
                done.add((seed, var_level, 'PRODEN'))
                print(f'  PRODEN  var={var_level}  acc={acc:.2f}%  ({elapsed/60:.1f} min)', flush=True)

            if need_comco:
                t0  = time.perf_counter()
                acc = _train_comco(cl_ds, orig_targets, test_loader, args.epochs,
                                   comco_config, device,
                                   f'ComCo v{var_level} seed={seed}')
                elapsed = time.perf_counter() - t0
                _append_result(csv_path, seed, var_level, 'ComCo', acc, args.epochs, elapsed)
                done.add((seed, var_level, 'ComCo'))
                print(f'  ComCo   var={var_level}  acc={acc:.2f}%  ({elapsed/60:.1f} min)', flush=True)

        n_done = sum(1 for (s, vl, a) in done if a == 'PRODEN' and vl == 1)
        make_plot(args.out_dir, args.plots_dir, n_done)

    print('\nAll done.')


if __name__ == '__main__':
    main()
