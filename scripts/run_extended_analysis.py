"""
Extended analysis: 5 methods × k ∈ {5,10,15,19} × 500 epochs.

Methods:
  PiCO         — correct init (PL-masked uniform confidence)
  PiCO-Uniform — adam_comparison-style init (uniform over ALL C classes)
  PiCO-CLS     — cls-only, no contrastive
  PiCO-SC      — full PiCO arch, confidence from cls softmax
  ComCo        — complementary contrastive

Results saved to: results/extended_analysis/{alg}/C{C}_k{k}/
  loss_curve.csv      — per-epoch losses + overall_acc
  per_class_loss.csv  — per-class loss & acc every LOG_EVERY epochs
  logits/ep{N}.csv    — full logit dump every LOG_EVERY epochs

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_extended_analysis.py --alg PiCO         --k 5
  CUDA_VISIBLE_DEVICES=1 python scripts/run_extended_analysis.py --alg PiCO-Uniform --k 10
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import prepare_cifar100_subset, get_subset_dataloaders_full
from src.models import create_model
from src.pico.model import PiCOModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.pico_cls_loss import PiCOCLSLoss
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss

# ─── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_ALGS = ['PiCO', 'PiCO-Uniform', 'PiCO-CLS', 'PiCO-SC', 'ComCo']
LOG_EVERY      = 10

_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.247,  0.2435, 0.2616]

_TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ─── Indexed dataset for PiCO-CLS ─────────────────────────────────────────────

class _IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _TRAIN_TF(Image.fromarray(self.data[idx])), idx

# ─── CSV helpers ──────────────────────────────────────────────────────────────

def _ensure_csv(path, fieldnames):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_row(path, fieldnames, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def _load_done_epochs(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline='') as f:
        return {int(r['epoch']) for r in csv.DictReader(f) if r.get('epoch')}

# ─── Confidence init ──────────────────────────────────────────────────────────

def _build_pico_conf_masked(pl_ds, C):
    """Uniform over PL candidates only (correct init)."""
    N    = len(pl_ds)
    conf = torch.zeros(N, C)
    for i, cands in enumerate(pl_ds.targets):
        k = max(len(cands), 1)
        for j in cands:
            conf[i, j.item()] = 1.0 / k
    return conf


def _build_pico_conf_uniform(pl_ds, C):
    """Uniform over ALL C classes (adam_comparison-style init)."""
    return torch.ones(len(pl_ds), C) / C

# ─── Validation logging ───────────────────────────────────────────────────────

def _log_validation(model, test_loader, device, C, epoch, out_dir):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if isinstance(model, (PiCOModel, ComCoModel)):
                logits = model(images, eval_only=True)
            else:
                logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits  = torch.cat(all_logits, dim=0)
    all_labels  = torch.cat(all_labels, dim=0)
    pred_labels = all_logits.argmax(dim=1)

    # Logit dump
    logit_dir  = os.path.join(out_dir, 'logits')
    os.makedirs(logit_dir, exist_ok=True)
    logit_path = os.path.join(logit_dir, f'ep{epoch:04d}.csv')
    fields     = ['sample_idx', 'true_label', 'pred_label'] + [f'logit_{c}' for c in range(C)]
    with open(logit_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(all_labels)):
            row = [i, all_labels[i].item(), pred_labels[i].item()] + \
                  [f'{v:.6f}' for v in all_logits[i].tolist()]
            w.writerow(row)

    # Per-class loss & accuracy
    log_probs      = F.log_softmax(all_logits, dim=1)
    ce_per_sample  = -log_probs[torch.arange(len(all_labels)), all_labels]
    per_class_loss, per_class_acc = [], []
    for c in range(C):
        mask = (all_labels == c)
        if mask.sum() == 0:
            per_class_loss.append(float('nan'))
            per_class_acc.append(float('nan'))
        else:
            per_class_loss.append(ce_per_sample[mask].mean().item())
            per_class_acc.append((pred_labels[mask] == c).float().mean().item() * 100)

    overall_acc = (pred_labels == all_labels).float().mean().item() * 100

    pcl_path   = os.path.join(out_dir, 'per_class_loss.csv')
    pcl_fields = ['epoch'] + \
                 [f'loss_class_{c}' for c in range(C)] + \
                 [f'acc_class_{c}'  for c in range(C)] + \
                 ['overall_acc']
    _ensure_csv(pcl_path, pcl_fields)
    row = {'epoch': epoch, 'overall_acc': round(overall_acc, 4)}
    for c in range(C):
        row[f'loss_class_{c}'] = round(per_class_loss[c], 6)
        row[f'acc_class_{c}']  = round(per_class_acc[c],  4)
    _append_row(pcl_path, pcl_fields, row)

    return overall_acc

# ─── Training loops ───────────────────────────────────────────────────────────

def _train_epoch_pico(model, loader, loss_fn, loss_cont_fn, optimizer,
                      epoch, pico_args, device, sc_mode=False):
    model.train()
    sum_cls = sum_cont = sum_total = 0.0
    n = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    for images_w, images_s, partial_Y, true_labels, index in loader:
        images_w  = images_w.to(device)
        images_s  = images_s.to(device)
        partial_Y = partial_Y.to(device)
        index     = index.to(device)

        cls_out, features, pseudo_target_cont, score_prot = model(
            images_w, images_s, partial_Y, pico_args
        )
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            if sc_mode:
                loss_fn.update_confidence(cls_out.detach(), index)
            else:
                loss_fn.confidence_update(score_prot.detach(), index, partial_Y)

        mask = (
            torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1),
                     pseudo_target_cont.unsqueeze(0)).float()
            if start_upd_prot else None
        )

        loss_cls  = loss_fn(cls_out, index)
        loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
        loss      = loss_cls + pico_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = cls_out.shape[0]
        sum_cls   += loss_cls.item()  * bs
        sum_cont  += loss_cont.item() * bs
        sum_total += loss.item()      * bs
        n         += bs

    return sum_cls / n, sum_cont / n, sum_total / n


def _train_epoch_pico_cls(model, loader, loss_fn, optimizer, epoch, device):
    model.train()
    loss_fn.set_conf_ema_m(epoch)
    sum_cls = 0.0
    n = 0
    for imgs, indices in loader:
        imgs, indices = imgs.to(device), indices.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = loss_fn(out, indices)
        loss.backward()
        optimizer.step()
        loss_fn.update_confidence(out.detach(), indices)
        sum_cls += loss.item() * imgs.shape[0]
        n       += imgs.shape[0]
    avg = sum_cls / n
    return avg, 0.0, avg


def _train_epoch_comco(model, loader, cls_loss_fn, cont_loss_fn, optimizer,
                       epoch, comco_args, device):
    model.train()
    warmup_pos = epoch >= comco_args['warmup_pos']
    warmup_neg = epoch >= comco_args['warmup_neg']
    sum_cls = sum_cont = sum_total = 0.0
    n = 0
    for images_w, images_s, comp_mask, true_labels, index in loader:
        images_w  = images_w.to(device)
        images_s  = images_s.to(device)
        comp_mask = comp_mask.to(device)

        cls_out, q, all_feats, all_pseudo, all_comp = model(
            images_w, images_s, comp_mask, comco_args
        )
        pseudo_q  = cls_out.argmax(dim=1)
        loss_cls  = cls_loss_fn(cls_out, comp_mask)
        loss_cont = cont_loss_fn(q, all_feats, all_pseudo, all_comp, pseudo_q,
                                 warmup_pos, warmup_neg)
        loss = loss_cls + comco_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = cls_out.shape[0]
        sum_cls   += loss_cls.item()  * bs
        sum_cont  += loss_cont.item() * bs
        sum_total += loss.item()      * bs
        n         += bs

    return sum_cls / n, sum_cont / n, sum_total / n

# ─── Main run function ────────────────────────────────────────────────────────

def run(args, cfg, device):
    alg    = args.alg
    C      = args.C
    k      = args.k
    epochs = args.epochs

    out_dir = os.path.join(args.out_dir, alg, f'C{C}_k{k}')
    os.makedirs(out_dir, exist_ok=True)

    lc_fields       = ['epoch', 'cls_loss', 'cont_loss', 'total_loss',
                        'cls_ratio', 'wcont_ratio', 'overall_acc']
    loss_curve_path = os.path.join(out_dir, 'loss_curve.csv')
    _ensure_csv(loss_curve_path, lc_fields)
    done_epochs = _load_done_epochs(loss_curve_path)

    if done_epochs and max(done_epochs) >= epochs:
        print(f'[{alg} C={C} k={k}] already complete ({epochs} epochs). Skipping.')
        return

    # ── Data ──────────────────────────────────────────────────────────────────
    pl_ds, cl_ds, orig_targets, test_info, log_info = prepare_cifar100_subset(
        total_classes=C, n_partial_labels=k,
        data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
    )
    print(f'  classes: {log_info["selected_class_names"]}', flush=True)

    loaders     = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, args.batch_size)
    test_loader = loaders['test']

    # ── Model & optimizer ─────────────────────────────────────────────────────
    pico_args  = dict(cfg['pico'])
    comco_args = dict(cfg['comco'])
    pico_args.update({'num_class': C, 'epochs': epochs})
    comco_args.update({'num_class': C, 'epochs': epochs})

    if alg == 'PiCO':
        model    = PiCOModel(pico_args).to(device)
        loss_fn  = PartialLoss(_build_pico_conf_masked(pl_ds, C)).to(device)
        loss_fn.confidence = loss_fn.confidence.to(device)
        cont_fn  = SupConLoss(temperature=0.07).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loader   = loaders['pico']

    elif alg == 'PiCO-Uniform':
        # Same as PiCO but confidence initialised uniformly over ALL classes
        model    = PiCOModel(pico_args).to(device)
        loss_fn  = PartialLoss(_build_pico_conf_uniform(pl_ds, C)).to(device)
        loss_fn.confidence = loss_fn.confidence.to(device)
        cont_fn  = SupConLoss(temperature=0.07).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loader   = loaders['pico']

    elif alg == 'PiCO-SC':
        model    = PiCOModel(pico_args).to(device)
        loss_fn  = PiCOCLSLoss(pl_ds.targets, C, epochs=epochs).to(device)
        cont_fn  = SupConLoss(temperature=0.07).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loader   = loaders['pico']

    elif alg == 'PiCO-CLS':
        model    = create_model(C).to(device)
        loss_fn  = PiCOCLSLoss(pl_ds.targets, C, epochs=epochs).to(device)
        cont_fn  = None
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loader   = DataLoader(_IndexedDataset(pl_ds.data),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)

    elif alg == 'ComCo':
        model    = ComCoModel(comco_args).to(device)
        loss_fn  = ComCoCLSLoss().to(device)
        cont_fn  = ComCoContrastiveLoss(
            temperature=comco_args['temperature'],
            top_k=comco_args['top_k'],
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loader   = loaders['comco']

    # ── Training loop ─────────────────────────────────────────────────────────
    start_epoch = max(done_epochs) + 1 if done_epochs else 0
    print(f'[{alg} C={C} k={k}] Training epochs {start_epoch+1}→{epochs}', flush=True)

    for ep in range(start_epoch, epochs):
        t0 = time.perf_counter()

        if alg in ('PiCO', 'PiCO-Uniform'):
            cls_l, cont_l, tot_l = _train_epoch_pico(
                model, loader, loss_fn, cont_fn, optimizer,
                ep, pico_args, device, sc_mode=False)
        elif alg == 'PiCO-SC':
            cls_l, cont_l, tot_l = _train_epoch_pico(
                model, loader, loss_fn, cont_fn, optimizer,
                ep, pico_args, device, sc_mode=True)
        elif alg == 'PiCO-CLS':
            cls_l, cont_l, tot_l = _train_epoch_pico_cls(
                model, loader, loss_fn, optimizer, ep, device)
        elif alg == 'ComCo':
            cls_l, cont_l, tot_l = _train_epoch_comco(
                model, loader, loss_fn, cont_fn, optimizer, ep, comco_args, device)

        # Validation every LOG_EVERY epochs
        acc = 0.0
        if (ep + 1) % LOG_EVERY == 0 or ep + 1 == epochs:
            acc = _log_validation(model, test_loader, device, C, ep + 1, out_dir)

        cls_ratio   = cls_l / tot_l if tot_l > 0 else 0.0
        wcont_ratio = (tot_l - cls_l) / tot_l if tot_l > 0 else 0.0

        _append_row(loss_curve_path, lc_fields, {
            'epoch':       ep + 1,
            'cls_loss':    round(cls_l,       6),
            'cont_loss':   round(cont_l,      6),
            'total_loss':  round(tot_l,       6),
            'cls_ratio':   round(cls_ratio,   6),
            'wcont_ratio': round(wcont_ratio, 6),
            'overall_acc': round(acc,         4),
        })

        elapsed = time.perf_counter() - t0
        print(f'  ep {ep+1:>3}/{epochs}  cls={cls_l:.4f}  cont={cont_l:.4f}'
              f'  tot={tot_l:.4f}  acc={acc:.1f}%  ({elapsed:.1f}s)', flush=True)

        if (ep + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ─── Argument parsing ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg',        required=True, choices=SUPPORTED_ALGS)
    parser.add_argument('--C',          type=int,   default=20)
    parser.add_argument('--k',          type=int,   default=5)
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--batch_size', type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--wd',         type=float, default=1e-4)
    parser.add_argument('--data_dir',   default='./data')
    parser.add_argument('--out_dir',    default='results/extended_analysis/')
    parser.add_argument('--log_dir',    default='logs/cifar100_subset')
    parser.add_argument('--config',     default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}  alg={args.alg}  C={args.C}  k={args.k}  '
          f'epochs={args.epochs}  lr={args.lr}', flush=True)
    run(args, cfg, device)
    print('Done.')


if __name__ == '__main__':
    main()
