"""cls-loss solo experiment (2026-10-12): every cls loss of the swap
experiment (src/clsloss_swap.py) trained ALONE -- no contrastive branch, no
MoCo key encoder / queue, no prototypes. Same backbone (PiCO's SupConResNet
encoder_q; its projection head receives no gradient), same weak augmentation
(PicoDataset.weak_transform), same Adam 3e-4 / wd 1e-4, 200 epochs, bs 512,
same candidate-masked confidence init and loss objects as the swap run, so
the swap results minus these isolate what the contrastive framework adds.

PiCOPartial is dropped: its pseudo-target is driven by prototype scores,
which do not exist without the contrastive branch, and swapping in the
classifier softmax turns it into PiCOCLS exactly.

Per (loss, k, seed) cell this also writes, under
results/<run>/detail/Solo-<loss>/C20_k<k>/:

  dynamics_s<seed>.csv      every epoch: test acc, train-set (true-label)
                            acc, disambiguation acc (argmax over Y == y),
                            candidate mass S, normalized in-candidate
                            entropy H(p|Y)/log k, p_true, confidence-buffer
                            acc for stateful losses.
  biasvariance_s<seed>.csv  epoch 1 and every --bv_every epochs: bias^2 /
                            variance / MSE / cos of the cls-loss gradient wrt
                            the logits over M resampled candidate sets vs.
                            the true-label CE gradient (raw and scale-aligned),
                            the gradient norm ratio to CE, and -- for losses
                            that are a sum of two terms (Wu2022, SoLar) -- the
                            norm ratio and cosine between the terms' grads.
"""

import copy
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.clsloss_swap import PiCOEMALoss, PRODENLoss, SoLarLoss, build_cls_loss
from src.data_utils import PicoDataset
from src.pipeline.biasvariance import _resample_batch, oracle_cls_grad

SOLO_LOSSES = ['UCE', 'CLPL', 'Wu2022', 'PRODEN', 'PiCOCLS', 'SoLar',
               'MCLLOG', 'MCLLOGFixed', 'SCLNL', 'OP', 'OPW']

_MEAN = [0.4914, 0.4822, 0.4465]
_STD = [0.247, 0.2435, 0.2616]

DYN_FIELDS = ['epoch', 'train_loss', 'test_acc', 'train_true_acc', 'disamb_acc', 'cand_mass',
              'cand_entropy', 'p_true', 'conf_acc']
BV_FIELDS = ['epoch', 'bias2', 'variance', 'mse', 'cos_sim', 'bias2_n', 'variance_n', 'mse_n',
             'grad_norm_ratio', 'comp_norm_ratio', 'comp_cos']


def solo_algorithm_name(loss: str) -> str:
    return f'Solo-{loss}'


# ─── data ──────────────────────────────────────────────────────────────────


class WeakOnlyDataset(Dataset):
    """PicoDataset minus the (unused) strong view: same weak transform, same
    one-hot candidate vector, same (true_label, index)."""

    def __init__(self, pl_ds, orig_targets):
        self.base = PicoDataset(pl_ds, orig_targets)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base
        y = torch.zeros(b.num_classes, dtype=torch.float)
        y[b.given_label_matrix_sparse[i]] = 1.0
        return b.weak_transform(b.images[i]), y, int(b.true_labels[i]), i


def make_train_loader(pl_ds, orig_targets, batch_size, num_workers=3):
    return DataLoader(WeakOnlyDataset(pl_ds, orig_targets), batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)


def build_train_eval_tensors(pl_ds, orig_targets, C, device):
    """Whole train set, ToTensor+Normalize only (no augmentation), plus dense
    candidate matrix and true labels -- for the per-epoch dynamics pass."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    imgs = torch.stack([tf(Image.fromarray(x)) for x in pl_ds.data])
    Y = torch.zeros(len(pl_ds.data), C)
    for i, cands in enumerate(pl_ds.targets):
        Y[i, [int(j) for j in cands]] = 1.0
    true = torch.as_tensor([int(t) for t in orig_targets], dtype=torch.long)
    return imgs, Y.to(device), true.to(device)


# ─── confidence buffers of the stateful losses ─────────────────────────────


def conf_buffer(loss_fn):
    if isinstance(loss_fn, (PRODENLoss, SoLarLoss)):
        return loss_fn.conf
    if isinstance(loss_fn, PiCOEMALoss):
        return loss_fn.partial.confidence
    return None


# ─── per-epoch dynamics ────────────────────────────────────────────────────


@torch.no_grad()
def _logits(model, imgs, device, bs=1024):
    model.eval()
    out = [model(imgs[i:i + bs].to(device, non_blocking=True))[0] for i in range(0, len(imgs), bs)]
    return torch.cat(out)


@torch.no_grad()
def dynamics_row(model, loss_fn, train_imgs, train_Y, train_true, test_loader, device, epoch, train_loss):
    p = F.softmax(_logits(model, train_imgs, device), dim=1)
    n = p.shape[0]
    ar = torch.arange(n, device=device)
    S = (p * train_Y).sum(dim=1)
    q = p * train_Y / S.clamp(min=1e-12).unsqueeze(1)
    k = train_Y.sum(dim=1)
    H = -(q * torch.log(q.clamp(min=1e-12))).sum(dim=1)
    Hn = torch.where(k > 1, H / torch.log(k.clamp(min=2)), torch.zeros_like(H))

    correct = total = 0
    model.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x)[0].argmax(1) == y).sum().item()
        total += y.numel()

    conf = conf_buffer(loss_fn)
    conf_acc = '' if conf is None else round((conf.argmax(1) == train_true).float().mean().item() * 100, 3)
    return dict(epoch=epoch, train_loss=round(train_loss, 6), test_acc=round(100 * correct / total, 3),
                train_true_acc=round((p.argmax(1) == train_true).float().mean().item() * 100, 3),
                disamb_acc=round((q.argmax(1) == train_true).float().mean().item() * 100, 3),
                cand_mass=round(S.mean().item(), 5), cand_entropy=round(Hn.mean().item(), 5),
                p_true=round(p[ar, train_true].mean().item(), 5), conf_acc=conf_acc)


# ─── gradient bias / variance ──────────────────────────────────────────────


def _two_term_split(loss_name, loss_copy, leaf, partial_Y):
    """The two summands whose gradients may conflict, or None."""
    if loss_name == 'Wu2022':
        C = leaf.shape[1]
        log_sm = F.log_softmax(leaf, dim=1)
        k = partial_Y.sum(dim=1).clamp(min=1)
        cand = ((C - 1.0) / (C - k) * -(log_sm * partial_Y).sum(dim=1)).mean()
        allt = (-(k - 1.0) / (C - k) * -log_sm.sum(dim=1)).mean()
        return cand, allt
    if loss_name == 'SoLar':
        return getattr(loss_copy, 'last_terms', None)
    return None


def _resampled_grad(loss_name, loss_fn, logits, partial_Y, eval_index):
    """Gradient of the REAL current loss wrt the logits for one resampled
    candidate set, run on a deep copy so the loss's own state (confidence
    buffer, SoLar queue) is never mutated. A stateful loss's confidence rows
    for the eval samples are its real rows restricted and renormalized to
    the resampled candidate set (uniform over it if no mass falls inside)."""
    if isinstance(loss_fn, SoLarLoss):
        loss_fn.last_terms = None  # holds the last training step's graph, which deepcopy rejects
    lc = copy.deepcopy(loss_fn)
    conf = conf_buffer(lc)
    if conf is not None:
        masked = conf[eval_index] * partial_Y
        mass = masked.sum(dim=1, keepdim=True)
        uni = partial_Y / partial_Y.sum(dim=1, keepdim=True)
        conf[eval_index] = torch.where(mass > 1e-8, masked / mass.clamp(min=1e-8), uni)
    if isinstance(lc, SoLarLoss):
        lc.last_terms = None
    leaf = logits.detach().clone().requires_grad_(True)
    loss = lc(leaf, eval_index, partial_Y)
    terms = _two_term_split(loss_name, lc, leaf, partial_Y)
    comp = None
    if terms is not None:
        ga = torch.autograd.grad(terms[0], leaf, retain_graph=True)[0]
        gb = torch.autograd.grad(terms[1], leaf, retain_graph=True)[0]
        comp = ((ga.norm() / gb.norm().clamp(min=1e-12)).item(),
                F.cosine_similarity(ga.flatten(), gb.flatten(), dim=0).item())
    g = torch.autograd.grad(loss, leaf)[0]
    return g.detach(), comp


def _stats(G, oracle):
    mean = G.mean(dim=0)
    return dict(bias2=((mean - oracle) ** 2).mean().item(),
                variance=((G - mean) ** 2).mean().item(),
                mse=((G - oracle) ** 2).mean().item())


def biasvariance_row(loss_name, model, loss_fn, eval_imgs, eval_true, eval_index, C, k, epoch, seed,
                     m_resamples, device):
    model.eval()
    with torch.no_grad():
        logits = model(eval_imgs)[0].detach()
    oracle = oracle_cls_grad(logits, eval_true)
    grads, comps = [], []
    for m in range(m_resamples):
        rng_seed = (seed * 1_000_003 + epoch * 1009 + m) & 0xFFFFFFFF
        partial_Y, _ = _resample_batch(eval_true, C, k, rng_seed, device)
        g, comp = _resampled_grad(loss_name, loss_fn, logits, partial_Y, eval_index)
        grads.append(g)
        if comp is not None:
            comps.append(comp)
    G = torch.stack(grads)
    mean = G.mean(dim=0)
    raw = _stats(G, oracle)
    # Scale-aligned: rescale every resampled gradient by the one scalar that
    # gives the mean gradient the oracle's norm, so losses whose gradient
    # carries a constant factor (MCLLOG's (C-1)/k, ...) are compared on
    # direction and spread, not on raw magnitude.
    scale = (oracle.norm() / mean.norm().clamp(min=1e-12)).item()
    nrm = _stats(G * scale, oracle)
    row = dict(epoch=epoch, **{k2: round(v, 10) for k2, v in raw.items()},
               cos_sim=round(F.cosine_similarity(mean.flatten(), oracle.flatten(), dim=0).item(), 6),
               bias2_n=round(nrm['bias2'], 10), variance_n=round(nrm['variance'], 10), mse_n=round(nrm['mse'], 10),
               grad_norm_ratio=round((mean.norm() / oracle.norm().clamp(min=1e-12)).item(), 6),
               comp_norm_ratio='', comp_cos='')
    if comps:
        row['comp_norm_ratio'] = round(float(np.mean([c[0] for c in comps])), 6)
        row['comp_cos'] = round(float(np.mean([c[1] for c in comps])), 6)
    return row


def append_csv(path, fields, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


# ─── training ──────────────────────────────────────────────────────────────


def train_solo_epoch(model, loader, loss_fn, optimizer, epoch, epochs, prot_start, conf_ema_range, device):
    model.train()
    loss_fn.set_epoch(epoch, conf_ema_range, epochs)
    start_upd = epoch >= prot_start
    total, n = 0.0, 0
    for images_w, partial_Y, _true, index in loader:
        images_w = images_w.to(device, non_blocking=True)
        partial_Y, index = partial_Y.to(device), index.to(device)
        logits = model(images_w)[0]
        loss_fn.pre_update(None, logits, index, partial_Y, start_upd)
        loss = loss_fn(logits, index, partial_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def run_solo(loss_name, *, pl_ds, orig_targets, test_loader, C, k, seed, epochs, batch_size, raw_cfg,
             make_opt, out_dir, device, bv_every=20, bv_m=50, bv_eval=256, log=print):
    """Trains one cell; returns (final_acc, best_acc, best_epoch)."""
    from src.pico.resnet import SupConResNet
    from src.pipeline.biasvariance import build_fixed_eval_batch
    from src.pll_init import candidate_masked_init

    pico_cfg = raw_cfg['pico']
    model = SupConResNet(num_class=C, feat_dim=pico_cfg['low_dim']).to(device)
    init_conf = candidate_masked_init(pl_ds.targets, C).to(device)
    loss_fn = build_cls_loss(loss_name, init_conf, C, raw_cfg['solar'], batch_size)
    opt = make_opt(model)
    prot_start = pico_cfg.get('prot_start_fixed', 1)

    loader = make_train_loader(pl_ds, orig_targets, batch_size)
    tr_imgs, tr_Y, tr_true = build_train_eval_tensors(pl_ds, orig_targets, C, device)
    bv_imgs, bv_true, bv_index = build_fixed_eval_batch(pl_ds, orig_targets, bv_eval, device)
    dyn_path = os.path.join(out_dir, f'dynamics_s{seed}.csv')
    bv_path = os.path.join(out_dir, f'biasvariance_s{seed}.csv')
    for p in (dyn_path, bv_path):  # a retried cell starts its curves over
        if os.path.isfile(p):
            os.remove(p)

    best_acc, best_ep, acc = -1.0, 0, 0.0
    for ep in range(epochs):
        tl = train_solo_epoch(model, loader, loss_fn, opt, ep, epochs, prot_start,
                              pico_cfg['conf_ema_range'], device)
        row = dynamics_row(model, loss_fn, tr_imgs, tr_Y, tr_true, test_loader, device, ep + 1, tl)
        append_csv(dyn_path, DYN_FIELDS, row)
        acc = row['test_acc']
        if acc > best_acc:
            best_acc, best_ep = acc, ep + 1
        if ep == 0 or (ep + 1) % bv_every == 0:
            append_csv(bv_path, BV_FIELDS, biasvariance_row(loss_name, model, loss_fn, bv_imgs, bv_true, bv_index,
                                                            C, k, ep + 1, seed, bv_m, device))
        if (ep + 1) % 10 == 0 or ep + 1 == epochs:
            log(f'  [Solo-{loss_name} k={k} s{seed}] ep {ep + 1}/{epochs}  loss {tl:.4f}  test {acc:.2f}  '
                f'train {row["train_true_acc"]:.2f}  disamb {row["disamb_acc"]:.2f}  S {row["cand_mass"]:.3f}  '
                f'H {row["cand_entropy"]:.3f}  best {best_acc:.2f}@{best_ep}')
    return acc, best_acc, best_ep


# ─── same diagnostics for the framework (swap) runs ────────────────────────


class CellDiagnostics:
    """The solo cells' dynamics/bias-variance logging, packaged for
    runners._run_swap_variant (enabled there by --biasvariance). `net` is the
    module whose forward returns (logits, feat) -- encoder_q for PiCO/ComCo,
    read-only, so the MoCo queue / prototypes are never touched. The cls
    loss's gradient is measured exactly as in the solo cells; the
    contrastive loss is not part of it."""

    def __init__(self, loss_name, pl_ds, orig_targets, test_loader, C, k, seed, out_dir, device,
                 bv_every=20, bv_m=50, bv_eval=256):
        from src.pipeline.biasvariance import build_fixed_eval_batch
        self.loss_name, self.C, self.k, self.seed, self.device = loss_name, C, k, seed, device
        self.test_loader, self.bv_every, self.bv_m = test_loader, bv_every, bv_m
        self.tr = build_train_eval_tensors(pl_ds, orig_targets, C, device)
        self.bv = build_fixed_eval_batch(pl_ds, orig_targets, bv_eval, device)
        self.dyn_path = os.path.join(out_dir, f'dynamics_s{seed}.csv')
        self.bv_path = os.path.join(out_dir, f'biasvariance_s{seed}.csv')
        for p in (self.dyn_path, self.bv_path):
            if os.path.isfile(p):
                os.remove(p)
        self.best_acc, self.best_ep = -1.0, 0

    def after_epoch(self, net, loss_fn, epoch, train_loss):
        row = dynamics_row(net, loss_fn, *self.tr, self.test_loader, self.device, epoch, train_loss)
        append_csv(self.dyn_path, DYN_FIELDS, row)
        if row['test_acc'] > self.best_acc:
            self.best_acc, self.best_ep = row['test_acc'], epoch
        if epoch == 1 or epoch % self.bv_every == 0:
            append_csv(self.bv_path, BV_FIELDS,
                       biasvariance_row(self.loss_name, net, loss_fn, *self.bv, self.C, self.k, epoch, self.seed,
                                        self.bv_m, self.device))
        return row
