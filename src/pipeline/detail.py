"""Optional diagnostic logging/plotting, off by default (`run --detail`):

1. Per-class accuracy/loss on the test set, logged every --detail_log_every
   epochs -- reproduces what scripts/legacy/run_extended_analysis.py's
   per_class_loss.csv used to do (a separate one-off script), now built into
   the main pipeline. Written to:
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/per_class_loss.csv
   Wired into every runner that has a genuine per-epoch loop (PRODEN, the
   PiCO family, ComCo family, CPE). Not wired into the '_train_simple_shape'
   family (CLPL/Wu2022/MCL-LOG/SCL-NL/OP/OP-W): those delegate whole chunks
   of `report_every` epochs to src.engine.train_algorithm at once, so there's
   no mid-chunk model checkpoint to evaluate at a finer cadence than
   report_every -- add support there if per-class detail on those algorithms
   is ever needed.

2. PiCO-specific: per-BATCH precision of the contrastive loss's positive/
   negative pair selection against ground truth (the model updates every
   batch, so this is logged at batch granularity, not averaged per epoch) --
   of the pairs SupConLoss's `mask` treats as "same class" (positive), what
   fraction actually share a true label; of the pairs it treats as
   "different class" (negative), what fraction actually differ. Only
   measurable for the within-batch block of `mask` (the MoCo queue's
   historical entries don't carry true labels). Also logs the raw pair
   confusion-matrix counts (tp/fp/tn/fn) behind those two precision numbers,
   so recall and other stats can be derived without re-deriving them from
   precision alone -- see log_pico_selection_stats_batch's docstring for the
   exact tp/fp/tn/fn definitions. Rows are buffered and written once per
   epoch (one row per batch) rather than once per batch, to avoid a
   file-open per batch over a shared NFS results dir. Written to:
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/pico_selection_stats.csv

3. t-SNE snapshots of the contrastive projection-head representation
   (the L2-normalized embedding SupConLoss/ComCoContrastiveLoss actually
   operate on -- see SupConResNet.forward in src/pico/resnet.py), every
   --tsne_every epochs. Independent of --detail (its own `run --tsne` flag):
   only meaningful for dual-encoder models with an `.encoder_q` (PiCO family,
   ComCo family) -- no-ops for anything else (PRODEN, PiCO-CLS, the
   '_train_simple_shape' baselines have no separate contrastive
   representation to visualize). Saves both the raw embeddings (for
   re-plotting with different t-SNE parameters later) and a rendered PNG to:
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/tsne/ep{epoch:04d}.{npz,png}

Both (1)/(2) and (3) add real cost -- (1) an extra full test-set forward pass
every detail_log_every epochs, (2) a few extra tensor reductions per batch,
(3) a t-SNE fit (CPU-bound, seconds) every tsne_every epochs -- which is why
they're all opt-in rather than always-on.

Also provides plot_heatmap()/plot_pico_selection_stats(), reproducing
scripts/legacy/plot_combined_heatmap_pair.py's figure and a new line chart
for (2), exposed via `scripts/run_pipeline.py detail-plot` /
`detail-plot-pico`. (3)'s PNGs are rendered inline during training (see
maybe_plot_tsne), not via a separate plotting subcommand.

4. Prediction-concentration logging (`run --concentration`): per-sample and
   averaged entropy + max-softmax-prob of the model's own predicted
   distribution, from a full non-augmented forward pass over the TRAINING
   set, every --concentration_log_every epochs. Ports the metrics from
   scripts/legacy/plot_logit_concentration.py (previously only computable
   offline from a disconnected legacy script) into the live pipeline.
   Generic across algorithm families (PiCO/PRODEN/ComCo) since it only
   depends on the model's own softmax output. Written to:
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/concentration_summary.csv  (averaged)
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/concentration/ep{epoch:04d}.npz  (per-sample)

5. ComCo-specific: per-BATCH precision of the contrastive loss's SELECTED
   positive pairs against ground truth (ComCo counterpart of (2) above) --
   see train_comco_epoch_with_selection_stats. Written to:
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/comco_selection_stats.csv

6. kNN accuracy evaluation (`run --knn_eval`): standard MoCo/SimCLR protocol
   -- train set (full, non-augmented) as a labeled reference bank, test set
   as queries, cosine-similarity-weighted top-k vote, top-1 accuracy. Only
   meaningful for dual-encoder models with an `.encoder_q` (PiCO/ComCo
   family); no-ops otherwise. Runs once at the end of training (the full
   O(N_test x N_train) similarity matrix is comparatively expensive, so this
   isn't repeated every checkpoint by default). Written to:
       results/<run_name>/detail/<algorithm>/C{C}_k{k}/knn_eval.csv
"""

import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.comco.model import ComCoModel
from src.pico.model import PiCOModel, PiCOOracleModel

_MEAN = [0.4914, 0.4822, 0.4465]
_STD = [0.247, 0.2435, 0.2616]

# ─── config helpers ────────────────────────────────────────────────────────


def _seed_matches(raw_cfg: dict) -> bool:
    """True unless a 'diagnostics_seed' has been designated (see runner.py,
    set when --seeds sweeps more than one seed) AND the CURRENT seed doesn't
    match it. All diagnostic gates (--detail, --tsne, --concentration,
    --knn_eval) route through this so that, when sweeping multiple seeds,
    the (fairly expensive, and not seed-scoped in its output paths)
    instrumentation only runs for one designated seed instead of multiplying
    overhead and interleaving different seeds' curves into the same files."""
    cfg = _detail_cfg(raw_cfg)
    diag_seed = cfg.get('diagnostics_seed')
    if diag_seed is None:
        return True
    return (raw_cfg or {}).get('_current_seed') == diag_seed


def is_enabled(raw_cfg: dict) -> bool:
    return bool((raw_cfg or {}).get('_detail', {}).get('enabled')) and _seed_matches(raw_cfg)


def _detail_cfg(raw_cfg: dict) -> dict:
    return (raw_cfg or {}).get('_detail') or {}


def cell_dir(raw_cfg: dict, algorithm: str, C: int) -> str:
    """results/<run_name>/detail/<algorithm>/C{C}_k{k}/ -- k comes from
    runner.py stashing '_current_k' in raw_cfg once per (C, k) cell, the same
    pattern already used for '_dataset_spec'."""
    base = _detail_cfg(raw_cfg)['out_dir']
    k = raw_cfg.get('_current_k', 0)
    return os.path.join(base, algorithm, f'C{C}_k{k}')


# ─── (1) per-class accuracy/loss checkpoint logging ────────────────────────


def _per_class_fields(C):
    return ['epoch'] + [f'loss_class_{c}' for c in range(C)] + [f'acc_class_{c}' for c in range(C)] + ['overall_acc']


@torch.no_grad()
def log_per_class_checkpoint(model, test_loader, device, C, epoch, out_dir, predict='argmax') -> float:
    """Full test-set forward pass; appends one row to out_dir/per_class_loss.csv
    with per-class CE loss + accuracy for this epoch checkpoint. Returns
    overall accuracy (%).

    predict='argmin' for CPE, which trains f to predict P(ybar|x) so the
    lowest-scoring class is the true-class prediction (see
    runners.py::_evaluate_argmin) -- everything else uses the standard
    argmax. Getting this wrong would silently produce a per-class accuracy
    breakdown inconsistent with the algorithm's actual (already-logged)
    overall accuracy.
    """
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in test_loader:
        images = images.to(device)
        if isinstance(model, (PiCOModel, PiCOOracleModel, ComCoModel)):
            logits = model(images, eval_only=True)
        else:
            logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    pred_labels = all_logits.argmin(dim=1) if predict == 'argmin' else all_logits.argmax(dim=1)

    log_probs = F.log_softmax(all_logits, dim=1)
    ce_per_sample = -log_probs[torch.arange(len(all_labels)), all_labels]

    per_class_loss, per_class_acc = [], []
    for c in range(C):
        cmask = (all_labels == c)
        if cmask.sum() == 0:
            per_class_loss.append(float('nan'))
            per_class_acc.append(float('nan'))
        else:
            per_class_loss.append(ce_per_sample[cmask].mean().item())
            per_class_acc.append((pred_labels[cmask] == c).float().mean().item() * 100)

    overall_acc = (pred_labels == all_labels).float().mean().item() * 100

    fields = _per_class_fields(C)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'per_class_loss.csv')
    new_file = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        row = {'epoch': epoch, 'overall_acc': round(overall_acc, 4)}
        for c in range(C):
            row[f'loss_class_{c}'] = round(per_class_loss[c], 6)
            row[f'acc_class_{c}'] = round(per_class_acc[c], 4)
        w.writerow(row)

    return overall_acc


def maybe_log_checkpoint(raw_cfg, model, test_loader, device, C, epoch, algorithm, predict='argmax'):
    """No-op unless --detail is enabled and `epoch` lands on a
    --detail_log_every boundary. Call once per epoch (1-indexed) from a
    runner's training loop -- decoupled from that runner's own
    `report_every` ETA-printing cadence on purpose, so detail resolution
    doesn't silently degrade if report_every is set coarser."""
    cfg = _detail_cfg(raw_cfg)
    if not is_enabled(raw_cfg):
        return
    log_every = cfg.get('log_every', 10)
    if epoch % log_every != 0:
        return
    log_per_class_checkpoint(model, test_loader, device, C, epoch, cell_dir(raw_cfg, algorithm, C), predict=predict)


# ─── shared: deterministic full-train-set forward pass ─────────────────────
# Used by both (4) concentration logging and (6) kNN eval below. Unlike
# algorithms.runners._IndexedDataset (RandomCrop+Flip train_transform), this
# is deterministic (ToTensor+Normalize only) so successive checkpoints
# measure genuine model change, not transform-induced noise.


class _DeterministicEvalDataset(Dataset):
    def __init__(self, data, labels, image_size=32, mean=_MEAN, std=_STD, modality='image'):
        self.data = data
        self.labels = labels
        self.modality = modality
        if modality == 'image':
            self._tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.modality == 'tabular':
            return torch.as_tensor(self.data[idx], dtype=torch.float32), self.labels[idx]
        img = Image.fromarray(self.data[idx])
        return self._tf(img), self.labels[idx]


def _build_train_eval_loader(pl_ds, spec, labels, batch_size=512):
    if spec is not None:
        ds = _DeterministicEvalDataset(pl_ds.data, labels, image_size=spec.image_size,
                                        mean=spec.mean, std=spec.std, modality=spec.modality)
    else:
        ds = _DeterministicEvalDataset(pl_ds.data, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def _predict_probs_over_trainset(model, loader, device):
    """Returns softmax probs [N, C] in dataset order. Dual-encoder models
    (PiCO/ComCo family) use eval_only=True; everything else calls model(imgs)
    directly -- same convention as log_per_class_checkpoint above."""
    model.eval()
    all_probs = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        if isinstance(model, (PiCOModel, PiCOOracleModel, ComCoModel)):
            logits = model(imgs, eval_only=True)
        else:
            logits = model(imgs)
        all_probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(all_probs, dim=0)


# ─── (4) prediction-concentration logging ──────────────────────────────────


def log_prediction_concentration_checkpoint(model, pl_ds, spec, device, C, epoch, out_dir, batch_size=512):
    """Full non-augmented train-set forward pass; computes per-sample entropy
    and max-softmax-prob of the model's own predicted distribution, appends
    averaged summary stats to concentration_summary.csv and saves the full
    per-sample arrays to concentration/ep{epoch:04d}.npz."""
    idx = np.arange(len(pl_ds))
    loader = _build_train_eval_loader(pl_ds, spec, idx, batch_size)
    probs = _predict_probs_over_trainset(model, loader, device)          # [N, C]
    eps = 1e-12
    entropy = -(probs * (probs + eps).log()).sum(dim=1)                  # [N]
    max_prob = probs.max(dim=1).values                                   # [N]

    os.makedirs(out_dir, exist_ok=True)
    fields = ['epoch', 'mean_entropy', 'std_entropy', 'median_entropy',
              'mean_max_prob', 'std_max_prob', 'median_max_prob']
    path = os.path.join(out_dir, 'concentration_summary.csv')
    new_file = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({
            'epoch': epoch,
            'mean_entropy': round(entropy.mean().item(), 6), 'std_entropy': round(entropy.std().item(), 6),
            'median_entropy': round(entropy.median().item(), 6),
            'mean_max_prob': round(max_prob.mean().item(), 6), 'std_max_prob': round(max_prob.std().item(), 6),
            'median_max_prob': round(max_prob.median().item(), 6),
        })

    conc_dir = os.path.join(out_dir, 'concentration')
    os.makedirs(conc_dir, exist_ok=True)
    np.savez(os.path.join(conc_dir, f'ep{epoch:04d}.npz'), entropy=entropy.numpy(), max_prob=max_prob.numpy())


def maybe_log_concentration(raw_cfg, model, pl_ds, device, C, epoch, algorithm, batch_size=512):
    """No-op unless --concentration is enabled, the current seed matches
    '_detail'.diagnostics_seed (see _seed_matches), and `epoch` lands on a
    --concentration_log_every boundary. Independent of --detail (same
    convention as --tsne)."""
    cfg = _detail_cfg(raw_cfg)
    conc_cfg = cfg.get('concentration') or {}
    if not conc_cfg.get('enabled') or not _seed_matches(raw_cfg):
        return
    log_every = conc_cfg.get('log_every', 10)
    if epoch % log_every != 0:
        return
    log_prediction_concentration_checkpoint(model, pl_ds, (raw_cfg or {}).get('_dataset_spec'),
                                             device, C, epoch, cell_dir(raw_cfg, algorithm, C), batch_size)


# ─── (2) PiCO contrastive pair-selection precision ─────────────────────────


def train_pico_epoch_with_selection_stats(pico_args, model, loader, loss_fn, loss_cont_fn,
                                           optimizer, epoch, device, raw_cfg, algorithm, C):
    """Identical to src.engine.train_pico_epoch, plus: measures, for every
    single BATCH (the model updates every batch, so an epoch-level average
    would hide how selection quality moves within an epoch), how often the
    contrastive loss's pseudo-label-based positive/negative pair selection
    (the `mask` built from pseudo_target_cont) agrees with ground truth, for
    anchor/candidate pairs that are both in the current batch (the MoCo
    queue's historical entries don't carry true labels, so only the
    within-batch block of `mask` is checkable).

    Per-batch rows are buffered in memory and written once at the end of the
    epoch (a single file open/append for the whole epoch) rather than once
    per batch -- results/ is often a shared NFS mount when training across
    multiple GPUs, and a file-open per batch over thousands of batches would
    add real, avoidable latency. If the process crashes mid-epoch, only that
    epoch's not-yet-flushed batches are lost; training itself (and its
    resume logic) is unaffected since this is a diagnostic-only log.

    Returns avg_total_loss for this epoch. pos/neg precision are NaN for any
    batch before pico_args['prot_start'] (mask is None until then, same as
    the original function's own behavior).
    """
    model.train()
    total_loss = 0.0
    start_upd_prot = epoch >= pico_args['prot_start']
    batch_rows = []   # (batch_idx, pos_precision, neg_precision)

    progress_bar = tqdm(loader, desc=f"PiCO Epoch {epoch + 1}/{pico_args['epochs']} [detail]")
    for batch_idx, (images_w, images_s, partial_Y, true_labels, index) in enumerate(progress_bar):
        images_w, images_s, partial_Y, index = (images_w.to(device), images_s.to(device),
                                                  partial_Y.to(device), index.to(device))

        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        mask = (torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
                if start_upd_prot else None)

        if mask is not None:
            true_dev = true_labels.to(device)
            same_true = torch.eq(true_dev.unsqueeze(0), true_dev.unsqueeze(1)).float()
            eye = torch.eye(batch_size, device=device)
            within_batch_mask = mask[:, :batch_size]
            m_pos = within_batch_mask * (1 - eye)          # selected-positive, excluding self-pairs
            m_neg = (1 - within_batch_mask) * (1 - eye)    # selected-negative, excluding self-pairs
            tp = (m_pos * same_true).sum().item()
            fp = (m_pos * (1 - same_true)).sum().item()
            tn = (m_neg * (1 - same_true)).sum().item()
            fn = (m_neg * same_true).sum().item()
            pos_total = m_pos.sum().item()
            neg_total = m_neg.sum().item()
            pos_prec = tp / pos_total if pos_total > 0 else float('nan')
            neg_prec = tn / neg_total if neg_total > 0 else float('nan')
        else:
            tp = fp = tn = fn = 0
            pos_prec = neg_prec = float('nan')
        batch_rows.append((batch_idx, pos_prec, neg_prec, tp, fp, tn, fn))

        loss_cls = loss_fn(cls_out, index)
        loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
        loss = loss_cls + pico_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    log_pico_selection_stats_batch(raw_cfg, algorithm, C, epoch + 1, batch_rows)

    return total_loss / len(loader)


def train_pico_epoch_fixed_with_selection_stats(pico_args, model, loader, loss_fn, loss_cont_fn,
                                                  optimizer, epoch, device, raw_cfg, algorithm, C):
    """Fixed-warm-up counterpart to train_pico_epoch_with_selection_stats above
    -- same per-batch pos/neg selection-precision measurement, but the loss
    itself follows src.fixed_pico_engine.train_pico_epoch_fixed's paper-exact
    warm-up (loss = loss_cls only, L_cont omitted entirely, not just switched
    to unsupervised MoCo). `mask` is still built unconditionally (purely for
    this diagnostic; NaN before prot_start, same convention as the original)
    so PiCO vs PiCO-Fixed selection-precision curves stay directly comparable
    even though Fixed doesn't use the mask for anything during warm-up.

    Added 2026-08-16: run_pico_fixed previously always called plain
    train_pico_epoch_fixed regardless of --detail, so pico_selection_stats.csv
    was never written for PiCO-Fixed -- see docs/pico_explanation.md."""
    model.train()
    total_loss = 0.0
    start_upd_prot = epoch >= pico_args['prot_start']
    batch_rows = []

    progress_bar = tqdm(loader, desc=f"PiCO-Fixed Epoch {epoch + 1}/{pico_args['epochs']} [detail]")
    for batch_idx, (images_w, images_s, partial_Y, true_labels, index) in enumerate(progress_bar):
        images_w, images_s, partial_Y, index = (images_w.to(device), images_s.to(device),
                                                  partial_Y.to(device), index.to(device))

        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        mask = (torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
                if start_upd_prot else None)

        if mask is not None:
            true_dev = true_labels.to(device)
            same_true = torch.eq(true_dev.unsqueeze(0), true_dev.unsqueeze(1)).float()
            eye = torch.eye(batch_size, device=device)
            within_batch_mask = mask[:, :batch_size]
            m_pos = within_batch_mask * (1 - eye)
            m_neg = (1 - within_batch_mask) * (1 - eye)
            tp = (m_pos * same_true).sum().item()
            fp = (m_pos * (1 - same_true)).sum().item()
            tn = (m_neg * (1 - same_true)).sum().item()
            fn = (m_neg * same_true).sum().item()
            pos_total = m_pos.sum().item()
            neg_total = m_neg.sum().item()
            pos_prec = tp / pos_total if pos_total > 0 else float('nan')
            neg_prec = tn / neg_total if neg_total > 0 else float('nan')
        else:
            tp = fp = tn = fn = 0
            pos_prec = neg_prec = float('nan')
        batch_rows.append((batch_idx, pos_prec, neg_prec, tp, fp, tn, fn))

        loss_cls = loss_fn(cls_out, index)
        if start_upd_prot:
            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            loss = loss_cls   # paper-exact warm-up: L_cont omitted entirely

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    log_pico_selection_stats_batch(raw_cfg, algorithm, C, epoch + 1, batch_rows)

    return total_loss / len(loader)


def log_pico_selection_stats_batch(raw_cfg, algorithm, C, epoch, batch_rows):
    """batch_rows: list of (batch_idx, pos_precision, neg_precision, tp, fp,
    tn, fn) for every batch in this epoch, written in one file open/append
    (see train_pico_epoch_with_selection_stats's docstring for why).

    tp/fp/tn/fn are the raw contrastive within-batch pair confusion-matrix
    counts (self-pairs excluded) behind pos_precision/neg_precision:
    - tp: mask says same pseudo-class AND true labels actually match
    - fp: mask says same pseudo-class BUT true labels differ
    - tn: mask says different pseudo-class AND true labels actually differ
    - fn: mask says different pseudo-class BUT true labels actually match
    pos_precision = tp/(tp+fp), neg_precision = tn/(tn+fn); recall
    (tp/(tp+fn)) and other confusion-matrix stats can be derived from the
    raw counts. All four are 0 (not NaN) for batches before prot_start,
    where mask is None and pos/neg_precision are NaN."""
    cfg = _detail_cfg(raw_cfg)
    if not cfg.get('enabled') or not batch_rows:
        return
    out_dir = cell_dir(raw_cfg, algorithm, C)
    fields = ['epoch', 'batch', 'pos_precision', 'neg_precision', 'tp', 'fp', 'tn', 'fn']
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pico_selection_stats.csv')
    new_file = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for batch_idx, pos_precision, neg_precision, tp, fp, tn, fn in batch_rows:
            w.writerow({
                'epoch': epoch,
                'batch': batch_idx,
                'pos_precision': '' if pos_precision != pos_precision else round(pos_precision, 6),
                'neg_precision': '' if neg_precision != neg_precision else round(neg_precision, 6),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            })


def train_pico_oracle_graded_epoch_with_stats(pico_args, model, loader, loss_fn, loss_cont_fn,
                                                optimizer, epoch, device, raw_cfg, algorithm, C,
                                                precision_threshold):
    """--detail-logging counterpart to
    src.oracle_pico_engine.train_pico_oracle_graded_epoch -- identical
    graduated-correction logic (see that function's docstring), plus:
    measures and buffers, per batch, both the NATURAL (pre-correction)
    selected-positive-pair precision the model's own pseudo-label mask
    would have had (what plain PiCO-Fixed would have used unmodified) and
    the precision actually used for this batch's contrastive loss after
    correction, plus the (untouched-target, but composition-shifted by any
    flips) negative-pair precision and how many pairs were flipped. Written
    to pico_oracle_correction_stats.csv via
    log_pico_oracle_correction_stats_batch."""
    model.train()
    total_loss = 0.0
    start_upd_prot = epoch >= pico_args['prot_start']
    batch_rows = []   # (batch_idx, precision_before, precision_after, neg_precision, pos_total, n_flipped)

    progress_bar = tqdm(loader, desc=f"PiCO-Oracle Epoch {epoch + 1}/{pico_args['epochs']} [detail]")
    for batch_idx, (images_w, images_s, partial_Y, true_labels, index) in enumerate(progress_bar):
        images_w, images_s, partial_Y, true_labels, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device),
            true_labels.to(device), index.to(device))

        cls_out, features, true_targets, pseudo_targets, score_prot = model(
            images_w, images_s, partial_Y, true_labels, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        loss_cls = loss_fn(cls_out, index)

        prec_before = prec_after = neg_prec = float('nan')
        pos_total = n_flipped = 0

        if start_upd_prot:
            mask = torch.eq(pseudo_targets[:batch_size].unsqueeze(1), pseudo_targets.unsqueeze(0)).float()
            same_true = torch.eq(true_targets[:batch_size].unsqueeze(1), true_targets.unsqueeze(0)).float()

            pos_total = int(mask.sum().item())
            true_pos = int((mask * same_true).sum().item()) if pos_total > 0 else 0
            if pos_total > 0:
                prec_before = true_pos / pos_total
                if prec_before < precision_threshold:
                    false_pos = pos_total - true_pos
                    n_flipped = max(0, min(false_pos, math.ceil(pos_total - true_pos / precision_threshold)))
                    if n_flipped > 0:
                        wrong_idx = ((mask == 1) & (same_true == 0)).nonzero(as_tuple=False)
                        perm = torch.randperm(wrong_idx.shape[0], device=device)[:n_flipped]
                        sel = wrong_idx[perm]
                        mask[sel[:, 0], sel[:, 1]] = 0.0
                new_pos_total = pos_total - n_flipped
                prec_after = (true_pos / new_pos_total) if new_pos_total > 0 else float('nan')

            neg_mask = (mask == 0).float()
            neg_total = neg_mask.sum().item()
            neg_prec = ((neg_mask * (1 - same_true)).sum() / neg_total).item() if neg_total > 0 else float('nan')

            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            loss = loss_cls

        batch_rows.append((batch_idx, prec_before, prec_after, neg_prec, pos_total, n_flipped))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    log_pico_oracle_correction_stats_batch(raw_cfg, algorithm, C, epoch + 1, batch_rows)

    return total_loss / len(loader)


def log_pico_oracle_correction_stats_batch(raw_cfg, algorithm, C, epoch, batch_rows):
    """batch_rows: list of (batch_idx, precision_before, precision_after,
    neg_precision, pos_total, n_flipped) for every batch in this epoch (see
    train_pico_oracle_graded_epoch_with_stats). precision_before is NaN
    before pico_args['prot_start'] (mask is None until then, same
    convention as the PiCO/PiCO-Fixed selection-stats logs)."""
    cfg = _detail_cfg(raw_cfg)
    if not cfg.get('enabled') or not batch_rows:
        return
    out_dir = cell_dir(raw_cfg, algorithm, C)
    fields = ['epoch', 'batch', 'precision_before', 'precision_after', 'neg_precision', 'pos_total', 'n_flipped']
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pico_oracle_correction_stats.csv')
    new_file = not os.path.isfile(path)

    def _r(v):
        return '' if v != v else round(v, 6)

    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for batch_idx, prec_before, prec_after, neg_prec, pos_total, n_flipped in batch_rows:
            w.writerow({
                'epoch': epoch, 'batch': batch_idx,
                'precision_before': _r(prec_before), 'precision_after': _r(prec_after),
                'neg_precision': _r(neg_prec), 'pos_total': pos_total, 'n_flipped': n_flipped,
            })


def train_pico_oracle_add_graded_epoch_with_stats(pico_args, model, loader, loss_fn, loss_cont_fn,
                                                    optimizer, epoch, device, raw_cfg, algorithm, C,
                                                    precision_threshold, max_add_ratio=1.0):
    """--detail-logging counterpart to
    src.oracle_pico_engine.train_pico_oracle_add_graded_epoch -- additive
    counterpart to train_pico_oracle_graded_epoch_with_stats above: instead
    of REMOVING false positives from the natural mask, ADDS randomly-chosen
    genuine true positives (same_true==1 pairs the natural mask did not
    select) until precision reaches precision_threshold, capped at
    max_add_ratio * pos_total additions per batch (see the engine
    function's docstring for why the cap exists). Logs precision_before/
    after, negative precision, pos_total, how many pairs were added
    (n_added), the actual achieved add_ratio (= n_added / pos_total, for
    directly comparing against how much of max_add_ratio's cap was used),
    and whether the cap bound (n_capped, so a `precision_after` that still
    falls short of the threshold can be told apart from one that genuinely
    reached it) to pico_oracle_add_correction_stats.csv via
    log_pico_oracle_add_correction_stats_batch."""
    model.train()
    total_loss = 0.0
    start_upd_prot = epoch >= pico_args['prot_start']
    # (batch_idx, precision_before, precision_after, neg_precision, pos_total, n_added, n_capped, add_ratio)
    batch_rows = []

    progress_bar = tqdm(loader, desc=f"PiCO-Oracle-Add Epoch {epoch + 1}/{pico_args['epochs']} [detail]")
    for batch_idx, (images_w, images_s, partial_Y, true_labels, index) in enumerate(progress_bar):
        images_w, images_s, partial_Y, true_labels, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device),
            true_labels.to(device), index.to(device))

        cls_out, features, true_targets, pseudo_targets, score_prot = model(
            images_w, images_s, partial_Y, true_labels, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        loss_cls = loss_fn(cls_out, index)

        prec_before = prec_after = neg_prec = float('nan')
        pos_total = n_added = 0
        n_capped = False

        if start_upd_prot:
            mask = torch.eq(pseudo_targets[:batch_size].unsqueeze(1), pseudo_targets.unsqueeze(0)).float()
            same_true = torch.eq(true_targets[:batch_size].unsqueeze(1), true_targets.unsqueeze(0)).float()

            pos_total = int(mask.sum().item())
            if pos_total > 0:
                true_pos = int((mask * same_true).sum().item())
                prec_before = true_pos / pos_total
                if prec_before < precision_threshold and precision_threshold < 1.0:
                    ideal_n_add = max(0, math.ceil(
                        (precision_threshold * pos_total - true_pos) / (1.0 - precision_threshold)))
                    cap = max(1, math.ceil(max_add_ratio * pos_total))
                    avail_idx = ((mask == 0) & (same_true == 1)).nonzero(as_tuple=False)
                    n_added = min(ideal_n_add, cap, avail_idx.shape[0])
                    n_capped = n_added < ideal_n_add
                    if n_added > 0:
                        perm = torch.randperm(avail_idx.shape[0], device=device)[:n_added]
                        sel = avail_idx[perm]
                        mask[sel[:, 0], sel[:, 1]] = 1.0
                new_pos_total = pos_total + n_added
                prec_after = ((true_pos + n_added) / new_pos_total) if new_pos_total > 0 else float('nan')

            neg_mask = (mask == 0).float()
            neg_total = neg_mask.sum().item()
            neg_prec = ((neg_mask * (1 - same_true)).sum() / neg_total).item() if neg_total > 0 else float('nan')

            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            loss = loss_cls

        # Actual achieved add ratio (n_added relative to the batch's own
        # natural pos_total) -- lets you directly see how much of
        # max_add_ratio's cap got used, without having to divide n_added by
        # pos_total yourself from the raw columns.
        add_ratio = (n_added / pos_total) if pos_total > 0 else float('nan')
        batch_rows.append((batch_idx, prec_before, prec_after, neg_prec, pos_total, n_added, n_capped, add_ratio))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    log_pico_oracle_add_correction_stats_batch(raw_cfg, algorithm, C, epoch + 1, batch_rows)

    return total_loss / len(loader)


def log_pico_oracle_add_correction_stats_batch(raw_cfg, algorithm, C, epoch, batch_rows):
    """batch_rows: list of (batch_idx, precision_before, precision_after,
    neg_precision, pos_total, n_added, n_capped, add_ratio) for every batch
    in this epoch (see train_pico_oracle_add_graded_epoch_with_stats)."""
    cfg = _detail_cfg(raw_cfg)
    if not cfg.get('enabled') or not batch_rows:
        return
    out_dir = cell_dir(raw_cfg, algorithm, C)
    fields = ['epoch', 'batch', 'precision_before', 'precision_after', 'neg_precision',
              'pos_total', 'n_added', 'n_capped', 'add_ratio']
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'pico_oracle_add_correction_stats.csv')
    new_file = not os.path.isfile(path)

    def _r(v):
        return '' if v != v else round(v, 6)

    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for batch_idx, prec_before, prec_after, neg_prec, pos_total, n_added, n_capped, add_ratio in batch_rows:
            w.writerow({
                'epoch': epoch, 'batch': batch_idx, 'add_ratio': _r(add_ratio),
                'precision_before': _r(prec_before), 'precision_after': _r(prec_after),
                'neg_precision': _r(neg_prec), 'pos_total': pos_total, 'n_added': n_added,
                'n_capped': int(n_capped),
            })


# ─── (5) ComCo contrastive positive-pair selection precision ──────────────


def train_comco_epoch_with_selection_stats(comco_args, model, loader, cls_loss_fn, cont_loss_fn,
                                            optimizer, epoch, device, raw_cfg, algorithm, C):
    """ComCo counterpart to train_pico_epoch_with_selection_stats above:
    measures, per BATCH, what fraction of ComCoContrastiveLoss's SELECTED
    positive pairs (the always-included key-view pair, plus, after
    warmup_pos, the top-K pseudo-label-matched pool entries) genuinely share
    the anchor's TRUE label -- true labels used only for this diagnostic,
    never for training. Only pool indices < 2B (this batch's own q/k views,
    see ComCoModel.forward: all_feats = cat([q, k, queue])) carry a
    checkable true label; MoCo-queue entries (indices >= 2B) don't.

    No negative-pair precision (unlike PiCO): ComCo's negative set is a
    whole per-class pool subset chosen by complementary-label distance, not
    a pairwise same/different-class judgement, so "precision against true
    label" isn't the natural metric there.

    Written to results/<run_name>/detail/<algorithm>/C{C}_k{k}/comco_selection_stats.csv."""
    model.train()
    total_loss = 0.0
    warmup_pos = epoch >= comco_args['warmup_pos']
    warmup_neg = epoch >= comco_args['warmup_neg']
    batch_rows = []   # (batch_idx, pos_precision, pos_total)

    progress_bar = tqdm(loader, desc=f"ComCo Epoch {epoch + 1}/{comco_args['epochs']} [detail]")
    for batch_idx, (images_w, images_s, comp_mask, true_labels, index) in enumerate(progress_bar):
        images_w = images_w.to(device)
        images_s = images_s.to(device)
        comp_mask = comp_mask.to(device)
        B = images_w.shape[0]

        cls_out, q, all_feats, all_pseudo, all_comp = model(images_w, images_s, comp_mask, comco_args)
        pseudo_q = cls_out.argmax(dim=1)

        loss_cls = cls_loss_fn(cls_out, comp_mask)
        loss_cont, pos_mask, denom_mask = cont_loss_fn(
            q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos, warmup_neg, return_masks=True)
        loss = loss_cls + comco_args['loss_weight'] * loss_cont

        true_dev = true_labels.to(device)
        pool_true_2B = torch.cat([true_dev, true_dev], dim=0)   # pool[0:B]=q-view, pool[B:2B]=k-view, same order
        pos_mask_2B = pos_mask[:, :2 * B]
        same_true = torch.eq(true_dev.unsqueeze(1), pool_true_2B.unsqueeze(0)).float()
        pos_total = pos_mask_2B.sum().item()
        pos_prec = (pos_mask_2B * same_true).sum().item() / pos_total if pos_total > 0 else float('nan')
        batch_rows.append((batch_idx, pos_prec, int(pos_total)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    log_comco_selection_stats_batch(raw_cfg, algorithm, C, epoch + 1, batch_rows)

    return total_loss / len(loader)


def log_comco_selection_stats_batch(raw_cfg, algorithm, C, epoch, batch_rows):
    """batch_rows: list of (batch_idx, pos_precision, pos_total) for every
    batch in this epoch, written in one file open/append (see
    log_pico_selection_stats_batch's docstring for why)."""
    cfg = _detail_cfg(raw_cfg)
    if not cfg.get('enabled') or not batch_rows:
        return
    out_dir = cell_dir(raw_cfg, algorithm, C)
    fields = ['epoch', 'batch', 'pos_precision', 'pos_total']
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'comco_selection_stats.csv')
    new_file = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for batch_idx, pos_precision, pos_total in batch_rows:
            w.writerow({'epoch': epoch, 'batch': batch_idx,
                        'pos_precision': '' if pos_precision != pos_precision else round(pos_precision, 6),
                        'pos_total': pos_total})


# ─── (3) t-SNE snapshots of the contrastive representation ────────────────


@torch.no_grad()
def maybe_plot_tsne(raw_cfg, model, test_loader, device, C, epoch, algorithm):
    """No-op unless --tsne is enabled and `epoch` lands on a --tsne_every
    boundary, or `model` has no `.encoder_q` (i.e. it isn't a PiCO/ComCo
    -family dual-encoder model -- PRODEN/PiCO-CLS/the '_train_simple_shape'
    baselines have no separate contrastive representation to visualize).

    Extracts the L2-normalized projection-head embedding
    (model.encoder_q(images) -> (_, feat_c), see SupConResNet.forward) for
    up to tsne_max_points test-set samples, reduces to 2D with t-SNE, and
    saves both the raw embeddings (.npz, for re-plotting with different t-SNE
    parameters later without retraining) and a rendered scatter plot (.png)
    colored by ground-truth class.
    """
    cfg = _detail_cfg(raw_cfg)
    tsne_cfg = cfg.get('tsne') or {}
    if not tsne_cfg.get('enabled') or not _seed_matches(raw_cfg):
        return
    every = tsne_cfg.get('every', 50)
    if epoch % every != 0:
        return
    if not hasattr(model, 'encoder_q'):
        return

    max_points = tsne_cfg.get('max_points', 2000)
    model.eval()
    feats, labels = [], []
    n = 0
    for images, lab in test_loader:
        images = images.to(device)
        _, feat_c = model.encoder_q(images)
        feats.append(feat_c.cpu())
        labels.append(lab)
        n += images.shape[0]
        if n >= max_points:
            break
    feats = torch.cat(feats, dim=0)[:max_points].numpy()
    labels = torch.cat(labels, dim=0)[:max_points].numpy()

    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, len(feats) // 100))
    emb_2d = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42).fit_transform(feats)

    tsne_dir = os.path.join(cell_dir(raw_cfg, algorithm, C), 'tsne')
    os.makedirs(tsne_dir, exist_ok=True)
    np.savez(os.path.join(tsne_dir, f'ep{epoch:04d}.npz'), feats=feats, labels=labels, emb_2d=emb_2d)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', s=8, alpha=0.8)
    ax.set_title(f'{algorithm}  C={C}  epoch={epoch}\nt-SNE of contrastive representation ({len(feats)} pts)',
                 fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(*scatter.legend_elements(num=min(C, 20)), title='class', fontsize=6,
              loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.savefig(os.path.join(tsne_dir, f'ep{epoch:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── (6) kNN accuracy evaluation ───────────────────────────────────────────


@torch.no_grad()
def knn_eval(model, pl_ds, orig_targets, test_loader, device, C, spec=None,
             k_neighbors=20, batch_size=512, temperature=0.07):
    """Standard MoCo/SimCLR kNN protocol: L2-normalized encoder_q
    projection-head features over the FULL non-augmented training set as a
    labeled reference bank, test set as queries, cosine-similarity-weighted
    top-k_neighbors vote (weight = exp(sim/temperature)), top-1 accuracy.
    Returns None (no-op) if `model` has no .encoder_q (PRODEN and other
    single-encoder algorithms have no separate contrastive representation)."""
    if not hasattr(model, 'encoder_q'):
        return None
    model.eval()

    train_loader = _build_train_eval_loader(pl_ds, spec, orig_targets, batch_size)
    bank_feats, bank_labels = [], []
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        _, feat = model.encoder_q(imgs)
        bank_feats.append(F.normalize(feat, dim=1).cpu())
        bank_labels.append(labels if torch.is_tensor(labels) else torch.as_tensor(labels))
    bank_feats = torch.cat(bank_feats, dim=0).to(device)
    bank_labels = torch.cat(bank_labels, dim=0).to(device)

    correct = total = 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, feat = model.encoder_q(imgs)
        feat = F.normalize(feat, dim=1)
        sim = feat @ bank_feats.T
        topk_sim, topk_idx = sim.topk(min(k_neighbors, bank_feats.shape[0]), dim=1)
        topk_labels = bank_labels[topk_idx]
        weights = (topk_sim / temperature).exp()
        one_hot = F.one_hot(topk_labels, C).float()
        scores = (one_hot * weights.unsqueeze(-1)).sum(dim=1)
        pred = scores.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)


def maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, test_loader, device, C, epoch, algorithm):
    """No-op unless --knn_eval is enabled, or `model` has no .encoder_q.
    Intended to be called once at the end of training (the full
    O(N_test x N_train) similarity matrix is comparatively expensive)."""
    cfg = _detail_cfg(raw_cfg)
    knn_cfg = cfg.get('knn') or {}
    if not knn_cfg.get('enabled') or not _seed_matches(raw_cfg):
        return
    acc = knn_eval(model, pl_ds, orig_targets, test_loader, device, C,
                    spec=(raw_cfg or {}).get('_dataset_spec'),
                    k_neighbors=knn_cfg.get('k', 20), temperature=knn_cfg.get('temperature', 0.07))
    if acc is None:
        return
    out_dir = cell_dir(raw_cfg, algorithm, C)
    fields = ['epoch', 'knn_top1_acc', 'k_neighbors', 'temperature']
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'knn_eval.csv')
    new_file = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({'epoch': epoch, 'knn_top1_acc': round(acc, 4),
                    'k_neighbors': knn_cfg.get('k', 20), 'temperature': knn_cfg.get('temperature', 0.07)})
    print(f'  [knn_eval] {algorithm} C={C} epoch={epoch}  top1={acc:.2f}%', flush=True)


# ─── plotting ───────────────────────────────────────────────────────────────


def _load_per_class_csv(path, C):
    if not os.path.isfile(path):
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    import numpy as np
    epochs = [int(r['epoch']) for r in rows]
    acc_mat = np.array([[float(r.get(f'acc_class_{c}', 'nan')) for c in range(C)] for r in rows])
    loss_mat = np.array([[float(r.get(f'loss_class_{c}', 'nan')) for c in range(C)] for r in rows])
    overall = np.array([float(r['overall_acc']) for r in rows])
    return {'epochs': epochs, 'acc_mat': acc_mat, 'loss_mat': loss_mat, 'overall': overall}


def plot_heatmap(results_dir: str, alg_l: str, C: int, k: int, out_path: str,
                  alg_r: str = None, acc_only: bool = True, class_names=None,
                  alg_l_display: str = None, alg_r_display: str = None) -> str:
    """Per-class accuracy (and optionally loss) heatmap over epoch
    checkpoints, for one algorithm or two side-by-side. Reproduces
    scripts/legacy/plot_combined_heatmap_pair.py, reading this module's own
    per_class_loss.csv files instead of that script's separate results dir.

    alg_l/alg_r select which algorithm's data to read (must match the
    on-disk detail/<algorithm>/ folder name); alg_l_display/alg_r_display
    optionally override only the text shown in the title (e.g. presenting
    an internal name like 'PiCO-Fixed' or 'PiCO-Oracle' as plain 'PiCO' for
    a slide, without renaming anything on disk)."""
    alg_l_display = alg_l_display or alg_l
    alg_r_display = alg_r_display or alg_r
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = os.path.join(results_dir, 'detail')
    dL = _load_per_class_csv(os.path.join(base, alg_l, f'C{C}_k{k}', 'per_class_loss.csv'), C)
    dR = _load_per_class_csv(os.path.join(base, alg_r, f'C{C}_k{k}', 'per_class_loss.csv'), C) if alg_r else None

    if dL is None and dR is None:
        algs = alg_l if not alg_r else f'{alg_l} or {alg_r}'
        raise ValueError(f'No detail logs found for C={C} k={k}, algorithm(s) {algs} under {base}/ '
                          f'(did you run with --detail?)')

    epochs = (dL or dR)['epochs']
    T = len(epochs)
    n_cols = 2 if alg_r else 1
    n_rows = 1 if acc_only else 2

    def _draw(ax, mat, cmap, vmin, vmax, show_xlabel, show_ylabel, ylabel=''):
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_yticks(range(C))
        if show_ylabel:
            ax.set_yticklabels(class_names if class_names else range(C), fontsize=6)
            ax.set_ylabel(ylabel, fontsize=9)
        else:
            ax.set_yticklabels([])
        if show_xlabel:
            ax.set_xticks(range(T))
            ax.set_xticklabels(epochs, rotation=45, fontsize=6)
            ax.set_xlabel('Epoch checkpoint', fontsize=9)
        else:
            ax.set_xticks([])
        return im

    fig_w = max(8 * n_cols, T * 0.32 * n_cols + 2)
    fig_h = max(5 * n_rows, C * 0.32 * n_rows + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              gridspec_kw={'hspace': 0.06, 'wspace': 0.04}, squeeze=False)

    acc_l = dL['acc_mat'] if dL else np.full((T, C), np.nan)
    fa_l = f'{dL["overall"][-1]:.1f}%' if dL else 'N/A'
    im_acc_l = _draw(axes[0, 0], acc_l, 'RdYlGn', 0, 100, show_xlabel=acc_only, show_ylabel=True,
                      ylabel='Accuracy (%)')
    axes[0, 0].set_title(f'{alg_l_display}  (final {fa_l})', fontsize=11, fontweight='bold')

    if alg_r:
        acc_r = dR['acc_mat'] if dR else np.full((T, C), np.nan)
        fa_r = f'{dR["overall"][-1]:.1f}%' if dR else 'N/A'
        _draw(axes[0, 1], acc_r, 'RdYlGn', 0, 100, show_xlabel=acc_only, show_ylabel=False)
        axes[0, 1].set_title(f'{alg_r_display}  (final {fa_r})', fontsize=11, fontweight='bold')
    fig.colorbar(im_acc_l, ax=axes[0, :], label='Accuracy (%)', shrink=0.85, pad=0.01)

    if not acc_only:
        loss_l = dL['loss_mat'] if dL else np.full((T, C), np.nan)
        mats = [m for m in ([loss_l] + ([dR['loss_mat']] if dR else [])) if not np.all(np.isnan(m))]
        loss_vmax = np.nanpercentile(np.concatenate(mats), 97) if mats else 1.0

        im_loss_l = _draw(axes[1, 0], loss_l, 'RdYlGn_r', 0, loss_vmax, show_xlabel=True, show_ylabel=True,
                           ylabel='CE Loss')
        if alg_r:
            loss_r = dR['loss_mat'] if dR else np.full((T, C), np.nan)
            _draw(axes[1, 1], loss_r, 'RdYlGn_r', 0, loss_vmax, show_xlabel=True, show_ylabel=False)
        fig.colorbar(im_loss_l, ax=axes[1, :], label='CE loss', shrink=0.85, pad=0.01)

    title = f'{alg_l_display} vs {alg_r_display}' if alg_r else alg_l_display
    fig.suptitle(f'{title}  —  C={C}  k={k}', fontsize=13, fontweight='bold')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_heatmap_multi(entries: list, C: int, k: int, out_path: str,
                        acc_only: bool = True, class_names=None) -> str:
    """Per-class accuracy (and optionally loss) heatmap for N algorithms
    side by side in one row per metric -- generalizes plot_heatmap's
    two-algorithm (alg_l/alg_r) case to an arbitrary number of columns.

    entries: list of (results_dir, algorithm, display_name) tuples, one
    column per entry, in the given order. Each algorithm can live in a
    different results_dir/run_name (e.g. one run per algorithm, as with
    run_main_pipeline_batch.sh's naming scheme) -- unlike plot_heatmap,
    which assumes both algorithms share one results_dir."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = []
    for results_dir, alg, display in entries:
        d = _load_per_class_csv(os.path.join(results_dir, 'detail', alg, f'C{C}_k{k}', 'per_class_loss.csv'), C)
        data.append((alg, display, d))

    loaded = [d for _, _, d in data if d is not None]
    if not loaded:
        algs = ', '.join(alg for alg, _, _ in data)
        raise ValueError(f'No detail logs found for C={C} k={k}, algorithm(s) {algs} '
                          f'(did you run with --detail?)')

    epochs = loaded[0]['epochs']
    T = len(epochs)
    n_cols = len(entries)
    n_rows = 1 if acc_only else 2

    def _draw(ax, mat, cmap, vmin, vmax, show_xlabel, show_ylabel, ylabel=''):
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_yticks(range(C))
        if show_ylabel:
            ax.set_yticklabels(class_names if class_names else range(C), fontsize=6)
            ax.set_ylabel(ylabel, fontsize=9)
        else:
            ax.set_yticklabels([])
        if show_xlabel:
            ax.set_xticks(range(T))
            ax.set_xticklabels(epochs, rotation=45, fontsize=6)
            ax.set_xlabel('Epoch checkpoint', fontsize=9)
        else:
            ax.set_xticks([])
        return im

    fig_w = max(6 * n_cols, T * 0.28 * n_cols + 2)
    fig_h = max(5 * n_rows, C * 0.32 * n_rows + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              gridspec_kw={'hspace': 0.06, 'wspace': 0.06}, squeeze=False)

    im_acc = None
    for i, (alg, display, d) in enumerate(data):
        acc = d['acc_mat'] if d else np.full((T, C), np.nan)
        fa = f'{d["overall"][-1]:.1f}%' if d else 'N/A'
        im = _draw(axes[0, i], acc, 'RdYlGn', 0, 100, show_xlabel=acc_only, show_ylabel=(i == 0),
                   ylabel='Accuracy (%)')
        axes[0, i].set_title(f'{display}  (final {fa})', fontsize=11, fontweight='bold')
        im_acc = im_acc or im
    fig.colorbar(im_acc, ax=axes[0, :], label='Accuracy (%)', shrink=0.85, pad=0.01)

    if not acc_only:
        loss_mats = [d['loss_mat'] if d else np.full((T, C), np.nan) for _, _, d in data]
        valid_mats = [m for m in loss_mats if not np.all(np.isnan(m))]
        loss_vmax = np.nanpercentile(np.concatenate(valid_mats), 97) if valid_mats else 1.0

        im_loss = None
        for i, loss_mat in enumerate(loss_mats):
            im = _draw(axes[1, i], loss_mat, 'RdYlGn_r', 0, loss_vmax, show_xlabel=True, show_ylabel=(i == 0),
                       ylabel='CE Loss')
            im_loss = im_loss or im
        fig.colorbar(im_loss, ax=axes[1, :], label='CE loss', shrink=0.85, pad=0.01)

    title = ' vs '.join(display for _, display, _ in data)
    fig.suptitle(f'{title}  —  C={C}  k={k}', fontsize=13, fontweight='bold')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_heatmap_side_by_side(results_dir: str, alg: str, C: int, k: int, out_path: str,
                               display_name: str = None, class_names=None) -> str:
    """Single-algorithm per-class accuracy (left) and CE loss (right) heatmaps,
    side by side in one row -- a horizontal counterpart to plot_heatmap's
    default vertical (accuracy-on-top, loss-below) stacking for the
    single-algorithm case. Reads the same per_class_loss.csv as plot_heatmap."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    display_name = display_name or alg
    d = _load_per_class_csv(os.path.join(results_dir, 'detail', alg, f'C{C}_k{k}', 'per_class_loss.csv'), C)
    if d is None:
        raise ValueError(f'No detail logs found for C={C} k={k}, algorithm {alg} under '
                          f'{results_dir}/detail/ (did you run with --detail?)')

    epochs = d['epochs']
    T = len(epochs)

    def _draw(ax, mat, cmap, vmin, vmax, ylabel):
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_yticks(range(C))
        ax.set_yticklabels(class_names if class_names else range(C), fontsize=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(T))
        ax.set_xticklabels(epochs, rotation=45, fontsize=6)
        ax.set_xlabel('Epoch checkpoint', fontsize=9)
        return im

    fig_w = max(16, T * 0.32 * 2 + 3)
    fig_h = max(5, C * 0.32 + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw={'wspace': 0.28})

    fa = f'{d["overall"][-1]:.1f}%'
    im_acc = _draw(axes[0], d['acc_mat'], 'RdYlGn', 0, 100, ylabel='Accuracy (%)')
    axes[0].set_title('Accuracy', fontsize=11, fontweight='bold')
    fig.colorbar(im_acc, ax=axes[0], label='Accuracy (%)', shrink=0.85, pad=0.02)

    loss_mat = d['loss_mat']
    loss_vmax = np.nanpercentile(loss_mat, 97) if not np.all(np.isnan(loss_mat)) else 1.0
    im_loss = _draw(axes[1], loss_mat, 'RdYlGn_r', 0, loss_vmax, ylabel='CE Loss')
    axes[1].set_title('CE loss', fontsize=11, fontweight='bold')
    fig.colorbar(im_loss, ax=axes[1], label='CE loss', shrink=0.85, pad=0.02)

    fig.suptitle(f'{display_name}  (final {fa})  —  C={C}  k={k}', fontsize=13, fontweight='bold')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _rolling_mean(values, window):
    """NaN-aware centered rolling mean (window counted in valid/non-NaN
    points within the window, not raw index count)."""
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < 2:
        return arr
    half = window // 2
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        seg = arr[max(0, i - half):min(len(arr), i + half + 1)]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            out[i] = seg.mean()
    return out


def plot_pico_selection_stats(results_dir: str, algorithm: str, C: int, k: int, out_path: str,
                               display_name: str = None) -> str:
    """Line chart of positive/negative contrastive pair-selection precision
    vs. ground truth, one point per BATCH in chronological (file) order --
    the log has one row per batch (see log_pico_selection_stats_batch). Raw
    per-batch values are plotted thin/faded; a rolling mean (window = one
    epoch's worth of batches) is overlaid bold since per-batch precision is
    typically too noisy to read trends from directly.

    display_name optionally overrides only the title text (algorithm still
    selects which detail/<algorithm>/ folder to read from)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    display_name = display_name or algorithm
    path = os.path.join(results_dir, 'detail', algorithm, f'C{C}_k{k}', 'pico_selection_stats.csv')
    if not os.path.isfile(path):
        raise ValueError(f'No selection-stats log found at {path} (did you run with --detail?)')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path} has no rows')

    def _f(v):
        return float('nan') if v in ('', 'nan') else float(v)

    steps = list(range(len(rows)))   # chronological training step (rows are written in batch order)
    pos = [_f(r['pos_precision']) for r in rows]
    neg = [_f(r['neg_precision']) for r in rows]

    # batches-per-epoch, for the rolling-mean window and epoch-boundary gridlines
    epochs = [int(r['epoch']) for r in rows]
    batches_per_epoch = sum(1 for e in epochs if e == epochs[0]) or 1

    pos_smooth = _rolling_mean(pos, batches_per_epoch)
    neg_smooth = _rolling_mean(neg, batches_per_epoch)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, pos, color='#2ca02c', linewidth=0.6, alpha=0.35)
    ax.plot(steps, neg, color='#d62728', linewidth=0.6, alpha=0.35)
    ax.plot(steps, pos_smooth, label='positive-pair precision  P(same true class | selected positive)',
            color='#2ca02c', linewidth=2)
    ax.plot(steps, neg_smooth, label='negative-pair precision  P(different true class | selected negative)',
            color='#d62728', linewidth=2)
    ax.set_xlabel(f'Training step (batch, chronological — ~{batches_per_epoch} batches/epoch)')
    ax.set_ylabel('Precision vs. ground truth')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{display_name} contrastive pair-selection precision  —  C={C}  k={k}  '
                 f'(faint = per-batch, bold = 1-epoch rolling mean)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_pico_selection_stats_multi_k(run_dirs: dict, algorithm: str, C: int, out_path: str,
                                       display_name: str = None) -> str:
    """Overlay positive-pair selection precision (rolling-mean only, no raw
    per-batch faint lines -- multiple k's on one axis gets too busy with
    those) for several k values on one chart, one line per k. Shows only
    the natural/uncorrected 'positive-pair precision' -- no negative-pair
    line -- since the point of this view is comparing how precision
    degrades as k grows, not the full pos/neg breakdown a single-k plot
    (plot_pico_selection_stats) already covers.

    run_dirs: {k: results_dir} -- each k's data may live in its own
    run_name/results dir (as with run_main_pipeline_batch.sh's one-run_name-
    per-algorithm-per-k scheme), so this takes a dict instead of a single
    results_dir + list of k's."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    display_name = display_name or algorithm

    def _f(v):
        return float('nan') if v in ('', 'nan') else float(v)

    ks = sorted(run_dirs, reverse=True)
    series = {}   # k -> pos_smooth array
    for k in ks:
        path = os.path.join(run_dirs[k], 'detail', algorithm, f'C{C}_k{k}', 'pico_selection_stats.csv')
        if not os.path.isfile(path):
            print(f'  [plot_pico_selection_stats_multi_k] skip k={k}: no data at {path}', flush=True)
            continue
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        pos = [_f(r['pos_precision']) for r in rows]
        epochs = [int(r['epoch']) for r in rows]
        batches_per_epoch = sum(1 for e in epochs if e == epochs[0]) or 1
        series[k] = _rolling_mean(pos, batches_per_epoch)

    # Different k's can end up with different row counts (most often because
    # a crashed-and-retried run appended a second pass's rows on top of an
    # earlier partial attempt's, since these CSVs are opened in append mode
    # and never truncated between process launches) -- clip every series to
    # the shortest one so lines are visually comparable over the same
    # x-range rather than some trailing off early.
    if series:
        min_len = min(len(v) for v in series.values())
        for k in series:
            series[k] = series[k][:min_len]

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap('viridis')
    for i, k in enumerate(ks):
        if k not in series:
            continue
        steps = list(range(len(series[k])))
        color = cmap(i / max(1, len(ks) - 1))
        ax.plot(steps, series[k], label=f'k={k}', color=color, linewidth=2)

    ax.set_xlabel('Training step (batch, chronological)')
    ax.set_ylabel('Positive-pair precision  P(same true class | selected positive)')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{display_name}  —  C={C}  —  positive-pair precision vs. k  (1-epoch rolling mean)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_pico_oracle_correction_stats(results_dir: str, C: int, k: int, out_path: str,
                                       show_neg: bool = True) -> str:
    """PiCO-Oracle's graduated correction: natural (pre-correction)
    selected-positive-pair precision vs. the precision actually used for
    training after correction, one point per BATCH in chronological order.
    Reads pico_oracle_correction_stats.csv (see
    log_pico_oracle_correction_stats_batch / train_pico_oracle_graded_epoch_with_stats).
    The gap between the two lines is exactly what the correction mechanism
    is doing each batch; 'after' should track at or above whatever
    pico.oracle_precision_threshold the run used once training reaches
    prot_start (flat 'before'≈'after' before that point means warm-up)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = os.path.join(results_dir, 'detail', 'PiCO-Oracle', f'C{C}_k{k}', 'pico_oracle_correction_stats.csv')
    if not os.path.isfile(path):
        raise ValueError(f'No oracle correction-stats log found at {path} '
                          f'(did you run PiCO-Oracle with --detail?)')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path} has no rows')

    def _f(v):
        return float('nan') if v in ('', 'nan') else float(v)

    steps = list(range(len(rows)))
    before = [_f(r['precision_before']) for r in rows]
    after = [_f(r['precision_after']) for r in rows]
    neg = [_f(r['neg_precision']) for r in rows]

    epochs = [int(r['epoch']) for r in rows]
    batches_per_epoch = sum(1 for e in epochs if e == epochs[0]) or 1

    before_smooth = _rolling_mean(before, batches_per_epoch)
    after_smooth = _rolling_mean(after, batches_per_epoch)
    neg_smooth = _rolling_mean(neg, batches_per_epoch)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, before, color='#d62728', linewidth=0.6, alpha=0.25)
    ax.plot(steps, after, color='#2ca02c', linewidth=0.6, alpha=0.25)
    ax.plot(steps, before_smooth, label='positive-pair precision BEFORE correction (natural, = PiCO-Fixed)',
            color='#d62728', linewidth=2)
    ax.plot(steps, after_smooth, label='positive-pair precision AFTER correction (used for training)',
            color='#2ca02c', linewidth=2)
    if show_neg:
        ax.plot(steps, neg_smooth, label='negative-pair precision (as used for training)',
                color='#1f77b4', linewidth=1.5, linestyle='--')
    ax.set_xlabel(f'Training step (batch, chronological — ~{batches_per_epoch} batches/epoch)')
    ax.set_ylabel('Precision vs. ground truth')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'PiCO-Oracle graduated correction  —  C={C}  k={k}  '
                 f'(faint = per-batch, bold = 1-epoch rolling mean)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _load_concentration_csv(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {
        'epochs': [int(r['epoch']) for r in rows],
        'mean_entropy': [float(r['mean_entropy']) for r in rows],
        'std_entropy': [float(r['std_entropy']) for r in rows],
        'mean_max_prob': [float(r['mean_max_prob']) for r in rows],
        'std_max_prob': [float(r['std_max_prob']) for r in rows],
    }


def plot_concentration_trend(entries: list, C: int, k: int, out_path: str) -> str:
    """Overlay per-algorithm prediction-concentration trends -- mean entropy
    (top panel) and mean max-softmax-prob (bottom panel), both vs. epoch,
    +-1 std shaded -- for N algorithms on one figure. Reads
    concentration_summary.csv (see maybe_log_concentration / `run
    --concentration`).

    entries: list of (results_dir, algorithm, display_name) tuples, one line
    per entry, in order -- same convention as plot_heatmap_multi, so each
    algorithm can live in a different results_dir/run_name."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = []
    for results_dir, alg, display in entries:
        d = _load_concentration_csv(
            os.path.join(results_dir, 'detail', alg, f'C{C}_k{k}', 'concentration_summary.csv'))
        data.append((alg, display, d))

    loaded = [d for _, _, d in data if d is not None]
    if not loaded:
        algs = ', '.join(alg for alg, _, _ in data)
        raise ValueError(f'No concentration logs found for C={C} k={k}, algorithm(s) {algs} '
                          f'(did you run with --concentration?)')

    fig, (ax_ent, ax_prob) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    cmap = plt.get_cmap('tab10')
    for i, (alg, display, d) in enumerate(data):
        if d is None:
            print(f'  [plot_concentration_trend] skip {alg}: no concentration_summary.csv '
                  f'(did you run it with --concentration?)', flush=True)
            continue
        color = cmap(i % 10)
        epochs = d['epochs']
        me, se = np.array(d['mean_entropy']), np.array(d['std_entropy'])
        mp, sp = np.array(d['mean_max_prob']), np.array(d['std_max_prob'])
        ax_ent.plot(epochs, me, label=display, color=color, linewidth=2)
        ax_ent.fill_between(epochs, me - se, me + se, color=color, alpha=0.15)
        ax_prob.plot(epochs, mp, label=display, color=color, linewidth=2)
        ax_prob.fill_between(epochs, mp - sp, mp + sp, color=color, alpha=0.15)

    ax_ent.set_ylabel('Mean prediction entropy')
    ax_ent.set_title(f'Prediction concentration  —  C={C}  k={k}  (shaded = ±1 std across samples)')
    ax_ent.grid(True, alpha=0.3)
    ax_ent.legend(fontsize=8, loc='best')

    ax_prob.set_ylabel('Mean max softmax prob')
    ax_prob.set_xlabel('Epoch checkpoint')
    ax_prob.set_ylim(0, 1.05)
    ax_prob.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_knn_eval_bar(entries: list, C: int, k: int, out_path: str) -> str:
    """Bar chart comparing final kNN top-1 accuracy across N algorithms.
    Reads knn_eval.csv (see maybe_run_knn_eval / `run --knn_eval`); if
    knn_eval.csv has more than one row (e.g. maybe_run_knn_eval called more
    than once for the same cell), uses the last (most recent) row.
    Algorithms with no knn_eval.csv (no .encoder_q, e.g. PRODEN, or
    --knn_eval not used) are skipped with a console note rather than
    erroring the whole plot.

    entries: list of (results_dir, algorithm, display_name) tuples, same
    convention as plot_heatmap_multi / plot_concentration_trend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels, accs = [], []
    for results_dir, alg, display in entries:
        path = os.path.join(results_dir, 'detail', alg, f'C{C}_k{k}', 'knn_eval.csv')
        if not os.path.isfile(path):
            print(f'  [plot_knn_eval_bar] skip {alg}: no knn_eval.csv '
                  f'(no .encoder_q, or --knn_eval not used)', flush=True)
            continue
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        labels.append(display)
        accs.append(float(rows[-1]['knn_top1_acc']))

    if not labels:
        algs = ', '.join(alg for _, alg, _ in entries)
        raise ValueError(f'No knn_eval logs found for C={C} k={k}, algorithm(s) {algs} '
                          f'(did you run with --knn_eval, and is the model a PiCO/ComCo-family '
                          f'dual-encoder model?)')

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(labels))]
    bars = ax.bar(labels, accs, color=colors)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1, f'{acc:.1f}%', ha='center', fontsize=9)

    ax.set_ylabel('kNN top-1 accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'kNN accuracy (train-set reference bank → test-set query)  —  C={C}  k={k}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path
