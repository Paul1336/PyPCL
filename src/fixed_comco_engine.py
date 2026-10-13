"""ComCo-Weighted's training epoch -- the ComCo-side mirror of
src/fixed_pico_engine.py's train_pico_weighted_epoch.

ComCo-Fixed's own architecture/loop (src/engine.py::train_comco_epoch) needs
no warm-up: FixedComCoCLSLoss is stateless. ComCoWeightedClsLoss
(src/comco/weighted_cls_loss.py) mixes in PartialLoss, though, which is
driven by a prototype-similarity confidence signal (PiCO Eq. 6/7) that isn't
meaningful until the (newly-added, see ComCoWeightedModel) prototypes have
accumulated for a while -- so this loop gates PartialLoss's
confidence_update behind `comco_args['prot_start']`, same rationale as
PiCO-Fixed's warm-up, while leaving everything else (loss_weight, the
contrastive loss's own warmup_pos/warmup_neg schedule) identical to
train_comco_epoch, active from epoch 0 as in ComCo-Fixed. Unlike PiCO-Fixed,
L_cont is never omitted here -- that omission was specific to reproducing
the PiCO paper's stated warm-up behavior, and has no analogue in ComCo's own
design.
"""

import random

from tqdm import tqdm


def train_comco_weighted_epoch(comco_args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch, device):
    model.train()
    total_loss = 0
    warmup_pos = epoch >= comco_args['warmup_pos']
    warmup_neg = epoch >= comco_args['warmup_neg']
    start_upd_prot = epoch >= comco_args['prot_start']

    progress_bar = tqdm(loader, desc=f"ComCo-Weighted Epoch {epoch + 1}/{comco_args['epochs']}")
    for (images_w, images_s, comp_mask, true_labels, index) in progress_bar:
        images_w = images_w.to(device)
        images_s = images_s.to(device)
        comp_mask = comp_mask.to(device)
        index = index.to(device)
        partial_Y = 1.0 - comp_mask

        cls_out, q, all_feats, all_pseudo, all_comp, score_prot = model(
            images_w, images_s, comp_mask, partial_Y, comco_args)
        pseudo_q = cls_out.argmax(dim=1)

        if start_upd_prot:
            cls_loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        loss_cls = cls_loss_fn(cls_out, index, comp_mask)
        loss_cont = cont_loss_fn(q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos, warmup_neg)
        loss = loss_cls + comco_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)


# ─── per-batch alpha switching (variable-average-k experiment) ────────────
#
# Under variable-size candidate sets (--only_q, mean size != fixed k), a
# batch's OWN realized average candidate-set size fluctuates around the
# dataset's target mean. These two strategies pick a HARD alpha in {0.0,
# 1.0} (never a blend) fresh each batch instead of using one fixed alpha for
# the whole run -- see train_comco_weighted_switch_epoch below. Threshold is
# a fixed absolute size (10), not scaled to whatever mean the dataset was
# built with, matching the experiment's own design: at mean=10 this is a
# genuine ~50/50 adaptive switch, at mean=5/15 it degenerates toward
# almost-always-A000/almost-always-A100.

BATCH_COND_THRESHOLD = 10.0


def batch_cond_alpha(partial_Y, threshold: float = BATCH_COND_THRESHOLD) -> float:
    """alpha=1.0 (ComCo-Fixed's own loss) if this batch's average candidate-
    set size is ABOVE `threshold`, else 0.0 (PartialLoss)."""
    avg_size = partial_Y.sum(dim=1).mean().item()
    return 1.0 if avg_size > threshold else 0.0


def batch_random_alpha(partial_Y) -> float:
    """alpha in {0.0, 1.0}, independent 50/50 coin flip each batch --
    `partial_Y` is unused, kept only so this has the same signature as
    batch_cond_alpha (both are called as `alpha_fn(partial_Y)`)."""
    del partial_Y
    return 1.0 if random.random() < 0.5 else 0.0


def train_comco_weighted_switch_epoch(comco_args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch,
                                       device, alpha_fn):
    """Same as train_comco_weighted_epoch, except `cls_loss_fn`'s alpha is
    picked fresh each batch by `alpha_fn(partial_Y_batch) -> float in {0, 1}`
    (see batch_cond_alpha/batch_random_alpha above) instead of using the
    ComCoWeightedClsLoss instance's own fixed self.alpha. PartialLoss's
    confidence_update still runs every batch once past warm-up regardless of
    that batch's realized alpha -- same as train_comco_weighted_epoch -- so
    its confidence buffer stays live even on batches where this step's loss
    was driven entirely by the other term."""
    model.train()
    total_loss = 0
    warmup_pos = epoch >= comco_args['warmup_pos']
    warmup_neg = epoch >= comco_args['warmup_neg']
    start_upd_prot = epoch >= comco_args['prot_start']

    progress_bar = tqdm(loader, desc=f"ComCo-Weighted-Switch Epoch {epoch + 1}/{comco_args['epochs']}")
    for (images_w, images_s, comp_mask, true_labels, index) in progress_bar:
        images_w = images_w.to(device)
        images_s = images_s.to(device)
        comp_mask = comp_mask.to(device)
        index = index.to(device)
        partial_Y = 1.0 - comp_mask

        cls_out, q, all_feats, all_pseudo, all_comp, score_prot = model(
            images_w, images_s, comp_mask, partial_Y, comco_args)
        pseudo_q = cls_out.argmax(dim=1)

        if start_upd_prot:
            cls_loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        alpha = alpha_fn(partial_Y)
        loss_cls = cls_loss_fn(cls_out, index, comp_mask, alpha=alpha)
        loss_cont = cont_loss_fn(q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos, warmup_neg)
        loss = loss_cls + comco_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1), alpha=alpha)
    return total_loss / len(loader)
