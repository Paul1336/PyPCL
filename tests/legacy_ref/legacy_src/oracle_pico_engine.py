"""Training loop for PiCOOracleModel (src/pico/model.py) -- graduated
precision-controlled contrastive pair-selection oracle.

Aligned with PiCO-Fixed (src.fixed_pico_engine.train_pico_epoch_fixed), not
plain PiCO: warm-up (epoch < pico_args['prot_start']) omits L_cont entirely
(loss = loss_cls only), rather than falling back to an unsupervised MoCo
variant the way plain PiCO's warm-up does.

Post-warmup, the contrastive `mask` starts as the model's own natural
pseudo-label-driven mask (identical construction to PiCO-Fixed's). This
batch's selected-positive-pair precision against ground truth is then
measured: if it's already >= `precision_threshold`, the mask is used
unchanged. Otherwise, just enough of the WRONG selected-positive pairs
(mask==1 but not actually same true class) are picked at random and flipped
to negative to bring precision up to the threshold -- pairs the model
already got right (mask==1 and same true class) are never touched, and no
pair the model selected as negative is ever promoted to positive (so this
only ever removes precision-hurting false positives, it never recovers
recall). threshold=0 is a no-op (precision is always >= 0), reproducing
plain PiCO-Fixed exactly. threshold=1 corrects every false positive out of
the model's own selected set -- an upper bound on what precision-only
correction can achieve, but NOT identical to the old ground-truth-mask
PiCO-Oracle (mask = true-label equality), since a true-class pair the model
never selected as positive in the first place is still never added.
"""

import math

import torch
from tqdm import tqdm


def train_pico_oracle_graded_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                                    precision_threshold):
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    progress_bar = tqdm(loader, desc=f"PiCO-Oracle Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, true_labels, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device),
            true_labels.to(device), index.to(device))

        cls_out, features, true_targets, pseudo_targets, score_prot = model(
            images_w, images_s, partial_Y, true_labels, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        loss_cls = loss_fn(cls_out, index)

        if start_upd_prot:
            # Natural, uncorrected mask -- identical construction to
            # PiCO-Fixed's post-warmup mask (train_pico_epoch_fixed).
            mask = torch.eq(pseudo_targets[:batch_size].unsqueeze(1), pseudo_targets.unsqueeze(0)).float()

            same_true = torch.eq(true_targets[:batch_size].unsqueeze(1), true_targets.unsqueeze(0)).float()
            pos_total = int(mask.sum().item())
            if pos_total > 0:
                true_pos = int((mask * same_true).sum().item())
                precision = true_pos / pos_total
                if precision < precision_threshold:
                    false_pos = pos_total - true_pos
                    n_flip = max(0, min(false_pos, math.ceil(pos_total - true_pos / precision_threshold)))
                    if n_flip > 0:
                        wrong_idx = ((mask == 1) & (same_true == 0)).nonzero(as_tuple=False)
                        perm = torch.randperm(wrong_idx.shape[0], device=device)[:n_flip]
                        sel = wrong_idx[perm]
                        mask[sel[:, 0], sel[:, 1]] = 0.0

            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            # Warm-up: L_cont omitted entirely, matching PiCO-Fixed
            # (paper Appendix B.1), not plain PiCO's unsupervised fallback.
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)


def train_pico_oracle_add_graded_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                                        precision_threshold, max_add_ratio=1.0):
    """Additive counterpart to train_pico_oracle_graded_epoch above: instead
    of REMOVING false positives from the model's own natural mask to raise
    precision to `precision_threshold`, this ADDS randomly-chosen genuine
    TRUE positives (same_true==1 pairs the natural mask did NOT select)
    until precision reaches the threshold. The natural mask's existing
    selections -- including its false positives -- are never touched or
    removed, only grown; precision moves toward the target purely by
    diluting the fixed false-positive count with more genuine true
    positives, not by shrinking the denominator.

    `max_add_ratio` caps how many pairs can be added per batch, as a
    multiple of that batch's own natural pos_total (e.g. max_add_ratio=1.0
    means at most doubling the natural positive-set size). Without this cap,
    a high threshold (e.g. 0.9) against a naturally low-precision mask can
    demand adding an unboundedly large number of pairs; when the cap binds,
    the achieved precision honestly falls short of precision_threshold for
    that batch. Guards precision_threshold >= 1.0 by skipping correction
    entirely (exact 1.0 precision is unreachable by finite addition while
    any false positive remains, and the closed-form solution below divides
    by (1 - precision_threshold)) -- not exercised by any of this repo's
    threshold sweeps, which stay below 1.0.

    Same pos_total == 0 convention as train_pico_oracle_graded_epoch:
    nothing to correct (precision undefined) if the natural mask selected
    zero positives in the first place."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    progress_bar = tqdm(loader, desc=f"PiCO-Oracle-Add Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, true_labels, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device),
            true_labels.to(device), index.to(device))

        cls_out, features, true_targets, pseudo_targets, score_prot = model(
            images_w, images_s, partial_Y, true_labels, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        loss_cls = loss_fn(cls_out, index)

        if start_upd_prot:
            mask = torch.eq(pseudo_targets[:batch_size].unsqueeze(1), pseudo_targets.unsqueeze(0)).float()
            same_true = torch.eq(true_targets[:batch_size].unsqueeze(1), true_targets.unsqueeze(0)).float()

            pos_total = int(mask.sum().item())
            if pos_total > 0:
                true_pos = int((mask * same_true).sum().item())
                precision = true_pos / pos_total
                if precision < precision_threshold and precision_threshold < 1.0:
                    ideal_n_add = max(0, math.ceil(
                        (precision_threshold * pos_total - true_pos) / (1.0 - precision_threshold)))
                    cap = max(1, math.ceil(max_add_ratio * pos_total))
                    avail_idx = ((mask == 0) & (same_true == 1)).nonzero(as_tuple=False)
                    n_add = min(ideal_n_add, cap, avail_idx.shape[0])
                    if n_add > 0:
                        perm = torch.randperm(avail_idx.shape[0], device=device)[:n_add]
                        sel = avail_idx[perm]
                        mask[sel[:, 0], sel[:, 1]] = 1.0

            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)
