"""Fixed version of src/engine.py's train_pico_epoch.

See docs/pico_explanation.md ("演算法保真度比對" -> 第 4 項 warm-up 機制) for the
full comparison. Wang et al., ICLR 2022 (PiCO) describes the warm-up period
(Appendix B.1: 1 epoch by default, 100 epochs for the CIFAR-100 q=0.1 setting)
as simply *omitting L_cont from the total loss* for the first N epochs --
Algorithm 1's pseudocode runs prototype updates, pseudo-target updates, and
L_cls unconditionally from epoch 0; only L_cont is described as disabled
during warm-up.

The original train_pico_epoch (src/engine.py) instead keeps L_cont active
throughout, but swaps SupConLoss from its masked (label-similarity) mode to
a plain unsupervised MoCo InfoNCE mode during warm-up, and additionally
gates the confidence/pseudo-target EMA update (Eq. 6) off until prot_start.
That is a reasonable engineering choice but not what the paper describes.

This module reproduces the paper's literal warm-up behavior: during
epoch < prot_start, L_cont is omitted entirely (loss = loss_cls only); the
confidence/pseudo-target update stays gated off during warm-up too, since
Eq. 6's prototype-similarity signal is not meaningful before the prototypes
have been updated for a while (consistent with the paper's own stated
intuition, even though Algorithm 1's box doesn't show an explicit gate for
it) -- this part of the original code is kept unchanged.
"""

import torch
from tqdm import tqdm


def train_pico_epoch_fixed(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device,
                            conf_source: str = 'prototype', conf_hard: bool = True):
    """Runs a single training epoch for the PiCO model, paper-faithful warm-up.

    conf_source ('prototype' | 'classifier'): which quantity feeds
    loss_fn.confidence_update as temp_un_conf. 'prototype' (default, paper
    Eq. 6) is score_prot -- similarity between the sample's contrastive
    embedding and the per-class prototype vectors, i.e. a signal one hop
    removed from the classifier, filtered through the (separately,
    proto_m-EMA'd) prototype memory. 'classifier' instead reuses
    softmax(cls_out) -- the SAME kind of candidate-masked classifier signal
    that already drives pseudo_target_cont / prototype assignment below
    (see predicted_scores/pseudo_labels_b in src/pico/model.py's forward),
    now also driving the confidence buffer. This is Factor A ('A'') of the
    2026-09-15 confidence-update-mechanism ablation -- see
    scripts/run_ab_sweep.py and docs discussion; it does NOT disconnect the
    contrastive/representation-learning branch (L_cont still trains the
    shared backbone and pseudo_target_cont still gates the SupCon mask
    below), it only stops routing the *confidence buffer* through the
    prototype-similarity readout.

    conf_hard: forwarded to PartialLoss.confidence_update -- Factor B."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']
    if conf_source not in ('prototype', 'classifier'):
        raise ValueError(f"conf_source must be 'prototype' or 'classifier', got {conf_source!r}")

    progress_bar = tqdm(loader, desc=f"PiCO-Fixed Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device))

        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            temp_un_conf = (torch.softmax(cls_out, dim=1) if conf_source == 'classifier' else score_prot).detach()
            loss_fn.confidence_update(temp_un_conf=temp_un_conf, batch_index=index, batchY=partial_Y, hard=conf_hard)

        loss_cls = loss_fn(cls_out, index)

        if start_upd_prot:
            # Post warm-up: identical to the original -- masked SupCon over the
            # candidate-restricted pseudo-label pool (paper Eq. 3/4), added to loss_cls.
            mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            # Warm-up: L_cont is omitted from the total loss entirely (paper
            # Appendix B.1), not merely switched to an unsupervised variant.
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)


def train_pico_weighted_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """Single training epoch for PiCO-weighted-cls-loss: identical to
    train_pico_epoch_fixed's paper-faithful warm-up (L_cont omitted entirely
    during warm-up), except loss_fn is a PiCOWeightedClsLoss
    (src/pico/weighted_cls_loss.py) -- alpha * PartialLoss + (1-alpha) *
    PiCOMCLLoss -- which needs both `index` (for PartialLoss's confidence
    buffer) and `partial_Y` (for PiCOMCLLoss) rather than just one or the
    other."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    progress_bar = tqdm(loader, desc=f"PiCO-Weighted Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device))

        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        loss_cls = loss_fn(cls_out, index, partial_Y)

        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
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


def train_pico_mcl_epoch_fixed(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """Single training epoch for PiCO-MCL-Fixed: same paper-faithful warm-up
    fix as train_pico_epoch_fixed (L_cont omitted entirely during warm-up,
    governed by prot_start_fixed), applied to PiCO-MCL's cls loss
    (PiCOMCLLoss, src/pico/mcl_cls_loss.py) instead of PartialLoss.

    PiCOMCLLoss is stateless -- forward(outputs, partial_Y), no confidence
    buffer -- so unlike train_pico_epoch_fixed there is no
    confidence_update call to gate."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    progress_bar = tqdm(loader, desc=f"PiCO-MCL-Fixed Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w = images_w.to(device)
        images_s = images_s.to(device)
        partial_Y = partial_Y.to(device)

        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        loss_cls = loss_fn(cls_out, partial_Y)

        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
            loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
            loss = loss_cls + pico_args['loss_weight'] * loss_cont
        else:
            # Warm-up: L_cont omitted entirely (paper Appendix B.1), same fix
            # as train_pico_epoch_fixed.
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)
