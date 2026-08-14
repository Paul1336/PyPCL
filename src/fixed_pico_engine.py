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


def train_pico_epoch_fixed(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """Runs a single training epoch for the PiCO model, paper-faithful warm-up."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    progress_bar = tqdm(loader, desc=f"PiCO-Fixed Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device))

        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

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
