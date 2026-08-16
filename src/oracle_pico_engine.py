"""Training loop for PiCOOracleModel (src/pico/model.py).

Identical to src/engine.py's train_pico_epoch except the contrastive `mask`
is built from ground-truth labels (true_target_cont, from the model's
queue_true buffer) instead of prototype-derived pseudo-labels. This isolates
one question: how much does PiCO's contrastive loss suffer from noisy
pseudo-label-based positive/negative pair selection, versus an oracle upper
bound? Everything else -- classification loss (PartialLoss), confidence
tracking, prototype EMA update, the prot_start warm-up gate on `mask` itself
-- is unchanged from plain PiCO, so this ablation isolates that one variable.

Not a real algorithm for actual use: true labels are not available in a
genuine partial-label setting.
"""

import torch
from tqdm import tqdm


def train_pico_oracle_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']

    progress_bar = tqdm(loader, desc=f"PiCO-Oracle Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in progress_bar:
        images_w, images_s, partial_Y, true_labels, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device),
            true_labels.to(device), index.to(device))

        cls_out, features, true_target_cont, score_prot = model(
            images_w, images_s, partial_Y, true_labels, pico_args)
        batch_size = cls_out.shape[0]

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        # Oracle mask: ground-truth label equality, not pseudo-label
        # equality -- the one thing this ablation changes vs. plain PiCO.
        mask = (torch.eq(true_target_cont[:batch_size].unsqueeze(1), true_target_cont.unsqueeze(0)).float()
                if start_upd_prot else None)

        loss_cls = loss_fn(cls_out, index)
        loss_cont = loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
        loss = loss_cls + pico_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    return total_loss / len(loader)
