"""cls-loss swap experiment (2026-10-06): framework epoch loops. The losses moved to
src/losses/swap.py (2026-11-22) and are re-exported here for old imports."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.losses.swap import (CLS_LOSSES, FRAMEWORKS, CandidatePruneLoss, CLPLLoss, MCLLOGFixedLoss,  # noqa: F401
                             MCLLOGLoss, MCLNLLoss, OPLoss, OPWLoss, PiCOEMALoss, PRODENLoss, SCLNLLoss,
                             SoLarLoss, SwapMixClsLoss, UCELoss, Wu2022Loss, _conf_buffer, _Stateless,
                             build_cls_loss, swap_algorithm_name)


def train_pico_swap_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """train_pico_epoch_fixed with the cls loss swapped: L_cont omitted
    entirely during warm-up (epoch < prot_start), masked SupCon afterwards."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']
    loss_fn.set_epoch(epoch, pico_args['conf_ema_range'], pico_args['epochs'])

    bar = tqdm(loader, desc=f"SwapPiCO Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in bar:
        images_w, images_s, partial_Y, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device))
        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        loss_fn.pre_update(score_prot, cls_out, index, partial_Y, start_upd_prot)
        loss_cls = loss_fn(cls_out, index, partial_Y)
        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
            loss = loss_cls + pico_args['loss_weight'] * loss_cont_fn(features=features, mask=mask,
                                                                       batch_size=batch_size)
        else:
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=total_loss / (bar.n + 1))
    return total_loss / len(loader)


def train_comco_swap_epoch(comco_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """train_comco_weighted_epoch with the cls loss swapped (ComCoWeightedModel
    supplies score_prot for PiCOPartial; L_cont active from epoch 0)."""
    model.train()
    total_loss = 0
    warmup_pos = epoch >= comco_args['warmup_pos']
    warmup_neg = epoch >= comco_args['warmup_neg']
    start_upd_prot = epoch >= comco_args['prot_start']
    loss_fn.set_epoch(epoch, comco_args['conf_ema_range'], comco_args['epochs'])

    bar = tqdm(loader, desc=f"SwapComCo Epoch {epoch + 1}/{comco_args['epochs']}")
    for (images_w, images_s, comp_mask, true_labels, index) in bar:
        images_w, images_s, comp_mask, index = (
            images_w.to(device), images_s.to(device), comp_mask.to(device), index.to(device))
        partial_Y = 1.0 - comp_mask
        cls_out, q, all_feats, all_pseudo, all_comp, score_prot = model(
            images_w, images_s, comp_mask, partial_Y, comco_args)
        pseudo_q = cls_out.argmax(dim=1)

        loss_fn.pre_update(score_prot, cls_out, index, partial_Y, start_upd_prot)
        loss_cls = loss_fn(cls_out, index, partial_Y)
        loss_cont = loss_cont_fn(q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos, warmup_neg)
        loss = loss_cls + comco_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=total_loss / (bar.n + 1))
    return total_loss / len(loader)
