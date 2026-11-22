"""ONE PiCO-framework training epoch, parameterised (2026-11-22 refactor).

Replaces 15 near-copies: engine.train_pico_epoch / _moco_ / _mclloss_ / _sc_,
fixed_pico_engine.train_pico_epoch_fixed / _weighted_ / _mcl_epoch_fixed / _switch_,
oracle_pico_engine.train_pico_oracle_graded / _add_graded, clsloss_swap.train_pico_swap_epoch
and the four instrumented copies in pipeline/detail.py. Those names still exist as thin
wrappers (src/engine.py etc.) that call pico_epoch with the matching PiCOSpec.

Per batch, in this order (the order is part of the contract -- it fixes RNG consumption and
which confidence the loss sees):
    forward -> confidence update (gated by prot_start) -> alpha_fn(Y) -> L_cls
    -> SupCon mask -> mask_transform (oracle) -> hooks.on_mask -> L_cont -> backward/step
"""

import math
from dataclasses import dataclass, replace

import torch
from tqdm import tqdm


@dataclass(frozen=True)
class PiCOSpec:
    desc: str = 'PiCO'
    warmup_cont: str = 'omit'    # before prot_start: 'omit' L_cont (paper) | 'moco' (unsupervised MoCo L_cont)
    cont_mask: str = 'pseudo'    # 'pseudo' (same pseudo-label positives after prot_start) | 'none' (MoCo always)
    conf: str = 'prototype'      # 'prototype' | 'classifier' | 'self' (PiCOCLSLoss.update_confidence)
                                 # | 'swap' (loss.pre_update, gate inside the loss) | 'none'
    conf_hard: bool = None       # pass hard= to confidence_update (None: use the loss default)
    cls_call: str = 'index'      # 'index' L(out, idx) | 'Y' L(out, Y) | 'index_Y' L(out, idx, Y)
                                 # | 'index_Y_alpha' L(out, idx, Y, alpha=alpha_fn(Y))
    oracle: bool = False         # PiCOOracleModel: forward takes true labels, returns (true, pseudo) pools
    set_epoch: bool = False      # swap losses: loss.set_epoch(epoch, conf_ema_range, epochs) at epoch start

    def with_(self, **kw):
        return replace(self, **kw)


SPECS = {
    'PiCO': PiCOSpec('PiCO', warmup_cont='moco'),
    'PiCO-MOCO': PiCOSpec('PiCO-MOCO', warmup_cont='moco', cont_mask='none'),
    'PiCO-MCL': PiCOSpec('PiCO-MCL', warmup_cont='moco', conf='none', cls_call='Y'),
    'PiCO-SC': PiCOSpec('PiCO-SC', warmup_cont='moco', conf='self'),
    'PiCO-Fixed': PiCOSpec('PiCO-Fixed'),
    'PiCO-Weighted': PiCOSpec('PiCO-Weighted', cls_call='index_Y'),
    'PiCO-MCL-Fixed': PiCOSpec('PiCO-MCL-Fixed', conf='none', cls_call='Y'),
    'PiCO-Switch': PiCOSpec('PiCO-Switch', cls_call='index_Y_alpha'),
    'PiCO-Oracle': PiCOSpec('PiCO-Oracle', oracle=True),
    'PiCO-Oracle-Add': PiCOSpec('PiCO-Oracle-Add', oracle=True),
    'SwapPiCO': PiCOSpec('SwapPiCO', conf='swap', cls_call='index_Y', set_epoch=True),
}


# ─── oracle mask transforms (PiCO-Oracle / -Add): graded correction towards a precision ──────


class OracleRemoveFP:
    """Flip random false-positive pairs off until precision >= threshold (full pool mask, self and
    key pairs included). Exact original arithmetic incl. torch.randperm on the device."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, mask, same_true, device, stats):
        pos_total = int(mask.sum().item())
        stats.update(precision_before=float('nan'), precision_after=float('nan'), pos_total=pos_total,
                     n_flipped=0)
        if pos_total > 0:
            true_pos = int((mask * same_true).sum().item())
            stats['precision_before'] = precision = true_pos / pos_total
            if precision < self.threshold:
                false_pos = pos_total - true_pos
                n_flip = max(0, min(false_pos, math.ceil(pos_total - true_pos / self.threshold)))
                if n_flip > 0:
                    wrong_idx = ((mask == 1) & (same_true == 0)).nonzero(as_tuple=False)
                    perm = torch.randperm(wrong_idx.shape[0], device=device)[:n_flip]
                    sel = wrong_idx[perm]
                    mask[sel[:, 0], sel[:, 1]] = 0.0
                stats['n_flipped'] = n_flip
            new_total = pos_total - stats['n_flipped']
            stats['precision_after'] = true_pos / new_total if new_total > 0 else float('nan')
        return mask


class OracleAddTP:
    """Add random missed true-positive pairs until precision >= threshold, capped at
    ceil(max_add_ratio * pos_total)."""

    def __init__(self, threshold, max_add_ratio=1.0):
        self.threshold, self.max_add_ratio = threshold, max_add_ratio

    def __call__(self, mask, same_true, device, stats):
        pos_total = int(mask.sum().item())
        stats.update(precision_before=float('nan'), precision_after=float('nan'), pos_total=pos_total,
                     n_added=0, n_capped=False)
        if pos_total > 0:
            true_pos = int((mask * same_true).sum().item())
            stats['precision_before'] = precision = true_pos / pos_total
            thr = self.threshold
            if precision < thr and thr < 1.0:
                ideal_n_add = max(0, math.ceil((thr * pos_total - true_pos) / (1.0 - thr)))
                cap = max(1, math.ceil(self.max_add_ratio * pos_total))
                avail_idx = ((mask == 0) & (same_true == 1)).nonzero(as_tuple=False)
                n_add = min(ideal_n_add, cap, avail_idx.shape[0])
                stats['n_capped'] = n_add < ideal_n_add
                if n_add > 0:
                    perm = torch.randperm(avail_idx.shape[0], device=device)[:n_add]
                    sel = avail_idx[perm]
                    mask[sel[:, 0], sel[:, 1]] = 1.0
                stats['n_added'] = n_add
            new_total = pos_total + stats['n_added']
            stats['precision_after'] = (true_pos + stats['n_added']) / new_total if new_total > 0 else float('nan')
            stats['add_ratio'] = stats['n_added'] / pos_total
        else:
            stats['add_ratio'] = float('nan')
        return mask


# ─── the epoch ───────────────────────────────────────────────────────────────────────────────


def pico_epoch(args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, spec: PiCOSpec,
               alpha_fn=None, mask_transform=None, hooks=(), conf_source=None):
    """args: the pico_args dict (prot_start, loss_weight, epochs, conf_ema_range, ...).
    hooks: objects with optional on_mask(ctx) / end_epoch(epoch) (see src/train/hooks.py).
    conf_source overrides spec.conf for 'prototype' | 'classifier' (PiCO-Fixed A/B ablation)."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= args['prot_start']
    conf = conf_source or spec.conf
    if spec.set_epoch:
        loss_fn.set_epoch(epoch, args['conf_ema_range'], args['epochs'])
    bar = tqdm(loader, desc=f"{spec.desc} Epoch {epoch + 1}/{args['epochs']}" + (' [detail]' if hooks else ''))
    for batch_idx, (images_w, images_s, partial_Y, true_labels, index) in enumerate(bar):
        images_w, images_s, partial_Y, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device))
        if spec.oracle:
            true_labels = true_labels.to(device)
            cls_out, features, true_targets, pseudo_target_cont, score_prot = model(
                images_w, images_s, partial_Y, true_labels, args)
        else:
            cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, args)
            true_targets = None
        batch_size = cls_out.shape[0]

        # 1) confidence update -- before L_cls, so L_cls sees this batch's update
        if conf == 'swap':
            loss_fn.pre_update(score_prot, cls_out, index, partial_Y, start_upd_prot)
        elif start_upd_prot and conf in ('prototype', 'classifier'):
            temp_un_conf = (torch.softmax(cls_out, dim=1) if conf == 'classifier' else score_prot).detach()
            if spec.conf_hard is None:
                loss_fn.confidence_update(temp_un_conf=temp_un_conf, batch_index=index, batchY=partial_Y)
            else:
                loss_fn.confidence_update(temp_un_conf=temp_un_conf, batch_index=index, batchY=partial_Y,
                                          hard=spec.conf_hard)
        elif start_upd_prot and conf == 'self':
            loss_fn.update_confidence(cls_out.detach(), index)

        # 2) classification loss
        alpha = None
        if spec.cls_call == 'index':
            loss_cls = loss_fn(cls_out, index)
        elif spec.cls_call == 'Y':
            loss_cls = loss_fn(cls_out, partial_Y)
        elif spec.cls_call == 'index_Y':
            loss_cls = loss_fn(cls_out, index, partial_Y)
        else:
            alpha = alpha_fn(partial_Y)
            loss_cls = loss_fn(cls_out, index, partial_Y, alpha=alpha)

        # 3) contrastive mask (+ oracle correction, + stats hooks)
        mask = None
        if spec.cont_mask == 'pseudo' and start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
        ctx = {'batch_idx': batch_idx, 'mask_pre': None, 'stats': {}, 'true': true_labels,
               'batch_size': batch_size, 'device': device}
        if mask is not None and mask_transform is not None:
            same_true = torch.eq(true_targets[:batch_size].unsqueeze(1), true_targets.unsqueeze(0)).float()
            mask = mask_transform(mask, same_true, device, ctx['stats'])
            ctx['same_true_pool'] = same_true
        ctx['mask'] = mask
        for h in hooks:
            if hasattr(h, 'on_mask'):
                h.on_mask(ctx)

        # 4) total loss
        if mask is not None:
            loss = loss_cls + args['loss_weight'] * loss_cont_fn(features=features, mask=mask, batch_size=batch_size)
        elif spec.warmup_cont == 'moco' or spec.cont_mask == 'none':
            loss = loss_cls + args['loss_weight'] * loss_cont_fn(features=features, mask=None, batch_size=batch_size)
        else:
            loss = loss_cls  # paper-exact warm-up: L_cont omitted entirely

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        post = dict(loss=total_loss / (bar.n + 1))
        if alpha is not None:
            post['alpha'] = float(alpha.mean()) if torch.is_tensor(alpha) else float(alpha)
        bar.set_postfix(**post)
    for h in hooks:
        if hasattr(h, 'end_epoch'):
            h.end_epoch(epoch)
    return total_loss / len(loader)
