"""ONE ComCo-framework training epoch, parameterised (2026-11-22 refactor).

Replaces engine.train_comco_epoch, fixed_comco_engine.train_comco_weighted_epoch /
_weighted_switch_epoch, clsloss_swap.train_comco_swap_epoch and
pipeline/detail.train_comco_epoch_with_selection_stats (thin wrappers keep those names).

Per batch: forward -> pseudo_q = argmax -> confidence update (weighted/swap, gated by
prot_start) -> alpha_fn(Y) -> L_cls -> L_cont (ComCo selection) -> backward/step.
"""

from dataclasses import dataclass, replace

import torch
from tqdm import tqdm


@dataclass(frozen=True)
class ComCoSpec:
    desc: str = 'ComCo'
    weighted_model: bool = False   # ComCoWeightedModel: forward also takes Y, returns score_prot
    conf: str = 'none'             # 'none' | 'prototype' (PartialLoss inside the loss) | 'swap' (pre_update)
    cls_call: str = 'comp'         # 'comp' L(out, comp) | 'index_comp' L(out, idx, comp)
                                   # | 'index_comp_alpha' L(out, idx, comp, alpha=) | 'index_Y' L(out, idx, Y)
    set_epoch: bool = False

    def with_(self, **kw):
        return replace(self, **kw)


SPECS = {
    'ComCo': ComCoSpec('ComCo'),
    'ComCo-Weighted': ComCoSpec('ComCo-Weighted', weighted_model=True, conf='prototype', cls_call='index_comp'),
    'ComCo-Weighted-Switch': ComCoSpec('ComCo-Weighted-Switch', weighted_model=True, conf='prototype',
                                       cls_call='index_comp_alpha'),
    'SwapComCo': ComCoSpec('SwapComCo', weighted_model=True, conf='swap', cls_call='index_Y', set_epoch=True),
}


def comco_epoch(args, model, loader, cls_loss_fn, cont_loss_fn, optimizer, epoch, device, spec: ComCoSpec,
                alpha_fn=None, alpha_trace: list = None, hooks=()):
    model.train()
    total_loss = 0
    warmup_pos = epoch >= args['warmup_pos']
    warmup_neg = epoch >= args['warmup_neg']
    start_upd_prot = epoch >= args['prot_start'] if spec.conf != 'none' else False
    if spec.set_epoch:
        cls_loss_fn.set_epoch(epoch, args['conf_ema_range'], args['epochs'])
    want_masks = any(hasattr(h, 'on_masks') for h in hooks)
    bar = tqdm(loader, desc=f"{spec.desc} Epoch {epoch + 1}/{args['epochs']}" + (' [detail]' if hooks else ''))
    for batch_idx, (images_w, images_s, comp_mask, true_labels, index) in enumerate(bar):
        images_w, images_s, comp_mask = images_w.to(device), images_s.to(device), comp_mask.to(device)
        if spec.weighted_model:
            index = index.to(device)
            partial_Y = 1.0 - comp_mask
            cls_out, q, all_feats, all_pseudo, all_comp, score_prot = model(
                images_w, images_s, comp_mask, partial_Y, args)
        else:
            cls_out, q, all_feats, all_pseudo, all_comp = model(images_w, images_s, comp_mask, args)
        pseudo_q = cls_out.argmax(dim=1)

        if spec.conf == 'swap':
            cls_loss_fn.pre_update(score_prot, cls_out, index, partial_Y, start_upd_prot)
        elif spec.conf == 'prototype' and start_upd_prot:
            cls_loss_fn.confidence_update(temp_un_conf=score_prot.detach(), batch_index=index, batchY=partial_Y)

        alpha_mean = None
        if spec.cls_call == 'comp':
            loss_cls = cls_loss_fn(cls_out, comp_mask)
        elif spec.cls_call == 'index_comp':
            loss_cls = cls_loss_fn(cls_out, index, comp_mask)
        elif spec.cls_call == 'index_Y':
            loss_cls = cls_loss_fn(cls_out, index, partial_Y)
        else:
            alpha = alpha_fn(partial_Y)
            alpha_mean = float(alpha.mean()) if torch.is_tensor(alpha) else float(alpha)
            if alpha_trace is not None:
                alpha_trace.append(alpha_mean)
            loss_cls = cls_loss_fn(cls_out, index, comp_mask, alpha=alpha)

        if want_masks:
            loss_cont, pos_mask, denom_mask = cont_loss_fn(q, all_feats, all_pseudo, all_comp, pseudo_q,
                                                           warmup_pos, warmup_neg, return_masks=True)
            ctx = {'batch_idx': batch_idx, 'pos_mask': pos_mask, 'denom_mask': denom_mask,
                   'true': true_labels, 'batch_size': cls_out.shape[0], 'device': device}
            for h in hooks:
                if hasattr(h, 'on_masks'):
                    h.on_masks(ctx)
        else:
            loss_cont = cont_loss_fn(q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos, warmup_neg)
        loss = loss_cls + args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        post = dict(loss=total_loss / (bar.n + 1))
        if alpha_mean is not None:
            post['alpha'] = alpha_mean
        bar.set_postfix(**post)
    for h in hooks:
        if hasattr(h, 'end_epoch'):
            h.end_epoch(epoch)
    return total_loss / len(loader)
