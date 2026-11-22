"""Contrastive losses: ONE masked InfoNCE core + mask builders (2026-11-22 refactor).

    masked_infonce(anchor, pool, pos, denom, tau)   -(1/|P_i|) sum_{p in P_i} log softmax_{D_i}(s_ip / tau)
    moco_infonce(features, B, tau)                 MoCo: key positive vs queue negatives (CE form, kept
                                                    exactly as SupConLoss's MoCo branch)
    pico_masks(pair_lab, pool_lab, B)              PiCO: positives = same pseudo-label, denominator = all but self
    comco_masks(sim, ...)                          ComCo positive/negative selection (top-K neighbour, c*)

SupConLoss (src/pico/utils_loss.py) and ComCoContrastiveLoss (src/comco/utils_loss.py) are thin
wrappers over these; pcdiag uses them directly. Pool layout: [q (B); k (B); queue (Q)].
"""

import torch
import torch.nn.functional as F


def masked_infonce(q, feats, pos, denom, tau):
    """-(1/|P_i|) sum_{p in P_i} log( exp(s_ip/tau) / sum_{a in D_i} exp(s_ia/tau) ), mean over i.
    Same arithmetic as SupConLoss (mask mode, D = all but self) and ComCoContrastiveLoss."""
    logits = torch.mm(q, feats.t()) / tau
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    exp_logits = torch.exp(logits) * denom
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    return -((pos * log_prob).sum(1) / (pos.sum(1) + 1e-12)).mean()


def moco_infonce(features, batch_size, temperature):
    """MoCo loss: positive = own key, negatives = queue (in-batch q/k excluded). CE form."""
    q = features[:batch_size]
    k = features[batch_size:batch_size * 2]
    queue = features[batch_size * 2:]
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,kc->nk', [q, queue])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    return F.cross_entropy(logits, labels)


def self_mask(B, A, device):
    """[B, A] ones with the anchor itself (column i) zeroed."""
    return torch.scatter(torch.ones(B, A, device=device), 1, torch.arange(B, device=device).view(-1, 1), 0)


@torch.no_grad()
def comco_masks(sim, lab_q, pool_lab, pool_comp, warm_pos, warm_neg, top_k, compat=None, force_cstar=None,
                pos_all=False, neg_all=False):
    """Selection masks of ComCoContrastiveLoss (same code path); compat (CandFilter) restricts the
    top-K neighbour search to pool entries sharing a candidate class; force_cstar fixes c*.
    pos_all / neg_all: PiCO+ AllClean rules (all same-label positives / all-but-self denominator)."""
    B, A = sim.shape
    C = pool_comp.shape[1]
    device = sim.device
    ar = torch.arange(B, device=device)
    key_idx = ar + B
    pos = torch.zeros(B, A, device=device)
    pos.scatter_(1, key_idx[:, None], 1.0)
    if neg_all:
        warm_neg = False  # -> denominator = ones minus self below
    if warm_pos and pos_all:
        same_lab = (lab_q[:, None] == pool_lab[None, :]).float()
        same_lab.scatter_(1, ar[:, None], 0.0)
        pos = ((pos + same_lab) > 0).float()
    elif warm_pos and top_k > 0:
        isim = (lab_q[:, None] == pool_lab[None, :]).float() * 0.5 * (1.0 + sim)
        isim.scatter_(1, ar[:, None], -1e9)
        isim.scatter_(1, key_idx[:, None], -1e9)
        if compat is None:
            idx = isim.topk(top_k, dim=1)[1]
            pos.scatter_(1, idx, 1.0)
        else:
            isim = isim.masked_fill(compat == 0, -1e9)
            vals, idx = isim.topk(top_k, dim=1)
            pos.scatter_add_(1, idx, (vals > -1e8).float())
            pos.clamp_(max=1.0)
    cstar = None
    if warm_neg:
        if force_cstar is None:
            mx = torch.full((B, C), -1e9, device=device)
            for c in range(C):
                mc = pool_comp[:, c].bool()
                if mc.any():
                    mx[:, c] = sim[:, mc].max(dim=1).values
            dist = 0.5 * (1.0 - mx)
            dist[mx <= -1e8] = -1e9
            cstar = dist.argmax(dim=1)
        else:
            cstar = force_cstar
        denom = (pos + pool_comp[:, cstar].t()).clamp(max=1.0)
    else:
        denom = torch.ones(B, A, device=device)
    denom.scatter_(1, ar[:, None], 0.0)
    return pos, denom, cstar

