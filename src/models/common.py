"""Building blocks shared by every dual-encoder model (PiCO, PiCO-Oracle, PCD diag models,
PiCO+, ComCo, ComCo-Weighted, Hybrid) -- 2026-11-22 refactor; previously copy-pasted 3-4x.

    init_key_encoder(enc_q, enc_k)          copy q -> k, freeze k
    momentum_update(enc_q, enc_k, m)        MoCo EMA of the key encoder
    ema_prototypes_loop(protos, f, y, m)    PiCO Eq. 7, sequential per sample (exact original loop)
    seq_ema_prototypes(protos, f, y, m)     vectorised equivalent (~1e-6 apart; PCD / PiCO+ only)
    predict_logits(model, x)                logits for ANY model (replaces isinstance whitelists)
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def init_key_encoder(enc_q, enc_k):
    for param_q, param_k in zip(enc_q.parameters(), enc_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False


@torch.no_grad()
def momentum_update(enc_q, enc_k, m):
    for param_q, param_k in zip(enc_q.parameters(), enc_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


def ema_prototypes_loop(prototypes, feats, labels, m):
    """proto[l] <- m * proto[l] + (1 - m) * f, one sample at a time, then L2-normalise.
    Updates `prototypes` in place and returns the normalised, detached result (assign it back)."""
    for feat, label in zip(feats, labels):
        prototypes[label] = prototypes[label] * m + (1 - m) * feat
    return F.normalize(prototypes, p=2, dim=1).detach()


def seq_ema_prototypes(protos, feats, labels, m):
    """Vectorised, exact equivalent of ema_prototypes_loop's sequential per-sample loop
    `proto[l] = m * proto[l] + (1 - m) * f` followed by one L2 normalisation."""
    C = protos.shape[0]
    n = labels.shape[0]
    order = torch.sort(labels, stable=True)[1]
    sl = labels[order]
    counts = torch.bincount(labels, minlength=C)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(n, device=labels.device) - starts[sl]
    w = (1 - m) * torch.pow(torch.tensor(m, device=labels.device, dtype=feats.dtype),
                            (counts[sl] - 1 - rank).to(feats.dtype))
    contrib = torch.zeros_like(protos).index_add_(0, sl, w[:, None] * feats[order])
    decay = torch.pow(torch.tensor(m, device=labels.device, dtype=feats.dtype), counts.to(feats.dtype))
    return F.normalize(protos * decay[:, None] + contrib, p=2, dim=1)


def predict_logits(model, x):
    """Classification logits of any model: dual-encoder models expose .predict(x)."""
    return model.predict(x) if hasattr(model, 'predict') else model(x)
