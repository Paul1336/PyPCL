"""Gradient diagnostics: alignment with the ground-truth gradient, and the
resampling bias / variance decomposition (SCL-NL / Chou et al. 2020 Sec. 4 style).

Origins
-------
* grad_alignment / cross_grad_stats  -- pcdiag.grad_stats (grad w.r.t. backbone feature h)
* flat_param_grad / param_grad_alignment -- switch_diag.SwitchDiag.grads (grad w.r.t. encoder params)
* bias_variance / bias_variance_normalised -- biasvariance._reduce, clsloss_solo._stats/biasvariance_row
* resample_candidate_sets / resample_complementary_labels -- biasvariance._resample_batch,
  biasvariance_cll.resample_single_cl_batch
* resampled_bias_variance -- generic driver of the above (the training code had one per loss type)
"""

import numpy as np
import torch
import torch.nn.functional as F


def grad_wrt(loss, x, retain_graph=True):
    return torch.autograd.grad(loss, x, retain_graph=retain_graph)[0]


@torch.no_grad()
def grad_alignment(g, g_ref, name=None):
    """Compare a gradient g with a reference g_ref (both [B, D], e.g. w.r.t. feature h).
    cos        = cosine of the flattened gradients (batch-level direction)
    cos_sample = per-sample cosine, averaged over the batch
    norm_ratio = ||g|| / ||g_ref||
    With name, keys are g_{name}_cos / _cos_sample / _norm_ratio (pcdiag columns)."""
    gf, rf = g.flatten(), g_ref.flatten()
    out = {'cos': F.cosine_similarity(gf, rf, dim=0, eps=1e-12).item(),
           'cos_sample': F.cosine_similarity(g, g_ref, dim=1, eps=1e-12).mean().item(),
           'norm_ratio': (gf.norm() / rf.norm().clamp(min=1e-12)).item()}
    return {f'g_{name}_{k}': v for k, v in out.items()} if name else out


@torch.no_grad()
def cross_grad_stats(g_cls, g_cont, lam):
    """Interaction of the classification and contrastive gradients:
    g_cls_vs_cont_cos and g_cont_over_cls_norm = ||lam g_cont|| / ||g_cls||."""
    return {'g_cls_vs_cont_cos': F.cosine_similarity(g_cls.flatten(), g_cont.flatten(), dim=0, eps=1e-12).item(),
            'g_cont_over_cls_norm': ((lam * g_cont).norm() / g_cls.norm().clamp(min=1e-12)).item()}


def flat_param_grad(params, loss):
    """Flattened gradient of loss w.r.t. a list of parameters (zeros for unused ones)."""
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])


@torch.no_grad()
def param_grad_alignment(grads: dict, ref_key='true'):
    """SwitchDiag-style: grads = {'0': g0, '1': g1, 'true': g_ce} flattened parameter grads.
    Returns cos{a} vs reference, cos{a}{b} pairwise, norm{a}."""
    out, keys = {}, [k for k in grads if k != ref_key]
    ref = grads[ref_key]
    for a in keys:
        out[f'cos{a}'] = F.cosine_similarity(grads[a], ref, dim=0).item()
        out[f'norm{a}'] = grads[a].norm().item()
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f'cos{a}{b}'] = F.cosine_similarity(grads[a], grads[b], dim=0).item()
    out[f'norm_{ref_key}'] = ref.norm().item()
    return out


# ─── bias / variance ───────────────────────────────────────────────────────


@torch.no_grad()
def bias_variance(grads, oracle):
    """grads: list or [M, ...] tensor of gradients from M resampled label sets.
    bias2 = mean((E g - g*)^2), variance = mean((g - E g)^2), mse = mean((g - g*)^2)
    (= bias2 + variance), cos_sim = cos(E g, g*)."""
    G = torch.stack(list(grads)) if not torch.is_tensor(grads) else grads
    mean = G.mean(0)
    return {'bias2': ((mean - oracle) ** 2).mean().item(), 'variance': ((G - mean) ** 2).mean().item(),
            'mse': ((G - oracle) ** 2).mean().item(),
            'cos_sim': F.cosine_similarity(mean.flatten(), oracle.flatten(), dim=0).item()}


@torch.no_grad()
def bias_variance_normalised(grads, oracle):
    """bias_variance plus scale-aligned versions (*_n): every resampled gradient is rescaled by
    ||g*|| / ||E g|| so losses differing only by a constant factor compare on direction/spread.
    Also grad_norm_ratio = ||E g|| / ||g*|| (clsloss_solo.biasvariance_row)."""
    G = torch.stack(list(grads)) if not torch.is_tensor(grads) else grads
    mean = G.mean(0)
    out = bias_variance(G, oracle)
    scale = (oracle.norm() / mean.norm().clamp(min=1e-12)).item()
    n = bias_variance(G * scale, oracle)
    out.update(bias2_n=n['bias2'], variance_n=n['variance'], mse_n=n['mse'],
               grad_norm_ratio=(mean.norm() / oracle.norm().clamp(min=1e-12)).item())
    return out


def oracle_ce_grad(logits, y):
    """d CE(logits, y) / d logits -- the paradigm-agnostic reference gradient."""
    leaf = logits.detach().clone().requires_grad_(True)
    F.cross_entropy(leaf, y).backward()
    return leaf.grad.detach()


def resample_candidate_sets(y, C, k, seed, device=None):
    """Fresh uniform fixed-k candidate sets (y + k-1 random others) for each row, from a local
    RandomState (global RNG untouched). Returns (Y [B, C], comp = 1 - Y). Same draws as
    biasvariance._resample_batch for the same seed."""
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    yl = y.detach().cpu().numpy()
    Y = np.zeros((len(yl), C), dtype=np.float32)
    for i, t in enumerate(yl):
        Y[i, t] = 1.0
        if k > 1:
            Y[i, rng.choice(np.delete(np.arange(C), t), size=k - 1, replace=False)] = 1.0
    Y = torch.from_numpy(Y).to(device if device is not None else y.device)
    return Y, 1.0 - Y


def resample_complementary_labels(y, C, seed, device=None):
    """One uniform complementary label per row (ordinary CLL), as [B, 1] long."""
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    yl = y.detach().cpu().numpy()
    cl = np.array([rng.choice(np.delete(np.arange(C), t)) for t in yl], dtype=np.int64)
    return torch.from_numpy(cl).view(-1, 1).to(device if device is not None else y.device)


def resampled_bias_variance(loss_on_logits, logits, y, C, k, M=50, seed=0, normalised=True):
    """Generic gradient bias/variance of a classification loss w.r.t. the logits.

    loss_on_logits(leaf_logits, Y, comp) -> scalar loss for one resampled candidate set.
    Stateful losses: wrap a deepcopy whose confidence rows are restricted to Y (see
    clsloss_solo._resampled_grad) inside loss_on_logits.
    Seeds are deterministic: seed * 1_000_003 + m (no Python hash())."""
    oracle = oracle_ce_grad(logits, y)
    grads = []
    for m in range(M):
        Y, comp = resample_candidate_sets(y, C, k, seed * 1_000_003 + m, logits.device)
        leaf = logits.detach().clone().requires_grad_(True)
        grads.append(torch.autograd.grad(loss_on_logits(leaf, Y, comp), leaf)[0].detach())
    return (bias_variance_normalised if normalised else bias_variance)(grads, oracle)


def two_term_conflict(term_a, term_b, leaf):
    """Norm ratio and cosine between the gradients of two loss summands
    (comp_norm_ratio / comp_cos: Wu2022 candidate vs all-class term, SoLar reliable vs rest)."""
    ga = torch.autograd.grad(term_a, leaf, retain_graph=True)[0]
    gb = torch.autograd.grad(term_b, leaf, retain_graph=True)[0]
    return {'comp_norm_ratio': (ga.norm() / gb.norm().clamp(min=1e-12)).item(),
            'comp_cos': F.cosine_similarity(ga.flatten(), gb.flatten(), dim=0).item()}
