"""Label-side diagnostics: pseudo-label / disambiguation quality, confidence buffer,
prototypes, candidate-set mass/entropy, sample selection, candidate pruning.

All functions take tensors and return python floats (means over the given rows), so
they work per batch (feed into EpochAccumulator.mean_many) or over a full set.

Shapes: logits/probs [N, C]; Y = candidate mask [N, C] (1 = candidate);
comp = complementary mask = 1 - Y; y = true labels [N] (long).
"""

import torch
import torch.nn.functional as F


def _m(x):
    return x.float().mean().item()


def entropy(p, dim=1):
    """Shannon entropy (nats) with 0 log 0 = 0."""
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=dim)


def masked_argmax(scores, Y):
    """argmax restricted to the candidate set (argmax of scores * Y, i.e. PiCO's pseudo-label)."""
    return (scores * Y).argmax(dim=1)


# ─── pseudo-labels (pcdiag / picoplus) ─────────────────────────────────────


@torch.no_grad()
def pseudo_label_stats(yhat, y, Y=None, comp=None, prefix='pl'):
    """{prefix}_acc = P(yhat == y); {prefix}_in_comp = P(yhat outside the candidate set)
    (give Y or comp). PiCO: yhat = masked argmax (in_comp ~ 0 by construction);
    ComCo: yhat = unmasked argmax (in_comp > 0 means the PL lands in Ybar)."""
    out = {f'{prefix}_acc': _m(yhat == y)}
    if comp is None and Y is not None:
        comp = 1.0 - Y
    if comp is not None:
        out[f'{prefix}_in_comp'] = _m(comp.gather(1, yhat[:, None]) > 0)
    return out


@torch.no_grad()
def classifier_stats(logits, Y, y):
    """cls_acc_unmasked = argmax(p) == y; cls_masked_acc = argmax(p * Y) == y."""
    p = F.softmax(logits, dim=1)
    return {'cls_acc_unmasked': _m(p.argmax(1) == y), 'cls_masked_acc': _m(masked_argmax(p, Y) == y)}


@torch.no_grad()
def ema_pseudo_label_stats(ema_pl, raw_pl, y, comp):
    """PL-EMA variants: accuracy / in-comp rate of the EMA pseudo-label and its agreement
    with the raw per-batch argmax (ema_pl_acc, ema_pl_in_comp, ema_pl_eq_yhat)."""
    return {'ema_pl_acc': _m(ema_pl == y), 'ema_pl_in_comp': _m(comp.gather(1, ema_pl[:, None]) > 0),
            'ema_pl_eq_yhat': _m(ema_pl == raw_pl)}


@torch.no_grad()
def cstar_stats(cstar, y, Y):
    """ComCo negative class c*: cstar_hit = P(c* == y) (bad: pushes away own class),
    cstar_in_cand = P(c* in Y)."""
    return {'cstar_hit': _m(cstar == y), 'cstar_in_cand': _m(Y.gather(1, cstar[:, None]) > 0)}


# ─── confidence buffer (PartialLoss / PRODEN / PiCO+ / SoLar) ──────────────


@torch.no_grad()
def confidence_stats(conf, y):
    """conf [N, C] soft label buffer rows. conf_acc = argmax == y; conf_true_mass = conf[y];
    conf_sharp = max > 0.9; conf_ent (nats)."""
    return {'conf_acc': _m(conf.argmax(1) == y), 'conf_true_mass': conf.gather(1, y[:, None]).mean().item(),
            'conf_sharp': _m(conf.max(1).values > 0.9), 'conf_ent': entropy(conf).mean().item()}


@torch.no_grad()
def label_flip_rate(cur_conf_or_arg, prev_conf_or_arg):
    """s_flip: fraction of rows whose argmax changed since the previous call (SFlip mode).
    Accepts [N, C] buffers or [N] argmax vectors."""
    a = cur_conf_or_arg.argmax(1) if cur_conf_or_arg.dim() == 2 else cur_conf_or_arg
    b = prev_conf_or_arg.argmax(1) if prev_conf_or_arg.dim() == 2 else prev_conf_or_arg
    return _m(a != b)


# ─── prototypes (PiCO family) ──────────────────────────────────────────────


@torch.no_grad()
def prototype_stats(score_prot, Y, y, prefix='proto'):
    """score_prot = softmax(q @ prototypes.T) [N, C]. {prefix}_acc (unmasked) and
    {prefix}_masked_acc (restricted to candidates)."""
    return {f'{prefix}_acc': _m(score_prot.argmax(1) == y),
            f'{prefix}_masked_acc': _m(masked_argmax(score_prot, Y) == y)}


@torch.no_grad()
def prototype_scores(q, prototypes):
    return F.softmax(q @ prototypes.t(), dim=1)


@torch.no_grad()
def prototype_oracle_cosine(prototypes, oracle_prototypes):
    """Per-class cosine between learned and oracle (true-label EMA) prototypes ->
    proto_cos_oracle (mean) / proto_cos_oracle_min."""
    c = F.cosine_similarity(prototypes, oracle_prototypes, dim=1)
    return {'proto_cos_oracle': c.mean().item(), 'proto_cos_oracle_min': c.min().item()}


@torch.no_grad()
def oracle_prototypes_ema(protos, feats, labels, m):
    """Sequential per-sample EMA prototype update with given labels, vectorised
    (pcdiag.seq_ema_prototypes); use true labels to build 'oracle' prototypes."""
    C = protos.shape[0]
    order = torch.argsort(labels, stable=True)
    sl = labels[order]
    counts = torch.bincount(labels, minlength=C)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(sl), device=labels.device) - starts[sl]
    after = counts[sl] - 1 - rank
    mt = torch.tensor(m, device=labels.device, dtype=feats.dtype)
    w = (1 - mt) * torch.pow(mt, after.to(feats.dtype))
    contrib = torch.zeros_like(protos).index_add_(0, sl, w[:, None] * feats[order])
    decay = torch.pow(mt, counts.to(feats.dtype))
    return F.normalize(protos * decay[:, None] + contrib, p=2, dim=1)


# ─── candidate-set view of the classifier (switch_diag / clsloss_solo) ─────


@torch.no_grad()
def candidate_distribution_stats(probs, Y, y):
    """cand_mass = sum_{j in Y} p_j; disamb_acc = argmax_{Y} p == y; cand_entropy =
    H(p restricted to Y, renormalised) / log|Y| (0 when |Y| = 1); p_true = p_y;
    train_true_acc = argmax p == y. (clsloss_solo.dynamics_row; switch_diag uses
    log(max(|Y|,2)) and calls cand_entropy 'p_ent' -- same for |Y| >= 2.)"""
    S = (probs * Y).sum(1)
    q = probs * Y / S.clamp(min=1e-12)[:, None]
    k = Y.sum(1)
    H = entropy(q)
    Hn = torch.where(k > 1, H / torch.log(k.clamp(min=2)), torch.zeros_like(H))
    return {'cand_mass': S.mean().item(), 'disamb_acc': _m(q.argmax(1) == y),
            'cand_entropy': Hn.mean().item(), 'p_true': probs.gather(1, y[:, None]).mean().item(),
            'train_true_acc': _m(probs.argmax(1) == y)}


@torch.no_grad()
def switch_state_stats(conf, probs, Y, y, score_prot=None):
    """SwitchDiag.state: confidence buffer s vs classifier p on the same rows.
    s_acc, s_ent (= H(s)/log max(|Y|,2)), s_sharp (max s > .9), cand_mass, disamb_acc,
    p_ent (normalised entropy of p within Y), agree_ps (within-Y argmax p == argmax s),
    proto_acc (masked prototype argmax, if score_prot given)."""
    k = Y.sum(1)
    logk = torch.log(k.clamp(min=2.0))
    pS = probs * Y
    P = pS.sum(1)
    pn = pS / P.clamp_min(1e-12)[:, None]
    p_arg, s_arg = pS.argmax(1), conf.argmax(1)
    out = {'s_acc': _m(s_arg == y), 's_ent': (entropy(conf) / logk).mean().item(),
           's_sharp': _m(conf.max(1).values > 0.9), 'cand_mass': P.mean().item(),
           'disamb_acc': _m(p_arg == y), 'p_ent': (entropy(pn) / logk).mean().item(),
           'agree_ps': _m(p_arg == s_arg)}
    if score_prot is not None:
        out['proto_acc'] = _m(masked_argmax(score_prot, Y) == y)
    return out


# ─── noisy-PLL / sample selection (PiCO+) ──────────────────────────────────


@torch.no_grad()
def selection_quality(is_selected_clean, true_in_Y):
    """Clean-sample selection vs ground truth "y in Y".
    frac_true_clean = P(y in Y); sel_clean_prec = P(y in Y | selected clean);
    sel_noisy_recall = P(not selected | y not in Y)."""
    rel, t = is_selected_clean.bool(), true_in_Y.bool()
    return {'frac_true_clean': _m(t),
            'sel_clean_prec': _m(t[rel]) if rel.any() else float('nan'),
            'sel_noisy_recall': ((~rel) & (~t)).sum().item() / max(1, (~t).sum().item())}


def true_in_candidates(Y, y):
    return Y.gather(1, y[:, None]).squeeze(1) > 0


@torch.no_grad()
def reliable_split_sums(is_rel, within, allmax, y, Y=None, guess_arg=None):
    """PiCO+ per-batch sums for the reliable (rel) / unreliable (ur) split; turn into
    ratios with RELIABLE_SPLIT_METRICS (rel_frac uses rel_n / (rel_n + ur_n))."""
    w = is_rel.float()
    hit = (within == y).float()
    s = {'rel_n': w.sum(), 'rel_hit': (hit * w).sum(), 'ur_n': (1 - w).sum(),
         'ur_hit_within': (hit * (1 - w)).sum(), 'ur_hit_allmax': ((allmax == y).float() * (1 - w)).sum()}
    if guess_arg is not None:
        s['ur_guess_hit'] = ((guess_arg == y).float() * (1 - w)).sum()
        s['ur_guess_out'] = ((Y.gather(1, guess_arg[:, None])[:, 0] == 0).float() * (1 - w)).sum()
    return s


RELIABLE_SPLIT_METRICS = {
    'rel_pl_acc': ('rel_hit', 'rel_n'), 'ur_pl_acc_within': ('ur_hit_within', 'ur_n'),
    'ur_pl_acc_allmax': ('ur_hit_allmax', 'ur_n'), 'ur_guess_acc': ('ur_guess_hit', 'ur_n'),
    'ur_guess_out_of_cand': ('ur_guess_out', 'ur_n'),
}


def rel_frac(sums):
    r, u = float(sums.get('rel_n', 0.0)), float(sums.get('ur_n', 0.0))
    return r / max(r + u, 1.0)


# ─── candidate pruning (SwapComCo/SwapPiCO-Prune) ──────────────────────────


@torch.no_grad()
def prune_stats(S, q, y):
    """S = pruned candidate mask [N, C]; q = EMA candidate distribution. S_size = mean |S|;
    retention = P(y in S); q_top1 = P(argmax q == y)."""
    rows = torch.arange(S.shape[0], device=S.device)
    return {'S_size': S.sum(1).mean().item(), 'retention': S[rows, y].float().mean().item(),
            'q_top1': _m(q.argmax(1) == y)}
