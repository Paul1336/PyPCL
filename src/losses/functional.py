"""Canonical classification-loss formulas (one implementation each).

Every function takes logits [B, C] and the DENSE candidate mask Y [B, C] (1 = candidate);
the complementary set is Ybar = 1 - Y, k = |Y|, m = |Ybar| = C - k. Per-sample values are
averaged over the batch. Padded label lists and comp-mask callers convert with
src.losses.masks before calling.

Before 2026-11-22 these formulas existed 2-6 times each (mcl_losses, fixed_mcl_losses,
pico/mcl_cls_loss, comco/utils_loss, comco/fixed_utils_loss, scl_loss, op_loss, cpe_loss,
clpl_loss, cour_loss, wu_loss, clsloss_swap, pcdiag) with slightly different epsilon
handling; they now all route here. Differences to the old copies are <= 1e-6 per step
(tests/step_equiv.py).
"""

import torch
import torch.nn.functional as F

EPS = 1e-7


def _k(Y):
    return Y.sum(dim=1).clamp(min=1.0)


def _m(Y):
    return (Y.shape[1] - Y.sum(dim=1)).clamp(min=1.0)


# ─── PLL ─────────────────────────────────────────────────────────────────────


def uce(logits, Y):
    """-(1/k) sum_{j in Y} log p_j  (uniform candidate CE; PRODEN-naive)."""
    return (-(Y * F.log_softmax(logits, dim=1)).sum(dim=1) / _k(Y)).mean()


def clpl(logits, Y):
    """Cour et al. 2011 CLPL with squared hinge on raw scores:
    relu(1 - mean_{Y} g)^2 + sum_{j not in Y} relu(1 + g_j)^2."""
    pos = F.relu(1.0 - (logits * Y).sum(dim=1) / _k(Y)).pow(2)
    neg = ((1.0 - Y) * F.relu(1.0 + logits).pow(2)).sum(dim=1)
    return (pos + neg).mean()


def wu2022(logits, Y):
    """Wu et al. 2022 proper PLL estimator:
    (C-1)/(C-k) sum_{Y} -log p - (k-1)/(C-k) sum_{all} -log p   (C-k clamped >= 1)."""
    C = logits.shape[1]
    log_sm = F.log_softmax(logits, dim=1)
    k = _k(Y)
    d = (C - k).clamp(min=1.0)
    return ((C - 1.0) / d * -(log_sm * Y).sum(dim=1) - (k - 1.0) / d * -log_sm.sum(dim=1)).mean()


def weighted_ce(logits, conf):
    """-sum_j conf_j log p_j  (PRODEN / PiCO PartialLoss / PiCO-CLS / SoLar with a soft target)."""
    return -(conf * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# ─── CLL / MCLL ──────────────────────────────────────────────────────────────

MCL_SCALES = ('none', 'c1_over_k', 'feng_fixed')


def mcl(logits, Y, base='log', scale='none', eps='add'):
    """Feng et al. 2020 MCL with s = sum_{j in Y} p_j (probability mass outside Ybar):
        base  log: -log(s + 1e-7)   mae: 1 - s   exp: exp(-s)
        eps   'add'   -log(s + 1e-7)           (MCL-LOG / -Fixed / MCL-NL / pcdiag)
              'clamp' -log(max(s, 1e-8))       (PiCO-MCL, original ComCo) -- NOT interchangeable:
                      they differ a lot once s < 1e-7 (confidently wrong samples)
        scale none        MCL-NL (ComCo's cls loss under MCLL)
              c1_over_k   (C-1)/(C-m) = (C-1)/k   original MCL-LOG / PiCO-MCL / original ComCo
              feng_fixed  2(C-1)/m                MCL-LOG-Fixed (paper Eq. 12)"""
    C = logits.shape[1]
    s = (F.softmax(logits, dim=1) * Y).sum(dim=1)
    if base == 'log':
        per = -torch.log(s + EPS) if eps == 'add' else -torch.log(s.clamp(min=1e-8))
    elif base == 'mae':
        per = 1.0 - s
    elif base == 'exp':
        per = torch.exp(-s)
    else:
        raise ValueError(base)
    if scale == 'c1_over_k':
        per = per * ((C - 1.0) / _k(Y))
    elif scale == 'feng_fixed':
        per = per * (2.0 * (C - 1.0) / _m(Y))
    elif scale != 'none':
        raise ValueError(scale)
    return per.mean()


def scl_nl(logits, Y):
    """Chou et al. 2020 SCL-NL: (1/m) sum_{ybar} -log(1 - p_ybar)  (== FixedComCoCLSLoss)."""
    comp = 1.0 - Y
    per = -torch.log1p(-F.softmax(logits, dim=1).clamp(max=1.0 - EPS))
    return ((per * comp).sum(dim=1) / _m(Y)).mean()


def op(logits, Y):
    """Liu et al. 2023 OP: (1/m) sum_{ybar} -log softmax(-g)_ybar."""
    comp = 1.0 - Y
    return ((-F.log_softmax(-logits, dim=1) * comp).sum(dim=1) / _m(Y)).mean()


def opw(logits, Y, eps=1e-6):
    """Liu et al. 2023 OP-W: weight = softmax(1/softmax(-g) + 1) * softmax(g) + eps."""
    comp = 1.0 - Y
    base = -F.log_softmax(-logits, dim=1)
    u = 1.0 / F.softmax(-logits, dim=1).clamp(min=EPS)
    w = F.softmax(u + 1.0, dim=1) * F.softmax(logits, dim=1) + eps
    return ((w * base * comp).sum(dim=1) / _m(Y)).mean()


def cpe(logits, Y):
    """Complementary probability estimate (CPE-I): (1/m) sum_{ybar} -log p_ybar, predict argmin."""
    comp = 1.0 - Y
    return ((-F.log_softmax(logits, dim=1) * comp).sum(dim=1) / _m(Y)).mean()


# name -> fn(logits, Y): the stateless losses usable in any framework (swap / solo / PCD)
STATELESS = {
    'UCE': uce, 'CLPL': clpl, 'Wu2022': wu2022,
    'MCLLOG': lambda g, Y: mcl(g, Y, 'log', 'c1_over_k'),
    'MCLLOGFixed': lambda g, Y: mcl(g, Y, 'log', 'feng_fixed'),
    'MCLNL': lambda g, Y: mcl(g, Y, 'log', 'none'),
    'SCLNL': scl_nl, 'OP': op, 'OPW': opw, 'CPE': cpe,
}
