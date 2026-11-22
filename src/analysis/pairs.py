"""Contrastive pair-selection quality (positive precision / recall, negative purity, ...).

Conventions. Anchors are the B query views of the batch. The pool has A columns,
laid out as ``[q-view (B); k-view (B); queue (Q)]`` (A = 2B+Q), the MoCo layout used
by PiCO / ComCo / PiCO+. Masks are float 0/1 tensors of shape [B, A].

Three different pair sets appear in older code, so their numbers are NOT
comparable with one another. The function names below say which set each one uses:

==========================  ==================================================  ===========================
function                    pair set                                            origin
==========================  ==================================================  ===========================
pool_pair_sums/_metrics     full pool, self + unfilled queue slots excluded,    pcdiag.pair_stats/_row_common
                            own key kept (separate *nonkey* columns)
within_batch_confusion      q x q block only, self excluded                     detail.py (PiCO --detail)
oracle_remove/add_*         full pool incl. self + key                          detail.py PiCO-Oracle(-Add)
pos_precision_2b            q+k block (2B), key included, no queue              detail.py ComCo --detail
supcon_mask_precision       full pool, self excluded, key included             picoplus.py (pos_prec)
==========================  ==================================================  ===========================
"""

import math

import torch


# ─── helpers ────────────────────────────────────────────────────────────────


def self_key_masks(B, A, device=None):
    """eye[i, i] = 1 (anchor itself), key[i, B+i] = 1 (own momentum key view). [B, A] each."""
    eye = torch.zeros(B, A, device=device)
    eye[torch.arange(B), torch.arange(B)] = 1.0
    key = torch.zeros(B, A, device=device)
    key[torch.arange(B), torch.arange(B) + B] = 1.0
    return eye, key


def same_class_mask(true, pool_true):
    """[B, A] 1 where anchor and pool entry share the true label."""
    return (true[:, None] == pool_true[None, :]).float()


def valid_pool_mask(pool_true):
    """[1, A] 1 for filled pool slots (unfilled queue slots hold labels < 0)."""
    return (pool_true >= 0).float()[None, :]


def candidate_compat_pll(Y, pool_Y):
    """PLL: 1 where the anchor's and the pool entry's candidate sets intersect."""
    return ((Y @ pool_Y.t()) > 0).float()


def candidate_compat_cll(Y, pool_comp, valid):
    """CLL (ComCo): pool stores complementary sets; invalid slots count as compatible."""
    return ((Y @ (1.0 - pool_comp).t()) > 0).float() * valid + (1 - valid)


def same_label_positive_mask(anchor_lab, pool_lab, eye=None):
    """PiCO SupCon positives: pool entries whose pseudo-label equals the anchor's."""
    pos = (anchor_lab.float()[:, None] == pool_lab.float()[None, :]).float()
    return pos * (1 - eye) if eye is not None else pos


# ─── full-pool statistics (pcdiag) ─────────────────────────────────────────

POOL_PAIR_METRICS = {
    # column: (numerator, denominator)  -- both summed over the epoch
    'pos_per_anchor': ('pos_n', 'anchors'),
    'pos_prec': ('pos_tp', 'pos_n'),
    'pos_nonkey_per_anchor': ('pos_nk_n', 'anchors'),
    'pos_nonkey_prec': ('pos_nk_tp', 'pos_nk_n'),
    'pos_recall': ('pos_tp', 'same_all'),
    'pos_viol_rate': ('pos_viol', 'pos_n'),
    'pos_nonkey_viol_rate': ('pos_nk_viol', 'pos_nk_n'),
    'fp_viol_share': ('pos_fp_viol', 'pos_fp'),
    'pos_prec_if_filtered': ('pos_ok_tp', 'pos_ok_n'),
    'neg_per_anchor': ('neg_n', 'anchors'),
    'neg_fn_rate': ('neg_fn', 'neg_n'),
    'neg_certified_rate': ('neg_cert', 'neg_n'),
}
"""Meaning of each column:
pos_prec              selected positives that share the true class
pos_nonkey_prec       same, excluding the anchor's own augmented key (the "real" neighbour positives)
pos_recall            fraction of all same-class pool entries that got selected
pos_viol_rate         positives whose candidate sets are disjoint (cannot be same class)
fp_viol_share         share of false positives a shared-candidate filter would remove
pos_prec_if_filtered  precision if disjoint-candidate positives were dropped (CandFilter)
neg_fn_rate           negatives (denominator minus positives) that are actually same class
neg_certified_rate    negatives guaranteed different-class (disjoint candidate sets)
"""


@torch.no_grad()
def pool_pair_sums(pos, denom, same, compat, valid, eye, key):
    """Per-batch sums for POOL_PAIR_METRICS (feed into EpochAccumulator.add_many).

    pos/denom: masks the algorithm SELECTED ([B, A]); denom = contrastive denominator
    set (positives included). same/compat: [B, A]. valid: [1, A]. eye/key: self_key_masks.
    """
    B = pos.shape[0]
    v = valid * (1 - eye)
    pv = pos * v
    nonkey = pv * (1 - key)
    fp = pv * (1 - same)
    neg = denom * (1 - pos) * v
    return {
        'pos_n': pv.sum(), 'pos_tp': (pv * same).sum(),
        'pos_nk_n': nonkey.sum(), 'pos_nk_tp': (nonkey * same).sum(),
        'pos_viol': (pv * (1 - compat)).sum(), 'pos_nk_viol': (nonkey * (1 - compat)).sum(),
        'pos_fp': fp.sum(), 'pos_fp_viol': (fp * (1 - compat)).sum(),
        'pos_ok_n': (pv * compat).sum(), 'pos_ok_tp': (pv * compat * same).sum(),
        'same_all': (same * v).sum(),
        'neg_n': neg.sum(), 'neg_fn': (neg * same).sum(), 'neg_cert': (neg * (1 - compat)).sum(),
        'anchors': B,
    }


def pool_pair_metrics(sums: dict) -> dict:
    """Ratios from (epoch-summed) pool_pair_sums. Works on a single batch too."""
    def r(a, b):
        d = float(sums.get(b, 0.0))
        return float(sums.get(a, 0.0)) / d if d > 0 else float('nan')
    return {col: r(a, b) for col, (a, b) in POOL_PAIR_METRICS.items()}


# ─── within-batch confusion (detail.py, PiCO / PiCO-Fixed --detail) ────────


@torch.no_grad()
def within_batch_confusion(mask, true):
    """mask: SupCon mask [B, >=B] (pseudo-label equality); only the q x q block is scored,
    self-pairs excluded. Returns tp/fp/tn/fn + pos_precision/neg_precision (NaN if empty).
    Note: neg_precision here = 1 - neg_fn_rate of the pool version, on a different pair set."""
    B = true.shape[0]
    same = (true[None, :] == true[:, None]).float()
    eye = torch.eye(B, device=mask.device)
    wb = mask[:, :B]
    m_pos = wb * (1 - eye)
    m_neg = (1 - wb) * (1 - eye)
    tp, fp = (m_pos * same).sum().item(), (m_pos * (1 - same)).sum().item()
    tn, fn = (m_neg * (1 - same)).sum().item(), (m_neg * same).sum().item()
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, **confusion_ratios(tp, fp, tn, fn))


def confusion_ratios(tp, fp, tn, fn):
    """pos_precision, neg_precision, recall, and each cell's share of all pairs (in %)."""
    tot = tp + fp + tn + fn
    out = {
        'pos_precision': tp / (tp + fp) if (tp + fp) else float('nan'),
        'neg_precision': tn / (tn + fn) if (tn + fn) else float('nan'),
        'recall': tp / (tp + fn) if (tp + fn) else float('nan'),
    }
    for name, v in (('tp', tp), ('fp', fp), ('tn', tn), ('fn', fn)):
        out[f'{name}_pct'] = 100.0 * v / tot if tot else float('nan')
    return out


# ─── PiCO-Oracle graded correction (detail.py) ─────────────────────────────


@torch.no_grad()
def oracle_remove_correction(mask, same, threshold, generator=None):
    """Flip random false positives to 0 until precision >= threshold (PiCO-Oracle).
    mask/same: [B, A] (full pool, self + key included). Returns (new_mask, stats)."""
    mask = mask.clone()
    st = dict(precision_before=float('nan'), precision_after=float('nan'), neg_precision=float('nan'),
              pos_total=0, n_flipped=0)
    pos_total = int(mask.sum().item())
    st['pos_total'] = pos_total
    if pos_total > 0:
        tp = int((mask * same).sum().item())
        st['precision_before'] = tp / pos_total
        if st['precision_before'] < threshold:
            n = max(0, min(pos_total - tp, math.ceil(pos_total - tp / threshold)))
            if n > 0:
                wrong = ((mask == 1) & (same == 0)).nonzero(as_tuple=False)
                sel = wrong[torch.randperm(wrong.shape[0], generator=generator)[:n].to(wrong.device)]
                mask[sel[:, 0], sel[:, 1]] = 0.0
            st['n_flipped'] = n
        new_total = pos_total - st['n_flipped']
        st['precision_after'] = tp / new_total if new_total > 0 else float('nan')
    st['neg_precision'] = negative_precision(mask, same)
    return mask, st


@torch.no_grad()
def oracle_add_correction(mask, same, threshold, max_add_ratio, generator=None):
    """Add random missed true positives until precision >= threshold, capped at
    ceil(max_add_ratio * pos_total) (PiCO-Oracle-Add). Returns (new_mask, stats)."""
    mask = mask.clone()
    st = dict(precision_before=float('nan'), precision_after=float('nan'), neg_precision=float('nan'),
              pos_total=0, n_added=0, n_capped=False, add_ratio=float('nan'))
    pos_total = int(mask.sum().item())
    st['pos_total'] = pos_total
    if pos_total > 0:
        tp = int((mask * same).sum().item())
        st['precision_before'] = tp / pos_total
        if st['precision_before'] < threshold < 1.0:
            ideal = max(0, math.ceil((threshold * pos_total - tp) / (1.0 - threshold)))
            cap = max(1, math.ceil(max_add_ratio * pos_total))
            avail = ((mask == 0) & (same == 1)).nonzero(as_tuple=False)
            n = min(ideal, cap, avail.shape[0])
            st['n_capped'] = n < ideal
            if n > 0:
                sel = avail[torch.randperm(avail.shape[0], generator=generator)[:n].to(avail.device)]
                mask[sel[:, 0], sel[:, 1]] = 1.0
            st['n_added'] = n
        new_total = pos_total + st['n_added']
        st['precision_after'] = (tp + st['n_added']) / new_total if new_total > 0 else float('nan')
        st['add_ratio'] = st['n_added'] / pos_total
    st['neg_precision'] = negative_precision(mask, same)
    return mask, st


def negative_precision(mask, same):
    """Fraction of non-selected pairs (mask == 0) that are truly different-class."""
    neg = (mask == 0).float()
    n = neg.sum().item()
    return ((neg * (1 - same)).sum() / n).item() if n > 0 else float('nan')


# ─── simple precisions ─────────────────────────────────────────────────────


@torch.no_grad()
def pos_precision_2b(pos_mask, true):
    """ComCo --detail: precision of selected positives over the q+k block (first 2B pool
    columns; key positive included, queue ignored). Returns (precision, pos_total)."""
    B = true.shape[0]
    pm = pos_mask[:, :2 * B]
    same = (true[:, None] == torch.cat([true, true])[None, :]).float()
    n = pm.sum().item()
    return ((pm * same).sum().item() / n if n > 0 else float('nan')), int(n)


@torch.no_grad()
def supcon_mask_precision_sums(mask, true, pool_true):
    """PiCO+ pos_prec: final SupCon mask with self removed, valid pool entries only.
    Returns sums {'pos_n', 'pos_tp'} (ratio-of-sums over the epoch)."""
    B = true.shape[0]
    pv = mask.clone()
    pv[torch.arange(B), torch.arange(B)] = 0.0
    valid = valid_pool_mask(pool_true)
    same = same_class_mask(true, pool_true)
    return {'pos_n': (pv * valid).sum(), 'pos_tp': (pv * valid * same).sum()}
