"""Shared initial-confidence-construction strategies for partial-label
algorithms (PiCO family, PRODEN), factored out so both share one
implementation of "which wrong candidate gets the biased weight" rather than
each re-deriving it (see src/pipeline/algorithms/runners.py's
_candidate_masked_init_conf / _uniform_all_init_conf / _biased_oracle_init_conf
and src/proden_loss.py::ProdenLoss's init_mode)."""

import torch

# Independent of the pipeline's --seed (which drives class-selection/
# candidate-set generation) so that changing --seed for an unrelated reason
# doesn't silently reshuffle which wrong candidate gets the biased weight.
DEFAULT_BIASED_INIT_SEED = 20260820


def candidate_masked_init(partial_targets: list, num_classes: int) -> torch.Tensor:
    """s_j = 1/|Y| * I(j in Y) -- uniform WITHIN the candidate set only, zero
    outside it (PiCO Eq. 6 / PRODEN's existing default)."""
    N = len(partial_targets)
    conf = torch.zeros(N, num_classes)
    for i, cands in enumerate(partial_targets):
        k = max(len(cands), 1)
        for j in cands:
            conf[i, int(j)] = 1.0 / k
    return conf


def uniform_all_init(n: int, num_classes: int) -> torch.Tensor:
    """Uniform over ALL C classes regardless of candidate-set membership
    (plain PiCO's original, non-paper-faithful init)."""
    return torch.ones(n, num_classes) / num_classes


def biased_oracle_init(partial_targets: list, orig_targets, num_classes: int,
                        seed: int = DEFAULT_BIASED_INIT_SEED) -> torch.Tensor:
    """True class gets weight 0.2; ONE other candidate from that sample's OWN
    partial-label set, chosen uniformly at random with a fixed/reproducible
    seed, gets weight 0.8; everything else 0. The choice is made ONCE here
    (at construction time) and must never be resampled per epoch by the
    caller. Requires true labels -- oracle-style diagnostic use only, never
    fed back into training beyond this one-time initialization."""
    N = len(partial_targets)
    conf = torch.zeros(N, num_classes)
    g = torch.Generator().manual_seed(seed)
    for i, cands in enumerate(partial_targets):
        true_c = int(orig_targets[i])
        cand_list = [int(c) for c in cands]
        others = [c for c in cand_list if c != true_c]
        if not others:
            # Degenerate k=1 case: candidate set == {true_label}, nothing to
            # bias toward -- put all mass on the true class.
            conf[i, true_c] = 1.0
            continue
        pick = others[torch.randint(len(others), (1,), generator=g).item()]
        conf[i, true_c] = 0.2
        conf[i, pick] = 0.8
    return conf


def random_candidate_init(partial_targets: list, num_classes: int, epsilon: float = 1e-3,
                           seed: int = DEFAULT_BIASED_INIT_SEED) -> torch.Tensor:
    """Picks ONE candidate uniformly at random from each sample's own
    partial-label candidate set (fixed/reproducible seed, chosen once at
    construction time, never resampled per epoch) -- NOT necessarily the
    true class, picked independent of any ground truth -- and assigns it
    weight (1 - epsilon); the remaining candidates in that same set share
    epsilon uniformly.

    epsilon keeps every OTHER candidate's confidence entry nonzero (instead
    of exactly 0) specifically so PRODEN's mask = (conf > 0) still covers
    the FULL candidate set from the very first batch onward. An exactly
    one-hot init (epsilon=0) would make that mask permanently collapse to
    just the single random candidate forever: ProdenLoss.forward's update
    renormalizes candidate_mask * softmax(outputs), and a single-nonzero-
    entry mask always renormalizes back to that same single index (1.0
    there, 0 elsewhere) no matter what the model predicts -- the support can
    never grow again once it's down to one class. That degenerates PRODEN
    into plain supervised learning on a fixed, often-wrong label instead of
    a partial-label method that can still recover via its own predictions.

    Degenerate k=1 case (candidate set == {true_label}, so 'random pick' has
    only one option): all mass on that one class -- also correct with
    epsilon, since there's no OTHER candidate left for it to go to."""
    N = len(partial_targets)
    conf = torch.zeros(N, num_classes)
    g = torch.Generator().manual_seed(seed)
    for i, cands in enumerate(partial_targets):
        cand_list = [int(c) for c in cands]
        if len(cand_list) == 1:
            conf[i, cand_list[0]] = 1.0
            continue
        pick = cand_list[torch.randint(len(cand_list), (1,), generator=g).item()]
        others = [c for c in cand_list if c != pick]
        conf[i, pick] = 1.0 - epsilon
        share = epsilon / len(others)
        for c in others:
            conf[i, c] = share
    return conf


def biased_candidates_init(partial_targets: list, orig_targets, num_classes: int,
                            true_weight: float) -> torch.Tensor:
    """True class gets weight `true_weight`; the remaining (1 - true_weight)
    is spread UNIFORMLY across every OTHER candidate in that sample's own
    partial-label set (not just one -- see biased_oracle_init for the
    single-random-candidate variant). Deterministic given the candidate set,
    no random seed needed. Requires true labels (oracle-style diagnostic use
    only, never fed back into training beyond this one-time initialization).
    Degenerate k=1 case (candidate set == {true_label}): all mass on the
    true class, same convention as biased_oracle_init."""
    N = len(partial_targets)
    conf = torch.zeros(N, num_classes)
    for i, cands in enumerate(partial_targets):
        true_c = int(orig_targets[i])
        others = [int(c) for c in cands if int(c) != true_c]
        if not others:
            conf[i, true_c] = 1.0
            continue
        conf[i, true_c] = true_weight
        share = (1.0 - true_weight) / len(others)
        for c in others:
            conf[i, c] = share
    return conf


def biased_all_init(partial_targets: list, orig_targets, num_classes: int,
                     true_weight: float) -> torch.Tensor:
    """True class gets weight `true_weight`; the remaining (1 - true_weight)
    is spread UNIFORMLY across ALL OTHER (C - 1) classes, regardless of
    candidate-set membership (unlike biased_candidates_init, this puts
    nonzero weight on classes that aren't even in the sample's partial-label
    set). Deterministic, no random seed needed. Requires true labels."""
    N = len(partial_targets)
    conf = torch.zeros(N, num_classes)
    share = (1.0 - true_weight) / (num_classes - 1) if num_classes > 1 else 0.0
    for i in range(N):
        true_c = int(orig_targets[i])
        conf[i, :] = share
        conf[i, true_c] = true_weight
    return conf


def biased_partial_random_init(partial_targets: list, orig_targets, num_classes: int,
                                true_weight: float, wf: int,
                                seed: int = DEFAULT_BIASED_INIT_SEED) -> torch.Tensor:
    """True class gets weight `true_weight`; the remaining (1 - true_weight)
    is spread UNIFORMLY across `wf` OTHER candidates chosen uniformly at
    random (fixed/reproducible seed, chosen once at construction time, never
    resampled per epoch) from that sample's own partial-label set.

    Interpolates between biased_oracle_init (wf=1, fixed true_weight=0.2) and
    biased_candidates_init (wf = every other candidate) as a single
    continuously-tunable parameter: wf controls how many wrong candidates
    the "misplaced" confidence gets concentrated onto versus spread across.

    If a sample's candidate set has fewer than `wf` OTHER candidates (e.g.
    k <= wf), uses all of them instead -- same degenerate-case convention as
    biased_oracle_init/biased_candidates_init (all mass on the true class if
    there are no others at all)."""
    N = len(partial_targets)
    conf = torch.zeros(N, num_classes)
    g = torch.Generator().manual_seed(seed)
    for i, cands in enumerate(partial_targets):
        true_c = int(orig_targets[i])
        others = [int(c) for c in cands if int(c) != true_c]
        if not others:
            conf[i, true_c] = 1.0
            continue
        n_pick = min(wf, len(others))
        perm = torch.randperm(len(others), generator=g)[:n_pick]
        picks = [others[j] for j in perm.tolist()]
        conf[i, true_c] = true_weight
        share = (1.0 - true_weight) / n_pick
        for c in picks:
            conf[i, c] = share
    return conf


def biased_random_all_init(orig_targets, num_classes: int, true_weight: float, n: int,
                            seed: int = DEFAULT_BIASED_INIT_SEED) -> torch.Tensor:
    """True class gets weight `true_weight`; the remaining (1 - true_weight)
    is spread UNIFORMLY across `n` OTHER classes chosen uniformly at random
    from ALL (num_classes - 1) other classes -- fixed/reproducible seed,
    chosen once at construction time, never resampled per epoch.

    Unlike biased_partial_random_init, the random pool here is every class
    in [0, num_classes) except the true one, NOT restricted to the sample's
    own partial-label candidate set -- so picks can land outside the
    candidate set entirely. Needs no `partial_targets` for that reason (the
    pool doesn't depend on each sample's own candidate set). At n =
    num_classes - 1 this is equivalent to biased_all_init (every other class
    gets an equal share); smaller n interpolates toward concentrating the
    misplaced weight onto fewer, still-random classes.

    Degenerate case (num_classes == 1, no other classes to pick from): all
    mass on the true class, same convention as the other biased_*_init
    functions."""
    N = len(orig_targets)
    conf = torch.zeros(N, num_classes)
    g = torch.Generator().manual_seed(seed)
    for i in range(N):
        true_c = int(orig_targets[i])
        others = [c for c in range(num_classes) if c != true_c]
        if not others:
            conf[i, true_c] = 1.0
            continue
        n_pick = min(n, len(others))
        perm = torch.randperm(len(others), generator=g)[:n_pick]
        picks = [others[j] for j in perm.tolist()]
        conf[i, true_c] = true_weight
        share = (1.0 - true_weight) / n_pick
        for c in picks:
            conf[i, c] = share
    return conf


# The set of true-class weights swept by the biased_candidates_init /
# biased_all_init experiment family, and the naming convention for the
# resulting algorithm registry entries (see
# src/pipeline/algorithms/runners.py's BIASED_SWEEP_RUNNERS and
# src/pipeline/algorithms/hparams.py). 0.045/0.052/0.066/0.083 added
# 2026-08-29 for a finer-grained PiCO-Fixed-BiasedCand sweep.
BIAS_WEIGHTS = [0.20, 0.10, 0.08, 0.06, 0.05, 0.045, 0.052, 0.066, 0.083]


def weight_pct_str(true_weight: float) -> str:
    """The numeric part of weight_tag, without the leading 'W' -- factored
    out so callers that build their own 'W{tag}'-shaped template (e.g.
    src/pipeline/plotting.py's plot_accuracy_vs_weight, which formats
    'PiCO-Fixed-BiasedCand-W{w}'.format(w=...)) can reuse the EXACT same
    percentage-string logic as weight_tag/biased_variant_name instead of
    re-deriving their own zero-padding rule that can silently drift out of
    sync with it (as happened before 2026-08-29: plot_accuracy_vs_weight's
    own f'{w:02d}' couldn't represent the fractional weights added below).

    '{:02d}' for a whole-number percentage (unchanged from before, so
    existing W20/W10/W08/W06/W05 algorithm names/results/output paths stay
    identical), the minimal decimal representation otherwise (e.g. 0.045 ->
    '4.5') -- plain round-to-2-digits would collide (0.05 and 0.052 both
    round to '05')."""
    pct = round(true_weight * 100, 4)
    if pct == int(pct):
        return f'{int(pct):02d}'
    return f'{pct:.4f}'.rstrip('0').rstrip('.')


def weight_tag(true_weight: float) -> str:
    """Wxx for a whole-number percentage, W<pct> with the minimal decimal
    representation otherwise -- see weight_pct_str."""
    return f'W{weight_pct_str(true_weight)}'


def biased_variant_name(base: str, strategy: str, true_weight: float) -> str:
    """base: 'PiCO-Fixed' | 'PRODEN'. strategy: 'cand' (biased_candidates_init)
    | 'all' (biased_all_init)."""
    suffix = 'BiasedCand' if strategy == 'cand' else 'BiasedAll'
    return f'{base}-{suffix}-{weight_tag(true_weight)}'


# The (true_weight, wf) grid swept by the biased_partial_random_init
# experiment family -- see src/pipeline/algorithms/runners.py's
# BIASED_RAND_SWEEP_RUNNERS and src/pipeline/algorithms/hparams.py. Distinct
# from BIAS_WEIGHTS/biased_variant_name above (biased_candidates_init /
# biased_all_init have no `wf` parameter).
BIAS_RAND_WEIGHTS = [0.10, 0.08]
BIAS_RAND_WF_VALUES = [5, 8, 10, 12, 15]


def biased_rand_variant_name(base: str, true_weight: float, wf: int) -> str:
    """base: 'PiCO-Fixed' | 'PRODEN'."""
    return f'{base}-BiasedRand-{weight_tag(true_weight)}-Wf{wf}'


# The (true_weight, n) grid swept by the biased_random_all_init experiment
# family -- see src/pipeline/algorithms/runners.py's
# BIASED_RAND_ALL_SWEEP_RUNNERS and src/pipeline/algorithms/hparams.py.
# Distinct from BIAS_RAND_WEIGHTS/BIAS_RAND_WF_VALUES above: that family
# picks its `wf` random others from each sample's own candidate set (capped
# at k-1); this one picks its `n` random others from ALL num_classes-1 other
# classes, so n can exceed k-1 (added 2026-09-03 for a PRODEN true_weight=20%
# sweep over n=4/9/14/19 at C=20 k=5, where k-1=4 was too small a candidate
# pool for n>4).
BIAS_RAND_ALL_WEIGHTS = [0.20]
BIAS_RAND_ALL_N_VALUES = [4, 9, 14, 19]


def biased_rand_all_variant_name(base: str, true_weight: float, n: int) -> str:
    """base: 'PiCO-Fixed' | 'PRODEN'."""
    return f'{base}-BiasedRandAll-{weight_tag(true_weight)}-N{n}'


# ─── conf_ema_m sweep (2026-09-15) ─────────────────────────────────────────
# Five levels of the confidence-update EMA momentum, applied identically to
# PiCO-Fixed and PRODEN so their init-sensitivity profiles can be compared
# under the SAME update rule. Each level scales PiCO's whole per-epoch
# schedule config.yaml pico.conf_ema_range = [start, end] by a factor:
#   1.00 -> [0.95, 0.80]  original PiCO (PRODEN gets PiCO-like memory)
#   0.75 -> [0.71, 0.60]
#   0.50 -> [0.475, 0.40]
#   0.25 -> [0.24, 0.20]
#   0.00 -> [0.00, 0.00]  hard overwrite every update == original PRODEN
# (see src/proden_loss.py ProdenLoss.set_conf_ema_m / src/pico/utils_loss.py
# PartialLoss.set_conf_ema_m, and runners.CONF_EMA_SWEEP_RUNNERS).
CONF_EMA_SCALES = [1.0, 0.75, 0.5, 0.25, 0.0]


def ema_tag(scale: float) -> str:
    """'EMA100' / 'EMA075' / 'EMA050' / 'EMA025' / 'EMA000' -- the scale as a
    zero-padded percentage, so names sort in sweep order."""
    return f'EMA{round(scale * 100):03d}'


def ema_variant_name(base_name: str, scale: float) -> str:
    """Appends the conf_ema scale tag to any existing algorithm name, e.g.
    'PiCO-Fixed-BiasedCand-W20' -> 'PiCO-Fixed-BiasedCand-W20-EMA050'."""
    return f'{base_name}-{ema_tag(scale)}'


def scaled_conf_ema_range(base_range, scale: float) -> list:
    """[start, end] * scale, e.g. ([0.95, 0.8], 0.5) -> [0.475, 0.4]."""
    return [float(v) * scale for v in base_range]


CONF_EMA_SWEEP_BASES = ('PiCO-Fixed', 'PRODEN')


# ─── confidence-update-MECHANISM sweep (2026-09-16): Factor A x Factor B ──
#
# Follow-up to the conf_ema sweep above: matching the EMA *coefficient*
# between PiCO-Fixed and PRODEN (CONF_EMA_SWEEP_RUNNERS) did not make
# PiCO-Fixed's init-sensitivity (W/N sweep) curves converge toward PRODEN's,
# so this isolates two further, EMA-independent differences in *what* PiCO's
# confidence update actually does each step (see src/pico/utils_loss.py
# PartialLoss.confidence_update, src/fixed_pico_engine.py
# train_pico_epoch_fixed):
#
#   Factor A (conf_source): 'prototype' (native, paper Eq. 6) -- confidence
#       is driven by score_prot, i.e. embedding-vs-class-prototype cosine
#       similarity, one hop removed from the classifier and filtered
#       through the separately-EMA'd (proto_m=0.99, untouched here)
#       prototype memory -- vs 'classifier' (A') -- confidence instead
#       reuses softmax(cls_out) masked to candidates, the SAME kind of
#       signal PRODEN's own update uses. Note: A' does NOT disconnect the
#       contrastive/representation-learning branch -- L_cont still trains
#       the shared backbone and the SupCon positive-pair mask is unaffected
#       (it was already classifier-sourced, see pseudo_target_cont in
#       src/pico/model.py's forward) -- it only stops routing the
#       confidence buffer specifically through prototype similarity.
#   Factor B (conf_hard): True (native, paper Eq. 6) -- the update target is
#       one-hotted (argmax over the masked conf_source) before EMA-blending
#       into confidence -- vs False (B') -- the full masked/renormalized
#       distribution is blended in instead, structurally the same shape of
#       update ProdenLoss.forward does.
#
# Four combinations per (base biased-init variant, EMA scale): A+B (native
# PiCO-Fixed -- already fully covered by CONF_EMA_SWEEP_RUNNERS/the
# ema_sweep_0915 results, not re-trained here), A'+B, A+B', A'+B'.
AB_VARIANT_SOURCE_HARD = {
    'PiCO-Fixed':                       ('prototype', True),    # A + B  (native; reuse ema_sweep_0915)
    'PiCO-Fixed-SrcSoftmax':            ('classifier', True),   # A'+ B
    'PiCO-Fixed-SoftUpdate':            ('prototype', False),   # A + B'
    'PiCO-Fixed-SrcSoftmax-SoftUpdate': ('classifier', False),  # A'+ B'
}
# Bases this ablation actually trains new cells for (excludes 'PiCO-Fixed'
# itself -- see above). scripts/run_ab_sweep.py builds cells from this list.
AB_SWEEP_BASES = ('PiCO-Fixed-SrcSoftmax', 'PiCO-Fixed-SoftUpdate', 'PiCO-Fixed-SrcSoftmax-SoftUpdate')
# Per user request: only the highest/middle/lowest of CONF_EMA_SCALES (not
# all five) -- still lets ema_sweep_0915's own EMA100/EMA050/EMA000 rows for
# base 'PiCO-Fixed' be reused directly as the A+B reference column.
AB_SWEEP_SCALES = [1.0, 0.5, 0.0]


def conf_ema_sweep_base_names(base: str) -> list:
    """Every init variant the conf_ema sweep covers for one base algorithm,
    as the UN-suffixed algorithm names (ema_variant_name is applied on top):
    the unbiased baseline, every TC-PLS weight (BiasedCand-W*), and every
    TC-n-PLS-from-all-classes (BiasedRandAll-W*-N*) combination. Shared by
    runners.CONF_EMA_SWEEP_RUNNERS, hparams.py and scripts/run_ema_sweep.py
    so the three can never disagree on which names exist."""
    names = [base]
    names += [biased_variant_name(base, 'cand', w) for w in BIAS_WEIGHTS]
    names += [biased_rand_all_variant_name(base, w, n)
              for w in BIAS_RAND_ALL_WEIGHTS for n in BIAS_RAND_ALL_N_VALUES]
    return names


# The alpha values swept by the PiCO-weighted-cls-loss experiment family --
# see src/pipeline/algorithms/runners.py's PICO_WEIGHTED_SWEEP_RUNNERS and
# src/pipeline/algorithms/hparams.py. alpha weights PiCO-Fixed's PartialLoss
# term; (1 - alpha) weights PiCOMCLLoss's term.
ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def pico_weighted_variant_name(alpha: float) -> str:
    """e.g. 0.3 -> 'PiCO-Weighted-A030'."""
    return f'PiCO-Weighted-A{round(alpha * 100):03d}'
