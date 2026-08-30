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
