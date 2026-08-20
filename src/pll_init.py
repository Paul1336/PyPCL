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
