"""Label-free accuracy estimates from candidate sets (bandit rewards) and the
data-level inclusion matrix Pi.

* candidate_hit_estimates -- comco_bandit.holdout_scores: chance-corrected candidate hit rate,
  unbiased for P(correct) under UNIFORM candidate generation.
* rho-corrected (bias-aware) versions and Pi estimation are re-exported from
  src/bias_aware_reward.py (already pure; not duplicated here).
* plateau_gain -- runners._run_generic_bandit 'plateau' mode.
"""

import torch

from src.bias_aware_reward import (MixtureAccuracy, RhoCorrected, estimate_inclusion,  # noqa: F401
                                   estimate_inclusion_shrunk, oracle_inclusion, uniform_inclusion)


@torch.no_grad()
def candidate_hit_estimates(probs, Y, C, y=None):
    """r_i = (|Y_i|-1)/(C-1) = chance a random wrong label is a candidate.
    est_soft = mean((sum_{Y} p - r)/(1 - r)); est_acc = mean((1[argmax p in Y] - r)/(1 - r)).
    With y: also true_acc and true_soft (= mean p_y; the TrueAcc oracle reward)."""
    r = (Y.sum(1) - 1) / (C - 1)
    hit = Y.gather(1, probs.argmax(1, keepdim=True)).squeeze(1)
    out = {'est_soft': (((probs * Y).sum(1) - r) / (1 - r)).mean().item(),
           'est_acc': ((hit - r) / (1 - r)).mean().item()}
    if y is not None:
        out['true_acc'] = (probs.argmax(1) == y).float().mean().item()
        out['true_soft'] = probs.gather(1, y.view(-1, 1)).mean().item()
    return out


@torch.no_grad()
def all_estimates(probs, Y, C, y=None, scorer: 'RhoCorrected' = None):
    """candidate_hit_estimates + rho_soft / rho_acc from a RhoCorrected scorer
    (built once from Pi: RhoCorrected(holdout_Y, Pi))."""
    out = candidate_hit_estimates(probs, Y, C, y)
    if scorer is not None:
        s = scorer.scores(probs)
        out['rho_soft'], out['rho_acc'] = s['est_soft'], s['est_acc']
    return out


def moving_average(xs, i, window=10):
    lo = max(0, i - window + 1)
    seg = xs[lo:i + 1]
    return sum(seg) / len(seg)


def plateau_gain(history, window=10, lag=20):
    """MA_window(history)[-1] - MA_window(history)[-1-lag]; NaN if too short."""
    i = len(history) - 1
    if i < lag:
        return float('nan')
    return moving_average(history, i, window) - moving_average(history, i - lag, window)
