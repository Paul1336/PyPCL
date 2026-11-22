"""Bias-aware label-free held-out accuracy estimate for the bandit (2026-10-20).

comco_bandit.holdout_scores corrects the candidate-set hit rate with
r = (|S|-1)/(C-1), i.e. it assumes every wrong label is equally likely to be a
false candidate. Under biased generation (--cl_bias) some wrong labels co-occur
with the true one almost always, so a model that predicts them still "hits"
the candidate set and the estimate drifts far above the true accuracy
(bias_c20_set1_1017, k=10: estimate 0.77 vs true 0.12).

Model. Pi[c, l] = P(l in S | y = c) (Pi[c, c] = 1) is a property of the data.
Treat each candidate set as product-Bernoulli given its class:
    log P(S | c) = sum_l s_l log Pi[c,l] + (1 - s_l) log(1 - Pi[c,l]).
1. estimate_inclusion: latent-class EM on the TRAINING candidate sets alone
   (no images, no labels, no model): q_i(c) ∝ prior_c P(S_i | c) over c in S_i,
   then Pi[c,l] = sum_i q_ic s_il / sum_i q_ic. Under uniform generation this
   recovers the uniform r; under bias it recovers the per-class structure.
2. MixtureAccuracy: for one classifier, group held-out samples by predicted
   class j; their candidate sets are a mixture over the true class c with
   KNOWN components P(S | c) and unknown weights W[j, c] = P(y = c | pred = j).
   Fit W by EM; then est_acc = mean_i P(y_i = pred_i | S_i, W) and
   est_soft = the same with the classifier's probabilities as soft predictions.
   With uniform Pi this reduces (in expectation) to the old correction.
Pi can also be the oracle (empirical with true labels) for an upper bound.
"""

import torch
import torch.nn.functional as F

_EPS = 1e-4


def _log_lik(partial, Pi):
    """partial [N, C] binary, Pi [C, C] -> log P(S_i | c) [N, C] (-inf where c not in S_i)."""
    P = Pi.clamp(_EPS, 1 - _EPS)
    ll = partial @ torch.log(P).T + (1 - partial) @ torch.log(1 - P).T
    return ll.masked_fill(partial == 0, float('-inf'))


@torch.no_grad()
def estimate_inclusion(partial, n_iter: int = 100, tol: float = 1e-6):
    """Label-free EM estimate of Pi from candidate masks [N, C] -> (Pi [C, C], prior [C])."""
    partial = partial.float()
    N, C = partial.shape
    q = partial / partial.sum(dim=1, keepdim=True)          # start: uniform over candidates
    prev = None
    for _ in range(n_iter):
        mass = q.sum(dim=0).clamp(min=1e-8)
        Pi = (q.T @ partial) / mass[:, None]
        Pi.fill_diagonal_(1.0)
        prior = mass / N
        logq = _log_lik(partial, Pi) + torch.log(prior.clamp(min=1e-8))[None, :]
        q = torch.softmax(logq, dim=1)
        if prev is not None and (Pi - prev).abs().max() < tol:
            break
        prev = Pi
    return Pi, prior


@torch.no_grad()
def oracle_inclusion(partial, y, C: int):
    """Empirical Pi with true labels (diagnostic upper bound)."""
    onehot = F.one_hot(y.long(), C).float()
    Pi = (onehot.T @ partial.float()) / onehot.sum(dim=0).clamp(min=1)[:, None]
    Pi.fill_diagonal_(1.0)
    return Pi


@torch.no_grad()
def uniform_inclusion(partial, C: int):
    """The old assumption: every wrong label is a false candidate w.p. (k-1)/(C-1)."""
    r = ((partial.sum(dim=1) - 1) / (C - 1)).mean().item()
    Pi = torch.full((C, C), r, device=partial.device)
    Pi.fill_diagonal_(1.0)
    return Pi


class MixtureAccuracy:
    """Label-free accuracy of a classifier on a fixed held-out split, given Pi."""

    def __init__(self, partial, Pi, n_iter: int = 50, alpha: float = 1.0):
        self.partial = partial.float()
        self.C = partial.shape[1]
        self.lik = torch.exp(_log_lik(self.partial, Pi) - _log_lik(self.partial, Pi).max(dim=1, keepdim=True).values)
        self.n_iter, self.alpha = n_iter, alpha

    @torch.no_grad()
    def _fit(self, assign):
        """assign [N, C] soft prediction weights -> posterior P(y = c | S_i, pred) per (i, j) folded into [N, C]
        of P(y_i = j) for the predicted j, plus W."""
        C = self.C
        W = torch.full((C, C), 1.0 / C, device=assign.device)
        for _ in range(self.n_iter):
            # resp[i, j, c] ∝ W[j, c] * lik[i, c]  for each predicted class j
            resp = W[None, :, :] * self.lik[:, None, :]
            resp = resp / resp.sum(dim=2, keepdim=True).clamp(min=1e-30)
            num = (assign[:, :, None] * resp).sum(dim=0) + self.alpha / C
            W = num / num.sum(dim=1, keepdim=True)
        diag = resp.diagonal(dim1=1, dim2=2)                   # [N, C]: P(y_i = j | S_i, pred = j)
        return diag, W

    @torch.no_grad()
    def scores(self, probs):
        hard = F.one_hot(probs.argmax(dim=1), self.C).float()
        d_hard, _ = self._fit(hard)
        d_soft, _ = self._fit(probs)
        return {'est_acc': (d_hard * hard).sum(dim=1).mean().item(),
                'est_soft': (d_soft * probs).sum(dim=1).mean().item()}


@torch.no_grad()
def _mixture_loglik(partial, Pi, prior):
    ll = _log_lik(partial, Pi) + torch.log(prior.clamp(min=1e-8))[None, :]
    return torch.logsumexp(ll, dim=1).mean().item()


@torch.no_grad()
def estimate_inclusion_shrunk(partial, lambdas=None, seed: int = 0):
    """Label-free Pi with a safety net: Pi = U + lam * (Pi_EM - U), U = uniform_inclusion.

    lam is picked by 2-fold cross-validated mixture log-likelihood of the
    candidate sets (fit EM on one half, score the other, both directions), so
    when no bias is detectable the EM structure is overfitting noise, lam -> 0,
    and the reward falls back to the old uniform correction exactly.
    Returns (Pi, lam, cv_curve)."""
    partial = partial.float()
    N, C = partial.shape
    lambdas = lambdas if lambdas is not None else [i / 10 for i in range(11)]
    U = uniform_inclusion(partial, C)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).to(partial.device)
    halves = (perm[: N // 2], perm[N // 2:])
    fits = [estimate_inclusion(partial[h]) for h in halves]
    curve = []
    for lam in lambdas:
        ll = 0.0
        for (Pi_a, prior_a), h_b in zip(fits, halves[::-1]):
            ll += _mixture_loglik(partial[h_b], U + lam * (Pi_a - U), prior_a)
        curve.append(ll / 2)
    lam = lambdas[max(range(len(lambdas)), key=lambda i: curve[i])]
    Pi_all, _ = estimate_inclusion(partial)
    return U + lam * (Pi_all - U), lam, curve


class RhoCorrected:
    """Bias-aware generalization of comco_bandit.holdout_scores' correction.

    Old: credit (1[j in S] - r) / (1 - r) with one scalar r = (|S|-1)/(C-1).
    New: r becomes rho[i, j] = E[Pi[c, j] | S_i, c != j] -- the chance that
    a WRONG prediction j still lands in S_i, averaged over the true classes
    c consistent with S_i (posterior over c from P(S_i | c), flat prior).
    Uniform Pi gives rho = r exactly, i.e. the old estimator. rho is capped
    (near-1 co-occurrence is not identifiable without labels anyway)."""

    def __init__(self, partial, Pi, cap: float = 0.95):
        partial = partial.float()
        post = torch.softmax(_log_lik(partial, Pi), dim=1)            # P(c | S_i), c in S_i
        num = post @ Pi - post * Pi.diagonal()[None, :]
        rho = (num / (1 - post).clamp(min=1e-6)).clamp(max=cap)
        self.credit = (partial - rho) / (1 - rho)                    # [N, C]: credit for predicting j

    def scores(self, probs):
        credit = self.credit.to(probs.device)
        hard = credit.gather(1, probs.argmax(dim=1, keepdim=True)).squeeze(1)
        return {'est_acc': hard.mean().item(), 'est_soft': (probs * credit).sum(dim=1).mean().item()}
