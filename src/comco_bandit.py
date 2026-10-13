"""Bandit-chosen hard alpha in {0, 1} for ComCo-Weighted (2026-10-13).

Two arms, picked fresh every epoch instead of a hand-set threshold:
    arm 0 = PartialLoss (alpha=0), arm 1 = ComCo-Fixed's own cls loss (alpha=1)

Reward is label-free: on a held-out split with candidate sets S_i drawn
uniformly (true label + |S_i|-1 random wrong labels), a wrong class lands in
S_i w.p. r_i = (|S_i|-1)/(C-1), so

    E[sum_{j in S_i} p_ij] = p_i,true + r_i (1 - p_i,true)
    =>  (sum_{j in S_i} p_ij - r_i) / (1 - r_i)   is unbiased for p_i,true

and the same correction applied to the hard hit indicator 1[argmax in S_i] is
unbiased for accuracy. holdout_scores() returns both (plus the true
held-out accuracy, for diagnostics only -- never used to decide anything).

Two bandits:
  * PairedTS (ComCo-Weighted-BanditPaired): every epoch, snapshot the full
    training state, run BOTH arms for the same L pre-drawn batches from that
    snapshot, score each on the held-out split, observe d = score1 - score0
    (paired -> held-out noise largely cancels), roll back, then Thompson-sample
    the sign of the discounted mean of d to pick the arm for the real epoch.
    Full-information: both arms are observed every round, exploration is
    paid with lookahead compute rather than with epochs on the worse arm.
  * TwoArmTS (ComCo-Weighted-BanditEpoch): classic partial-feedback bandit,
    reward = change in held-out score over the epoch the chosen arm was
    played. Control for the paired version.
"""

import copy
import math
import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def holdout_scores(model, holdout, C, device, chunk: int = 500) -> dict:
    was_training = model.training
    model.eval()
    soft, hard, correct = [], [], []
    n = holdout['x'].shape[0]
    for i in range(0, n, chunk):
        x = holdout['x'][i:i + chunk].to(device)
        partial = holdout['partial'][i:i + chunk].to(device)
        y = holdout['y'][i:i + chunk].to(device)
        p = F.softmax(model(x, eval_only=True), dim=1)
        r = (partial.sum(dim=1) - 1) / (C - 1)
        soft.append(((p * partial).sum(dim=1) - r) / (1 - r))
        pred = p.argmax(dim=1)
        hit = partial.gather(1, pred.view(-1, 1)).squeeze(1)
        hard.append((hit - r) / (1 - r))
        correct.append((pred == y).float())
    if was_training:
        model.train()
    return {'est_soft': torch.cat(soft).mean().item(),
            'est_acc': torch.cat(hard).mean().item(),
            'true_acc': torch.cat(correct).mean().item()}


class PairedTS:
    """Discounted Gaussian Thompson sampling on the paired advantage
    d = score(arm1) - score(arm0). Picks arm 1 iff a posterior sample of
    E[d] is > 0."""

    def __init__(self, gamma: float = 0.9, sigma_floor: float = 0.01):
        self.gamma, self.sigma_floor = gamma, sigma_floor
        self.w = self.s = self.ss = 0.0

    def update(self, d: float):
        self.w = self.gamma * self.w + 1.0
        self.s = self.gamma * self.s + d
        self.ss = self.gamma * self.ss + d * d

    def posterior(self):
        if self.w <= 0:
            return 0.0, float('inf')
        mu = self.s / self.w
        var = max(self.ss / self.w - mu * mu, self.sigma_floor ** 2)
        return mu, math.sqrt(var / self.w)

    def choose(self) -> int:
        mu, se = self.posterior()
        if math.isinf(se):
            return random.randint(0, 1)
        return int(random.gauss(mu, se) > 0)


class TwoArmTS:
    """Discounted Gaussian Thompson sampling, one posterior per arm; only the
    played arm's reward is observed. All arms' statistics decay every round
    (so a long-unplayed arm's posterior widens again)."""

    def __init__(self, gamma: float = 0.9, sigma_floor: float = 0.01):
        self.gamma, self.sigma_floor = gamma, sigma_floor
        self.w, self.s, self.ss = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

    def update(self, arm: int, reward: float):
        for a in (0, 1):
            self.w[a] *= self.gamma
            self.s[a] *= self.gamma
            self.ss[a] *= self.gamma
        self.w[arm] += 1.0
        self.s[arm] += reward
        self.ss[arm] += reward * reward

    def posterior(self, a):
        if self.w[a] < 1e-3:
            return 0.0, float('inf')
        mu = self.s[a] / self.w[a]
        var = max(self.ss[a] / self.w[a] - mu * mu, self.sigma_floor ** 2)
        return mu, math.sqrt(var / self.w[a])

    def choose(self) -> int:
        draws = []
        for a in (0, 1):
            mu, se = self.posterior(a)
            draws.append(random.random() * 1e9 if math.isinf(se) else random.gauss(mu, se))
        return int(draws[1] > draws[0])


def snapshot(model, optimizer, cls_loss):
    return (copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
            cls_loss.partial_loss.confidence.detach().clone())


def restore(snap, model, optimizer, cls_loss):
    model_sd, opt_sd, conf = snap
    model.load_state_dict(model_sd)
    optimizer.load_state_dict(copy.deepcopy(opt_sd))
    cls_loss.partial_loss.confidence.copy_(conf)


def draw_batches(loader, n: int) -> list:
    """First n batches of a fresh pass over `loader`, materialized so both
    lookahead arms train on the identical (augmented) images."""
    out = []
    for b in loader:
        out.append(b)
        if len(out) == n:
            break
    return out


class ContinuousRhoTS:
    """Continuous rho in [0, 1], paired-lookahead hill climbing.

    Each round: probe a = max(0, rho - delta) and b = min(1, rho + delta)
    from the same snapshot on the same batches, observe the paired slope
    (score_b - score_a) / (b - a), keep discounted Gaussian statistics of
    that slope, Thompson-sample it, and move rho one `step` in the sampled
    direction. Randomness of the sample is the exploration; a confidently
    positive/negative slope walks rho steadily toward 1/0."""

    def __init__(self, rho0: float = 0.5, delta: float = 0.15, step: float = 0.05,
                 gamma: float = 0.9, sigma_floor: float = 0.01):
        self.rho, self.delta, self.step = rho0, delta, step
        self.slope = PairedTS(gamma=gamma, sigma_floor=sigma_floor)

    def probe_pair(self):
        return max(0.0, self.rho - self.delta), min(1.0, self.rho + self.delta)

    def update(self, a: float, b: float, score_a: float, score_b: float):
        self.slope.update((score_b - score_a) / (b - a))

    def posterior(self):
        return self.slope.posterior()

    def choose(self) -> float:
        mu, se = self.slope.posterior()
        g = random.gauss(mu, se) if not math.isinf(se) else random.gauss(0.0, 1.0)
        self.rho = min(1.0, max(0.0, self.rho + (self.step if g > 0 else -self.step)))
        return self.rho
