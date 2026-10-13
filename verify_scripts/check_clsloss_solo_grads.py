"""Numerical check of the closed-form logit gradients in
docs/clsloss_solo_gradients.md against autograd on the exact loss objects
the solo/swap experiments train with (src/clsloss_swap.py). Also checks the
two equivalences the doc relies on:
  * PRODEN with w = q (no lag)    == MCLLOG / ((C-1)/k)    (gradient)
  * MCLLOGFixed / MCLLOG           == constant 2k/(C-k)     (gradient, fixed k)
Random logits, C = 20, every k of the sweep, per-sample (B = 1 per row)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import clsloss_swap as S  # noqa: E402

torch.manual_seed(0)
C, B = 20, 32


def rand_Y(k):
    Y = torch.zeros(B, C)
    for i in range(B):
        Y[i, torch.randperm(C)[:k]] = 1.0
    return Y


def autograd(loss_fn, z, Y, index=None):
    leaf = z.clone().requires_grad_(True)
    loss_fn(leaf, index if index is not None else torch.arange(B), Y).backward()
    return leaf.grad * B  # undo the batch mean -> per-sample gradient


def closed(name, z, Y, w=None):
    p = F.softmax(z, dim=1)
    k = Y.sum(1, keepdim=True)
    m = C - k
    Yb = 1 - Y
    Ssum = (p * Y).sum(1, keepdim=True)
    q = p * Y / Ssum
    # the implementations take log(S + 1e-7); exact gradient keeps that eps
    p_eps, q_eps = p * Ssum / (Ssum + 1e-7), p * Y / (Ssum + 1e-7)
    if name == 'UCE':
        return p - Y / k
    if name == 'CLPL':
        zbar = (z * Y).sum(1, keepdim=True) / k
        return Y * (-(2 / k) * F.relu(1 - zbar)) + Yb * 2 * F.relu(1 + z)
    if name == 'Wu2022':
        return p - ((C - 1) * Y - (k - 1)) / (C - k)
    if name in ('PRODEN', 'PiCOCLS'):
        return p - w
    if name == 'MCLLOG':
        return (C - 1) / k * (p_eps - q_eps)
    if name == 'MCLLOGFixed':
        return 2 * (C - 1) / m * (p_eps - q_eps)
    if name == 'SCLNL':
        r = p / (1 - p)
        R = (Yb * r).sum(1, keepdim=True)
        return (Yb * r - p * R) / m
    if name == 'OP':
        return Yb / m - F.softmax(-z, dim=1)
    if name == 'OPW_detached':
        pt = F.softmax(-z, dim=1)
        u = 1.0 / pt.clamp(min=1e-7)
        a = F.softmax(u + 1.0, dim=1) * p + 1e-6
        return (Yb * a - pt * (Yb * a).sum(1, keepdim=True)) / m
    raise KeyError(name)


class OPWDetached(S.OPWLoss):
    def forward(self, logits, index, partial_Y):
        comp = 1.0 - partial_Y
        m = comp.sum(dim=1).clamp(min=1)
        base = -F.log_softmax(-logits, dim=1)
        with torch.no_grad():
            u = 1.0 / F.softmax(-logits, dim=1).clamp(min=1e-7)
            weight = F.softmax(u + 1.0, dim=1) * F.softmax(logits, dim=1) + self.eps
        return ((weight * base * comp).sum(dim=1) / m).mean()


worst = {}
for k in [1, 3, 5, 10, 15, 19]:
    Y = rand_Y(k)
    z = torch.randn(B, C) * 2
    w = torch.rand(B, C) * Y
    w = w / w.sum(1, keepdim=True)
    cases = {
        'UCE': (S.UCELoss(), None), 'CLPL': (S.CLPLLoss(), None), 'Wu2022': (S.Wu2022Loss(), None),
        'MCLLOG': (S.MCLLOGLoss(), None), 'MCLLOGFixed': (S.MCLLOGFixedLoss(), None),
        'SCLNL': (S.SCLNLLoss(), None), 'OP': (S.OPLoss(), None), 'OPW_detached': (OPWDetached(), None),
        'PRODEN': (S.PRODENLoss(w.clone()), w),
        'PiCOCLS': (S.PiCOEMALoss(w.clone(), 'classifier'), w),
    }
    for name, (fn, ww) in cases.items():
        err = (autograd(fn, z, Y) - closed(name, z, Y, ww)).abs().max().item()
        worst[name] = max(worst.get(name, 0.0), err)

    p = F.softmax(z, 1)
    Ss = (p * Y).sum(1, keepdim=True)
    g_proden_q = p * Ss / (Ss + 1e-7) - p * Y / (Ss + 1e-7)  # p - q, with MCLLOG's eps
    g_mcl = autograd(S.MCLLOGLoss(), z, Y)
    worst['PRODEN(w=q)==MCLLOG/((C-1)/k)'] = max(worst.get('PRODEN(w=q)==MCLLOG/((C-1)/k)', 0.0),
                                                  (g_proden_q - g_mcl * k / (C - 1)).abs().max().item())
    g_fix = autograd(S.MCLLOGFixedLoss(), z, Y)
    worst['MCLLOGFixed==MCLLOG*2k/(C-k)'] = max(worst.get('MCLLOGFixed==MCLLOG*2k/(C-k)', 0.0),
                                                 (g_fix - g_mcl * 2 * k / (C - k)).abs().max().item())

# OPW: how big is the d(weight)/dz term the closed form leaves out?
Y, z = rand_Y(5), torch.randn(B, C) * 2
full, det = autograd(S.OPWLoss(), z, Y), autograd(OPWDetached(), z, Y)
print(f"{'OPW weight-path share of grad norm (k=5)':45s} {((full - det).norm() / full.norm()).item():.3f}")

ok = True
for name, e in worst.items():
    flag = 'OK' if e < 1e-4 else 'MISMATCH'
    ok &= flag == 'OK'
    print(f'{name:45s} max|err| = {e:.2e}  {flag}')
sys.exit(0 if ok else 1)
