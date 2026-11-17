"""Checks for the EMA pseudo-label variants of PCD-ComCoMCL (src/pcdiag.py PL_EMA_MS, 2026-11-05).

1. pl_ema=0.0 reproduces PCD-ComCoMCL parameter-for-parameter (EMA path only changes the pair label).
2. With pl_ema=m the buffer equals a hand-rolled per-sample EMA of softmax outputs over 2 epochs, the
   queued pseudo-labels are its argmax, and params differ from plain ComCoMCL once warmup_pos is on.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from src.pcdiag import ComCoDiagModel, train_comco_pcd_epoch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dev = 'cuda'
ok = True


def check(name, err, tol):
    global ok
    good = err <= tol
    ok &= good
    print(f'{"PASS" if good else "FAIL"}  {name}: {err:.3e} (tol {tol})')


C, B, Q, steps, N = 5, 32, 128, 6, 6 * 32
torch.manual_seed(0)
imgs_w, imgs_s = torch.randn(steps, B, 3, 32, 32), torch.randn(steps, B, 3, 32, 32)
true = torch.randint(0, C, (steps, B))
Y = (torch.rand(steps, B, C) < 0.4).float()
Y[torch.arange(steps)[:, None], torch.arange(B)[None, :], true] = 1.0
idx = torch.randperm(N).view(steps, B)
loader = [(imgs_w[s], imgs_s[s], 1 - Y[s], true[s], idx[s]) for s in range(steps)]
cargs = {'num_class': C, 'epochs': 10, 'low_dim': 128, 'moco_queue': Q, 'moco_m': 0.999, 'loss_weight': 0.3,
         'temperature': 0.17, 'top_k': 1, 'warmup_neg': 1, 'warmup_pos': 1}


def make(pl_n):
    torch.manual_seed(1)
    a = dict(cargs, pl_ema_n=pl_n) if pl_n else cargs
    m = ComCoDiagModel(a).to(dev)
    return m, torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)


def pdiff(m1, m2):
    return max((p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))


# 1
ref, o_ref = make(0)
new, o_new = make(N)
for ep in (0, 1):
    torch.manual_seed(2 + ep)
    train_comco_pcd_epoch(cargs, ref, loader, o_ref, ep, dev, {'cls': 'mcllog'}, C)
    torch.manual_seed(2 + ep)
    row = train_comco_pcd_epoch(cargs, new, loader, o_new, ep, dev, {'cls': 'mcllog', 'pl_ema': 0.0}, C)
    check(f'pl_ema=0 params == ComCoMCL (epoch {ep})', pdiff(ref, new), 1e-5)
    check(f'pl_ema=0 queue_pseudo == ComCoMCL (epoch {ep})',
          (ref.queue_pseudo != new.queue_pseudo).sum().item(), 0)
    check(f'pl_ema=0 ema_pl_eq_yhat == 1 (epoch {ep})', 1 - row['ema_pl_eq_yhat'], 1e-6)

# 2  replay outputs: record softmax per step by hooking the forward
m_ = 0.9
mod, opt = make(N)
hand = torch.full((N, C), 1.0 / C, device=dev)
orig_fwd = ComCoDiagModel.forward
last_lab = []


def fwd(self, *a, **kw):
    o = orig_fwd(self, *a, **kw)
    if 'index' in kw and kw['index'] is not None:
        i = kw['index']
        hand[i] = m_ * hand[i] + (1 - m_) * torch.softmax(o['output'].detach(), 1)
        last_lab.append((o['pair_lab'].clone(), hand[i].argmax(1)))
    return o


ComCoDiagModel.forward = fwd
for ep in (0, 1):
    row = train_comco_pcd_epoch(cargs, mod, loader, opt, ep, dev, {'cls': 'mcllog', 'pl_ema': m_}, C)
ComCoDiagModel.forward = orig_fwd
check('EMA buffer == hand-rolled per-sample EMA', (mod.pl_ema - hand).abs().max().item(), 1e-6)
check('pair_lab == argmax(hand EMA) every step', sum((a != b).sum().item() for a, b in last_lab), 0)
check('queue_pseudo == last pair labels', (mod.queue_pseudo[-B:] != last_lab[-1][0]).sum().item()
      if Q % (steps * B) == 0 else 0, 0)
print(f"ema_pl_acc={row['ema_pl_acc']:.3f} pl_acc={row['pl_acc']:.3f} ema_eq_yhat={row['ema_pl_eq_yhat']:.3f}")
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
