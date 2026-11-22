"""src/analysis must reproduce the original in-training implementations exactly.

Run from the repo root:  python verify_scripts/check_analysis.py
Every check compares the new pure function with the original code path on random inputs.
"""

import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import pcdiag  # noqa: E402
from src import comco_bandit  # noqa: E402
from src.pipeline import biasvariance as bv  # noqa: E402
from src import clsloss_solo  # noqa: E402
from src.analysis import accum, estimates, gradients, labels, pairs, prediction, traces  # noqa: E402

torch.manual_seed(0)
FAILS = []


def close(name, a, b, tol=1e-6):
    a, b = float(a), float(b)
    ok = (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1.0, abs(b))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {a:.8g} vs {b:.8g}")
    if not ok:
        FAILS.append(name)


def rand_setting(B=16, Q=32, C=6, k=3):
    A = 2 * B + Q
    true = torch.randint(0, C, (B,))
    pool_true = torch.cat([true, true, torch.randint(0, C, (Q,))])
    pool_true[-5:] = torch.arange(-5, 0)  # unfilled queue slots
    Y = torch.zeros(B, C)
    for i in range(B):
        Y[i, true[i]] = 1
        Y[i, torch.randperm(C)[:k - 1]] = 1
    pool_Y = (torch.rand(A, C) < 0.4).float()
    pool_Y[:B] = Y
    pool_Y[B:2 * B] = Y
    lab = torch.randint(0, C, (B,))
    pool_lab = torch.cat([lab, lab, torch.randint(0, C, (Q,))])
    return B, A, C, true, pool_true, Y, pool_Y, lab, pool_lab


# 1. pool pair stats == pcdiag.pair_stats + _row_common, accumulated over 3 batches
acc_old, acc_new = pcdiag._Acc(), accum.EpochAccumulator()
for _ in range(3):
    B, A, C, true, pool_true, Y, pool_Y, lab, pool_lab = rand_setting()
    eye, key = pcdiag._self_key(B, A, None)
    e2, k2 = pairs.self_key_masks(B, A)
    assert torch.equal(eye, e2) and torch.equal(key, k2)
    valid = pairs.valid_pool_mask(pool_true)
    same = pairs.same_class_mask(true, pool_true)
    compat = pairs.candidate_compat_pll(Y, pool_Y)
    pos = pairs.same_label_positive_mask(lab, pool_lab, eye)
    denom = 1 - eye
    pcdiag.pair_stats(acc_old, pos, denom, same, compat, valid, eye, key, B)
    acc_new.add_many(pairs.pool_pair_sums(pos, denom, same, compat, valid, eye, key))
old_row = pcdiag._row_common(acc_old, 0)
new_row = acc_new.ratios(pairs.POOL_PAIR_METRICS)
for c in pairs.POOL_PAIR_METRICS:
    close(f'pool_pair:{c}', new_row[c], old_row[c])

# 2. grad alignment == pcdiag.grad_stats
g, gr = torch.randn(16, 32), torch.randn(16, 32)
a_old = pcdiag._Acc()
pcdiag.grad_stats(a_old, 'x', g, gr)
new = gradients.grad_alignment(g, gr, 'x')
for kk in new:
    close(f'grad:{kk}', new[kk], a_old.avg(kk))

# 3. within-batch confusion == detail.py inline code
B, C = 12, 5
true = torch.randint(0, C, (B,))
ptc = torch.cat([torch.randint(0, C, (B,)), torch.randint(0, C, (20,))])
mask = (ptc[:B, None] == ptc[None, :]).float()
same_t = (true[None, :] == true[:, None]).float()
eye = torch.eye(B)
wb = mask[:, :B]
mp, mn = wb * (1 - eye), (1 - wb) * (1 - eye)
tp, fp = (mp * same_t).sum().item(), (mp * (1 - same_t)).sum().item()
tn, fn = (mn * (1 - same_t)).sum().item(), (mn * same_t).sum().item()
r = pairs.within_batch_confusion(mask, true)
close('wb:pos_precision', r['pos_precision'], tp / mp.sum().item())
close('wb:neg_precision', r['neg_precision'], tn / mn.sum().item())
close('wb:fn', r['fn'], fn)

# 4. oracle remove / add: deterministic parts (counts, precisions) vs the detail.py formulas
pt = torch.randint(0, 3, (2 * B + 20,))
tt = torch.randint(0, 3, (2 * B + 20,))
m = (pt[:B, None] == pt[None, :]).float()
s = (tt[:B, None] == tt[None, :]).float()
thr = 0.9
_, st = pairs.oracle_remove_correction(m, s, thr)
tot, tpos = int(m.sum()), int((m * s).sum())
nf = max(0, min(tot - tpos, math.ceil(tot - tpos / thr)))
close('oracle_rm:n_flipped', st['n_flipped'], nf)
close('oracle_rm:prec_after', st['precision_after'], tpos / (tot - nf))
_, st = pairs.oracle_add_correction(m, s, thr, 0.5)
ideal = max(0, math.ceil((thr * tot - tpos) / (1 - thr)))
n_add = min(ideal, max(1, math.ceil(0.5 * tot)), int(((m == 0) & (s == 1)).sum()))
close('oracle_add:n_added', st['n_added'], n_add)
close('oracle_add:prec_after', st['precision_after'], (tpos + n_add) / (tot + n_add))


# 5. candidate-hit estimates == comco_bandit.holdout_scores (via a fake model)
class Fake(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.l = logits

    def forward(self, x, eval_only=True):
        return self.l[x.long().view(-1)]


N, C = 300, 10
logits = torch.randn(N, C) * 2
y = torch.randint(0, C, (N,))
Yh = (torch.rand(N, C) < 0.3).float()
Yh[torch.arange(N), y] = 1
hold = {'x': torch.arange(N).float().view(-1, 1), 'partial': Yh, 'y': y}
old = comco_bandit.holdout_scores(Fake(logits), hold, C, 'cpu', chunk=77)
new = estimates.candidate_hit_estimates(F.softmax(logits, 1), Yh, C, y)
for kk in ('est_soft', 'est_acc', 'true_acc'):
    close(f'holdout:{kk}', new[kk], old[kk])

# 6. bias/variance reductions == biasvariance._reduce and clsloss_solo._stats
G = [torch.randn(8, 5) for _ in range(7)]
orc = torch.randn(8, 5)
old = bv._reduce(G, orc)
new = gradients.bias_variance(G, orc)
for kk in old:
    close(f'bv:{kk}', new[kk], old[kk])
Gs = torch.stack(G)
nn_ = gradients.bias_variance_normalised(G, orc)
scale = (orc.norm() / Gs.mean(0).norm()).item()
old_n = clsloss_solo._stats(Gs * scale, orc)
for kk in ('bias2', 'variance', 'mse'):
    close(f'bv_n:{kk}', nn_[kk + '_n'], old_n[kk])

# 7. resampling draws == biasvariance._resample_batch for the same seed
yy = torch.randint(0, 10, (20,))
Y_old, _ = bv._resample_batch(yy, 10, 4, 1234, 'cpu')
Y_new, _ = gradients.resample_candidate_sets(yy, 10, 4, 1234)
close('resample:identical', float(torch.equal(Y_old, Y_new)), 1.0)
close('oracle_ce_grad', (gradients.oracle_ce_grad(logits, y) - bv.oracle_cls_grad(logits, y)).abs().max(), 0.0)

# 8. candidate distribution stats == clsloss_solo.dynamics_row formulas
p = F.softmax(logits, 1)
Yd = Yh.clone()
Yd[:5] = F.one_hot(y[:5], C).float()  # |Y| = 1 rows -> entropy 0
S = (p * Yd).sum(1)
q = p * Yd / S[:, None]
kk_ = Yd.sum(1)
H = -(q * torch.log(q.clamp(min=1e-12))).sum(1)
Hn = torch.where(kk_ > 1, H / torch.log(kk_.clamp(min=2)), torch.zeros_like(H))
new = labels.candidate_distribution_stats(p, Yd, y)
close('dyn:cand_mass', new['cand_mass'], S.mean())
close('dyn:cand_entropy', new['cand_entropy'], Hn.mean())
close('dyn:disamb_acc', new['disamb_acc'], (q.argmax(1) == y).float().mean())

# 9. oracle prototypes == pcdiag.seq_ema_prototypes
protos = F.normalize(torch.randn(C, 16), dim=1)
feats = F.normalize(torch.randn(40, 16), dim=1)
lab = torch.randint(0, C, (40,))
close('proto_ema', (labels.oracle_prototypes_ema(protos, feats, lab, 0.99)
                    - pcdiag.seq_ema_prototypes(protos, feats, lab, 0.99)).abs().max(), 0.0, tol=1e-6)

# 10. kNN probe == detail.knn_eval core (inline replica of its scoring)
bank, bl = torch.randn(200, 8), torch.randint(0, 4, (200,))
qf, ql = torch.randn(50, 8), torch.randint(0, 4, (50,))
bn, qn = F.normalize(bank, dim=1), F.normalize(qf, dim=1)
ts, ti = (qn @ bn.T).topk(20, dim=1)
pred = (F.one_hot(bl[ti], 4).float() * (ts / 0.07).exp()[..., None]).sum(1).argmax(1)
close('knn', prediction.knn_top1(bank, bl, qf, ql, 4), 100 * (pred == ql).float().mean())

# 11. plateau gain == runners plateau-mode moving average
hist = list(np.random.rand(45))
ma = lambda j: sum(hist[max(0, j - 9):j + 1]) / len(hist[max(0, j - 9):j + 1])
close('plateau_gain', estimates.plateau_gain(hist), ma(44) - ma(24))

# 12. trace analysis on a real bandit trace (if present): sign agreement in [0, 1]
from src.analysis import io  # noqa: E402
if os.path.isdir('results/biasreward_c20_set1_1020'):
    tr = io.load_bandit_traces('biasreward_c20_set1_1020')
    sq = traces.estimate_quality_summary(tr)
    ok = sq.filter(like='agree').stack().between(0, 1).all()
    close('trace:sign_agreement_range', float(ok), 1.0)

print('\nALL PASS' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
sys.exit(1 if FAILS else 0)
