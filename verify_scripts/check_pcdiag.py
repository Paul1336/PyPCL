"""Numerical checks for src/pcdiag.py (2026-11-03).

1. seq_ema_prototypes == PiCOModel's sequential per-sample EMA loop.
2. masked_infonce (D = all but self) == SupConLoss mask mode.
3. comco_masks + masked_infonce == ComCoContrastiveLoss for every warm-up phase.
4. A few training steps of PCD-PiCO / PCD-ComCo (no flags) reproduce train_pico_epoch_fixed /
   train_comco_epoch(FixedComCoCLSLoss) parameter-for-parameter.
5. PiCO positives never violate the shared-candidate rule.
"""
import copy
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from src.comco.fixed_utils_loss import FixedComCoCLSLoss  # noqa: E402
from src.comco.model import ComCoModel  # noqa: E402
from src.comco.utils_loss import ComCoContrastiveLoss  # noqa: E402
from src.engine import train_comco_epoch  # noqa: E402
from src.fixed_pico_engine import train_pico_epoch_fixed  # noqa: E402
from src.pcdiag import (ComCoDiagModel, PiCODiagModel, comco_masks, masked_infonce,  # noqa: E402
                        seq_ema_prototypes, train_comco_pcd_epoch, train_pico_pcd_epoch)
from src.pico.model import PiCOModel  # noqa: E402
from src.pico.utils_loss import PartialLoss, SupConLoss  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dev = 'cuda'
torch.manual_seed(0)
ok = True


def check(name, a, b, tol=1e-5):
    global ok
    err = (a - b).abs().max().item() if torch.is_tensor(a) else abs(a - b)
    good = err <= tol
    ok &= good
    print(f'{"PASS" if good else "FAIL"}  {name}: max|diff| = {err:.3e}')


# 1
C, D, n, m = 6, 16, 200, 0.99
protos = F.normalize(torch.randn(C, D, device=dev), dim=1)
feats = F.normalize(torch.randn(n, D, device=dev), dim=1)
labels = torch.randint(0, C, (n,), device=dev)
ref = protos.clone()
for f, l in zip(feats, labels):
    ref[l] = ref[l] * m + (1 - m) * f
ref = F.normalize(ref, p=2, dim=1)
check('seq_ema_prototypes', seq_ema_prototypes(protos, feats, labels, m), ref)

# 2
B, A = 32, 32 * 2 + 64
q = F.normalize(torch.randn(B, D, device=dev), dim=1)
pool = torch.cat([q, F.normalize(torch.randn(A - B, D, device=dev), dim=1)])
lab = torch.randint(0, C, (A,), device=dev)
mask = (lab[:B, None] == lab[None, :]).float()
eye = torch.zeros(B, A, device=dev)
eye[torch.arange(B), torch.arange(B)] = 1
check('masked_infonce vs SupConLoss', masked_infonce(q, pool, mask * (1 - eye), 1 - eye, 0.07),
      SupConLoss()(features=pool, mask=mask, batch_size=B))

# 3
comp = (torch.rand(A, C, device=dev) < 0.5).float()
crit = ComCoContrastiveLoss(temperature=0.17, top_k=1)
for wp, wn in [(False, False), (False, True), (True, True)]:
    l_ref, p_ref, d_ref = crit(q, pool, lab, comp, lab[:B], wp, wn, return_masks=True)
    p, d, _ = comco_masks(torch.mm(q, pool.t()), lab[:B], lab, comp, wp, wn, 1)
    check(f'comco masks pos  (pos={wp}, neg={wn})', p, p_ref, 0)
    check(f'comco masks denom(pos={wp}, neg={wn})', d, d_ref, 0)
    check(f'comco loss       (pos={wp}, neg={wn})', masked_infonce(q, pool, p, d, 0.17), l_ref)

# 4  end-to-end steps
C, B, Q, steps = 5, 32, 128, 6
imgs_w = torch.randn(steps, B, 3, 32, 32)
imgs_s = torch.randn(steps, B, 3, 32, 32)
true = torch.randint(0, C, (steps, B))
Y = (torch.rand(steps, B, C) < 0.4).float()
Y[torch.arange(steps)[:, None], torch.arange(B)[None, :], true] = 1.0
idx = torch.arange(steps * B).view(steps, B)
loader = [(imgs_w[s], imgs_s[s], Y[s], true[s], idx[s]) for s in range(steps)]


def max_param_diff(m1, m2):
    return max((p1 - p2).abs().max().item() for p1, p2 in zip(m1.parameters(), m2.parameters()))


pargs = {'num_class': C, 'epochs': 10, 'low_dim': 128, 'moco_queue': Q, 'moco_m': 0.999, 'proto_m': 0.99,
         'loss_weight': 0.5, 'conf_ema_range': [0.95, 0.8], 'prot_start': 1}
for ep in (0, 1):
    torch.manual_seed(1)
    ref_m = PiCOModel(pargs).to(dev)
    new_m = PiCODiagModel(pargs).to(dev)
    new_m.load_state_dict(ref_m.state_dict(), strict=False)
    conf0 = Y.view(-1, C) / Y.view(-1, C).sum(1, keepdim=True)
    l_ref, l_new = PartialLoss(conf0.clone().to(dev)), PartialLoss(conf0.clone().to(dev))
    o_ref = torch.optim.Adam(ref_m.parameters(), lr=3e-4, weight_decay=1e-4)
    o_new = torch.optim.Adam(new_m.parameters(), lr=3e-4, weight_decay=1e-4)
    for L in (l_ref, l_new):
        L.set_conf_ema_m(ep, pargs)
    train_pico_epoch_fixed(pargs, ref_m, loader, l_ref, SupConLoss(), o_ref, ep, dev)
    row = train_pico_pcd_epoch(pargs, new_m, loader, l_new, o_new, ep, dev, {}, C)
    check(f'PiCO params after {steps} steps (epoch {ep})', max_param_diff(ref_m, new_m), 0, 1e-4)
    check(f'PiCO confidence (epoch {ep})', l_ref.confidence, l_new.confidence, 1e-5)
    check(f'PiCO prototypes (epoch {ep})', ref_m.prototypes, new_m.prototypes, 1e-4)
    check(f'PiCO positive violation rate (epoch {ep})', row['pos_viol_rate'], 0.0, 0)

cargs = {'num_class': C, 'epochs': 10, 'low_dim': 128, 'moco_queue': Q, 'moco_m': 0.999, 'loss_weight': 0.3,
         'temperature': 0.17, 'top_k': 1, 'warmup_neg': 1, 'warmup_pos': 2}
cl_loader = [(w, s, 1 - y, t, i) for (w, s, y, t, i) in loader]
for ep in (0, 1, 2):
    torch.manual_seed(1)
    ref_m = ComCoModel(cargs).to(dev)
    new_m = ComCoDiagModel(cargs).to(dev)
    new_m.load_state_dict(ref_m.state_dict(), strict=False)
    o_ref = torch.optim.Adam(ref_m.parameters(), lr=3e-4, weight_decay=1e-4)
    o_new = torch.optim.Adam(new_m.parameters(), lr=3e-4, weight_decay=1e-4)
    train_comco_epoch(cargs, ref_m, cl_loader, FixedComCoCLSLoss(),
                      ComCoContrastiveLoss(cargs['temperature'], cargs['top_k']), o_ref, ep, dev)
    row = train_comco_pcd_epoch(cargs, new_m, cl_loader, o_new, ep, dev, {}, C)
    check(f'ComCo params after {steps} steps (epoch {ep})', max_param_diff(ref_m, new_m), 0, 1e-4)
    print(f'      ComCo ep{ep}: viol={row["pos_viol_rate"]:.3f} nonkey_viol={row["pos_nonkey_viol_rate"]:.3f} '
          f'cstar_hit={row.get("cstar_hit", float("nan")):.3f}')

print('ALL PASS' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
