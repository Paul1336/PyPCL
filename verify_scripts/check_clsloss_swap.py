"""Numerical check: every dense-mask loss in src/clsloss_swap.py equals the
original implementation it re-implements (random logits, C=20, every k)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import clsloss_swap as S  # noqa: E402
from src.clpl_loss import CLPLSquaredHingeLoss  # noqa: E402
from src.comco.fixed_utils_loss import FixedComCoCLSLoss  # noqa: E402
from src.comco.utils_loss import ComCoCLSLoss  # noqa: E402
from src.cour_loss import UniformCandidateCrossEntropyLoss  # noqa: E402
from src.fixed_mcl_losses import FixedMCLLog  # noqa: E402
from src.mcl_losses import MCL_LOG  # noqa: E402
from src.op_loss import OPLoss, OPWLoss  # noqa: E402
from src.pico.mcl_cls_loss import PiCOMCLLoss  # noqa: E402
from src.proden_loss import ProdenLoss  # noqa: E402
from src.scl_loss import SCL_NL  # noqa: E402
from src.wu_loss import WuPLLLoss  # noqa: E402
from src.cpe_loss import CPELoss  # noqa: E402

torch.manual_seed(0)
C, B = 20, 64


def padded(mask):
    idx = [torch.where(r > 0)[0] for r in mask]
    L = max(len(i) for i in idx)
    out = torch.full((len(idx), L), -1, dtype=torch.long)
    for r, i in enumerate(idx):
        out[r, :len(i)] = i
    return out


ok = True
for k in [1, 3, 5, 10, 15, 19]:
    logits = torch.randn(B, C) * 3
    Y = torch.zeros(B, C)
    for r in range(B):
        Y[r, torch.randperm(C)[:k]] = 1
    comp = 1 - Y
    Yp, Cp = padded(Y), padded(comp)
    idx = torch.arange(B)
    pairs = {
        'UCE': (S.UCELoss()(logits, idx, Y), UniformCandidateCrossEntropyLoss()(logits, Yp)),
        'CLPL': (S.CLPLLoss()(logits, idx, Y), CLPLSquaredHingeLoss()(logits, Yp)),
        'MCLLOG': (S.MCLLOGLoss()(logits, idx, Y), MCL_LOG(C)(logits, Cp)),
        'MCLLOG=PiCOMCL': (S.MCLLOGLoss()(logits, idx, Y), PiCOMCLLoss()(logits, Y)),
        'MCLLOG=ComCo': (S.MCLLOGLoss()(logits, idx, Y), ComCoCLSLoss()(logits, comp)),
        'MCLLOGFixed': (S.MCLLOGFixedLoss()(logits, idx, Y), FixedMCLLog(C)(logits, Cp)),
        'SCLNL': (S.SCLNLLoss()(logits, idx, Y), SCL_NL()(logits, Cp)),
        'MCLNL=MCLLOGFixed/scale': (S.MCLNLLoss()(logits, idx, Y),
                                    S.MCLLOGFixedLoss()(logits, idx, Y) * max(C - k, 1) / (2.0 * (C - 1))),
        'SCLNL=ComCoFixed': (S.SCLNLLoss()(logits, idx, Y), FixedComCoCLSLoss()(logits, comp)),
        'OP': (S.OPLoss()(logits, idx, Y), OPLoss()(logits, Cp)),
        'OP=CPE(-g)': (S.OPLoss()(logits, idx, Y), CPELoss()(-logits, Cp)),
        'OPW': (S.OPWLoss()(logits, idx, Y), OPWLoss()(logits, Cp)),
    }
    if k == 1:
        pairs['MCLNL=CE'] = (S.MCLNLLoss()(logits, idx, Y), F.cross_entropy(logits, Y.argmax(1)))
    if k == C - 1:
        pairs['MCLNL=SCLNL'] = (S.MCLNLLoss()(logits, idx, Y), S.SCLNLLoss()(logits, idx, Y))
    if k < C:
        pairs['Wu2022'] = (S.Wu2022Loss()(logits, idx, Y), WuPLLLoss()(logits, Yp))
    # PRODEN: loss value and the updated confidence buffer.
    targets = [torch.where(r > 0)[0] for r in Y]
    ref = ProdenLoss(targets, C)
    mine = S.PRODENLoss(ref.conf.clone())
    pairs['PRODEN'] = (mine(logits, idx, Y), ref(logits, idx))
    pairs['PRODEN.conf'] = (mine.conf.sum() * 0 + (mine.conf - ref.conf).abs().max(), torch.tensor(0.))
    for name, (a, b) in pairs.items():
        # PiCOMCLLoss / ComCoCLSLoss guard log() with clamp(1e-8) instead of
        # +1e-7, which only matters when the candidate mass is ~1e-7 (same for
        # MCLNL's +1e-7 vs F.cross_entropy's exact log-softmax).
        rtol = 1e-3 if name in ('MCLLOG=PiCOMCL', 'MCLLOG=ComCo', 'MCLNL=CE') else 1e-5
        good = torch.allclose(a, b, atol=1e-5, rtol=rtol)
        ok &= good
        if not good:
            print(f'MISMATCH k={k} {name}: {a.item():.6f} vs {b.item():.6f}')
    print(f'k={k:2d}: checked {len(pairs)} pairs')

# stateful losses: run a few steps to make sure they execute and stay finite.
init = torch.zeros(B, C)
init[Y > 0] = 1
init = init / init.sum(1, keepdim=True)
cfg = dict(warmup_epoch=50, rho_range=[0.2, 0.6], lamd=3, eta=0.9, tau=0.99)
for name in ['PiCOPartial', 'PiCOCLS', 'SoLar']:
    fn = S.build_cls_loss(name, init.clone(), C, cfg, batch_size=16)
    for ep in range(3):
        fn.set_epoch(ep, [0.95, 0.8], 200)
        for s in range(0, B, 16):
            ii = torch.arange(s, s + 16)
            lg = torch.randn(16, C, requires_grad=True)
            fn.pre_update(torch.softmax(torch.randn(16, C), 1), lg, ii, Y[ii], ep >= 1)
            v = fn(lg, ii, Y[ii])
            v.backward()
            assert torch.isfinite(v), (name, ep)
    print(f'{name}: finite over 3 epochs, last loss {v.item():.4f}')
print('ALL OK' if ok else 'FAILURES ABOVE')
sys.exit(0 if ok else 1)
