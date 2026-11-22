"""Step-level equivalence: refactored code vs the frozen pre-refactor copy (tests/legacy_ref).

For every loss that was merged / rewritten, compare loss value and gradient w.r.t. the logits
on random inputs (several C, k, including k = 1 and k = C-1, and peaked logits so p -> 0/1
edge cases are hit). Tolerance: |a - b| <= ATOL + RTOL * max(|a|, |b|), elementwise.

    python tests/step_equiv.py
"""

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests', 'legacy_ref'))

RTOL, ATOL = 1e-6, 1e-6
FAILS = []
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)


def close(a, b):
    return bool(((a - b).abs() <= ATOL + RTOL * torch.maximum(a.abs(), b.abs())).all())


def report(name, ok, detail=''):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        FAILS.append(name)


def make_batch(B, C, k, scale):
    y = torch.randint(0, C, (B,))
    Y = torch.zeros(B, C)
    for i in range(B):
        Y[i, y[i]] = 1
        if k > 1:
            others = torch.randperm(C)
            others = others[others != y[i]][:k - 1]
            Y[i, others] = 1
    logits = torch.randn(B, C) * scale
    return logits, Y, y


def padded(mask):
    rows = [torch.nonzero(r).flatten() for r in mask]
    L = max(1, max(len(r) for r in rows))
    out = torch.full((len(rows), L), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def val_grad(fn, logits, *args):
    g = logits.clone().requires_grad_(True)
    v = fn(g, *args)
    v.backward()
    return v.detach(), g.grad.detach()


def compare(name, new_fn, old_fn, new_args, old_args, logits):
    v1, g1 = val_grad(new_fn, logits, *new_args)
    v0, g0 = val_grad(old_fn, logits, *old_args)
    ok = close(v1, v0) and close(g1, g0)
    return ok, f'(loss {v1.item():.6g} vs {v0.item():.6g}, max|dg| {(g1 - g0).abs().max().item():.2e})'


def test_stateless_losses():
    import legacy_src.clpl_loss as oclpl
    import legacy_src.clsloss_swap as oswap
    import legacy_src.comco.fixed_utils_loss as ofix
    import legacy_src.comco.utils_loss as ocomco
    import legacy_src.cour_loss as ocour
    import legacy_src.cpe_loss as ocpe
    import legacy_src.fixed_mcl_losses as ofmcl
    import legacy_src.mcl_losses as omcl
    import legacy_src.op_loss as oop
    import legacy_src.pcdiag as opcd
    import legacy_src.pico.mcl_cls_loss as opmcl
    import legacy_src.scl_loss as oscl
    import legacy_src.wu_loss as owu
    from src import pcdiag as npcd
    from src.losses import stateless as S
    from src.losses import swap as W

    for C, k, scale in [(10, 1, 1.0), (10, 3, 3.0), (20, 10, 5.0), (20, 19, 2.0), (100, 50, 8.0), (100, 99, 1.0)]:
        logits, Y, _ = make_batch(64, C, k, scale)
        comp = 1.0 - Y
        pl, cl = padded(Y), padded(comp)
        tag = f'C={C} k={k}'
        cases = [
            ('MCL_LOG', S.MCL_LOG(C), omcl.MCL_LOG(C), (cl,), (cl,)),
            ('MCL_MAE', S.MCL_MAE(C), omcl.MCL_MAE(C), (cl,), (cl,)),
            ('MCL_EXP', S.MCL_EXP(C), omcl.MCL_EXP(C), (cl,), (cl,)),
            ('SCL_NL', S.SCL_NL(), oscl.SCL_NL(), (cl,), (cl,)),
            ('OPLoss', S.OPLoss(), oop.OPLoss(), (cl,), (cl,)),
            ('OPWLoss', S.OPWLoss(), oop.OPWLoss(), (cl,), (cl,)),
            ('CPELoss', S.CPELoss(), ocpe.CPELoss(), (cl,), (cl,)),
            ('CLPL', S.CLPLSquaredHingeLoss(), oclpl.CLPLSquaredHingeLoss(), (pl,), (pl,)),
            ('UCE(cour)', S.UniformCandidateCrossEntropyLoss(), ocour.UniformCandidateCrossEntropyLoss(), (pl,), (pl,)),
            ('PiCOMCLLoss', S.PiCOMCLLoss(), opmcl.PiCOMCLLoss(), (Y,), (Y,)),
            ('ComCoCLSLoss', S.ComCoCLSLoss(), ocomco.ComCoCLSLoss(), (comp,), (comp,)),
            ('FixedComCoCLSLoss', S.FixedComCoCLSLoss(), ofix.FixedComCoCLSLoss(), (comp,), (comp,)),
            ('pcdiag.scl_nl_avg', npcd.scl_nl_avg, opcd.scl_nl_avg, (comp,), (comp,)),
            ('pcdiag.mcl_log', npcd.mcl_log, opcd.mcl_log, (comp,), (comp,)),
            ('pcdiag.mcl_nl', npcd.mcl_nl, opcd.mcl_nl, (comp,), (comp,)),
            ('pcdiag.mcl_log_fixed', npcd.mcl_log_fixed, opcd.mcl_log_fixed, (comp,), (comp,)),
        ]
        if k < C:  # the old Wu2022 / FixedMCL divided by zero at k = C or m = 0
            cases += [('WuPLLLoss', S.WuPLLLoss(), owu.WuPLLLoss(), (pl,), (pl,)),
                      ('FixedMCLLog', S.FixedMCLLog(C), ofmcl.FixedMCLLog(C), (cl,), (cl,)),
                      ('FixedMCLMae', S.FixedMCLMae(C), ofmcl.FixedMCLMae(C), (cl,), (cl,)),
                      ('FixedMCLExp', S.FixedMCLExp(C), ofmcl.FixedMCLExp(C), (cl,), (cl,))]
        idx = torch.arange(64)
        for nm in ['UCELoss', 'CLPLLoss', 'Wu2022Loss', 'MCLLOGLoss', 'MCLLOGFixedLoss', 'MCLNLLoss',
                   'SCLNLLoss', 'OPLoss', 'OPWLoss']:
            cases.append((f'swap.{nm}', getattr(W, nm)(), getattr(oswap, nm)(), (idx, Y), (idx, Y)))
        for name, new, old, na, oa in cases:
            ok, d = compare(name, new, old, na, oa, logits)
            report(f'{name} [{tag}]', ok, '' if ok else d)


def test_stateful_moved():
    """Moved verbatim: two training steps must give identical loss and buffers."""
    import legacy_src.proden_loss as oprod
    import legacy_src.pico.utils_loss as opart
    import legacy_src.pico_cls_loss as opcls
    from src.losses import confidence as Cf
    C, k, B = 10, 3, 32
    logits, Y, y = make_batch(B, C, k, 2.0)
    targets = [torch.nonzero(r).flatten() for r in Y]
    idx = torch.arange(B)
    pairs = [('ProdenLoss', Cf.ProdenLoss(targets, C), oprod.ProdenLoss(targets, C),
              lambda L, g: L(g, idx)),
             ('PartialLoss', Cf.PartialLoss(Y / Y.sum(1, keepdim=True)), opart.PartialLoss(Y / Y.sum(1, keepdim=True)),
              lambda L, g: (L.confidence_update(torch.softmax(g.detach(), 1), idx, Y), L(g, idx))[1]),
             ('PiCOCLSLoss', Cf.PiCOCLSLoss(targets, C, epochs=10), opcls.PiCOCLSLoss(targets, C, epochs=10),
              lambda L, g: (L(g, idx), L.update_confidence(g.detach(), idx))[0])]
    for name, new, old, step in pairs:
        ok = True
        for _ in range(2):
            v1, v0 = step(new, logits), step(old, logits)
            ok &= close(v1, v0)
        b1, b0 = Cf.conf_buffer(new), getattr(old, 'conf', getattr(old, 'confidence', None))
        ok &= close(b1, b0)
        report(f'{name} (moved)', ok)


def test_contrastive():
    import legacy_src.comco.utils_loss as oc
    import legacy_src.pico.utils_loss as op
    from src.comco.utils_loss import ComCoContrastiveLoss
    from src.pico.utils_loss import SupConLoss
    B, Q, D, C = 16, 32, 8, 6
    feats = torch.nn.functional.normalize(torch.randn(2 * B + Q, D), dim=1)
    lab = torch.randint(0, C, (2 * B + Q,))
    mask = (lab[:B, None] == lab[None, :]).float()
    for name, args in [('SupCon mask', (mask, B)), ('SupCon MoCo', (None, B))]:
        ok, d = compare(name, lambda f, *a: SupConLoss()(f, *a), lambda f, *a: op.SupConLoss()(f, *a),
                        args, args, feats)
        report(name, ok, '' if ok else d)
    comp = (torch.rand(2 * B + Q, C) < 0.5).float()
    pq = lab[:B]
    for wp, wn in [(False, False), (False, True), (True, True)]:
        f_new = lambda q, *a: ComCoContrastiveLoss()(q, *a)
        f_old = lambda q, *a: oc.ComCoContrastiveLoss()(q, *a)
        args = (feats, lab, comp, pq, wp, wn)
        ok, d = compare(f'ComCoContrastive wp={wp} wn={wn}', f_new, f_old, args, args, feats[:B].clone())
        report(f'ComCoContrastive wp={wp} wn={wn}', ok, '' if ok else d)


def test_models_forward():
    """Same seed -> same init; two forward passes must give identical outputs and buffers."""
    import legacy_src.comco.model as ocm
    import legacy_src.pico.model as opm
    from src.comco import model as ncm
    from src.pico import model as npm
    C, B = 5, 8
    args = dict(num_class=C, low_dim=16, moco_queue=16, moco_m=0.99, proto_m=0.9)
    x1, x2 = torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32)
    Y = (torch.rand(B, C) < 0.5).float()
    Y[:, 0] = 1
    y = torch.randint(0, C, (B,))
    comp = 1 - Y
    cases = [('PiCOModel', npm.PiCOModel, opm.PiCOModel, lambda M: M(x1, x2, Y, args)),
             ('PiCOOracleModel', npm.PiCOOracleModel, opm.PiCOOracleModel, lambda M: M(x1, x2, Y, y, args)),
             ('ComCoModel', ncm.ComCoModel, ocm.ComCoModel, lambda M: M(x1, x2, comp, args)),
             ('ComCoWeightedModel', ncm.ComCoWeightedModel, ocm.ComCoWeightedModel,
              lambda M: M(x1, x2, comp, Y, args))]
    for name, New, Old, fwd in cases:
        torch.manual_seed(1)
        mn = New(args)
        torch.manual_seed(1)
        mo = Old(args)
        ok = True
        for _ in range(2):
            on, oo = fwd(mn), fwd(mo)
            ok &= all(close(a.float(), b.float()) for a, b in zip(on, oo))
        bn, bo = dict(mn.named_buffers()), dict(mo.named_buffers())
        ok &= set(bn) == set(bo) and all(close(bn[k].float(), bo[k].float()) for k in bn)
        ok &= all(close(a, b) for a, b in zip(mn.encoder_k.parameters(), mo.encoder_k.parameters()))
        ok &= close(mn.predict(x1), mo(x1, eval_only=True))
        report(f'{name} forward x2 + buffers + predict', ok)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('\nALL PASS' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
