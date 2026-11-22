"""Cls-loss swap (SwapPiCO-<loss> / SwapComCo-<loss>) and candidate-pruning runners."""

from ._base import *  # noqa: F401,F403
from ._base import _candidate_masked_init_conf, _comco_args, _pico_args, _print_eta, _write_trace  # noqa: F401


# ─── cls-loss swap experiment: {PiCO-Fixed, ComCo-Fixed} x every cls loss ──
#
# Architecture/warm-up/contrastive loss are held at PiCO-Fixed (PiCOModel,
# prot_start_fixed, SupCon, L_cont omitted during warm-up) or ComCo-Fixed
# (ComCoWeightedModel -- ComCoModel plus a prototype buffer that only matters
# for PiCOPartial -- and ComCoContrastiveLoss); only L_cls changes. See
# src/clsloss_swap.py and scripts/run_clsloss_swap.py.


def _run_swap_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                      device, tag, report_every, framework: str, loss_name: str):
    from src.clsloss_swap import (build_cls_loss, train_comco_swap_epoch, train_pico_swap_epoch)
    pico_cfg = raw_cfg['pico']
    if framework == 'PiCO':
        fw_args = _pico_args(C, epochs, pico_cfg)
        fw_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
        model = PiCOModel(fw_args).to(device)
        cont_loss = SupConLoss()
        loader, epoch_fn = loaders['pico'], train_pico_swap_epoch
    else:
        comco_cfg = raw_cfg['comco']
        fw_args = _comco_args(C, epochs, raw_cfg, borrow_pico=True)
        model = ComCoWeightedModel(fw_args).to(device)
        cont_loss = ComCoContrastiveLoss(temperature=fw_args['temperature'], top_k=fw_args['top_k'])
        loader, epoch_fn = loaders['comco'], train_comco_swap_epoch
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    cls_loss = build_cls_loss(loss_name, init_conf, C, raw_cfg['solar'], batch_size)
    opt = make_optimizer(model, hparams)

    # --biasvariance: per-epoch cls-loss dynamics + gradient bias/variance,
    # identical to the solo cells (src/clsloss_solo.py), every seed.
    diag = None
    if detail.biasvariance_enabled(raw_cfg):
        from src.clsloss_solo import CellDiagnostics
        bv = detail.biasvariance_cfg(raw_cfg)
        diag = CellDiagnostics(loss_name, pl_ds, orig_targets, loaders['test'], C, raw_cfg.get('_current_k', 0),
                               raw_cfg.get('_current_seed'), detail.cell_dir(raw_cfg, f'Swap{framework}-{loss_name}', C),
                               device, bv.get('log_every', 20), bv.get('m_resamples', 50), bv.get('eval_size', 256))

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        tl = epoch_fn(fw_args, model, loader, cls_loss, cont_loss, opt, ep, device)
        if diag is not None:
            row = diag.after_epoch(model.encoder_q, cls_loss, ep + 1, tl)
            if (ep + 1) % report_every == 0 or ep + 1 == epochs:
                print(f"  [diag] ep {ep + 1}  test {row['test_acc']:.2f}  train {row['train_true_acc']:.2f}  "
                      f"disamb {row['disamb_acc']:.2f}  S {row['cand_mass']:.3f}  H {row['cand_entropy']:.3f}  "
                      f"best {diag.best_acc:.2f}@{diag.best_ep}", flush=True)
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def _make_swap_runner(framework: str, loss_name: str):
    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_swap_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                 device, tag, report_every, framework, loss_name)
    return _runner


def _build_swap_runners() -> dict:
    from src.clsloss_swap import CLS_LOSSES, FRAMEWORKS, swap_algorithm_name
    return {swap_algorithm_name(fw, l): _make_swap_runner(fw, l) for fw in FRAMEWORKS for l in CLS_LOSSES}


SWAP_RUNNERS = _build_swap_runners()


# ─── Candidate pruning on SwapComCo (2026-10-28, src/clsloss_swap.CandidatePruneLoss) ──
# MCL-LOG-Fixed on Y for PRUNE_WARM epochs, then a PLL loss on S subset of Y; S re-pruned every
# PRUNE_EVERY epochs, |S| ramped linearly from k to k* = ceil(PRUNE_TARGET_FRAC * C) over
# PRUNE_RAMP epochs. L_cont keeps the original complementary set. No held-out needed.
# Per-epoch |S| / true-label retention (labels used for logging only, except Oracle) ->
# results/<run>/prune_trace/.
PRUNE_WARM, PRUNE_EVERY, PRUNE_RAMP, PRUNE_TARGET_FRAC = 50, 5, 50, 0.25


def _run_swap_prune(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag,
                    report_every, algorithm: str, pl_name: str, oracle: bool, framework: str = 'ComCo'):
    from src.clsloss_swap import CandidatePruneLoss, train_comco_swap_epoch, train_pico_swap_epoch
    pico_cfg = raw_cfg['pico']
    if framework == 'PiCO':
        # Option A (2026-10-29): only L_cls sees the pruned set S; PiCOModel's forward (prototype
        # pseudo-labels for SupCon, score_prot) keeps the original candidate set Y.
        fw_args = _pico_args(C, epochs, pico_cfg)
        fw_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
        model = PiCOModel(fw_args).to(device)
        cont_loss = SupConLoss()
        loader, epoch_impl = loaders['pico'], train_pico_swap_epoch
    else:
        comco_cfg = raw_cfg['comco']
        fw_args = _comco_args(C, epochs, raw_cfg, borrow_pico=True)
        model = ComCoWeightedModel(fw_args).to(device)
        cont_loss = ComCoContrastiveLoss(temperature=fw_args['temperature'], top_k=fw_args['top_k'])
        loader, epoch_impl = loaders['comco'], train_comco_swap_epoch
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    y = torch.as_tensor(orig_targets, device=device).long()
    cls_loss = CandidatePruneLoss(init_conf, pl_name, C, PRUNE_WARM, true_labels=y if oracle else None,
                                  seed=raw_cfg.get('_current_seed') or 0)
    opt = make_optimizer(model, hparams)
    k = int(cls_loss.Y.sum(dim=1).max().item())
    k_star = min(k, math.ceil(PRUNE_TARGET_FRAC * C))
    rows = torch.arange(len(y), device=device)

    trace = []
    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        rec = {'epoch': ep + 1}
        if ep >= PRUNE_WARM and (ep - PRUNE_WARM) % PRUNE_EVERY == 0:
            r = round(k - (k - k_star) * min(1.0, (ep - PRUNE_WARM) / PRUNE_RAMP))
            cls_loss.prune(r)
            rec['r'] = r
        epoch_impl(fw_args, model, loader, cls_loss, cont_loss, opt, ep, device)
        with torch.no_grad():
            rec['S_size'] = round(cls_loss.S.sum(dim=1).mean().item(), 3)
            rec['retention'] = round(cls_loss.S[rows, y].mean().item(), 5)
            rec['q_top1'] = round((cls_loss.q.argmax(dim=1) == y).float().mean().item(), 5)
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            rec['test_acc'] = evaluate_model(model, loaders['test'], device)
        trace.append(rec)
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            _print_eta(tag, ep + 1, epochs, time.perf_counter() - chunk_t0, min(report_every, ep + 1))
            print(f"    [prune] |S|={rec['S_size']:.2f}  retention={rec['retention']:.3f}  "
                  f"q_top1={rec['q_top1']:.3f}", flush=True)
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    _write_trace(raw_cfg, 'prune_trace', algorithm, C, {'k': k, 'k_star': k_star, 'warm': PRUNE_WARM, 'every': PRUNE_EVERY, 'ramp': PRUNE_RAMP,
                   'trace': trace})
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def _make_prune_runner(name, pl_name, oracle, framework):
    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_swap_prune(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device,
                               tag, report_every, name, pl_name, oracle, framework)
    return _runner


# Swap<framework>-Prune[Oracle]-<PLL loss>; the PiCO-framework version (2026-10-29) only with PiCOPartial.
PRUNE_VARIANTS = [('ComCo', _pl) for _pl in ('PiCOPartial', 'PRODEN')] + [('PiCO', 'PiCOPartial')]
PRUNE_RUNNERS = {f'Swap{_fw}-Prune{"Oracle" if _o else ""}-{_pl}': _make_prune_runner(
                     f'Swap{_fw}-Prune{"Oracle" if _o else ""}-{_pl}', _pl, _o, _fw)
                 for _fw, _pl in PRUNE_VARIANTS for _o in (False, True)}
