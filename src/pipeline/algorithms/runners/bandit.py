"""Switch / bandit runners: ComCo-Weighted per-batch alpha rules, the legacy ComCo-Weighted bandit and the
generic {PiCO, ComCo, Hybrid, Solo, SwapComCo} x {Uni, Mix} bandit / schedule / plateau / s-flip runner."""

from ._base import *  # noqa: F401,F403
from ._base import _candidate_masked_init_conf, _comco_args, _pico_args, _print_eta, _write_trace  # noqa: F401


# ─── ComCo-Weighted, bandit-chosen hard alpha (2026-10-13) ────────────────
# See src/comco_bandit.py. Needs --holdout_frac > 0 (label-free reward is
# computed on loaders['holdout']). Per-epoch trace (arm, paired advantage,
# posterior, held-out estimated vs true acc, test acc every 5 epochs) goes to
# results/<run>/bandit_trace/.

BANDIT_LOOKAHEAD_BATCHES = 4
LONG_DECIDE_EVERY = 5  # 'paired01_long': decision interval (epochs); lookahead = one full epoch per arm
BANDIT_GAMMA = 0.9


def _run_comco_weighted_bandit(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                               device, tag, report_every, algorithm: str, paired: bool):
    from src import comco_bandit as cb
    if 'holdout' not in loaders:
        raise ValueError(f'{algorithm} needs a held-out split: run with --holdout_frac > 0')
    comco_cfg = raw_cfg['comco']
    pico_cfg = raw_cfg['pico']
    comco_args = _comco_args(C, epochs, raw_cfg, borrow_pico=True)
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    model = ComCoWeightedModel(comco_args).to(device)
    cls_loss = ComCoWeightedClsLoss(init_conf, alpha=0.0)  # alpha overridden every batch by the chosen arm
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'], top_k=comco_args['top_k'])
    opt = make_optimizer(model, hparams)
    holdout = loaders['holdout']
    bandit = cb.PairedTS(gamma=BANDIT_GAMMA) if paired else cb.TwoArmTS(gamma=BANDIT_GAMMA)

    trace = []
    prev_score = cb.holdout_scores(model, holdout, C, device)['est_soft']
    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, comco_args)
        rec = {'epoch': ep + 1}
        if paired:
            snap = cb.snapshot(model, opt, cls_loss)
            batches = cb.draw_batches(loaders['comco'], BANDIT_LOOKAHEAD_BATCHES)
            look = {}
            for arm in (0, 1):
                cb.restore(snap, model, opt, cls_loss)
                train_comco_weighted_switch_epoch(comco_args, model, batches, cls_loss, cont_loss, opt, ep,
                                                   device, lambda _y, a=float(arm): a)
                look[arm] = cb.holdout_scores(model, holdout, C, device)['est_soft']
            cb.restore(snap, model, opt, cls_loss)
            del snap, batches
            bandit.update(look[1] - look[0])
            mu, se = bandit.posterior()
            rec.update(look0=look[0], look1=look[1], d=look[1] - look[0], mu=mu, se=se)
        arm = bandit.choose()
        train_comco_weighted_switch_epoch(comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep,
                                           device, lambda _y, a=float(arm): a)
        scores = cb.holdout_scores(model, holdout, C, device)
        if not paired:
            bandit.update(arm, scores['est_soft'] - prev_score)
            rec.update(mu0=bandit.posterior(0)[0], mu1=bandit.posterior(1)[0])
        prev_score = scores['est_soft']
        rec.update(arm=arm, **scores)
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            rec['test_acc'] = evaluate_model(model, loaders['test'], device)
        trace.append(rec)
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            frac1 = sum(t['arm'] for t in trace[-report_every:]) / len(trace[-report_every:])
            print(f'    [bandit] last {report_every} ep: arm1 frac={frac1:.2f}  '
                  f'holdout est={scores["est_acc"]:.3f} true={scores["true_acc"]:.3f}', flush=True)
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    _write_trace(raw_cfg, 'bandit_trace', algorithm, C, trace)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def run_comco_weighted_bandit_paired(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                     device, tag, report_every):
    return _run_comco_weighted_bandit(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                      device, tag, report_every, 'ComCo-Weighted-BanditPaired', paired=True)


def run_comco_weighted_bandit_epoch(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every):
    return _run_comco_weighted_bandit(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                      device, tag, report_every, 'ComCo-Weighted-BanditEpoch', paired=False)


# ─── Unified loss (src/unified_loss.py) + generic bandit, PiCO & ComCo ──────
# (2026-10-13). One runner for both frameworks and three ways of picking the
# cls loss's mixing value each epoch:
#   'fixed'     constant value
#   'paired01'  PairedTS over {0, 1} (same paired lookahead as BanditPaired)
#   'cont'      ContinuousRhoTS over [0, 1] (paired lookahead at rho +- delta)
# loss_kind 'unified' -> value is rho of UnifiedClsLoss (0 = PartialLoss,
# 1 = original ComCo cls loss); 'mix' -> value is alpha of
# PartialLoss/ComCo-Fixed(SCL-NL) (the earlier ComCo-Weighted arms). Needs
# --holdout_frac > 0. Per-epoch trace -> results/<run>/bandit_trace/.


def _generic_bandit_setup(framework, loss_kind, pl_ds, C, hparams, raw_cfg, epochs, device):
    from src.unified_loss import PiCOFixedComCoMixClsLoss, UnifiedClsLoss, UnifiedClsLossComCo
    from src.fixed_pico_engine import train_pico_switch_epoch
    pico_cfg = raw_cfg['pico']
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    if framework == 'ComCo':
        comco_cfg = raw_cfg['comco']
        args = _comco_args(C, epochs, raw_cfg, borrow_pico=True)
        model = ComCoWeightedModel(args).to(device)
        cls_loss = UnifiedClsLossComCo(init_conf) if loss_kind == 'unified' else ComCoWeightedClsLoss(init_conf, 0.0)
        cont_loss = ComCoContrastiveLoss(temperature=args['temperature'], top_k=args['top_k'])
        loader_key, epoch_impl = 'comco', train_comco_weighted_switch_epoch
    elif framework == 'PiCO':
        args = _pico_args(C, epochs, pico_cfg)
        args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
        model = PiCOModel(args).to(device)
        cls_loss = UnifiedClsLoss(init_conf) if loss_kind == 'unified' else PiCOFixedComCoMixClsLoss(init_conf)
        cont_loss = SupConLoss()
        loader_key, epoch_impl = 'pico', train_pico_switch_epoch
    elif framework == 'Hybrid':
        # PiCO+PartialLoss (value 0) <-> ComCo + its cls loss (value 1), shared encoder; src/hybrid_switch.py.
        # loss_kind 'unified': ComCo arm uses MCL-LOG (rho=1); 'mix': ComCo arm uses SCL-NL (ComCo-Fixed).
        from src.hybrid_switch import ComCoBatchView, ComCoLossView, HybridModel
        comco_cfg = raw_cfg['comco']
        pico_args = _pico_args(C, epochs, pico_cfg)
        pico_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
        comco_args = _comco_args(C, epochs, raw_cfg, borrow_pico=True)
        model = HybridModel(pico_args, comco_args).to(device)
        cls_loss = UnifiedClsLoss(init_conf) if loss_kind == 'unified' else PiCOFixedComCoMixClsLoss(init_conf)
        comco_view = ComCoLossView(cls_loss)
        pico_cont = SupConLoss()
        comco_cont = ComCoContrastiveLoss(temperature=comco_args['temperature'], top_k=comco_args['top_k'])
        cont_loss = (pico_cont, comco_cont)
        opt = make_optimizer(model, hparams)

        def epoch_fn(ep, loader, value):
            if float(value) < 0.5:
                train_pico_switch_epoch(pico_args, model.pico, loader, cls_loss, pico_cont, opt, ep, device,
                                        lambda _y: 0.0)
                model.sync_prototypes('pico')
            else:
                train_comco_weighted_switch_epoch(comco_args, model.comco, ComCoBatchView(loader), comco_view,
                                                  comco_cont, opt, ep, device, lambda _y: 1.0)
                model.sync_prototypes('comco')

        def loss_at(logits, index, cand, value):
            return cls_loss(logits, index, cand, alpha=value)

        return model, cls_loss, cont_loss, opt, pico_args, 'pico', epoch_fn, init_conf, loss_at
    elif framework == 'SwapComCo':
        # Two clsloss_swap losses on SwapComCo (2026-10-27): loss_kind = '<arm0>+<arm1>', value 0 = arm0,
        # 1 = arm1. Each epoch is exactly train_comco_swap_epoch, so a fixed arm == SwapComCo-<arm>.
        from src.clsloss_swap import SwapMixClsLoss, build_cls_loss, train_comco_swap_epoch
        comco_cfg = raw_cfg['comco']
        args = _comco_args(C, epochs, raw_cfg, borrow_pico=True)
        model = ComCoWeightedModel(args).to(device)
        arm0, arm1 = (build_cls_loss(n, init_conf, C, raw_cfg['solar'], None) for n in loss_kind.split('+'))
        cls_loss = SwapMixClsLoss(arm0, arm1)
        cont_loss = ComCoContrastiveLoss(temperature=args['temperature'], top_k=args['top_k'])
        opt = make_optimizer(model, hparams)

        def epoch_fn(ep, loader, value):
            cls_loss.value = float(value)
            train_comco_swap_epoch(args, model, loader, cls_loss, cont_loss, opt, ep, device)

        def loss_at(logits, index, cand, value):
            cls_loss.value = float(value)
            return cls_loss(logits, index, cand)

        return model, cls_loss, cont_loss, opt, args, 'comco', epoch_fn, init_conf, loss_at
    elif framework == 'Solo':
        # no contrastive branch; s updated from the classifier (src/solo_switch.py)
        from src.solo_switch import SoloModel, train_solo_switch_epoch
        args = {'num_class': C, 'epochs': epochs, 'prot_start': pico_cfg.get('prot_start_fixed', 1),
                'conf_ema_range': pico_cfg['conf_ema_range']}
        model = SoloModel(C, pico_cfg['low_dim']).to(device)
        cls_loss = UnifiedClsLoss(init_conf) if loss_kind == 'unified' else PiCOFixedComCoMixClsLoss(init_conf)
        cont_loss = None
        loader_key, epoch_impl = 'pico', train_solo_switch_epoch
    else:
        raise ValueError(framework)
    opt = make_optimizer(model, hparams)

    def epoch_fn(ep, loader, value):
        epoch_impl(args, model, loader, cls_loss, cont_loss, opt, ep, device, lambda _y, v=float(value): v)

    if framework == 'ComCo':
        def loss_at(logits, index, cand, value):
            return cls_loss(logits, index, 1.0 - cand, alpha=value)
    else:
        def loss_at(logits, index, cand, value):
            return cls_loss(logits, index, cand, alpha=value)

    return model, cls_loss, cont_loss, opt, args, loader_key, epoch_fn, init_conf, loss_at


def _run_generic_bandit(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag,
                        report_every, algorithm: str, framework: str, loss_kind: str, mode: str,
                        fixed_value: float = None, reward: str = 'uniform'):
    from src import comco_bandit as cb
    if 'holdout' not in loaders:
        raise ValueError(f'{algorithm} needs a held-out split: run with --holdout_frac > 0')
    model, cls_loss, cont_loss, opt, args, loader_key, epoch_fn, init_conf, loss_at = _generic_bandit_setup(
        framework, loss_kind, pl_ds, C, hparams, raw_cfg, epochs, device)
    holdout = loaders['holdout']
    # Lookahead reward (src/bias_aware_reward.py): 'uniform' = the original est_soft; 'rho_em' /
    # 'rho_oracle' = bias-aware rho correction with label-free / true-label Pi; 'true' = held-out
    # mean true-class probability (uses labels -- upper bound only). Non-uniform rewards also log
    # every lookahead arm's old / new / true score so the sign of d can be checked against truth.
    scorer, reward_info = None, {}
    if reward != 'uniform':
        from src import bias_aware_reward as bar
        tr_idx = loaders['train_idx']
        tr_partial = torch.zeros(len(tr_idx), C)
        for row, i in enumerate(tr_idx):
            tr_partial[row, pl_ds.targets[i]] = 1.0
        if reward == 'rho_oracle':
            Pi = bar.oracle_inclusion(tr_partial, torch.as_tensor(orig_targets)[torch.as_tensor(tr_idx)], C)
        else:
            Pi, lam, curve = bar.estimate_inclusion_shrunk(tr_partial)
            reward_info = {'shrink_lambda': lam, 'cv_loglik': curve}
        scorer = bar.RhoCorrected(holdout['partial'], Pi)
        scorer.credit = scorer.credit.to(device)
        print(f'    [reward] {reward} {reward_info.get("shrink_lambda", "")}', flush=True)

    def look_score(m):
        if scorer is None:
            return cb.holdout_scores(m, holdout, C, device)['est_soft'], None
        s = cb.holdout_scores_all(m, holdout, C, device, scorer)
        key = {'rho_em': 'rho_soft', 'rho_oracle': 'rho_soft', 'true': 'true_soft'}[reward]
        return s[key], {k: round(v, 5) for k, v in s.items()}
    diag = None
    if raw_cfg.get('_switch_diag'):
        from src.switch_diag import SwitchDiag
        diag = SwitchDiag(pl_ds, orig_targets, loaders['train_idx'], C, device)
    if mode == 'paired01':
        bandit = cb.PairedTS(gamma=BANDIT_GAMMA)
    elif mode == 'cont':
        bandit = cb.ContinuousRhoTS(gamma=BANDIT_GAMMA)
    elif mode == 'paired01_long':
        bandit = cb.PairedTS(gamma=BANDIT_GAMMA)
        value = 1.0
    elif mode == 'plateau':
        est_hist, switched, value = [], False, 1.0
    elif mode == 'sflip':
        train_idx_t = torch.as_tensor(loaders['train_idx'], device=device).long()
        prev_arg, calm, switched, value = None, 0, False, 1.0
    elif mode not in ('fixed', 'sched', 'sched_frozen'):
        raise ValueError(mode)

    trace = [{'epoch': 0, 'reward': reward, **reward_info}] if reward != 'uniform' else []
    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, args)
        rec = {'epoch': ep + 1}
        if mode == 'fixed':
            value = fixed_value
        elif mode in ('sched', 'sched_frozen'):
            # fixed_value = switch epoch T: rho=1 for ep < T, then rho=0. sched_frozen also
            # stops s (the confidence) from being updated while rho=1.
            value = 1.0 if ep < fixed_value else 0.0
            if mode == 'sched_frozen':
                cls_loss.frozen = ep < fixed_value
        elif mode in ('sflip', 'plateau'):
            pass  # value decided at the end of the previous epoch
        elif mode == 'paired01_long':
            if ep % LONG_DECIDE_EVERY == 0:
                snap = cb.snapshot(model, opt, cls_loss)
                batches = cb.draw_batches(loaders[loader_key], len(loaders[loader_key]))
                look, look_all = [], []
                for v in (0.0, 1.0):
                    cb.restore(snap, model, opt, cls_loss)
                    epoch_fn(ep, batches, v)
                    sc, sa = look_score(model)
                    look.append(sc)
                    look_all.append(sa)
                cb.restore(snap, model, opt, cls_loss)
                del snap, batches
                bandit.update(look[1] - look[0])
                value = float(bandit.choose())
                mu, se = bandit.posterior()
                rec.update(probes=[0.0, 1.0], look=look, mu=mu, se=se)
        else:
            snap = cb.snapshot(model, opt, cls_loss)
            batches = cb.draw_batches(loaders[loader_key], BANDIT_LOOKAHEAD_BATCHES)
            probes = (0.0, 1.0) if mode == 'paired01' else bandit.probe_pair()
            look, look_all = [], []
            for v in probes:
                cb.restore(snap, model, opt, cls_loss)
                epoch_fn(ep, batches, v)
                sc, sa = look_score(model)
                look.append(sc)
                look_all.append(sa)
            cb.restore(snap, model, opt, cls_loss)
            del snap, batches
            if mode == 'paired01':
                bandit.update(look[1] - look[0])
                value = float(bandit.choose())
            else:
                bandit.update(probes[0], probes[1], look[0], look[1])
                value = bandit.choose()
            mu, se = bandit.posterior()
            rec.update(probes=list(probes), look=look, mu=mu, se=se)
            if scorer is not None:
                rec['look_all'] = look_all
        epoch_fn(ep, loaders[loader_key], value)
        scores = cb.holdout_scores(model, holdout, C, device)
        rec.update(value=value, **scores)
        if mode == 'plateau':
            est_hist.append(scores['est_soft'])
            i = len(est_hist) - 1
            if not switched and i >= 30:
                ma = lambda j: sum(est_hist[max(0, j - 9):j + 1]) / len(est_hist[max(0, j - 9):j + 1])
                gain = ma(i) - ma(i - 20)
                rec['plateau_gain'] = round(gain, 5)
                if gain < fixed_value:
                    switched = True
                    rec['switched_at'] = ep + 1
            value = 0.0 if switched else 1.0
        if mode == 'sflip':
            with torch.no_grad():
                cur_arg = cls_loss.partial_loss.confidence[train_idx_t].argmax(dim=1)
            if prev_arg is not None:
                flip = (cur_arg != prev_arg).float().mean().item()
                rec['s_flip'] = round(flip, 5)
                calm = calm + 1 if (ep >= 10 and flip < fixed_value) else 0
                if not switched and calm >= 3:
                    switched = True
                    rec['switched_at'] = ep + 1
            prev_arg = cur_arg
            value = 0.0 if switched else 1.0
        if diag is not None:
            rec.update(diag.state(model, cls_loss))
            rec.update(diag.grads(model, loss_at))
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            rec['test_acc'] = evaluate_model(model, loaders['test'], device)
        trace.append(rec)
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            recent = [t for t in trace[-report_every:] if 'value' in t]
            print(f'    [bandit] last {len(recent)} ep: mean value={sum(t["value"] for t in recent) / len(recent):.2f}  '
                  f'holdout est={scores["est_acc"]:.3f} true={scores["true_acc"]:.3f}', flush=True)
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    _write_trace(raw_cfg, 'bandit_trace', algorithm, C, trace)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


# name -> (framework, loss_kind, mode, fixed_value)
GENERIC_BANDIT_SPECS = {
    'ComCo-Uni-Fixed0': ('ComCo', 'unified', 'fixed', 0.0),
    'ComCo-Uni-Fixed1': ('ComCo', 'unified', 'fixed', 1.0),
    'ComCo-Uni-Bandit01': ('ComCo', 'unified', 'paired01', None),
    'ComCo-Uni-BanditCont': ('ComCo', 'unified', 'cont', None),
    'PiCO-Uni-Fixed0': ('PiCO', 'unified', 'fixed', 0.0),
    'PiCO-Uni-Fixed1': ('PiCO', 'unified', 'fixed', 1.0),
    'PiCO-Uni-Bandit01': ('PiCO', 'unified', 'paired01', None),
    'PiCO-Uni-BanditCont': ('PiCO', 'unified', 'cont', None),
    'PiCO-Mix-A100': ('PiCO', 'mix', 'fixed', 1.0),
    'PiCO-Mix-BanditPaired': ('PiCO', 'mix', 'paired01', None),
}
# Switch-cause analysis (2026-10-14): forced switch at epoch T, with and without s frozen before T.
for _fw in ('PiCO', 'ComCo'):
    for _kind, _tag in (('unified', 'Uni'), ('mix', 'Mix')):
        for _T in (25, 50, 100, 150):
            GENERIC_BANDIT_SPECS[f'{_fw}-{_tag}-Sched{_T}'] = (_fw, _kind, 'sched', _T)
        for _T in (50, 100):
            GENERIC_BANDIT_SPECS[f'{_fw}-{_tag}-Sched{_T}-Frozen'] = (_fw, _kind, 'sched_frozen', _T)
# Original {PartialLoss, ComCo-Fixed (SCL-NL)} pair through the generic runner on ComCo too, and the
# no-contrastive ('Solo') versions of both loss pairs (2026-10-14).
for _fw in ('PiCO', 'ComCo'):  # small-k fixes (2026-10-14)
    GENERIC_BANDIT_SPECS[f'{_fw}-Uni-BanditLong'] = (_fw, 'unified', 'paired01_long', None)
    GENERIC_BANDIT_SPECS[f'{_fw}-Uni-SFlip'] = (_fw, 'unified', 'sflip', 0.02)
    GENERIC_BANDIT_SPECS[f'{_fw}-Uni-Plateau'] = (_fw, 'unified', 'plateau', 0.01)
# PiCO+PartialLoss <-> ComCo+MCL-LOG framework switching (2026-10-14); value 0 = PiCO arm, 1 = ComCo arm.
GENERIC_BANDIT_SPECS['Hybrid-Bandit01'] = ('Hybrid', 'unified', 'paired01', None)
GENERIC_BANDIT_SPECS['Hybrid-BanditLong'] = ('Hybrid', 'unified', 'paired01_long', None)
GENERIC_BANDIT_SPECS['Hybrid-Plateau'] = ('Hybrid', 'unified', 'plateau', 0.01)
GENERIC_BANDIT_SPECS['Hybrid-FixedPiCO'] = ('Hybrid', 'unified', 'fixed', 0.0)
GENERIC_BANDIT_SPECS['Hybrid-FixedComCo'] = ('Hybrid', 'unified', 'fixed', 1.0)
GENERIC_BANDIT_SPECS['Hybrid-Sched150'] = ('Hybrid', 'unified', 'sched', 150)
# Same framework switch, but the ComCo arm uses SCL-NL (ComCo-Fixed) instead of MCL-LOG (2026-10-15).
GENERIC_BANDIT_SPECS['Hybrid-Mix-Bandit01'] = ('Hybrid', 'mix', 'paired01', None)
GENERIC_BANDIT_SPECS['Hybrid-Mix-FixedComCo'] = ('Hybrid', 'mix', 'fixed', 1.0)
GENERIC_BANDIT_SPECS['Hybrid-Mix-FixedPiCO'] = ('Hybrid', 'mix', 'fixed', 0.0)
GENERIC_BANDIT_SPECS['ComCo-Mix-A100'] = ('ComCo', 'mix', 'fixed', 1.0)
GENERIC_BANDIT_SPECS['ComCo-Mix-BanditPaired'] = ('ComCo', 'mix', 'paired01', None)
GENERIC_BANDIT_SPECS['Solo-Uni-Fixed0'] = ('Solo', 'unified', 'fixed', 0.0)
GENERIC_BANDIT_SPECS['Solo-Uni-Fixed1'] = ('Solo', 'unified', 'fixed', 1.0)
GENERIC_BANDIT_SPECS['Solo-Uni-Bandit01'] = ('Solo', 'unified', 'paired01', None)
GENERIC_BANDIT_SPECS['Solo-Uni-BanditCont'] = ('Solo', 'unified', 'cont', None)
GENERIC_BANDIT_SPECS['Solo-Mix-A100'] = ('Solo', 'mix', 'fixed', 1.0)
GENERIC_BANDIT_SPECS['Solo-Mix-BanditPaired'] = ('Solo', 'mix', 'paired01', None)
# Bias-aware lookahead reward (2026-10-20, src/bias_aware_reward.py): same bandit as
# ComCo-Mix-BanditPaired, only the score behind d changes.
GENERIC_BANDIT_REWARD = {}
for _suffix, _reward in (('RhoEM', 'rho_em'), ('RhoOracle', 'rho_oracle'), ('TrueAcc', 'true')):
    GENERIC_BANDIT_SPECS[f'ComCo-Mix-BanditPaired-{_suffix}'] = ('ComCo', 'mix', 'paired01', None)
    GENERIC_BANDIT_REWARD[f'ComCo-Mix-BanditPaired-{_suffix}'] = _reward
# Same bias-aware reward on the {PartialLoss, original MCL-LOG} pair (ComCo-Uni-Bandit01), 2026-10-28.
GENERIC_BANDIT_SPECS['ComCo-Uni-Bandit01-RhoEM'] = ('ComCo', 'unified', 'paired01', None)
GENERIC_BANDIT_REWARD['ComCo-Uni-Bandit01-RhoEM'] = 'rho_em'
# Paired bandit over two clsloss_swap losses on SwapComCo (2026-10-27): PL loss (value 0) vs MCL-LOG-Fixed (1).
SWAP_BANDIT_PAIRS = (('PiCOPartial', 'MCLLOGFixed'), ('PRODEN', 'MCLLOGFixed'))
for _a0, _a1 in SWAP_BANDIT_PAIRS:
    GENERIC_BANDIT_SPECS[f'SwapComCo-BanditPaired-{_a0}-{_a1}'] = ('SwapComCo', f'{_a0}+{_a1}', 'paired01', None)


def _make_generic_bandit_runner(name, spec):
    framework, loss_kind, mode, fixed_value = spec

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_generic_bandit(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device,
                                   tag, report_every, name, framework, loss_kind, mode, fixed_value,
                                   reward=GENERIC_BANDIT_REWARD.get(name, 'uniform'))
    return _runner


GENERIC_BANDIT_RUNNERS = {n: _make_generic_bandit_runner(n, sp) for n, sp in GENERIC_BANDIT_SPECS.items()}
