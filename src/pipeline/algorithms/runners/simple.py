"""Single-run runners: simple-shape losses, CPE, PRODEN, PiCO-CLS, SoLar, and the named PiCO / ComCo
variants (thin wrappers over families.run_pico_family / run_comco_family)."""

from ._base import *  # noqa: F401,F403
from ._base import _IndexedDataset, _biased_oracle_init_conf, _candidate_masked_init_conf, _print_eta, _uniform_all_init_conf  # noqa: F401


# ─── 'simple' shape: one loss_fn + standard loader + argmax eval ──────────


def _train_simple_shape(loss_fn, loader_key: str, loaders: dict, C: int, hparams: dict,
                         epochs: int, device, tag: str, report_every: int, raw_cfg: dict = None) -> float:
    spec = (raw_cfg or {}).get('_dataset_spec')
    model = create_model_for_spec(spec, C).to(device)
    opt = make_optimizer(model, hparams)
    last_accs = [0.0]

    for ep_start in range(0, epochs, report_every):
        chunk = min(report_every, epochs - ep_start)
        t0 = time.perf_counter()
        last_accs = train_algorithm(model, loaders[loader_key], loaders['test'], loss_fn, opt, chunk, device)
        elapsed = time.perf_counter() - t0
        _print_eta(tag, ep_start + chunk, epochs, elapsed, chunk)

    acc = last_accs[-1]
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def run_clpl(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _train_simple_shape(CLPLSquaredHingeLoss(), 'pl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


def run_wu(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _train_simple_shape(WuPLLLoss(), 'pl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


def run_mcl_log(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _train_simple_shape(MCL_LOG(num_classes=C), 'cl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


def run_mcl_log_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    # See docs/mcl_explanation.md: corrects the unbiased-risk-estimator scaling
    # factor to 2*(C-1)/m (Feng et al. 2020, Eq. 12), vs. the original's (C-1)/(C-m).
    return _train_simple_shape(FixedMCLLog(num_classes=C), 'cl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


def run_scl_nl(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _train_simple_shape(SCL_NL(), 'cl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


def run_op(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _train_simple_shape(OPLoss(), 'cl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


def run_op_w(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _train_simple_shape(OPWLoss(), 'cl', loaders, C, hparams, epochs, device, tag, report_every, raw_cfg)


# ─── CPE: same 'simple' shape but argmin inference (predicts P(ybar|x)) ────


@torch.no_grad()
def _evaluate_argmin(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmin(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run_cpe(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    model = create_model_for_spec((raw_cfg or {}).get('_dataset_spec'), C).to(device)
    opt = make_optimizer(model, hparams)
    loss_fn = CPELoss()

    final_acc = 0.0
    for ep_start in range(0, epochs, report_every):
        chunk = min(report_every, epochs - ep_start)
        t0 = time.perf_counter()
        for _ in range(chunk):
            model.train()
            for imgs, cl_labels in loaders['cl']:
                imgs, cl_labels = imgs.to(device), cl_labels.to(device)
                opt.zero_grad()
                loss_fn(model(imgs), cl_labels).backward()
                opt.step()
        elapsed = time.perf_counter() - t0
        final_acc = _evaluate_argmin(model, loaders['test'], device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep_start + chunk, 'CPE',
                                     predict='argmin')
        _print_eta(tag, ep_start + chunk, epochs, elapsed, chunk)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


# ─── PRODEN: index-based loader + cross-epoch confidence accumulation ─────


def _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                         device, tag, report_every, algorithm: str, init_mode: str, true_weight: float = None,
                         wf: int = None, conf_ema_range=None):
    """Shared body for PRODEN and its confidence-init ablations
    (PRODEN-UniformInit, PRODEN-BiasedInit, and the parametrized
    PRODEN-Biased{Cand,All}-W* / PRODEN-BiasedRand-W*-Wf* sweeps). PRODEN's
    own every-batch renormalize-to-candidate-set update (ProdenLoss.forward)
    is completely unchanged across all of them -- only the INITIAL
    confidence matrix differs (init_mode, see src/pll_init.py), which is
    deliberate: PRODEN's fast renormalization overwrites the initial
    distribution almost immediately, unlike PiCO's slow EMA, so this
    variant set is meant to demonstrate that contrast rather than hide it
    behind a slower update rule.

    conf_ema_range (optional [start, end]): the one exception to "update
    rule unchanged" -- the conf_ema sweep (CONF_EMA_SWEEP_RUNNERS) passes a
    PiCO-style EMA momentum schedule so PRODEN's confidence update keeps a
    memory of its previous value instead of hard-overwriting. None keeps
    the original PRODEN (conf_ema_m == 0)."""
    spec = (raw_cfg or {}).get('_dataset_spec')
    model = create_model_for_spec(spec, C).to(device)
    loss_fn = ProdenLoss(pl_ds.targets, C, init_mode=init_mode, orig_targets=orig_targets,
                          true_weight=true_weight, wf=wf, conf_ema_range=conf_ema_range).to(device)
    opt = make_optimizer(model, hparams)

    idx_ds = (_IndexedDataset(pl_ds.data, image_size=spec.image_size, mean=spec.mean, std=spec.std,
                               modality=spec.modality)
              if spec is not None else _IndexedDataset(pl_ds.data))
    if 'train_idx' in loaders:  # --holdout_frac: never train on the held-out split (indices stay global)
        idx_ds = Subset(idx_ds, [int(i) for i in loaders['train_idx']])
    idx_loader = DataLoader(idx_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    chunk_t0 = time.perf_counter()
    final_acc = 0.0
    for ep in range(epochs):
        loss_fn.set_conf_ema_m(ep, epochs)   # no-op (stays 0) unless conf_ema_range was given
        model.train()
        for imgs, indices in idx_loader:
            imgs, indices = imgs.to(device), indices.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), indices).backward()
            opt.step()

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
        detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, algorithm)

        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            final_acc = evaluate_model(model, loaders['test'], device)
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    del model, loss_fn, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


def run_proden(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                device, tag, report_every, 'PRODEN', init_mode='candidate_masked')


def run_proden_uniform_init(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                device, tag, report_every, 'PRODEN-UniformInit', init_mode='uniform_all')


def run_proden_biased_init(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                device, tag, report_every, 'PRODEN-BiasedInit', init_mode='biased_oracle')


def run_proden_random_cand_init(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """Ablation: initial confidence puts ~100% on ONE candidate chosen
    uniformly at random from each sample's own partial-label set -- NOT
    necessarily the true class -- with a tiny epsilon floor on the rest of
    the candidate set so PRODEN's candidate_mask still covers all of it from
    the start (see src/pll_init.py.random_candidate_init's docstring for why
    an exact one-hot init would permanently lock the mask to a single,
    often-wrong class). Tests PRODEN's fast-renormalization recovery under
    the most extreme initial concentration on a single (usually wrong)
    class."""
    return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                device, tag, report_every, 'PRODEN-RandomCandInit', init_mode='random_candidate')


# ─── PiCO family: dual-encoder + MoCo queue + prototypes ──────────────────


def run_pico(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO', algorithm='PiCO')


def run_pico_oracle(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO-Oracle', algorithm='PiCO-Oracle')


def run_pico_oracle_add(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO-Oracle-Add', algorithm='PiCO-Oracle-Add')


def run_pico_moco(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO-MOCO', algorithm='PiCO-MOCO')


def _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, algorithm: str, init_conf: torch.Tensor,
                             conf_ema_range=None, conf_source: str = 'prototype', conf_hard: bool = True):
    """PiCO-Fixed core shared by every init / conf-EMA / A-B sweep variant (families.run_pico_family)."""
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, variant='PiCO-Fixed',
                           algorithm=algorithm, init_conf=init_conf, conf_ema_range=conf_ema_range,
                           conf_source=conf_source, conf_hard=conf_hard)


def run_pico_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, 'PiCO-Fixed', init_conf)


def run_pico_fixed_uniform_init(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """Ablation: PiCO-Fixed's warm-up config, but with the initial confidence
    spread uniformly over ALL C classes (plain PiCO's init) instead of
    candidate-masked -- isolates whether PiCO's small-k advantage depends on
    the initial confidence distribution's shape."""
    init_conf = _uniform_all_init_conf(pl_ds, C, device)
    return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, 'PiCO-Fixed-UniformInit', init_conf)


def run_pico_fixed_biased_init(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """Ablation: PiCO-Fixed's warm-up config, but the initial confidence puts
    0.2 on the true class and 0.8 on one other (fixed-seed-random) candidate
    -- tests whether it's the *concentration* of initial confidence (even if
    partly misplaced) rather than its correctness that drives PiCO's
    small-k advantage. Uses plain PiCOModel (not PiCOOracleModel): true
    labels are consulted only once, at init-conf construction time, not
    during the training loop itself."""
    init_conf = _biased_oracle_init_conf(pl_ds, orig_targets, C, device)
    return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, 'PiCO-Fixed-BiasedInit', init_conf)


# ─── PiCO-weighted-cls-loss: alpha * PiCO-Fixed's PartialLoss + ──────────
#     (1 - alpha) * PiCOMCLLoss ────────────────────────────────────────────
#
# Same warm-up config/init as PiCO-Fixed (paper-faithful: L_cont omitted
# entirely during warm-up, candidate-masked initial confidence), only the
# cls loss is swapped for a PiCOWeightedClsLoss blending PartialLoss and
# PiCOMCLLoss by alpha. See src/pico/weighted_cls_loss.py and
# src/fixed_pico_engine.py's train_pico_weighted_epoch. Registered as a
# parametrized sweep over src/pll_init.py's ALPHA_VALUES, same pattern as
# BIASED_SWEEP_RUNNERS above.


def _run_pico_weighted_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, algorithm: str, alpha: float):
    """alpha * PartialLoss + (1 - alpha) * PiCO-MCL on PiCO-Fixed (families.run_pico_family)."""
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, variant='PiCO-Weighted',
                           algorithm=algorithm, alpha=alpha)


# ─── ComCo-weighted-cls-loss: alpha * ComCo-Fixed's own cls loss + ────────
#     (1 - alpha) * PartialLoss (borrowed onto ComCo's architecture) ───────
#
# ComCo-side mirror of PICO_WEIGHTED_SWEEP_RUNNERS above. ComCoWeightedModel
# adds the per-class prototype buffer PartialLoss needs (native ComCo-Fixed
# has none); see src/comco/model.py, src/comco/weighted_cls_loss.py,
# src/fixed_comco_engine.py::train_comco_weighted_epoch. prot_start/proto_m
# are borrowed from raw_cfg['pico'] (same "Fixed"-family defaults already
# used by PiCO-Fixed/PiCO-Weighted) rather than duplicated into config.yaml's
# 'comco' section, since they govern the same PartialLoss/prototype
# mechanism either way.


def _run_comco_weighted_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, algorithm: str, alpha: float):
    """alpha * SCL-NL + (1 - alpha) * PartialLoss on ComCo (families.run_comco_family)."""
    return run_comco_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, variant='ComCo-Weighted',
                            algorithm=algorithm, alpha=alpha)


# ─── ComCo-Weighted, per-batch alpha switching (variable-average-k experiment) ──
#
# Same architecture/losses as _run_comco_weighted_variant above, except alpha
# is picked fresh each batch by alpha_fn instead of held fixed for the whole
# run -- see train_comco_weighted_switch_epoch / batch_cond_alpha /
# batch_random_alpha (src/fixed_comco_engine.py). Meant to be trained under
# --only_q (variable per-sample candidate-set size, see src/pll_init.py's
# only_q derivation for a target mean k), where a batch's own realized
# average candidate-set size fluctuates around the dataset's target mean.


def _run_comco_weighted_switch_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, algorithm: str, alpha_fn):
    """ComCo-Weighted with per-batch alpha = alpha_fn(Y); writes alpha_trace JSON (families.run_comco_family)."""
    return run_comco_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every, variant='ComCo-Weighted',
                            algorithm=algorithm, alpha_fn=alpha_fn)


def run_comco_weighted_samplecond(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device,
                                   tag, report_every):
    return _run_comco_weighted_switch_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size,
                                               epochs, device, tag, report_every,
                                               'ComCo-Weighted-SampleCond', sample_cond_alpha)


def run_comco_weighted_batchcond(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device,
                                  tag, report_every):
    return _run_comco_weighted_switch_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size,
                                               epochs, device, tag, report_every,
                                               'ComCo-Weighted-BatchCond', batch_cond_alpha)


def run_comco_weighted_batchrandom(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device,
                                    tag, report_every):
    return _run_comco_weighted_switch_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size,
                                               epochs, device, tag, report_every,
                                               'ComCo-Weighted-BatchRandom', batch_random_alpha)


def run_pico_mcl(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO-MCL', algorithm='PiCO-MCL')


def run_pico_mcl_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO-MCL-Fixed', algorithm='PiCO-MCL-Fixed')


def run_pico_sc(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='PiCO-SC', algorithm='PiCO-SC')


def run_pico_cls(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """Standalone PiCO-CLS: PartialLoss-style cls loss without the contrastive
    branch, confidence driven by the model's own softmax (no dual encoder)."""
    spec = (raw_cfg or {}).get('_dataset_spec')
    model = create_model_for_spec(spec, C).to(device)
    loss_fn = PiCOCLSLoss(pl_ds.targets, C, epochs=epochs).to(device)
    opt = make_optimizer(model, hparams)

    idx_ds = (_IndexedDataset(pl_ds.data, image_size=spec.image_size, mean=spec.mean, std=spec.std,
                               modality=spec.modality)
              if spec is not None else _IndexedDataset(pl_ds.data))
    if 'train_idx' in loaders:  # --holdout_frac: never train on the held-out split (2026-11-22 fix)
        idx_ds = Subset(idx_ds, [int(i) for i in loaders['train_idx']])
    idx_loader = DataLoader(idx_ds, batch_size=batch_size,
                             shuffle=True, num_workers=2, drop_last=True)

    chunk_t0 = time.perf_counter()
    final_acc = 0.0
    for ep in range(epochs):
        loss_fn.set_conf_ema_m(ep)
        model.train()
        for imgs, indices in idx_loader:
            imgs, indices = imgs.to(device), indices.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, indices)
            loss.backward()
            opt.step()
            loss_fn.update_confidence(out.detach(), indices)

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-CLS')

        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            final_acc = evaluate_model(model, loaders['test'], device)
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    del model, loss_fn, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


# ─── ComCo: contrastive complementary-label learning ──────────────────────


def run_comco(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_comco_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='ComCo', algorithm='ComCo')


def run_comco_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    return run_comco_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every,
                    variant='ComCo-Fixed', algorithm='ComCo-Fixed')


# ─── SoLar: two-stage (pre-estimation + Sinkhorn-Knopp final training) ────


def _setup_solar(args, train_config, solar_config, solar_train_dataset, device):
    """SoLar model / loss / optimizer / queue (moved from the old-workflow src/model_setup.py)."""
    import torch.optim as optim
    from src.losses.confidence import SolarPartialLoss
    from src.models import create_model
    solar_args = {
        'num_class': train_config['num_classes'], 'epochs': args.epochs, 'warmup_epoch': solar_config['warmup_epoch'],
        'rho_range': solar_config['rho_range'], 'lamd': solar_config['lamd'], 'eta': solar_config['eta'],
        'tau': solar_config['tau'], 'est_epochs': solar_config['est_epochs'], 'gamma1': solar_config['gamma1'],
        'gamma2': solar_config['gamma2']
    }
    model = create_model(train_config['num_classes']).to(device)
    num_classes = train_config['num_classes']
    solar_given_label_matrix = torch.zeros(len(solar_train_dataset), num_classes)
    for i, p_label in enumerate(solar_train_dataset.given_label_matrix_sparse):
        solar_given_label_matrix[i, p_label] = 1.0
    loss_fn = SolarPartialLoss(solar_given_label_matrix, device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    queue = torch.zeros(64 * args.batch_size, train_config['num_classes']).to(device)
    return model, loss_fn, optimizer, solar_args, queue


def run_solar(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    solar_cfg = raw_cfg['solar']
    solar_ds = SoLarDataset(pl_ds, orig_targets)
    train_ds = (Subset(solar_ds, [int(i) for i in loaders['train_idx']])  # holdout fix (2026-11-22)
                if 'train_idx' in loaders else solar_ds)
    solar_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, collate_fn=solar_collate_fn, pin_memory=True)

    fake_args = SimpleNamespace(lr=hparams['lr'], momentum=hparams.get('momentum', 0.9),
                                 weight_decay=hparams['weight_decay'], epochs=epochs, batch_size=batch_size)
    train_config = {'num_classes': C}

    model, loss_fn, opt, solar_args, queue = _setup_solar(fake_args, train_config, solar_cfg, solar_ds, device)

    total_epochs = solar_cfg['est_epochs'] + epochs
    print(f'  [{tag}]  SoLar: est={solar_cfg["est_epochs"]} + train={epochs} = {total_epochs} total epochs',
          flush=True)

    accuracies = train_solar(solar_args, model, solar_loader, loaders['test'], loss_fn, opt, device, queue)
    final_acc = accuracies[-1] if accuracies else 0.0

    del model, loss_fn, opt, queue, solar_ds, solar_loader
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc
