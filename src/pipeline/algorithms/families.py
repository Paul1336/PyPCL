"""Shared runner machinery for the PiCO and ComCo families (2026-11-22 refactor).

    pico_args / comco_args   the per-run hyper-parameter dicts (were hand-written 11 / 10 times)
    run_epochs               the outer epoch loop every family runner shares: per-epoch diagnostics
                             (checkpoint -> t-SNE -> concentration -> bias/variance; this order is
                             kept because concentration's DataLoader draws from the torch RNG),
                             ETA / gc, end-of-run kNN probe and final evaluation
    run_pico_family          PiCO, PiCO-Oracle(-Add), PiCO-MOCO, PiCO-Fixed (+ every init / EMA / A-B
                             sweep), PiCO-Weighted, PiCO-MCL(-Fixed), PiCO-SC
    run_comco_family         ComCo, ComCo-Fixed, ComCo-Weighted (fixed alpha or per-batch alpha_fn)
"""

import gc
import time
from dataclasses import dataclass

import torch

from src.comco.model import ComCoModel, ComCoWeightedModel
from src.comco.utils_loss import ComCoContrastiveLoss
from src.engine import evaluate_model
from src.losses.composite import ComCoWeightedClsLoss, PiCOWeightedClsLoss
from src.losses.confidence import PartialLoss, PiCOCLSLoss
from src.losses.stateless import ComCoCLSLoss, FixedComCoCLSLoss, PiCOMCLLoss
from src.pico.model import PiCOModel, PiCOOracleModel
from src.pico.utils_loss import SupConLoss
from src.pipeline import biasvariance, detail
from src.pipeline.algorithms.hparams import make_optimizer
from src.pll_init import candidate_masked_init
from src.train import comco as tcomco
from src.train import pico as tpico
from src.train.hooks import ComCoSelectionStats, OracleCorrectionStats, PiCOSelectionStats


# ─── argument dicts ──────────────────────────────────────────────────────────────────────────


def pico_args(C, epochs, pico_cfg, fixed=False):
    """fixed=True: paper-faithful warm-up length prot_start_fixed (default 1) instead of prot_start."""
    a = {
        'num_class': C, 'epochs': epochs, 'low_dim': pico_cfg['low_dim'], 'moco_queue': pico_cfg['moco_queue'],
        'moco_m': pico_cfg['moco_m'], 'proto_m': pico_cfg['proto_m'], 'prot_start': pico_cfg['prot_start'],
        'loss_weight': pico_cfg['loss_weight'], 'conf_ema_range': pico_cfg['conf_ema_range'],
    }
    if fixed:
        a['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
    return a


def comco_args(C, epochs, raw_cfg, borrow_pico=False):
    """borrow_pico=True adds the PiCO keys a confidence-buffer cls loss needs inside ComCo
    (proto_m, prot_start = prot_start_fixed, conf_ema_range)."""
    cc = raw_cfg['comco']
    a = {
        'num_class': C, 'epochs': epochs, 'low_dim': cc['low_dim'], 'moco_queue': cc['moco_queue'],
        'moco_m': cc['moco_m'], 'loss_weight': cc['loss_weight'], 'temperature': cc['temperature'],
        'top_k': cc['top_k'], 'warmup_neg': cc['warmup_neg'], 'warmup_pos': cc['warmup_pos'],
    }
    if borrow_pico:
        pc = raw_cfg['pico']
        a.update(proto_m=pc['proto_m'], prot_start=pc.get('prot_start_fixed', 1), conf_ema_range=pc['conf_ema_range'])
    return a


def candidate_init_conf(pl_ds, C, device):
    return candidate_masked_init(pl_ds.targets, C).to(device)


# ─── the shared outer loop ───────────────────────────────────────────────────────────────────


def fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}min'
    return f'{seconds / 3600:.2f}h'


def print_eta(tag: str, ep_done: int, ep_total: int, t_chunk: float, chunk_size: int):
    avg_s = t_chunk / chunk_size
    eta = avg_s * (ep_total - ep_done)
    print(f'  [{tag}]  ep {ep_done:>3}/{ep_total}  {avg_s:.1f}s/ep  ETA {fmt_eta(eta)}', flush=True)


def run_epochs(epoch_fn, model, *, loaders, pl_ds, orig_targets, C, raw_cfg, device, algorithm, epochs, tag,
               report_every, cls_loss=None, cont_loss=None, concentration=True, biasvar=True, knn=True):
    """Calls epoch_fn(ep) for every epoch, then the diagnostics hooks (each gated by its own CLI
    flag), and returns the final test accuracy."""
    bv_on = biasvar and detail.biasvariance_enabled(raw_cfg)
    if bv_on:
        eval_size = detail.biasvariance_cfg(raw_cfg).get('eval_size', 256)
        bv_images, bv_labels, bv_index = biasvariance.build_fixed_eval_batch(pl_ds, orig_targets, eval_size, device)
    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        epoch_fn(ep)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
        if concentration:
            detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, algorithm)
        if bv_on:
            detail.maybe_log_biasvariance(raw_cfg, model, cls_loss, cont_loss, algorithm,
                                          bv_images, bv_labels, bv_index, C, ep + 1, device)
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            print_eta(tag, ep + 1, epochs, time.perf_counter() - chunk_t0, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()
    if knn:
        detail.maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, loaders['test'], device, C, epochs, algorithm)
    acc = evaluate_model(model, loaders['test'], device)
    gc.collect()
    torch.cuda.empty_cache()
    return acc


# ─── PiCO family ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PiCOVariant:
    epoch: str                  # src.train.pico.SPECS key
    fixed_warmup: bool = True   # prot_start_fixed (paper) vs config prot_start (original PiCO family)
    init: str = 'candidate'     # 'candidate' | 'uniform_all' -- unless the caller passes init_conf
    loss: str = 'partial'       # 'partial' | 'weighted' | 'mcl' | 'sc'
    oracle: str = None          # None | 'remove' | 'add'
    biasvar: bool = True        # --biasvariance supported for this loss


PICO_VARIANTS = {
    'PiCO': PiCOVariant('PiCO', fixed_warmup=False, init='uniform_all'),
    'PiCO-MOCO': PiCOVariant('PiCO-MOCO', fixed_warmup=False, init='uniform_all'),
    'PiCO-Oracle': PiCOVariant('PiCO-Oracle', oracle='remove'),
    'PiCO-Oracle-Add': PiCOVariant('PiCO-Oracle-Add', oracle='add'),
    'PiCO-Fixed': PiCOVariant('PiCO-Fixed'),
    'PiCO-Weighted': PiCOVariant('PiCO-Weighted', loss='weighted'),
    'PiCO-MCL': PiCOVariant('PiCO-MCL', fixed_warmup=False, init=None, loss='mcl'),
    'PiCO-MCL-Fixed': PiCOVariant('PiCO-MCL-Fixed', init=None, loss='mcl'),
    'PiCO-SC': PiCOVariant('PiCO-SC', fixed_warmup=False, init=None, loss='sc', biasvar=False),
}


def run_pico_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag,
                    report_every, *, variant, algorithm, init_conf=None, conf_ema_range=None,
                    conf_source='prototype', conf_hard=True, alpha=None):
    v = PICO_VARIANTS[variant]
    pico_cfg = raw_cfg['pico']
    args = pico_args(C, epochs, pico_cfg, fixed=v.fixed_warmup)
    if conf_ema_range is not None:
        args['conf_ema_range'] = list(conf_ema_range)
    model = (PiCOOracleModel if v.oracle else PiCOModel)(args).to(device)
    if init_conf is None and v.init == 'uniform_all':
        init_conf = torch.ones(len(pl_ds), C).to(device) / C
    elif init_conf is None and v.init == 'candidate':
        init_conf = candidate_init_conf(pl_ds, C, device)
    cls_loss = {'partial': lambda: PartialLoss(init_conf),
                'weighted': lambda: PiCOWeightedClsLoss(init_conf, alpha),
                'mcl': lambda: PiCOMCLLoss(),
                'sc': lambda: PiCOCLSLoss(pl_ds.targets, C, conf_ema_range=tuple(pico_cfg['conf_ema_range']),
                                          epochs=epochs).to(device)}[v.loss]()
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    spec = tpico.SPECS[v.epoch]
    if v.epoch == 'PiCO-Fixed':
        spec = spec.with_(conf_hard=conf_hard)
    mask_transform = None
    if v.oracle:
        thr = pico_cfg.get('oracle_precision_threshold', 1.0)
        mask_transform = (tpico.OracleRemoveFP(thr) if v.oracle == 'remove'
                          else tpico.OracleAddTP(thr, pico_cfg.get('oracle_max_add_ratio', 1.0)))
    detail_on = detail.is_enabled(raw_cfg)

    def epoch_fn(ep):
        if v.loss == 'sc':
            cls_loss.set_conf_ema_m(ep)
        elif v.loss != 'mcl':
            cls_loss.set_conf_ema_m(ep, args)
        hooks = []
        if detail_on:
            hooks = [OracleCorrectionStats(raw_cfg, algorithm, C, v.oracle) if v.oracle
                     else PiCOSelectionStats(raw_cfg, algorithm, C)]
        tpico.pico_epoch(args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device, spec,
                         mask_transform=mask_transform, hooks=hooks,
                         conf_source=conf_source if v.epoch == 'PiCO-Fixed' else None)

    acc = run_epochs(epoch_fn, model, loaders=loaders, pl_ds=pl_ds, orig_targets=orig_targets, C=C,
                     raw_cfg=raw_cfg, device=device, algorithm=algorithm, epochs=epochs, tag=tag,
                     report_every=report_every, cls_loss=cls_loss, cont_loss=cont_loss, biasvar=v.biasvar)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


# ─── ComCo family ────────────────────────────────────────────────────────────────────────────


def run_comco_family(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag,
                     report_every, *, variant, algorithm, alpha=None, alpha_fn=None):
    """variant: 'ComCo' (original cls loss = scaled MCL-NL) | 'ComCo-Fixed' (SCL-NL) |
    'ComCo-Weighted' (alpha * SCL-NL + (1-alpha) * PartialLoss; per-batch alpha_fn -> switch epoch
    + alpha_trace JSON)."""
    weighted = variant == 'ComCo-Weighted'
    args = comco_args(C, epochs, raw_cfg, borrow_pico=weighted)
    init_conf = candidate_init_conf(pl_ds, C, device) if weighted else None
    model = (ComCoWeightedModel if weighted else ComCoModel)(args).to(device)
    if weighted:
        cls_loss = ComCoWeightedClsLoss(init_conf, alpha if alpha_fn is None else 0.0)
    else:
        cls_loss = ComCoCLSLoss() if variant == 'ComCo' else FixedComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=args['temperature'], top_k=args['top_k'])
    opt = make_optimizer(model, hparams)
    detail_on = detail.is_enabled(raw_cfg)
    spec = tcomco.SPECS['ComCo-Weighted-Switch' if alpha_fn is not None else
                        ('ComCo-Weighted' if weighted else 'ComCo')]
    alpha_per_epoch = []

    def epoch_fn(ep):
        if weighted:
            cls_loss.set_conf_ema_m(ep, args)
        hooks = [ComCoSelectionStats(raw_cfg, algorithm, C)] if (detail_on and not weighted) else []
        trace = None
        if alpha_fn is not None:
            alpha_per_epoch.append([])
            trace = alpha_per_epoch[-1]
        tcomco.comco_epoch(args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device, spec,
                           alpha_fn=alpha_fn, alpha_trace=trace, hooks=hooks)

    acc = run_epochs(epoch_fn, model, loaders=loaders, pl_ds=pl_ds, orig_targets=orig_targets, C=C,
                     raw_cfg=raw_cfg, device=device, algorithm=algorithm, epochs=epochs, tag=tag,
                     report_every=report_every, cls_loss=cls_loss, cont_loss=cont_loss,
                     biasvar=variant != 'ComCo')
    if alpha_fn is not None:
        from src.core.io import write_trace
        write_trace(raw_cfg, 'alpha_trace', algorithm, C, alpha_per_epoch)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc
