"""Training 'shapes' for the 14 PLL/CLL algorithms.

These are relocated (not reinvented) from the training loops already
validated in scripts/run_adam_comparison.py, scripts/run_op_cpe_comparison.py,
scripts/run_pico_cls_comparison.py and scripts/run_solar_comparison.py.
Every function below builds on the shared training loops in src/engine.py.

Every run_* function shares one signature (see algorithms/__init__.py):
    (loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size,
     epochs, device, tag, report_every) -> final_accuracy: float
"""

import gc
import time
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.collate import solar_collate_fn
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss
from src.comco.fixed_utils_loss import FixedComCoCLSLoss
from src.clpl_loss import CLPLSquaredHingeLoss
from src.cpe_loss import CPELoss
from src.data_utils import SoLarDataset
from src.engine import (evaluate_model, train_algorithm, train_comco_epoch,
                         train_pico_epoch, train_pico_mclloss_epoch,
                         train_pico_moco_epoch, train_pico_sc_epoch, train_solar)
from src.fixed_pico_engine import train_pico_epoch_fixed, train_pico_mcl_epoch_fixed
from src.oracle_pico_engine import train_pico_oracle_add_graded_epoch, train_pico_oracle_graded_epoch
from src.mcl_losses import MCL_LOG
from src.fixed_mcl_losses import FixedMCLLog
from src.model_setup import setup_solar
from src.models import create_model_for_spec
from src.op_loss import OPLoss, OPWLoss
from src.pico.mcl_cls_loss import PiCOMCLLoss
from src.pico.model import PiCOModel, PiCOOracleModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.pico_cls_loss import PiCOCLSLoss
from src.proden_loss import ProdenLoss
from src.scl_loss import SCL_NL
from src.wu_loss import WuPLLLoss

from src.pipeline import detail

from .hparams import make_optimizer

_MEAN = [0.4914, 0.4822, 0.4465]
_STD = [0.247, 0.2435, 0.2616]

# ─── shared helpers ────────────────────────────────────────────────────────


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}min'
    return f'{seconds / 3600:.2f}h'


def _print_eta(tag: str, ep_done: int, ep_total: int, t_chunk: float, chunk_size: int):
    avg_s = t_chunk / chunk_size
    eta = avg_s * (ep_total - ep_done)
    print(f'  [{tag}]  ep {ep_done:>3}/{ep_total}  {avg_s:.1f}s/ep  ETA {_fmt_eta(eta)}', flush=True)


class _IndexedDataset(Dataset):
    """Wraps raw uint8 image array (or, for tabular datasets, raw feature
    vectors); returns (img_or_features, index) for index-based losses
    (PRODEN, PiCO-CLS).

    image_size/mean/std default to the original CIFAR values so existing call
    sites (no args passed) are unaffected. Previously cached the transform as
    a class-level singleton; now built per-instance so different datasets
    (different image_size/mean/std) don't share a stale cached transform.

    modality='tabular' skips the image transform pipeline entirely and just
    returns the raw feature vector as a float tensor -- Image.fromarray()
    doesn't apply to a feature vector."""

    def __init__(self, data, image_size=32, mean=_MEAN, std=_STD, modality='image'):
        self.modality = modality
        self.data = data
        if modality == 'image':
            self._train_tf = transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.modality == 'tabular':
            return torch.as_tensor(self.data[idx], dtype=torch.float32), idx
        img = Image.fromarray(self.data[idx])
        return self._train_tf(img), idx


def _pico_args(C: int, epochs: int, pico_cfg: dict) -> dict:
    return {
        'num_class': C,
        'epochs': epochs,
        'low_dim': pico_cfg['low_dim'],
        'moco_queue': pico_cfg['moco_queue'],
        'moco_m': pico_cfg['moco_m'],
        'proto_m': pico_cfg['proto_m'],
        'prot_start': pico_cfg['prot_start'],
        'loss_weight': pico_cfg['loss_weight'],
        'conf_ema_range': pico_cfg['conf_ema_range'],
    }


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
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    model = PiCOModel(pico_args).to(device)
    init_conf = torch.ones(len(pl_ds), C).to(device) / C
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    detail_on = detail.is_enabled(raw_cfg)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        if detail_on:
            # Instrumented copy of train_pico_epoch that also measures, per
            # BATCH, contrastive positive/negative pair-selection precision
            # against ground truth -- see src/pipeline/detail.py. Logs to
            # pico_selection_stats.csv internally (buffered, flushed once
            # per epoch).
            detail.train_pico_epoch_with_selection_stats(
                pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device,
                raw_cfg, 'PiCO', C)
        else:
            train_pico_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device)

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO')

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


def run_pico_oracle(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """Graduated precision-controlled contrastive-pair-selection oracle,
    aligned with PiCO-Fixed (not plain PiCO): candidate-set-masked
    pseudo-target init (Eq. 6, same as run_pico_fixed), prot_start_fixed
    warm-up length, and L_cont omitted entirely (not switched to
    unsupervised MoCo) during warm-up.

    Post-warmup, the contrastive mask is PiCO-Fixed's own natural
    pseudo-label-driven mask, corrected up to config.yaml's
    pico.oracle_precision_threshold (default 1.0) if it falls short -- see
    src/oracle_pico_engine.py::train_pico_oracle_graded_epoch for exactly
    how the correction works. threshold=0 reproduces PiCO-Fixed exactly;
    threshold=1 removes every false positive from the model's own selected
    set (not identical to the old ground-truth-mask oracle -- see that
    function's docstring). Not for real use: even at low thresholds this
    still requires true labels at train time, unavailable in a genuine
    partial-label setting."""
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    pico_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
    precision_threshold = pico_cfg.get('oracle_precision_threshold', 1.0)
    model = PiCOOracleModel(pico_args).to(device)
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    detail_on = detail.is_enabled(raw_cfg)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        if detail_on:
            detail.train_pico_oracle_graded_epoch_with_stats(
                pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device,
                raw_cfg, 'PiCO-Oracle', C, precision_threshold)
        else:
            train_pico_oracle_graded_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device,
                                            precision_threshold)

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-Oracle')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-Oracle')

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


def run_pico_oracle_add(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """Additive counterpart to run_pico_oracle: instead of REMOVING false
    positives from the natural mask to reach config.yaml's
    pico.oracle_precision_threshold, ADDS randomly-chosen genuine true
    positives (pairs the natural mask didn't select but which do share the
    true class) until the threshold is reached, capped at
    pico.oracle_max_add_ratio (default 1.0, i.e. at most doubling the
    natural positive-set size per batch) to avoid unboundedly large
    injections at high thresholds -- see
    src/oracle_pico_engine.py::train_pico_oracle_add_graded_epoch. Otherwise
    identical setup to run_pico_oracle (PiCO-Fixed-aligned warm-up,
    candidate-masked init). Not for real use: requires true labels at train
    time, unavailable in a genuine partial-label setting."""
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    pico_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
    precision_threshold = pico_cfg.get('oracle_precision_threshold', 1.0)
    max_add_ratio = pico_cfg.get('oracle_max_add_ratio', 1.0)
    model = PiCOOracleModel(pico_args).to(device)
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    detail_on = detail.is_enabled(raw_cfg)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        if detail_on:
            detail.train_pico_oracle_add_graded_epoch_with_stats(
                pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device,
                raw_cfg, 'PiCO-Oracle-Add', C, precision_threshold, max_add_ratio)
        else:
            train_pico_oracle_add_graded_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep,
                                                device, precision_threshold, max_add_ratio)

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-Oracle-Add')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-Oracle-Add')
        detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, 'PiCO-Oracle-Add')

        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    detail.maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, loaders['test'], device, C, epochs,
                               'PiCO-Oracle-Add')
    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def run_pico_moco(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """PiCO-MOCO: identical to run_pico except the contrastive positive pair
    is always the sample's own weak/strong augmentation pair, even after
    warm-up ends -- the candidate-set/prototype-pseudo-label-driven SupCon
    mask (mask = same predicted class in batch+queue) is never built. See
    src/engine.py::train_pico_moco_epoch."""
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    model = PiCOModel(pico_args).to(device)
    init_conf = torch.ones(len(pl_ds), C).to(device) / C
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        train_pico_moco_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device)

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-MOCO')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-MOCO')

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


def _candidate_masked_init_conf(pl_ds, C: int, device) -> torch.Tensor:
    """Paper-faithful pseudo-target initialization (PiCO Eq. 6): s_j =
    1/|Y| * I(j in Y) -- uniform WITHIN the candidate set only, zero outside
    it. See docs/pico_explanation.md's newly-found discrepancy: run_pico's
    `torch.ones(N, C) / C` is uniform over ALL C classes regardless of the
    candidate set, which the paper's own text ("we first initialize the
    pseudo targets with a uniform distribution, s_j = 1/|Y| * I(j in Y)")
    does not support -- it gives equal initial weight even to classes
    provably not the true label, diluting the classification loss's signal
    for the whole warm-up period. Same construction as ProdenLoss.__init__
    (src/proden_loss.py), which already does this correctly."""
    from src.pll_init import candidate_masked_init
    return candidate_masked_init(pl_ds.targets, C).to(device)


def _uniform_all_init_conf(pl_ds, C: int, device) -> torch.Tensor:
    """Uniform over ALL C classes regardless of candidate-set membership --
    same as plain run_pico's init, but usable with PiCO-Fixed's warm-up
    config (see PiCO-Fixed-UniformInit) to isolate the effect of the initial
    confidence *shape* from the warm-up-length/L_cont-omission fix."""
    from src.pll_init import uniform_all_init
    return uniform_all_init(len(pl_ds), C).to(device)


def _biased_oracle_init_conf(pl_ds, orig_targets, C: int, device) -> torch.Tensor:
    """True class gets 0.2, one (fixed-seed-random, decided once) other
    candidate from the sample's own partial-label set gets 0.8, everything
    else 0 -- see PiCO-Fixed-BiasedInit and src/pll_init.py.biased_oracle_init."""
    from src.pll_init import biased_oracle_init
    return biased_oracle_init(pl_ds.targets, orig_targets, C).to(device)


def _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                             device, tag, report_every, algorithm: str, init_conf: torch.Tensor,
                             conf_ema_range=None, conf_source: str = 'prototype', conf_hard: bool = True):
    """Shared body for PiCO-Fixed and its confidence-init ablations
    (PiCO-Fixed-UniformInit, PiCO-Fixed-BiasedInit): identical warm-up config
    (prot_start_fixed, L_cont omitted entirely during warm-up per
    docs/pico_explanation.md) and training loop; only the initial confidence
    matrix and the `algorithm` label (used for detail-log output paths)
    differ between the three callers below.

    conf_ema_range (optional [start, end]): overrides config.yaml's
    pico.conf_ema_range for PartialLoss.set_conf_ema_m -- used by the
    conf_ema sweep (CONF_EMA_SWEEP_RUNNERS); [0, 0] degenerates the EMA
    confidence update into a hard overwrite (PRODEN-like).

    conf_source / conf_hard: forwarded to train_pico_epoch_fixed -- Factor A
    / Factor B of the 2026-09-15 confidence-update-mechanism ablation (see
    scripts/run_ab_sweep.py, AB_SWEEP_RUNNERS below). Both default to
    PiCO-Fixed's native behaviour (prototype-sourced, hard one-hot), so
    every existing caller is unaffected."""
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    pico_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
    if conf_ema_range is not None:
        pico_args['conf_ema_range'] = list(conf_ema_range)
    model = PiCOModel(pico_args).to(device)
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    detail_on = detail.is_enabled(raw_cfg)
    if detail_on and (conf_source != 'prototype' or not conf_hard):
        raise NotImplementedError(
            '--detail (per-batch selection stats) is not wired up for the conf_source/conf_hard '
            'ablation -- detail.train_pico_epoch_fixed_with_selection_stats always uses the native '
            'prototype-sourced hard update. Run the AB sweep without --detail.')

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        if detail_on:
            # See detail.train_pico_epoch_fixed_with_selection_stats: previously
            # this branch didn't exist, so PiCO-Fixed never wrote
            # pico_selection_stats.csv regardless of --detail.
            detail.train_pico_epoch_fixed_with_selection_stats(
                pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device,
                raw_cfg, algorithm, C)
        else:
            train_pico_epoch_fixed(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device,
                                    conf_source=conf_source, conf_hard=conf_hard)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
        detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, algorithm)
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    detail.maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, loaders['test'], device, C, epochs, algorithm)
    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


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


# ─── Parametrized biased-init sweep: true_weight x {candidates, all classes} ─
#
# Generalizes run_pico_fixed_biased_init / run_proden_biased_init's fixed
# 20%/80%-one-random-candidate scheme into a sweep over (a) how much weight
# the true class gets and (b) whether the remainder is spread across the
# sample's own candidate set (biased_candidates_init) or across every class
# (biased_all_init) -- see src/pll_init.py. Registers one runner per
# (base_algorithm, strategy, true_weight) combination under a factory rather
# than writing 20 near-identical functions by hand; algorithms/__init__.py
# and hparams.py merge BIASED_SWEEP_RUNNERS into the normal registry.


def _make_pico_fixed_biased_runner(strategy: str, true_weight: float):
    """strategy: 'cand' -> biased_candidates_init (remainder spread across
    the sample's own candidates only) | 'all' -> biased_all_init (remainder
    spread across every class)."""
    from src.pll_init import biased_all_init, biased_candidates_init, biased_variant_name
    algorithm = biased_variant_name('PiCO-Fixed', strategy, true_weight)
    init_fn = biased_candidates_init if strategy == 'cand' else biased_all_init

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        init_conf = init_fn(pl_ds.targets, orig_targets, C, true_weight).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf)

    return algorithm, _runner


def _make_proden_biased_runner(strategy: str, true_weight: float):
    from src.pll_init import biased_variant_name
    algorithm = biased_variant_name('PRODEN', strategy, true_weight)
    init_mode = 'biased_candidates' if strategy == 'cand' else 'biased_all'

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, init_mode=init_mode,
                                    true_weight=true_weight)

    return algorithm, _runner


def _build_biased_sweep_runners() -> dict:
    from src.pll_init import BIAS_WEIGHTS
    runners = {}
    for strategy in ('cand', 'all'):
        for w in BIAS_WEIGHTS:
            name, fn = _make_pico_fixed_biased_runner(strategy, w)
            runners[name] = fn
            name, fn = _make_proden_biased_runner(strategy, w)
            runners[name] = fn
    return runners


BIASED_SWEEP_RUNNERS = _build_biased_sweep_runners()


# ─── Parametrized biased-init sweep #2: true_weight x wf (how many other ──
#     candidates the remainder is spread across, chosen at random)
#
# Interpolates between biased_oracle_init (wf=1) and biased_candidates_init
# (wf = every other candidate) -- see biased_partial_random_init in
# src/pll_init.py. Same factory pattern as BIASED_SWEEP_RUNNERS above.


def _make_pico_fixed_biased_rand_runner(true_weight: float, wf: int):
    from src.pll_init import biased_partial_random_init, biased_rand_variant_name
    algorithm = biased_rand_variant_name('PiCO-Fixed', true_weight, wf)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        init_conf = biased_partial_random_init(pl_ds.targets, orig_targets, C, true_weight, wf).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf)

    return algorithm, _runner


def _make_proden_biased_rand_runner(true_weight: float, wf: int):
    from src.pll_init import biased_rand_variant_name
    algorithm = biased_rand_variant_name('PRODEN', true_weight, wf)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, init_mode='biased_partial_random',
                                    true_weight=true_weight, wf=wf)

    return algorithm, _runner


def _build_biased_rand_sweep_runners() -> dict:
    from src.pll_init import BIAS_RAND_WEIGHTS, BIAS_RAND_WF_VALUES
    runners = {}
    for w in BIAS_RAND_WEIGHTS:
        for wf in BIAS_RAND_WF_VALUES:
            name, fn = _make_pico_fixed_biased_rand_runner(w, wf)
            runners[name] = fn
            name, fn = _make_proden_biased_rand_runner(w, wf)
            runners[name] = fn
    return runners


BIASED_RAND_SWEEP_RUNNERS = _build_biased_rand_sweep_runners()


# ─── Parametrized biased-init sweep #3: true_weight x n, random pool drawn ─
#     from ALL other classes (not just the sample's own candidate set)
#
# Originally PRODEN-only (requested for a PRODEN true_weight=20% sweep over
# n=4/9/14/19 at C=20 k=5, where k-1=4 other candidates was too small a
# pool for n>4); the PiCO-Fixed counterpart was added 2026-09-15 so the
# conf_ema sweep below can run the same n-sweep on both algorithms -- see
# src/pll_init.py.biased_random_all_init / BIAS_RAND_ALL_WEIGHTS /
# BIAS_RAND_ALL_N_VALUES. Same factory pattern as BIASED_RAND_SWEEP_RUNNERS.


def _make_pico_fixed_biased_rand_all_runner(true_weight: float, n: int):
    from src.pll_init import biased_rand_all_variant_name, biased_random_all_init
    algorithm = biased_rand_all_variant_name('PiCO-Fixed', true_weight, n)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        init_conf = biased_random_all_init(orig_targets, C, true_weight, n).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf)

    return algorithm, _runner


def _make_proden_biased_rand_all_runner(true_weight: float, n: int):
    from src.pll_init import biased_rand_all_variant_name
    algorithm = biased_rand_all_variant_name('PRODEN', true_weight, n)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, init_mode='biased_random_all',
                                    true_weight=true_weight, wf=n)

    return algorithm, _runner


def _build_biased_rand_all_sweep_runners() -> dict:
    from src.pll_init import BIAS_RAND_ALL_N_VALUES, BIAS_RAND_ALL_WEIGHTS
    runners = {}
    for w in BIAS_RAND_ALL_WEIGHTS:
        for n in BIAS_RAND_ALL_N_VALUES:
            name, fn = _make_pico_fixed_biased_rand_all_runner(w, n)
            runners[name] = fn
            name, fn = _make_proden_biased_rand_all_runner(w, n)
            runners[name] = fn
    return runners


BIASED_RAND_ALL_SWEEP_RUNNERS = _build_biased_rand_all_sweep_runners()


# ─── conf_ema_m sweep (2026-09-15): {init variant} x {PiCO-Fixed, PRODEN} ──
#     x CONF_EMA_SCALES
#
# Re-runs the TC-PLS W sweep and the TC-n-PLS n sweep on BOTH algorithms
# under five confidence-update EMA momentum levels, from PiCO's original
# schedule (scale 1.0) down to a hard overwrite every update (scale 0.0 ==
# original PRODEN) -- see src/pll_init.py CONF_EMA_SCALES / ema_variant_name
# and the orchestrator scripts/run_ema_sweep.py. The EMA scale is part of the
# algorithm name (e.g. 'PRODEN-BiasedCand-W20-EMA050') so every level is its
# own resumable results.csv row. The actual [start, end] range is computed
# at run time from config.yaml's pico.conf_ema_range so it follows config.


def _conf_ema_init_specs() -> list:
    """(name_fn(base) -> un-suffixed name, pico_init_fn(pl_ds, orig_targets, C)
    -> init_conf, proden_kwargs) for every init variant, in the same order
    as src.pll_init.conf_ema_sweep_base_names."""
    from src.pll_init import (BIAS_RAND_ALL_N_VALUES, BIAS_RAND_ALL_WEIGHTS, BIAS_WEIGHTS,
                              biased_candidates_init, biased_rand_all_variant_name,
                              biased_random_all_init, biased_variant_name, candidate_masked_init)
    specs = [(lambda base: base,
              lambda pl_ds, orig, C: candidate_masked_init(pl_ds.targets, C),
              dict(init_mode='candidate_masked'))]
    for w in BIAS_WEIGHTS:
        specs.append((lambda base, w=w: biased_variant_name(base, 'cand', w),
                      lambda pl_ds, orig, C, w=w: biased_candidates_init(pl_ds.targets, orig, C, w),
                      dict(init_mode='biased_candidates', true_weight=w)))
    for w in BIAS_RAND_ALL_WEIGHTS:
        for n in BIAS_RAND_ALL_N_VALUES:
            specs.append((lambda base, w=w, n=n: biased_rand_all_variant_name(base, w, n),
                          lambda pl_ds, orig, C, w=w, n=n: biased_random_all_init(orig, C, w, n),
                          dict(init_mode='biased_random_all', true_weight=w, wf=n)))
    return specs


def _make_conf_ema_runner(base: str, name_fn, pico_init_fn, proden_kwargs: dict, scale: float):
    from src.pll_init import ema_variant_name, scaled_conf_ema_range
    algorithm = ema_variant_name(name_fn(base), scale)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        conf_ema_range = scaled_conf_ema_range(raw_cfg['pico']['conf_ema_range'], scale)
        if base == 'PiCO-Fixed':
            init_conf = pico_init_fn(pl_ds, orig_targets, C).to(device)
            return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                            device, tag, report_every, algorithm, init_conf,
                                            conf_ema_range=conf_ema_range)
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, conf_ema_range=conf_ema_range,
                                    **proden_kwargs)

    return algorithm, _runner


def _build_conf_ema_sweep_runners() -> dict:
    from src.pll_init import CONF_EMA_SCALES, CONF_EMA_SWEEP_BASES, conf_ema_sweep_base_names
    runners = {}
    for name_fn, pico_init_fn, proden_kwargs in _conf_ema_init_specs():
        for base in CONF_EMA_SWEEP_BASES:
            for scale in CONF_EMA_SCALES:
                name, fn = _make_conf_ema_runner(base, name_fn, pico_init_fn, proden_kwargs, scale)
                runners[name] = fn
    # Guard: the name-only enumeration used by hparams.py / run_ema_sweep.py
    # must produce exactly the names registered here.
    from src.pll_init import ema_variant_name
    expected = {ema_variant_name(bn, s) for base in CONF_EMA_SWEEP_BASES
                for bn in conf_ema_sweep_base_names(base) for s in CONF_EMA_SCALES}
    assert set(runners) == expected, 'conf_ema sweep name enumeration out of sync'
    return runners


CONF_EMA_SWEEP_RUNNERS = _build_conf_ema_sweep_runners()


# ─── confidence-update-mechanism sweep (2026-09-16): Factor A x Factor B ──
# See src/pll_init.py's AB_VARIANT_SOURCE_HARD / AB_SWEEP_BASES / AB_SWEEP_SCALES
# docstring for the full rationale. Reuses _conf_ema_init_specs() (same
# biased-init variants as the conf_ema sweep) -- only the base's
# conf_source/conf_hard and the EMA scale list differ.


def _make_ab_runner(base: str, name_fn, pico_init_fn, scale: float):
    from src.pll_init import AB_VARIANT_SOURCE_HARD, ema_variant_name, scaled_conf_ema_range
    algorithm = ema_variant_name(name_fn(base), scale)
    conf_source, conf_hard = AB_VARIANT_SOURCE_HARD[base]

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        conf_ema_range = scaled_conf_ema_range(raw_cfg['pico']['conf_ema_range'], scale)
        init_conf = pico_init_fn(pl_ds, orig_targets, C).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf,
                                        conf_ema_range=conf_ema_range, conf_source=conf_source,
                                        conf_hard=conf_hard)

    return algorithm, _runner


def _build_ab_sweep_runners() -> dict:
    from src.pll_init import AB_SWEEP_BASES, AB_SWEEP_SCALES, conf_ema_sweep_base_names, ema_variant_name
    runners = {}
    for name_fn, pico_init_fn, _proden_kwargs in _conf_ema_init_specs():
        for base in AB_SWEEP_BASES:
            for scale in AB_SWEEP_SCALES:
                name, fn = _make_ab_runner(base, name_fn, pico_init_fn, scale)
                runners[name] = fn
    expected = {ema_variant_name(bn, s) for base in AB_SWEEP_BASES
                for bn in conf_ema_sweep_base_names(base) for s in AB_SWEEP_SCALES}
    assert set(runners) == expected, 'AB sweep name enumeration out of sync'
    return runners


AB_SWEEP_RUNNERS = _build_ab_sweep_runners()


def run_pico_mcl(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    model = PiCOModel(pico_args).to(device)
    cls_loss = PiCOMCLLoss()
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_pico_mclloss_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-MCL')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-MCL')
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def run_pico_mcl_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    """PiCO-MCL with PiCO-Fixed's paper-faithful warm-up fix (L_cont omitted
    entirely during warm-up, governed by prot_start_fixed) instead of
    PiCO-MCL's original warm-up (which keeps L_cont active throughout, same
    as plain PiCO -- see train_pico_mclloss_epoch). PiCOMCLLoss itself is
    stateless, so there is no confidence-buffer/init-conf to fix (unlike
    run_pico_fixed vs. run_pico); only the warm-up scheduling changes."""
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    pico_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
    model = PiCOModel(pico_args).to(device)
    cls_loss = PiCOMCLLoss()
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_pico_mcl_epoch_fixed(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-MCL-Fixed')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-MCL-Fixed')
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def run_pico_sc(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    model = PiCOModel(pico_args).to(device)
    cls_loss = PiCOCLSLoss(pl_ds.targets, C, conf_ema_range=tuple(pico_cfg['conf_ema_range']),
                            epochs=epochs).to(device)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep)
        train_pico_sc_epoch(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-SC')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-SC')
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc


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
    comco_cfg = raw_cfg['comco']
    comco_args = {
        'num_class': C,
        'epochs': epochs,
        'low_dim': comco_cfg['low_dim'],
        'moco_queue': comco_cfg['moco_queue'],
        'moco_m': comco_cfg['moco_m'],
        'loss_weight': comco_cfg['loss_weight'],
        'temperature': comco_cfg['temperature'],
        'top_k': comco_cfg['top_k'],
        'warmup_neg': comco_cfg['warmup_neg'],
        'warmup_pos': comco_cfg['warmup_pos'],
    }
    model = ComCoModel(comco_args).to(device)
    cls_loss = ComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'], top_k=comco_args['top_k'])
    opt = make_optimizer(model, hparams)

    detail_on = detail.is_enabled(raw_cfg)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        if detail_on:
            # Instrumented copy of train_comco_epoch that also measures, per
            # BATCH, contrastive positive-pair selection precision against
            # ground truth (ComCo counterpart to PiCO's selection-stats
            # logging) -- see src/pipeline/detail.py.
            detail.train_comco_epoch_with_selection_stats(
                comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device, raw_cfg, 'ComCo', C)
        else:
            train_comco_epoch(comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo')
        detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, 'ComCo')
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    detail.maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, loaders['test'], device, C, epochs, 'ComCo')
    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def run_comco_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    # See docs/comco_explanation.md: classification loss corrected to plain
    # (unscaled) SCL-NL per the paper's Section 5.1, instead of the original's
    # MCL-NL-style (C-1)/(C-m)-scaled formula borrowed from a different paper.
    # Contrastive loss / model architecture are unchanged (already verified faithful).
    comco_cfg = raw_cfg['comco']
    comco_args = {
        'num_class': C,
        'epochs': epochs,
        'low_dim': comco_cfg['low_dim'],
        'moco_queue': comco_cfg['moco_queue'],
        'moco_m': comco_cfg['moco_m'],
        'loss_weight': comco_cfg['loss_weight'],
        'temperature': comco_cfg['temperature'],
        'top_k': comco_cfg['top_k'],
        'warmup_neg': comco_cfg['warmup_neg'],
        'warmup_pos': comco_cfg['warmup_pos'],
    }
    model = ComCoModel(comco_args).to(device)
    cls_loss = FixedComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'], top_k=comco_args['top_k'])
    opt = make_optimizer(model, hparams)

    detail_on = detail.is_enabled(raw_cfg)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        if detail_on:
            detail.train_comco_epoch_with_selection_stats(
                comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device, raw_cfg,
                'ComCo-Fixed', C)
        else:
            train_comco_epoch(comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo-Fixed')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo-Fixed')
        detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, 'ComCo-Fixed')
        if (ep + 1) % report_every == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(report_every, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    detail.maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, loaders['test'], device, C, epochs, 'ComCo-Fixed')
    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc


# ─── SoLar: two-stage (pre-estimation + Sinkhorn-Knopp final training) ────


def run_solar(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    solar_cfg = raw_cfg['solar']
    solar_ds = SoLarDataset(pl_ds, orig_targets)
    solar_loader = DataLoader(solar_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, collate_fn=solar_collate_fn, pin_memory=True)

    fake_args = SimpleNamespace(lr=hparams['lr'], momentum=hparams.get('momentum', 0.9),
                                 weight_decay=hparams['weight_decay'], epochs=epochs, batch_size=batch_size)
    train_config = {'num_classes': C}

    model, loss_fn, opt, solar_args, queue = setup_solar(fake_args, train_config, solar_cfg, solar_ds, device)

    total_epochs = solar_cfg['est_epochs'] + epochs
    print(f'  [{tag}]  SoLar: est={solar_cfg["est_epochs"]} + train={epochs} = {total_epochs} total epochs',
          flush=True)

    accuracies = train_solar(solar_args, model, solar_loader, loaders['test'], loss_fn, opt, device, queue)
    final_acc = accuracies[-1] if accuracies else 0.0

    del model, loss_fn, opt, queue, solar_ds, solar_loader
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc
