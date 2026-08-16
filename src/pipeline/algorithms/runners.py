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
                         train_pico_sc_epoch, train_solar)
from src.fixed_pico_engine import train_pico_epoch_fixed
from src.mcl_losses import MCL_LOG
from src.fixed_mcl_losses import FixedMCLLog
from src.model_setup import setup_solar
from src.models import create_model_for_spec
from src.op_loss import OPLoss, OPWLoss
from src.pico.mcl_cls_loss import PiCOMCLLoss
from src.pico.model import PiCOModel
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


def run_proden(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    spec = (raw_cfg or {}).get('_dataset_spec')
    model = create_model_for_spec(spec, C).to(device)
    loss_fn = ProdenLoss(pl_ds.targets, C).to(device)
    opt = make_optimizer(model, hparams)

    idx_ds = (_IndexedDataset(pl_ds.data, image_size=spec.image_size, mean=spec.mean, std=spec.std,
                               modality=spec.modality)
              if spec is not None else _IndexedDataset(pl_ds.data))
    idx_loader = DataLoader(idx_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    chunk_t0 = time.perf_counter()
    final_acc = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, indices in idx_loader:
            imgs, indices = imgs.to(device), indices.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), indices).backward()
            opt.step()

        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PRODEN')

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
    conf = torch.zeros(len(pl_ds), C)
    for i, cands in enumerate(pl_ds.targets):
        conf[i, cands] = 1.0 / len(cands)
    return conf.to(device)


def run_pico_fixed(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
    # See docs/pico_explanation.md: paper-faithful warm-up (L_cont omitted
    # entirely, not switched to plain MoCo), prot_start default following
    # the paper's Appendix B.1 (1 epoch; use 100 for CIFAR-100 @ q=0.1 runs
    # by overriding config.yaml's pico.prot_start for that specific config),
    # and candidate-set-masked pseudo-target initialization (Eq. 6) instead
    # of run_pico's uniform-over-all-C-classes init.
    pico_cfg = raw_cfg['pico']
    pico_args = _pico_args(C, epochs, pico_cfg)
    pico_args['prot_start'] = pico_cfg.get('prot_start_fixed', 1)
    model = PiCOModel(pico_args).to(device)
    init_conf = _candidate_masked_init_conf(pl_ds, C, device)
    cls_loss = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt = make_optimizer(model, hparams)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        train_pico_epoch_fixed(pico_args, model, loaders['pico'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-Fixed')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'PiCO-Fixed')
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

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_comco_epoch(comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo')
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

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_comco_epoch(comco_args, model, loaders['comco'], cls_loss, cont_loss, opt, ep, device)
        detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo-Fixed')
        detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, 'ComCo-Fixed')
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
