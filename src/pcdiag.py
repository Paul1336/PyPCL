"""PiCO-Fixed vs ComCo-Fixed diagnostics, oracles and candidate filter (2026-11-03).

Three experiments share one implementation:

  Exp 1  per-epoch stage statistics of each algorithm -- pseudo-label accuracy, (PiCO) confidence and
         prototype quality, positive/negative pair correctness incl. the "positive pairs must share a
         candidate class" rule, loss values vs their ground-truth counterparts -- plus oracle
         variants that train with one component replaced by ground truth.
  Exp 2  gradient of each current loss vs its ground-truth counterpart, w.r.t. the shared backbone
         feature h (512-d, the only tensor both the classifier and the projection head read).
  Exp 3  candidate filter: drop selected positive pairs whose candidate sets are disjoint.

Algorithms (registered in src/pipeline/algorithms/__init__.py via PCD_VARIANTS):

  PCD-PiCO / PCD-ComCo     re-implementations of PiCO-Fixed / ComCo-Fixed with instrumentation
                           (numerically identical training; verify_scripts/check_pcdiag.py)
  -OraclePL                pseudo-label used for pair selection (and nothing else) = true label
  -OracleProto (PiCO)      prototypes are EMA-updated with the true label instead of the masked
                           classifier argmax (=> score_prot / confidence update see oracle prototypes)
  -OraclePos               selected positives that are not the same true class are removed
  -OracleNeg               false negatives (same true class, not selected as positive) are removed
                           from the contrastive denominator
  -OracleCls               classification loss = CE with the true label
  -CandFilter              positive pairs with Y_i . Y_j = 0 are removed (PiCO: pairs never violate
                           by construction since the pair label is argmax over Y; ComCo: the top-1
                           neighbour is searched only among compatible pool entries)

Stats go to results/<run>/pcdiag/<algorithm>/C{C}_k{k}_s{seed}.csv, one row per epoch. Pair stats
are computed on the pairs the algorithm SELECTED (after OraclePL / CandFilter, which change the
selection rule, but before the OraclePos / OracleNeg corrections), ratio-of-sums over the epoch,
excluding self and not-yet-filled queue slots.
"""

import csv
import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.comco.model import ComCoModel
from src.engine import evaluate_model
from src.pico.model import PiCOModel
from src.losses import functional as Fn
from src.losses.contrastive import comco_masks, masked_infonce  # noqa: F401  (moved 2026-11-22)
from src.models.common import seq_ema_prototypes  # noqa: F401  (moved 2026-11-22)
from src.analysis.accum import EpochAccumulator
from src.analysis.gradients import grad_alignment
from src.analysis.pairs import POOL_PAIR_METRICS, pool_pair_sums, self_key_masks
from src.pico.utils_loss import PartialLoss

PCD_VARIANTS = {
    'PCD-PiCO': ('pico', {}),
    'PCD-PiCO-OraclePL': ('pico', {'oracle_pl': True}),
    'PCD-PiCO-OracleProto': ('pico', {'oracle_proto': True}),
    'PCD-PiCO-OraclePos': ('pico', {'oracle_pos': True}),
    'PCD-PiCO-OracleNeg': ('pico', {'oracle_neg': True}),
    'PCD-PiCO-OracleCls': ('pico', {'oracle_cls': True}),
    'PCD-PiCO-CandFilter': ('pico', {'cand_filter': True}),
    'PCD-ComCo': ('comco', {}),
    'PCD-ComCo-OraclePL': ('comco', {'oracle_pl': True}),
    'PCD-ComCo-OraclePos': ('comco', {'oracle_pos': True}),
    'PCD-ComCo-OracleNeg': ('comco', {'oracle_neg': True}),
    'PCD-ComCo-OracleCls': ('comco', {'oracle_cls': True}),
    'PCD-ComCo-CandFilter': ('comco', {'cand_filter': True}),
    # ComCo-Fixed framework with the original MCL-LOG classification loss (2026-11-04):
    # -log(sum_{j in Y} p_j) * (C-1)/(C-m)  (== src/mcl_losses.MCL_LOG == clsloss_swap.MCLLOGLoss).
    # -OracleCls is omitted: it replaces the classification loss by CE, so it is PCD-ComCo-OracleCls.
    'PCD-ComCoMCL': ('comco', {'cls': 'mcllog'}),
    'PCD-ComCoMCL-OraclePL': ('comco', {'cls': 'mcllog', 'oracle_pl': True}),
    'PCD-ComCoMCL-OraclePos': ('comco', {'cls': 'mcllog', 'oracle_pos': True}),
    'PCD-ComCoMCL-OracleNeg': ('comco', {'cls': 'mcllog', 'oracle_neg': True}),
    'PCD-ComCoMCL-CandFilter': ('comco', {'cls': 'mcllog', 'cand_filter': True}),
}

# ComCo-Fixed + MCL-LOG with an EMA pseudo-label (2026-11-05). Each training sample i keeps a soft score
# s_i in R^C (init uniform 1/C); every time i is in a batch: s_i <- m * s_i + (1 - m) * softmax(f(x_i)), and
# ComCo's pseudo-label (used for top-K positive selection + stored in the queue) becomes argmax s_i instead of
# the per-batch argmax f(x_i). Unmasked, like ComCo's own pseudo-label. Each sample is updated once per epoch,
# so the half-life is ln(0.5)/ln(m) epochs: m=0.5 -> 1, 0.9 -> 6.6, 0.99 -> 69, 0.999 -> 693 (~ running mean).
# m=0 reproduces PCD-ComCoMCL exactly. The uniform init does not change the argmax.
# ComCo-Fixed framework, two more MCL-LOG scalings (2026-11-06, variable-size candidate sets):
#   MCLNL    -log(sum_{j in Y} p_j)                      (no coefficient)
#   MCLFixed -log(sum_{j in Y} p_j) * 2(C-1)/|Ybar|       (== clsloss_swap.MCLLOGFixedLoss, |Ybar| clamped >= 1)
PCD_VARIANTS['PCD-ComCoMCLNL'] = ('comco', {'cls': 'mclnl'})
PCD_VARIANTS['PCD-ComCoMCLFixed'] = ('comco', {'cls': 'mcllogfixed'})

# ComCo-Fixed + MCL-NL with PiCO+ AllClean pair selection (2026-11-16). Only the contrastive masks change;
# temperature / loss_weight / cls loss / ComCo's own pseudo-label (unmasked argmax) stay ComCo's.
#   PosAC     positives = own key + EVERY pool entry with the same pseudo-label (AllClean / PiCO SupCon rule)
#             instead of key + top-1 same-label neighbour; active from epoch ac_pos_start (= PiCO+ warm-up 5)
#             instead of ComCo's warmup_pos (100).
#   NegAC     denominator = whole pool except self (AllClean) instead of positives + entries with c* in Ybar.
#   PosNegAC  both.
AC_POS_START = 5
PCD_VARIANTS['PCD-ComCoMCLNL-PosAC'] = ('comco', {'cls': 'mclnl', 'pos_all': True})
PCD_VARIANTS['PCD-ComCoMCLNL-NegAC'] = ('comco', {'cls': 'mclnl', 'neg_all': True})
PCD_VARIANTS['PCD-ComCoMCLNL-PosNegAC'] = ('comco', {'cls': 'mclnl', 'pos_all': True, 'neg_all': True})

# 2026-11-17 (comco-smallk-1117):
#   -OracleSel   oracle sample selection = OraclePos + OracleNeg together (selected positives that are not the
#                same true class removed AND false negatives removed from the denominator); the pseudo-label
#                and therefore which pairs are *selected* stay the algorithm's own. One component (selection).
#   ComCo-Fixed + MCL-NL coefficient / contrastive-weight sensitivity:
#     -CLSx{s}   training classification loss = s * MCL-NL (diagnostic loss columns stay unscaled)
#     -LWx{s}    contrastive weight lambda = s * comco.loss_weight (config 0.3)
PCD_VARIANTS['PCD-PiCO-OracleSel'] = ('pico', {'oracle_pos': True, 'oracle_neg': True})
PCD_VARIANTS['PCD-ComCoMCL-OracleSel'] = ('comco', {'cls': 'mcllog', 'oracle_pos': True, 'oracle_neg': True})
SCALE_TAGS = {'0.25': 0.25, '0.5': 0.5, '2': 2.0, '4': 4.0}
for _t, _s in SCALE_TAGS.items():
    PCD_VARIANTS[f'PCD-ComCoMCLNL-CLSx{_t}'] = ('comco', {'cls': 'mclnl', 'cls_scale': _s})
    PCD_VARIANTS[f'PCD-ComCoMCLNL-LWx{_t}'] = ('comco', {'cls': 'mclnl', 'lam_scale': _s})

# 2026-11-18: ComCo-Fixed + MCL-NL + loss-level mixup ("option A"). No disambiguation target: the mixed image
# x~ = lam*x_a + (1-lam)*x_b (weak views, in-batch random pairing) is trained with
#   L_mix = lam * MCLNL(f(x~), S_a) + (1-lam) * MCLNL(f(x~), S_b),   lam ~ Beta(4,4), lam <- max(lam, 1-lam)
# (same lam distribution / pairing as PiCO+'s mixup, src/picoplus.py:169-175), added to the total loss with
# weight mix_weight (1.0, as L_mix in PiCO+). Everything else (pseudo-label, pair selection, tau, lambda) unchanged.
MIX_BETA = 4.0
PCD_VARIANTS['PCD-ComCoMCLNL-MixA'] = ('comco', {'cls': 'mclnl', 'mix_weight': 1.0})

PL_EMA_MS = {'EMA50': 0.5, 'EMA90': 0.9, 'EMA99': 0.99, 'EMA999': 0.999}
for _tag, _m in PL_EMA_MS.items():
    PCD_VARIANTS[f'PCD-ComCoMCL-{_tag}'] = ('comco', {'cls': 'mcllog', 'pl_ema': _m})

PICO_TAU = 0.07  # SupConLoss default (temperature = base_temperature -> scale 1)


# ─── shared pieces ─────────────────────────────────────────────────────────


# cls losses on the dense comp mask -> canonical formulas in src/losses/functional.py (2026-11-22)
def scl_nl_avg(logits, comp_mask):
    """FixedComCoCLSLoss: (1/m) sum_{j in Ybar} -log(1 - p_j)."""
    return Fn.scl_nl(logits, 1.0 - comp_mask)


def mcl_log(logits, comp_mask):
    """Original MCL-LOG (Feng et al. 2020 loss, (C-1)/(C-m) scaling as in src/mcl_losses.MCL_LOG)."""
    return Fn.mcl(logits, 1.0 - comp_mask, 'log', 'c1_over_k')


def mcl_nl(logits, comp_mask):
    return Fn.mcl(logits, 1.0 - comp_mask, 'log', 'none')


def mcl_log_fixed(logits, comp_mask):
    return Fn.mcl(logits, 1.0 - comp_mask, 'log', 'feng_fixed')


CLS_FNS = {'mcllog': mcl_log, 'mclnl': mcl_nl, 'mcllogfixed': mcl_log_fixed}


def _self_key(B, A, device):
    return self_key_masks(B, A, device)


_Acc = EpochAccumulator  # src/analysis/accum.py (ratio-of-sums + mean-of-batch-means)


def pair_stats(acc, pos, denom, same, compat, valid, eye, key, B):
    """pos/denom: selected masks (B x A). same: same true class. compat: Y_i.Y_j > 0. valid: filled
    pool slot (1 x A). All float 0/1. Sums -> src/analysis/pairs.pool_pair_sums."""
    acc.add_many(pool_pair_sums(pos, denom, same, compat, valid, eye, key))


def grad_stats(acc, name, g, g_ref):
    """g_{name}_cos / _cos_sample / _norm_ratio -> src/analysis/gradients.grad_alignment."""
    acc.mean_many(grad_alignment(g, g_ref, name))


def _row_common(acc, ep):
    r = {'epoch': ep + 1}
    r.update(acc.ratios(POOL_PAIR_METRICS))
    r.update(acc.means())
    return r


class _CsvLog:
    def __init__(self, raw_cfg, algorithm, C):
        d = os.path.join(raw_cfg.get('_results_dir', 'results/_pcdiag'), 'pcdiag', algorithm)
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"C{C}_k{raw_cfg.get('_current_k', 0)}_s{raw_cfg.get('_current_seed', 0)}.csv")
        self.fields, self.rows = [], []

    def write(self, row):
        """Rewrites the whole file each epoch so columns that first appear later (e.g. ComCo's
        cstar_* once warmup_neg starts) are kept; missing values are left empty."""
        self.rows.append(row)
        self.fields += [k for k in row if k not in self.fields]
        tmp = self.path + '.tmp'
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, self.fields)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)


def _encode(enc, x):
    h = enc.encoder(x)
    return h, enc.fc_layer(h), F.normalize(enc.head(h), dim=1)


# ─── PiCO ──────────────────────────────────────────────────────────────────


class PiCODiagModel(PiCOModel):
    """PiCOModel plus queue bookkeeping of true labels / candidate sets and a shadow prototype
    memory EMA-updated with TRUE labels (diagnostic; it drives training only under OracleProto)."""

    def __init__(self, args):
        super().__init__(args)
        Q, C, D = args['moco_queue'], args['num_class'], args['low_dim']
        self.register_buffer('queue_true', torch.arange(-Q, 0, dtype=torch.long))
        self.register_buffer('queue_Y', torch.zeros(Q, C))
        self.register_buffer('oracle_prototypes', torch.zeros(C, D))

    def forward(self, img_q, im_k=None, partial_Y=None, true=None, args=None, flags=None, eval_only=False):
        if eval_only:
            return self.encoder_q(img_q)[0]
        flags = flags or {}
        h, output, q = _encode(self.encoder_q, img_q)
        yhat = torch.max(torch.softmax(output, dim=1) * partial_Y, dim=1)[1]
        pair_lab = true if flags.get('oracle_pl') else yhat
        proto_lab = true if flags.get('oracle_proto') else yhat

        score_prot = torch.softmax(torch.mm(q, self.prototypes.clone().detach().t()), dim=1)
        score_oprot = torch.softmax(torch.mm(q.detach(), self.oracle_prototypes.clone().t()), dim=1)
        self.prototypes = seq_ema_prototypes(self.prototypes, q.detach(), proto_lab, args['proto_m']).detach()
        self.oracle_prototypes = seq_ema_prototypes(self.oracle_prototypes, q.detach(), true, args['proto_m'])

        with torch.no_grad():
            self._momentum_update_key_encoder(args)
            _, k = self.encoder_k(im_k)

        Q = args['moco_queue']
        B = k.shape[0]
        feats = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pool_lab = torch.cat((pair_lab.float(), pair_lab.float(), self.queue_pseudo.clone().detach()), dim=0)
        pool_true = torch.cat((true, true, self.queue_true.clone()), dim=0)
        pool_Y = torch.cat((partial_Y, partial_Y, self.queue_Y.clone()), dim=0)

        ptr = int(self.queue_ptr)
        assert Q % B == 0
        self.queue[ptr:ptr + B] = k
        self.queue_pseudo[ptr:ptr + B] = pair_lab.float()
        self.queue_true[ptr:ptr + B] = true
        self.queue_Y[ptr:ptr + B] = partial_Y
        self.queue_ptr[0] = (ptr + B) % Q
        return dict(h=h, output=output, q=q, yhat=yhat, pair_lab=pair_lab, score_prot=score_prot,
                    score_oprot=score_oprot, feats=feats, pool_lab=pool_lab, pool_true=pool_true, pool_Y=pool_Y)


def train_pico_pcd_epoch(pico_args, model, loader, cls_loss, optimizer, epoch, device, flags, C):
    model.train()
    acc = _Acc()
    cont_on = epoch >= pico_args['prot_start']
    lam = pico_args['loss_weight']
    for images_w, images_s, Y, true, index in loader:
        images_w, images_s, Y, true, index = (images_w.to(device), images_s.to(device), Y.to(device),
                                              true.to(device), index.to(device))
        o = model(images_w, images_s, Y, true, pico_args, flags)
        out, q, h, feats = o['output'], o['q'], o['h'], o['feats']
        B, A = out.shape[0], feats.shape[0]

        if cont_on:
            cls_loss.confidence_update(temp_un_conf=o['score_prot'].detach(), batch_index=index, batchY=Y)

        loss_cls = cls_loss(out, index)
        loss_ce = F.cross_entropy(out, true)

        with torch.no_grad():
            eye, key = _self_key(B, A, device)
            valid = (o['pool_true'] >= 0).float()[None, :]
            same = (true[:, None] == o['pool_true'][None, :]).float()
            compat = ((Y @ o['pool_Y'].t()) > 0).float()
            pos = (o['pair_lab'].float()[:, None] == o['pool_lab'][None, :]).float() * (1 - eye)
            if flags.get('cand_filter'):
                pos = pos * compat
            denom = 1 - eye
            pair_stats(acc, pos, denom, same, compat, valid, eye, key, B)
            pos_used = pos * same if flags.get('oracle_pos') else pos
            denom_used = denom * (1 - same * (1 - pos_used)) if flags.get('oracle_neg') else denom
            pos_gt = same * (1 - eye)

        loss_cont = masked_infonce(q, feats, pos_used, denom_used, PICO_TAU)
        loss_cont_gt = masked_infonce(q, feats, pos_gt, denom, PICO_TAU)
        train_cls = loss_ce if flags.get('oracle_cls') else loss_cls
        loss = train_cls + lam * loss_cont if cont_on else train_cls

        # Exp 2: gradients w.r.t. the shared backbone feature h
        g_cls, g_ce, g_cont, g_cgt = (torch.autograd.grad(L, h, retain_graph=True)[0]
                                      for L in (loss_cls, loss_ce, loss_cont, loss_cont_gt))
        grad_stats(acc, 'cls_vs_ce', g_cls, g_ce)
        grad_stats(acc, 'cont_vs_gt', g_cont, g_cgt)
        g_tot = g_cls + lam * g_cont if cont_on else g_cls
        g_tot_gt = g_ce + lam * g_cgt if cont_on else g_ce
        grad_stats(acc, 'total_vs_gt', g_tot, g_tot_gt)
        acc.mean('g_cls_vs_cont_cos', F.cosine_similarity(g_cls.flatten(), g_cont.flatten(), dim=0, eps=1e-12))
        acc.mean('g_cont_over_cls_norm', (lam * g_cont).norm() / g_cls.norm().clamp(min=1e-12))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            conf = cls_loss.confidence[index]
            sp, so = o['score_prot'], o['score_oprot']
            acc.mean('pl_acc', (o['yhat'] == true).float().mean())
            acc.mean('pl_in_comp', (Y.gather(1, o['yhat'][:, None]) == 0).float().mean())
            acc.mean('cls_acc_unmasked', (out.argmax(1) == true).float().mean())
            acc.mean('conf_acc', (conf.argmax(1) == true).float().mean())
            acc.mean('conf_true_mass', conf.gather(1, true[:, None]).mean())
            acc.mean('proto_acc', (sp.argmax(1) == true).float().mean())
            acc.mean('proto_masked_acc', ((sp * Y).argmax(1) == true).float().mean())
            acc.mean('oproto_masked_acc', ((so * Y).argmax(1) == true).float().mean())
            acc.mean('loss_cls', loss_cls)
            acc.mean('loss_cls_gt', loss_ce)
            acc.mean('loss_cont', loss_cont)
            acc.mean('loss_cont_gt', loss_cont_gt)
            acc.mean('loss_total', loss)
    row = _row_common(acc, epoch)
    with torch.no_grad():
        pc = F.cosine_similarity(model.prototypes, model.oracle_prototypes, dim=1)
    row.update({'cont_on': int(cont_on), 'conf_ema_m': cls_loss.conf_ema_m,
                'proto_cos_oracle': pc.mean().item(), 'proto_cos_oracle_min': pc.min().item()})
    return row


# ─── ComCo ─────────────────────────────────────────────────────────────────


class ComCoDiagModel(ComCoModel):
    def __init__(self, args):
        super().__init__(args)
        Q = args['moco_queue']
        self.register_buffer('queue_true', torch.arange(-Q, 0, dtype=torch.long))
        if args.get('pl_ema_n'):  # per-sample EMA soft pseudo-label scores (PL_EMA_MS variants)
            C = args['num_class']
            self.register_buffer('pl_ema', torch.full((args['pl_ema_n'], C), 1.0 / C))

    def forward(self, img_q, img_k=None, comp_mask=None, true=None, args=None, flags=None, eval_only=False,
                index=None):
        if eval_only:
            return self.encoder_q(img_q)[0]
        flags = flags or {}
        h, output, q = _encode(self.encoder_q, img_q)
        yhat = output.argmax(dim=1)
        pl_lab = yhat
        if flags.get('pl_ema') is not None:
            m = flags['pl_ema']
            with torch.no_grad():
                s = m * self.pl_ema[index] + (1.0 - m) * torch.softmax(output.detach(), dim=1)
                self.pl_ema[index] = s
                pl_lab = s.argmax(dim=1)
        pair_lab = true if flags.get('oracle_pl') else pl_lab
        with torch.no_grad():
            self._momentum_update_key_encoder(args['moco_m'])
            _, k = self.encoder_k(img_k)
        feats = torch.cat([q, k, self.queue.clone().detach()], dim=0)
        pool_lab = torch.cat([pair_lab, pair_lab, self.queue_pseudo.clone().detach()], dim=0)
        pool_comp = torch.cat([comp_mask, comp_mask, self.queue_comp.clone().detach()], dim=0)
        pool_true = torch.cat([true, true, self.queue_true.clone()], dim=0)
        ptr = int(self.queue_ptr)
        self._dequeue_and_enqueue(k, pair_lab, comp_mask, args['moco_queue'])
        self.queue_true[ptr:ptr + k.shape[0]] = true
        return dict(h=h, output=output, q=q, yhat=yhat, pair_lab=pair_lab, feats=feats, pool_lab=pool_lab,
                    pool_comp=pool_comp, pool_true=pool_true)


def train_comco_pcd_epoch(comco_args, model, loader, optimizer, epoch, device, flags, C):
    model.train()
    acc = _Acc()
    pos_all, neg_all = bool(flags.get('pos_all')), bool(flags.get('neg_all'))
    warm_pos = epoch >= (AC_POS_START if pos_all else comco_args['warmup_pos'])
    warm_neg = epoch >= comco_args['warmup_neg']
    lam, tau, top_k = comco_args['loss_weight'], comco_args['temperature'], comco_args['top_k']
    lam = lam * float(flags.get('lam_scale', 1.0))
    cls_scale = float(flags.get('cls_scale', 1.0))
    for images_w, images_s, comp, true, index in loader:
        images_w, images_s, comp, true = (images_w.to(device), images_s.to(device), comp.to(device),
                                          true.to(device))
        Y = 1.0 - comp
        o = model(images_w, images_s, comp, true, comco_args, flags, index=index.to(device))
        out, q, h, feats = o['output'], o['q'], o['h'], o['feats']
        B, A = out.shape[0], feats.shape[0]

        cls_fn = CLS_FNS.get(flags.get('cls'), scl_nl_avg)
        loss_cls = cls_fn(out, comp)
        loss_ce = F.cross_entropy(out, true)
        loss_cls_gt_own = cls_fn(out, 1.0 - F.one_hot(true, C).float())

        with torch.no_grad():
            sim = torch.mm(q, feats.t())
            eye, key = _self_key(B, A, device)
            valid = (o['pool_true'] >= 0).float()[None, :]
            same = (true[:, None] == o['pool_true'][None, :]).float()
            compat = ((Y @ (1.0 - o['pool_comp']).t()) > 0).float() * valid + (1 - valid)
            pos, denom, cstar = comco_masks(sim, o['pair_lab'], o['pool_lab'], o['pool_comp'], warm_pos, warm_neg,
                                            top_k, compat=compat if flags.get('cand_filter') else None,
                                            pos_all=pos_all, neg_all=neg_all)
            pair_stats(acc, pos, denom, same, compat, valid, eye, key, B)
            if cstar is not None:
                acc.mean('cstar_hit', (cstar == true).float().mean())
                acc.mean('cstar_in_cand', (Y.gather(1, cstar[:, None]) > 0).float().mean())
            pos_used = pos * same if flags.get('oracle_pos') else pos
            denom_used = denom * (1 - same * (1 - pos_used)) if flags.get('oracle_neg') else denom
            pos_gt, denom_gt, _ = comco_masks(sim, true, o['pool_true'], o['pool_comp'], warm_pos, warm_neg, top_k,
                                              force_cstar=true if (warm_neg and not neg_all) else None,
                                              pos_all=pos_all, neg_all=neg_all)

        loss_cont = masked_infonce(q, feats, pos_used, denom_used, tau)
        loss_cont_gt = masked_infonce(q, feats, pos_gt, denom_gt, tau)
        train_cls = loss_ce if flags.get('oracle_cls') else cls_scale * loss_cls
        loss = train_cls + lam * loss_cont
        mix_w = float(flags.get('mix_weight', 0.0))
        if mix_w > 0:
            mlam = float(np.random.beta(MIX_BETA, MIX_BETA))
            mlam = max(mlam, 1.0 - mlam)
            perm = torch.randperm(B, device=device)
            logits_mix, _ = model.encoder_q(mlam * images_w + (1.0 - mlam) * images_w[perm])
            loss_mix = mlam * cls_fn(logits_mix, comp) + (1.0 - mlam) * cls_fn(logits_mix, comp[perm])
            loss = loss + mix_w * loss_mix
            acc.mean('loss_mix', loss_mix.detach())
            acc.mean('mix_lam', mlam)

        g_cls = cls_scale * torch.autograd.grad(loss_cls, h, retain_graph=True)[0]
        g_ce = torch.autograd.grad(loss_ce, h, retain_graph=True)[0]
        g_own = torch.autograd.grad(loss_cls_gt_own, h, retain_graph=True)[0]
        g_cont = torch.autograd.grad(loss_cont, h, retain_graph=True)[0]
        g_cgt = torch.autograd.grad(loss_cont_gt, h, retain_graph=True)[0]
        grad_stats(acc, 'cls_vs_ce', g_cls, g_ce)
        grad_stats(acc, 'cls_vs_gtown', g_cls, g_own)
        grad_stats(acc, 'cont_vs_gt', g_cont, g_cgt)
        grad_stats(acc, 'total_vs_gt', g_cls + lam * g_cont, g_ce + lam * g_cgt)
        acc.mean('g_cls_vs_cont_cos', F.cosine_similarity(g_cls.flatten(), g_cont.flatten(), dim=0, eps=1e-12))
        acc.mean('g_cont_over_cls_norm', (lam * g_cont).norm() / g_cls.norm().clamp(min=1e-12))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc.mean('pl_acc', (o['yhat'] == true).float().mean())
            acc.mean('pl_in_comp', (comp.gather(1, o['yhat'][:, None]) > 0).float().mean())
            if flags.get('pl_ema') is not None:  # pair_lab = EMA pseudo-label; pl_acc above stays raw argmax
                acc.mean('ema_pl_acc', (o['pair_lab'] == true).float().mean())
                acc.mean('ema_pl_in_comp', (comp.gather(1, o['pair_lab'][:, None]) > 0).float().mean())
                acc.mean('ema_pl_eq_yhat', (o['pair_lab'] == o['yhat']).float().mean())
            acc.mean('cls_masked_acc', ((torch.softmax(out, 1) * Y).argmax(1) == true).float().mean())
            acc.mean('loss_cls', loss_cls)
            acc.mean('loss_cls_gt', loss_ce)
            acc.mean('loss_cls_gt_own', loss_cls_gt_own)
            acc.mean('loss_cont', loss_cont)
            acc.mean('loss_cont_gt', loss_cont_gt)
            acc.mean('loss_total', loss)
    row = _row_common(acc, epoch)
    row.update({'warm_neg': int(warm_neg), 'warm_pos': int(warm_pos)})
    return row


# ─── runners ───────────────────────────────────────────────────────────────


def _print_eta(tag, ep, epochs, elapsed, n):
    rem = (epochs - ep) * elapsed / max(n, 1)
    print(f'  [{tag}] epoch {ep}/{epochs}  {elapsed / max(n, 1):.1f}s/ep  ETA {rem / 60:.1f} min', flush=True)


def make_pcd_runner(algorithm):
    framework, flags = PCD_VARIANTS[algorithm]

    def run(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        from src.pipeline.algorithms.hparams import make_optimizer
        from src.pll_init import candidate_masked_init
        if framework == 'pico':
            pc = raw_cfg['pico']
            args = {'num_class': C, 'epochs': epochs, 'low_dim': pc['low_dim'], 'moco_queue': pc['moco_queue'],
                    'moco_m': pc['moco_m'], 'proto_m': pc['proto_m'], 'loss_weight': pc['loss_weight'],
                    'conf_ema_range': pc['conf_ema_range'], 'prot_start': pc.get('prot_start_fixed', 1)}
            model = PiCODiagModel(args).to(device)
            cls_loss = PartialLoss(candidate_masked_init(pl_ds.targets, C).to(device))
            loader = loaders['pico']
        else:
            cc = raw_cfg['comco']
            args = {'num_class': C, 'epochs': epochs, **{k: cc[k] for k in (
                'low_dim', 'moco_queue', 'moco_m', 'loss_weight', 'temperature', 'top_k', 'warmup_neg',
                'warmup_pos')}}
            if flags.get('pl_ema') is not None:
                ds = loaders['comco'].dataset
                args['pl_ema_n'] = len(getattr(ds, 'dataset', ds))  # Subset (held-out) yields base indices
            model = ComCoDiagModel(args).to(device)
            cls_loss = None
            loader = loaders['comco']
        opt = make_optimizer(model, hparams)
        log = _CsvLog(raw_cfg, algorithm, C)
        # standard detail diagnostics (2026-11-22): per-class, concentration, label_diag, t-SNE,
        # cls-loss gradient bias/variance (MCL family via a comp-mask adapter), kNN at the end
        from src.pipeline import biasvariance as _bv
        from src.pipeline import detail as _detail
        bv_loss = cls_loss
        if framework != 'pico':
            fn = CLS_FNS.get(flags.get('cls'), scl_nl_avg)
            bv_loss = type('CompLoss', (), {'takes_comp': True, '__call__': staticmethod(fn)})()
        bv_on = _detail.biasvariance_enabled(raw_cfg)
        if bv_on:
            bv_batch = _bv.build_fixed_eval_batch(pl_ds, orig_targets,
                                                  _detail.biasvariance_cfg(raw_cfg).get('eval_size', 256), device)
        t0 = time.perf_counter()
        for ep in range(epochs):
            te = time.perf_counter()
            if framework == 'pico':
                cls_loss.set_conf_ema_m(ep, args)
                row = train_pico_pcd_epoch(args, model, loader, cls_loss, opt, ep, device, flags, C)
            else:
                row = train_comco_pcd_epoch(args, model, loader, opt, ep, device, flags, C)
            row['test_acc'] = (evaluate_model(model, loaders['test'], device)
                               if (ep + 1) % 10 == 0 or ep + 1 == epochs or ep < 5 else float('nan'))
            row['epoch_time_s'] = time.perf_counter() - te
            log.write(row)
            _detail.maybe_log_checkpoint(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
            _detail.maybe_plot_tsne(raw_cfg, model, loaders['test'], device, C, ep + 1, algorithm)
            _detail.maybe_log_concentration(raw_cfg, model, pl_ds, device, C, ep + 1, algorithm)
            _detail.maybe_log_label_diag(raw_cfg, model, cls_loss, pl_ds, orig_targets, C, ep + 1, algorithm)
            if bv_on:
                _detail.maybe_log_biasvariance(raw_cfg, model, bv_loss, None, algorithm, *bv_batch, C, ep + 1, device)
            if (ep + 1) % report_every == 0 or ep + 1 == epochs:
                _print_eta(tag, ep + 1, epochs, time.perf_counter() - t0, min(report_every, ep + 1))
                t0 = time.perf_counter()
        _detail.maybe_run_knn_eval(raw_cfg, model, pl_ds, orig_targets, loaders['test'], device, C, epochs, algorithm)
        acc = evaluate_model(model, loaders['test'], device)
        del model, opt, cls_loss
        gc.collect()
        torch.cuda.empty_cache()
        return acc

    return run


PCD_RUNNERS = {name: make_pcd_runner(name) for name in PCD_VARIANTS}
