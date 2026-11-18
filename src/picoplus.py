"""PiCO+ (Wang et al., "PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning",
TPAMI 2023 / arXiv:2201.08984v3) -- 2026-11-08.

Ported from the official implementation (github.com/hbzju/PiCO, train_pico_plus.py + model.PiCO_PLUS +
utils_plus/utils_loss.py), single-GPU, with this repo's PiCO-Fixed backbone / data pipeline / Adam
optimiser so it is directly comparable with PCD-PiCO.

What PiCO+ adds on top of PiCO (paper Sec. 4, Alg. 2):
  * warm-up (epoch < plus_warmup): PiCO classification loss on uniform candidate targets + MoCo
    (instance) contrastive loss; no confidence update.
  * distance-based clean selection (Eq. 8): d_i = -q_i . mu_{yhat_i}, yhat_i = argmax_{j in Y_i} f_j(x_i);
    at the start of every epoch the pure_ratio fraction with the smallest d (computed during the previous
    epoch) is "reliable" (D_clean), the rest is D_noisy.
  * pseudo-label for pairs: within-candidate argmax for clean samples, all-class argmax for noisy (Eq. 9).
  * prototypes EMA-updated from clean samples only; queue also stores the clean flag.
  * contrastive: L_cont_rel (positives only among clean rows/cols; noisy rows use kNN positives from
    epoch knn_start, Eq. 10) weighted loss_weight (clean) / ur_weight (noisy) + ur_weight * L_n-cont
    (label-driven positives over all data, Eq. 9).
  * classification: cls_weight * PiCO loss on clean + ur_weight * CE(prototype label guess, Eq. 11,
    sharpened with temperature_guess) on noisy + mixup loss over all (Eq. 12, Beta(4,4), max(l,1-l)),
    targets = PiCO confidence (clean) / guess (noisy).
  * confidence (PiCO pseudo-target) EMA update over ALL samples, within the candidate set (as PiCO).
  Total (Eq. 13 as in the code): L_mix + 2 L_cls_clean + 0.1 L_cls_noisy + L_cont_rel/knn + 0.1 L_cont_all.

Variants:
  PCD-PiCOPlus           pure_ratio 0.8 (paper's delta for the mildest noise, eta=0.1)
  PCD-PiCOPlus-AllClean  pure_ratio 1.0 -> every sample clean: no label guessing / kNN; this is the paper's
                         own reading of PiCO+ on standard (noise-free) PLL = PiCO + mixup (Sec. 5.2.1).

Per-epoch diagnostics: results/<run>/pcdiag/<algorithm>/C{C}_k{k}_s{seed}.csv.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from src.engine import evaluate_model
from src.pcdiag import PiCODiagModel, _Acc, _CsvLog, _print_eta, seq_ema_prototypes
from src.pico.model import PiCOModel

PLUS_VARIANTS = {
    'PCD-PiCOPlus': {'pure_ratio': 0.8},
    'PCD-PiCOPlus-AllClean': {'pure_ratio': 1.0},
    'PCD-PiCOPlus-D60': {'pure_ratio': 0.6},  # noisy 11-17: paper's delta for eta=0.2 (kept for eta=0.3)
}

# Official defaults (README CIFAR commands / argparse), adapted where noted.
PLUS_DEFAULTS = dict(
    plus_warmup=5,          # --prot_start 5 (CIFAR-10 README); selection + PiCO disambiguation start here
    knn_start_frac=1 / 8,   # --knn_start 100 of 800 epochs -> scaled to our epoch budget (25 of 200)
    chosen_neighbors=5,     # 5 for CIFAR-100 (500 images / class, same as our class subsets)
    temperature_guess=0.07,
    ur_weight=0.1,
    cls_weight=2.0,
    mix_beta=4.0,
    tau=0.07,
)


class PiCOPlusModel(PiCODiagModel):
    """PiCODiagModel (PiCO + true-label/candidate queue bookkeeping) plus the clean flag queue."""

    def __init__(self, args):
        super().__init__(args)
        self.register_buffer('queue_rel', torch.zeros(args['moco_queue'], dtype=torch.bool))

    def forward(self, img_q, im_k=None, partial_Y=None, true=None, is_rel=None, args=None, eval_only=False):
        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        with torch.no_grad():
            scores = torch.softmax(output, dim=1)
            within = torch.max(scores * partial_Y, dim=1)[1]
            allmax = torch.max(scores, dim=1)[1]
            pl_b = torch.where(is_rel, within, allmax)
            protos = self.prototypes.clone().detach()
            score_prot = torch.softmax(torch.mm(q, protos.t()), dim=1)
            dist = -(q * protos[within]).sum(dim=1)
            if is_rel.any():  # == official sequential loop over clean samples
                self.prototypes = seq_ema_prototypes(self.prototypes, q[is_rel], pl_b[is_rel], args['proto_m'])
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach()
            self.oracle_prototypes = seq_ema_prototypes(self.oracle_prototypes, q.detach(), true, args['proto_m'])
            self._momentum_update_key_encoder(args)
            _, k = self.encoder_k(im_k)

        Q, B = args['moco_queue'], k.shape[0]
        feats = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pool_lab = torch.cat((pl_b.float(), pl_b.float(), self.queue_pseudo.clone().detach()), dim=0)
        pool_rel = torch.cat((is_rel, is_rel, self.queue_rel.clone()), dim=0)
        pool_true = torch.cat((true, true, self.queue_true.clone()), dim=0)

        ptr = int(self.queue_ptr)
        assert Q % B == 0
        self.queue[ptr:ptr + B] = k
        self.queue_pseudo[ptr:ptr + B] = pl_b.float()
        self.queue_rel[ptr:ptr + B] = is_rel
        self.queue_true[ptr:ptr + B] = true
        self.queue_Y[ptr:ptr + B] = partial_Y
        self.queue_ptr[0] = (ptr + B) % Q
        return dict(output=output, q=q, within=within, allmax=allmax, pl_b=pl_b, score_prot=score_prot,
                    dist=dist, feats=feats, pool_lab=pool_lab, pool_rel=pool_rel, pool_true=pool_true)


def supcon(feats, mask, B, tau, weights=None):
    """Official utils_plus SupConLoss, mask mode (temperature == base_temperature)."""
    logits = torch.mm(feats[:B], feats.t()) / tau
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    lmask = torch.ones_like(mask)
    lmask[torch.arange(B), torch.arange(B)] = 0.0
    mask = mask * lmask
    log_prob = logits - torch.log((torch.exp(logits) * lmask).sum(1, keepdim=True) + 1e-12)
    loss = -(mask * log_prob).sum(1) / mask.sum(1)
    return loss.mean() if weights is None else (loss * weights.detach()).mean()


def moco_loss(feats, B, tau):
    q, k, queue = feats[:B], feats[B:2 * B], feats[2 * B:]
    logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.t()], dim=1) / tau
    return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=feats.device))


def soft_ce(logits, targets):
    return -(F.log_softmax(logits, dim=1) * targets.detach()).sum(1)


def train_picoplus_epoch(a, model, loader, confidence, conf_ema_m, optimizer, epoch, device, sel, C):
    model.train()
    acc = _Acc()
    started = epoch >= a['plus_warmup']
    knn_on = epoch >= a['knn_start']
    tau = a['tau']
    for images_w, images_s, Y, true, index in loader:
        images_w, images_s, Y, true, index = (images_w.to(device), images_s.to(device), Y.to(device),
                                              true.to(device), index.to(device))
        is_rel = sel['is_rel'][index]
        w_rel = is_rel.float()
        o = model(images_w, images_s, Y, true, is_rel, a)
        out, feats = o['output'], o['feats']
        B = out.shape[0]
        diag = torch.eye(B, device=device)

        if started:
            with torch.no_grad():  # PiCO confidence update (Eq. 6) on all samples, within candidates
                z = F.one_hot((o['score_prot'] * Y).argmax(1), C).float()
                confidence[index] = conf_ema_m * confidence[index] + (1 - conf_ema_m) * z
            plab = o['pool_lab']
            mask_all = (plab[:B, None] == plab[None, :]).float()
            loss_cont_all = supcon(feats, mask_all, B, tau)
            mask = mask_all * w_rel[:, None] * o['pool_rel'].float()[None, :]
            if knn_on:
                with torch.no_grad():
                    knn_idx = (feats[:B] @ feats.t()).topk(a['chosen_neighbors'], dim=1)[1]
                    mask_knn = torch.zeros_like(mask).scatter_(1, knn_idx, 1.0)
                mask[~is_rel] = mask_knn[~is_rel]
            mask[:, B:2 * B] = ((mask[:, B:2 * B] + diag) > 0).float()
            mask[:, :B] = ((mask[:, :B] + diag) > 0).float()
            weights = a['loss_weight'] * w_rel + a['ur_weight'] * (1 - w_rel) if knn_on else None
            loss_cont_rel = supcon(feats, mask, B, tau, weights)
            loss_cont = loss_cont_rel + a['ur_weight'] * loss_cont_all

            conf_b = confidence[index]
            loss_cls_rel = soft_ce(out, conf_b)[is_rel].mean() if is_rel.any() else out.sum() * 0
            sp = o['score_prot'] ** (1 / a['temperature_guess'])
            guess = sp / sp.sum(dim=1, keepdim=True)
            loss_cls_ur = soft_ce(out, guess)[~is_rel].mean() if (~is_rel).any() else out.sum() * 0

            lam = np.random.beta(a['mix_beta'], a['mix_beta'])
            lam = max(lam, 1 - lam)
            target = conf_b.clone()
            target[~is_rel] = guess[~is_rel]
            perm = torch.randperm(B, device=device)
            logits_mix, _ = model.encoder_q(lam * images_w + (1 - lam) * images_w[perm])
            loss_mix = soft_ce(logits_mix, lam * target + (1 - lam) * target[perm]).mean()

            loss_cls = loss_mix + a['cls_weight'] * loss_cls_rel + a['ur_weight'] * loss_cls_ur
            loss = loss_cls + loss_cont
        else:
            loss_cls = soft_ce(out, confidence[index]).mean()
            loss_cont = moco_loss(feats, B, tau)
            loss = loss_cls + a['loss_weight'] * loss_cont

        sel['dist'][index] = o['dist'].detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            hit_w = (o['within'] == true).float()
            acc.mean('pl_acc', hit_w.mean())
            acc.mean('pl_used_acc', (o['pl_b'] == true).float().mean())
            acc.mean('cls_acc_unmasked', (o['allmax'] == true).float().mean())
            acc.mean('conf_acc', (confidence[index].argmax(1) == true).float().mean())
            acc.mean('proto_acc', (o['score_prot'].argmax(1) == true).float().mean())
            acc.mean('proto_masked_acc', ((o['score_prot'] * Y).argmax(1) == true).float().mean())
            acc.add('rel_n', w_rel.sum())
            acc.add('rel_hit', (hit_w * w_rel).sum())
            acc.add('ur_n', (1 - w_rel).sum())
            acc.add('ur_hit_within', (hit_w * (1 - w_rel)).sum())
            acc.add('ur_hit_allmax', ((o['allmax'] == true).float() * (1 - w_rel)).sum())
            if started:
                g = guess.argmax(1)
                acc.add('ur_guess_hit', ((g == true).float() * (1 - w_rel)).sum())
                acc.add('ur_guess_out', ((Y.gather(1, g[:, None])[:, 0] == 0).float() * (1 - w_rel)).sum())
                pv = mask.clone()
                pv[torch.arange(B), torch.arange(B)] = 0.0  # self (removed by SupCon's logits mask)
                valid = (o['pool_true'] >= 0).float()[None, :]
                same = (true[:, None] == o['pool_true'][None, :]).float()
                acc.add('pos_n', (pv * valid).sum())
                acc.add('pos_tp', (pv * valid * same).sum())
                acc.mean('loss_mix', loss_mix)
                acc.mean('loss_cls_rel', loss_cls_rel)
                acc.mean('loss_cls_ur', loss_cls_ur)
                acc.mean('loss_cont_all', loss_cont_all)
                acc.mean('loss_cont_rel', loss_cont_rel)
            acc.mean('loss_cls', loss_cls)
            acc.mean('loss_cont', loss_cont)
            acc.mean('loss_total', loss)
    row = {'epoch': epoch + 1, 'started': int(started), 'knn_on': int(knn_on), 'conf_ema_m': conf_ema_m,
           'rel_frac': acc.s.get('rel_n', 0.0) / max(acc.s.get('rel_n', 0.0) + acc.s.get('ur_n', 0.0), 1.0),
           'rel_pl_acc': acc.ratio('rel_hit', 'rel_n'), 'ur_pl_acc_within': acc.ratio('ur_hit_within', 'ur_n'),
           'ur_pl_acc_allmax': acc.ratio('ur_hit_allmax', 'ur_n'), 'ur_guess_acc': acc.ratio('ur_guess_hit', 'ur_n'),
           'ur_guess_out_of_cand': acc.ratio('ur_guess_out', 'ur_n'), 'pos_prec': acc.ratio('pos_tp', 'pos_n')}
    for k in sorted(acc.m):
        row[k] = acc.avg(k)
    return row


def reliable_set_selection(sel, pure_ratio):
    d = sel['dist']
    n = d.shape[0]
    is_rel = torch.zeros(n, dtype=torch.bool, device=d.device)
    n_valid = int(torch.isfinite(d).sum().item())  # held-out samples carry dist=inf (never selected)
    is_rel[torch.argsort(d)[:int(n_valid * pure_ratio)]] = True
    sel['is_rel'] = is_rel


def make_plus_runner(algorithm):
    flags = PLUS_VARIANTS[algorithm]

    def run(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        from src.pipeline.algorithms.hparams import make_optimizer
        from src.pll_init import candidate_masked_init
        pc = raw_cfg['pico']
        a = {'num_class': C, 'epochs': epochs, 'low_dim': pc['low_dim'], 'moco_queue': pc['moco_queue'],
             'moco_m': pc['moco_m'], 'proto_m': pc['proto_m'], 'loss_weight': pc['loss_weight'],
             'conf_ema_range': pc['conf_ema_range'], **PLUS_DEFAULTS, **flags, **raw_cfg.get('picoplus', {})}  # yaml picoplus: overrides (hptune 11-12)
        a['knn_start'] = a.get('knn_start', int(round(epochs * a['knn_start_frac'])))
        model = PiCOPlusModel(a).to(device)
        confidence = candidate_masked_init(pl_ds.targets, C).to(device)
        n = confidence.shape[0]
        sel = {'dist': torch.zeros(n, device=device), 'is_rel': torch.ones(n, dtype=torch.bool, device=device)}
        if 'train_idx' in loaders:  # --holdout_frac: held-out rows are never updated -> keep them out of the
            sel['dist'].fill_(float('inf'))  # reliable-set quota
            sel['dist'][torch.as_tensor(loaders['train_idx'], device=device)] = 0.0
        loader = loaders['pico']
        opt = make_optimizer(model, hparams)
        log = _CsvLog(raw_cfg, algorithm, C)
        true_all = torch.as_tensor(np.asarray(orig_targets), device=device) if orig_targets is not None else None
        true_in_Y = None  # noisy PLL diag (11-17): which rows really contain y
        if true_all is not None and true_all.shape[0] == n:
            cand = candidate_masked_init(pl_ds.targets, C).to(device) > 0
            true_in_Y = cand.gather(1, true_all.view(-1, 1)).squeeze(1)
        t0 = time.perf_counter()
        for ep in range(epochs):
            te = time.perf_counter()
            if ep >= a['plus_warmup']:
                reliable_set_selection(sel, a['pure_ratio'])
            conf_ema_m = 1.0 * ep / epochs * (a['conf_ema_range'][1] - a['conf_ema_range'][0]) + a['conf_ema_range'][0]
            row = train_picoplus_epoch(a, model, loader, confidence, conf_ema_m, opt, ep, device, sel, C)
            if true_all is not None and true_all.shape[0] == n:
                row['conf_acc_all'] = (confidence.argmax(1) == true_all).float().mean().item()
            if true_in_Y is not None:
                rel = sel['is_rel']
                row['frac_true_clean'] = true_in_Y.float().mean().item()
                row['sel_clean_prec'] = true_in_Y[rel].float().mean().item() if rel.any() else float('nan')
                row['sel_noisy_recall'] = ((~rel) & (~true_in_Y)).sum().item() / max(1, (~true_in_Y).sum().item())
            row['test_acc'] = (evaluate_model(model, loaders['test'], device)
                               if (ep + 1) % 10 == 0 or ep + 1 == epochs or ep < 5 else float('nan'))
            row['epoch_time_s'] = time.perf_counter() - te
            log.write(row)
            if (ep + 1) % report_every == 0 or ep + 1 == epochs:
                _print_eta(tag, ep + 1, epochs, time.perf_counter() - t0, min(report_every, ep + 1))
                t0 = time.perf_counter()
        acc = evaluate_model(model, loaders['test'], device)
        del model, opt, confidence
        torch.cuda.empty_cache()
        return acc

    return run


PLUS_RUNNERS = {name: make_plus_runner(name) for name in PLUS_VARIANTS}
