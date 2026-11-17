"""cls-loss swap experiment (2026-10-06): every distinct classification loss
used so far in this repo, plugged into the PiCO-Fixed and ComCo-Fixed
architectures with ONLY L_cls changed.

Each loss here is a dense-mask re-implementation of the original class
(partial_Y: [B, C] binary candidate mask, comp_mask = 1 - partial_Y), checked
numerically against the original in verify_scripts/check_clsloss_swap.py.
Every loss exposes the same interface so the two framework loops below can
drive any of them:

    loss_fn.set_epoch(epoch, conf_ema_range, epochs)     # once per epoch
    loss_fn.pre_update(score_prot, logits, index, partial_Y, start_upd_prot)
                                                         # before forward loss
    loss_fn(logits, index, partial_Y) -> scalar          # may update state

Loss keys (CLS_LOSSES):
  PLL side
    UCE          -(1/k) sum_{j in Y} log p_j                (PRODEN paper's "PRODEN-naive" ablation, i.e.
                 PRODEN with the uniform weights never updated; src/cour_loss.py. Not
                 Cour 2011 -- its "naive" loss (Jin & Ghahramani 2002) is psi on raw scores)
    CLPL         squared-hinge CLPL on raw logits          (src/clpl_loss.py)
    Wu2022       proper PLL estimator                      (src/wu_loss.py)
    PRODEN       -sum w_j log p_j, w <- renormalized softmax in Y, every batch
                                                           (src/proden_loss.py)
    PiCOPartial  -sum w_j log p_j, w <- EMA one-hot argmax of prototype score
                                                           (PartialLoss, native PiCO-Fixed)
    PiCOCLS      same, but argmax of classifier softmax    (src/pico_cls_loss.py / A' of ab_sweep)
    SoLar        Sinkhorn pseudo-label CE on reliable samples + PRODEN-style
                 loss on the rest (src/engine.py::train_solar_epoch, see SoLarLoss)
  CLL side (comp set = non-candidates, m = C - k)
    MCLLOG       -log sum_{j in Y} p_j * (C-1)/(C-m)       (MCL-LOG == PiCO-MCL == original ComCo)
    MCLLOGFixed  -log sum_{j in Y} p_j * 2(C-1)/m          (src/fixed_mcl_losses.py)
    MCLNL        -log sum_{j in Y} p_j, unscaled           (ComCo's MCL-NL: Feng et al.'s L_LOG without the
                 risk-estimator factor; ComCo Sec 5.4.2 "NL indicates SCL-NL/MCL-NL in
                 CLL / MCLL". Equals SCL-NL at m = 1 and CE at k = 1)
    SCLNL        (1/m) sum_{ybar} -log(1 - p_ybar)         (SCL-NL == ComCo-Fixed native)
    OP           (1/m) sum_{ybar} -log softmax(-g)_ybar    (src/op_loss.py)
    OPW          weighted OP                               (src/op_loss.py)

CPE-I is deliberately absent: -log softmax(g)_ybar with argmin prediction is
exactly OP under g -> -g (the last layer is sign-symmetric), so it is the
same experiment as OP.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.pico.utils_loss import PartialLoss
from src.solar.utils_algo import linear_rampup, sinkhorn


# ─── stateless losses ─────────────────────────────────────────────────────


class _Stateless(nn.Module):
    def set_epoch(self, epoch, conf_ema_range, epochs):
        pass

    def pre_update(self, score_prot, logits, index, partial_Y, start_upd_prot):
        pass


class UCELoss(_Stateless):
    def forward(self, logits, index, partial_Y):
        k = partial_Y.sum(dim=1).clamp(min=1)
        return (-(partial_Y * F.log_softmax(logits, dim=1)).sum(dim=1) / k).mean()


class CLPLLoss(_Stateless):
    def forward(self, logits, index, partial_Y):
        k = partial_Y.sum(dim=1).clamp(min=1)
        pos = F.relu(1.0 - (logits * partial_Y).sum(dim=1) / k).pow(2)
        neg = ((1.0 - partial_Y) * F.relu(1.0 + logits).pow(2)).sum(dim=1)
        return (pos + neg).mean()


class Wu2022Loss(_Stateless):
    def forward(self, logits, index, partial_Y):
        C = logits.shape[1]
        log_sm = F.log_softmax(logits, dim=1)
        k = partial_Y.sum(dim=1).clamp(min=1)
        sum_cand = -(log_sm * partial_Y).sum(dim=1)
        sum_all = -log_sm.sum(dim=1)
        return ((C - 1.0) / (C - k) * sum_cand - (k - 1.0) / (C - k) * sum_all).mean()


class MCLLOGLoss(_Stateless):
    """Original (buggy-scaling) MCL-LOG; identical to PiCOMCLLoss and the
    original ComCoCLSLoss, since (C-1)/(C-m) == (C-1)/k."""

    def forward(self, logits, index, partial_Y):
        C = logits.shape[1]
        k = partial_Y.sum(dim=1).clamp(min=1)
        s = (F.softmax(logits, dim=1) * partial_Y).sum(dim=1)
        return (-torch.log(s + 1e-7) * (C - 1.0) / k).mean()


class MCLLOGFixedLoss(_Stateless):
    def forward(self, logits, index, partial_Y):
        C = logits.shape[1]
        m = (C - partial_Y.sum(dim=1)).clamp(min=1)
        s = (F.softmax(logits, dim=1) * partial_Y).sum(dim=1)
        return (-torch.log(s + 1e-7) * 2.0 * (C - 1.0) / m).mean()


class MCLNLLoss(_Stateless):
    """Unscaled MCL-NL / L_LOG: ComCo's classification loss for multiple
    complementary labels. MCLLOGLoss / MCLLOGFixedLoss times a constant."""

    def forward(self, logits, index, partial_Y):
        s = (F.softmax(logits, dim=1) * partial_Y).sum(dim=1)
        return (-torch.log(s + 1e-7)).mean()


class SCLNLLoss(_Stateless):
    def forward(self, logits, index, partial_Y):
        comp = 1.0 - partial_Y
        m = comp.sum(dim=1).clamp(min=1)
        per = -torch.log1p(-F.softmax(logits, dim=1).clamp(max=1.0 - 1e-7))
        return ((per * comp).sum(dim=1) / m).mean()


class OPLoss(_Stateless):
    def forward(self, logits, index, partial_Y):
        comp = 1.0 - partial_Y
        m = comp.sum(dim=1).clamp(min=1)
        per = -F.log_softmax(-logits, dim=1)
        return ((per * comp).sum(dim=1) / m).mean()


class OPWLoss(_Stateless):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, index, partial_Y):
        comp = 1.0 - partial_Y
        m = comp.sum(dim=1).clamp(min=1)
        base = -F.log_softmax(-logits, dim=1)
        u = 1.0 / F.softmax(-logits, dim=1).clamp(min=1e-7)
        weight = F.softmax(u + 1.0, dim=1) * F.softmax(logits, dim=1) + self.eps
        return ((weight * base * comp).sum(dim=1) / m).mean()


# ─── confidence-buffer losses ─────────────────────────────────────────────


class PRODENLoss(nn.Module):
    """ProdenLoss.forward with conf_ema_m = 0: loss uses the stored conf,
    then conf[index] is hard-overwritten with the current candidate-renormalized
    softmax (one-step lag, every batch, no warm-up gate)."""

    def __init__(self, init_conf):
        super().__init__()
        self.conf = init_conf

    def set_epoch(self, epoch, conf_ema_range, epochs):
        pass

    def pre_update(self, score_prot, logits, index, partial_Y, start_upd_prot):
        pass

    def forward(self, logits, index, partial_Y):
        conf = self.conf[index]
        loss = -(conf * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        with torch.no_grad():
            new = (conf > 0).float() * torch.softmax(logits, dim=1)
            self.conf[index] = new / new.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return loss


class PiCOEMALoss(nn.Module):
    """PartialLoss with its EMA hard one-hot confidence update, gated by
    prot_start exactly as in train_pico_epoch_fixed. source='prototype' is
    native PiCO-Fixed (score_prot); source='classifier' is PiCO-CLS's signal
    (softmax(logits)), i.e. ab_sweep's A'+B."""

    def __init__(self, init_conf, source: str):
        super().__init__()
        assert source in ('prototype', 'classifier')
        self.source = source
        self.partial = PartialLoss(init_conf)

    def set_epoch(self, epoch, conf_ema_range, epochs):
        self.partial.set_conf_ema_m(epoch, {'conf_ema_range': conf_ema_range, 'epochs': epochs})

    def pre_update(self, score_prot, logits, index, partial_Y, start_upd_prot):
        if not start_upd_prot:
            return
        sig = score_prot if self.source == 'prototype' else torch.softmax(logits, dim=1)
        self.partial.confidence_update(temp_un_conf=sig.detach(), batch_index=index, batchY=partial_Y)

    def forward(self, logits, index, partial_Y):
        return self.partial(logits, index)


class SoLarLoss(nn.Module):
    """SoLar's classification objective (src/engine.py::train_solar_epoch),
    reduced to what can be computed from the framework's single weak-view
    forward pass:

      * pseudo-labels: Sinkhorn-Knopp over [queue; softmax(logits)*Y], with
        a UNIFORM target class marginal -- exact here, since every CIFAR-100
        class subset is balanced (SoLar's stage-1 distribution estimation is
        therefore skipped);
      * reliable set: per-class small-loss selection (rho ramped over
        warmup_epoch) plus high-confidence samples (> tau);
      * loss = eta * CE(pseudo) on reliable + (1 - eta) * confidence-weighted
        CE on the rest; epoch 0 uses the confidence-weighted CE only;
      * confidence buffer: candidate-renormalized softmax every batch.

    Dropped vs. full SoLar: the strong-view consistency term and the mixup
    term (both need extra forward passes through the classifier, i.e. they
    are augmentation regularizers, not part of L_cls), and the 100-epoch
    pre-estimation stage."""

    def __init__(self, init_conf, solar_cfg: dict, num_class: int, queue_len: int):
        super().__init__()
        self.conf = init_conf
        self.cfg = solar_cfg
        self.C = num_class
        self.queue = torch.zeros(queue_len, num_class, device=init_conf.device)
        self.emp_dist = (torch.ones(num_class, device=init_conf.device) / num_class).unsqueeze(1)
        self.epoch = 0

    def set_epoch(self, epoch, conf_ema_range, epochs):
        self.epoch = epoch

    def pre_update(self, score_prot, logits, index, partial_Y, start_upd_prot):
        pass

    def forward(self, logits, index, partial_Y):
        cfg, bs = self.cfg, logits.shape[0]
        ramp = linear_rampup(self.epoch, cfg['warmup_epoch'])
        eta = cfg['eta'] * ramp
        rho = cfg['rho_range'][0] + (cfg['rho_range'][1] - cfg['rho_range'][0]) * ramp

        log_sm = F.log_softmax(logits, dim=1)
        rn_loss_vec = -(self.conf[index] * log_sm).sum(dim=1)

        with torch.no_grad():
            prediction = torch.softmax(logits.detach(), dim=1)
            cost = prediction * partial_Y
            sk_in = cost if torch.all(self.queue[-1] == 0) else torch.cat((self.queue, cost))
            self.queue[bs:] = self.queue[:-bs].clone()
            self.queue[:bs] = cost
            pseudo = sinkhorn(sk_in, cfg['lamd'], r_in=self.emp_dist)[0][-bs:]
            self.conf[index] = cost / cost.sum(dim=1, keepdim=True).clamp(min=1e-8)

        if self.epoch < 1:
            return rn_loss_vec.mean()

        pseudo_loss_vec = -(pseudo * log_sm).sum(dim=1)
        with torch.no_grad():
            sel = torch.zeros(bs, dtype=torch.bool, device=logits.device)
            pidx = pseudo.argmax(dim=1)
            plv = pseudo_loss_vec.detach()
            for j in range(self.C):
                members = torch.where(pidx == j)[0]
                if members.numel() == 0:
                    continue
                n_j = max(min(int(math.ceil(bs * float(self.emp_dist[j]) * rho)), members.numel()), 1)
                sel[members[plv[members].argsort()[:n_j]]] = True
            sel |= (pseudo * prediction).sum(dim=1) > cfg['tau']
        if not sel.any():
            return rn_loss_vec.mean()
        loss_unrel = rn_loss_vec[~sel].mean() if (~sel).any() else rn_loss_vec.sum() * 0
        # Exposed for the gradient-conflict diagnostic (src/clsloss_solo.py);
        # no effect on training.
        self.last_terms = (pseudo_loss_vec[sel].mean() * eta, loss_unrel * (1 - eta))
        return self.last_terms[0] + self.last_terms[1]


# ─── per-epoch arm switch for the paired bandit ───────────────────────────


class SwapMixClsLoss(nn.Module):
    """Hard switch between two swap cls losses (2026-10-27), for the paired
    bandit in runners._generic_bandit_setup ('SwapComCo'): self.value 0 ->
    arm0, 1 -> arm1, set before each (lookahead or real) epoch. Both arms'
    confidence state stays live on every batch, whichever arm drives the
    gradient -- same convention as ComCoWeightedClsLoss in
    train_comco_weighted_switch_epoch. `partial_loss.confidence` exposes the
    (single) stateful arm's buffer so comco_bandit.snapshot/restore roll it
    back between lookahead arms."""

    def __init__(self, arm0, arm1):
        super().__init__()
        self.arm0, self.arm1 = arm0, arm1
        self.value = 0.0
        bufs = [b for b in (_conf_buffer(arm0), _conf_buffer(arm1)) if b is not None]
        assert len(bufs) <= 1, 'SwapMixClsLoss supports at most one stateful arm'
        self.partial_loss = SimpleNamespace(confidence=bufs[0] if bufs else torch.zeros(1))

    def set_conf_ema_m(self, epoch, args):
        self.set_epoch(epoch, args['conf_ema_range'], args['epochs'])

    def set_epoch(self, epoch, conf_ema_range, epochs):
        self.arm0.set_epoch(epoch, conf_ema_range, epochs)
        self.arm1.set_epoch(epoch, conf_ema_range, epochs)

    def pre_update(self, score_prot, logits, index, partial_Y, start_upd_prot):
        self.arm0.pre_update(score_prot, logits, index, partial_Y, start_upd_prot)
        self.arm1.pre_update(score_prot, logits, index, partial_Y, start_upd_prot)

    def forward(self, logits, index, partial_Y):
        active, idle = (self.arm1, self.arm0) if self.value >= 0.5 else (self.arm0, self.arm1)
        if isinstance(idle, PRODENLoss):  # PRODEN updates its buffer inside forward
            with torch.no_grad():
                idle(logits.detach(), index, partial_Y)
        return active(logits, index, partial_Y)


# ─── candidate pruning: reduce large-k (MCLL) to PLL (2026-10-28) ─────────


class CandidatePruneLoss(nn.Module):
    """Warm up with MCL-LOG-Fixed on the full candidate set Y_i, then train a
    PLL loss (PiCOPartial / PRODEN) on a pruned set S_i subset of Y_i.

    q_i = EMA (q_m) of the candidate-masked, renormalized softmax, updated on
    every batch from the training forward pass (no extra compute). prune(r)
    sets S_i = top-r of q_i within Y_i, recomputed from scratch each call, so a
    wrongly dropped label can come back. oracle=True instead keeps the true
    label plus r-1 false candidates in a fixed per-sample random order
    (nested sets; upper bound only, uses labels). The PLL arm's confidence is
    masked to S_i and renormalized at every prune, with a small floor so
    re-admitted labels are reachable for PRODEN's (conf > 0) mask."""

    def __init__(self, init_conf, pl_name: str, num_class: int, warm_epochs: int,
                 true_labels=None, q_m: float = 0.9, seed: int = 0):
        super().__init__()
        self.Y = (init_conf > 0).float()
        self.S = self.Y.clone()
        self.q = init_conf.clone()
        self.warm_loss = MCLLOGFixedLoss()
        self.pl = build_cls_loss(pl_name, init_conf.clone(), num_class, None, None)
        self.conf = _conf_buffer(self.pl)
        self.warm_epochs, self.q_m, self.epoch = warm_epochs, q_m, 0
        self.oracle_order = None
        if true_labels is not None:
            g = torch.Generator().manual_seed(seed)
            y = torch.as_tensor(true_labels, device=init_conf.device).long()
            # rank: true label first, false candidates in random order, non-candidates last
            key = torch.rand(self.Y.shape, generator=g).to(init_conf.device) + 2.0 * (1.0 - self.Y)
            key[torch.arange(len(y), device=y.device), y] = -1.0
            self.oracle_order = key.argsort(dim=1)

    def set_conf_ema_m(self, epoch, args):
        self.set_epoch(epoch, args['conf_ema_range'], args['epochs'])

    def set_epoch(self, epoch, conf_ema_range, epochs):
        self.epoch = epoch
        self.pl.set_epoch(epoch, conf_ema_range, epochs)

    def pre_update(self, score_prot, logits, index, partial_Y, start_upd_prot):
        if self.epoch >= self.warm_epochs:
            self.pl.pre_update(score_prot, logits, index, self.S[index], start_upd_prot)

    def forward(self, logits, index, partial_Y):
        with torch.no_grad():
            p = F.softmax(logits.detach(), dim=1) * self.Y[index]
            self.q[index] = self.q_m * self.q[index] + (1 - self.q_m) * p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if self.epoch < self.warm_epochs:
            return self.warm_loss(logits, index, partial_Y)
        return self.pl(logits, index, self.S[index])

    @torch.no_grad()
    def prune(self, r: int):
        r = max(1, min(int(r), self.Y.shape[1]))
        if self.oracle_order is not None:
            keep = self.oracle_order[:, :r]
        else:
            keep = self.q.masked_fill(self.Y == 0, -1.0).topk(r, dim=1).indices
        self.S = torch.zeros_like(self.Y).scatter_(1, keep, 1.0) * self.Y
        conf = self.conf * self.S + 1e-4 * self.S
        self.conf.copy_(conf / conf.sum(dim=1, keepdim=True))


def _conf_buffer(loss):
    if isinstance(loss, PRODENLoss):
        return loss.conf
    if isinstance(loss, PiCOEMALoss):
        return loss.partial.confidence
    if isinstance(loss, _Stateless):
        return None
    raise ValueError(f'SwapMixClsLoss: no snapshot support for {type(loss).__name__}')


# ─── registry ─────────────────────────────────────────────────────────────

CLS_LOSSES = ['UCE', 'CLPL', 'Wu2022', 'PRODEN', 'PiCOPartial', 'PiCOCLS', 'SoLar',
              'MCLLOG', 'MCLLOGFixed', 'MCLNL', 'SCLNL', 'OP', 'OPW']
FRAMEWORKS = ['PiCO', 'ComCo']


def swap_algorithm_name(framework: str, loss: str) -> str:
    return f'Swap{framework}-{loss}'


def build_cls_loss(name: str, init_conf, num_class: int, solar_cfg: dict, batch_size: int):
    stateless = {'UCE': UCELoss, 'CLPL': CLPLLoss, 'Wu2022': Wu2022Loss, 'MCLLOG': MCLLOGLoss,
                 'MCLLOGFixed': MCLLOGFixedLoss, 'MCLNL': MCLNLLoss, 'SCLNL': SCLNLLoss, 'OP': OPLoss, 'OPW': OPWLoss}
    if name in stateless:
        return stateless[name]()
    if name == 'PRODEN':
        return PRODENLoss(init_conf)
    if name == 'PiCOPartial':
        return PiCOEMALoss(init_conf, 'prototype')
    if name == 'PiCOCLS':
        return PiCOEMALoss(init_conf, 'classifier')
    if name == 'SoLar':
        return SoLarLoss(init_conf, solar_cfg, num_class, queue_len=64 * batch_size)
    raise ValueError(f'unknown cls loss {name!r}')


# ─── framework epochs ─────────────────────────────────────────────────────


def train_pico_swap_epoch(pico_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """train_pico_epoch_fixed with the cls loss swapped: L_cont omitted
    entirely during warm-up (epoch < prot_start), masked SupCon afterwards."""
    model.train()
    total_loss = 0
    start_upd_prot = epoch >= pico_args['prot_start']
    loss_fn.set_epoch(epoch, pico_args['conf_ema_range'], pico_args['epochs'])

    bar = tqdm(loader, desc=f"SwapPiCO Epoch {epoch + 1}/{pico_args['epochs']}")
    for (images_w, images_s, partial_Y, true_labels, index) in bar:
        images_w, images_s, partial_Y, index = (
            images_w.to(device), images_s.to(device), partial_Y.to(device), index.to(device))
        cls_out, features, pseudo_target_cont, score_prot = model(images_w, images_s, partial_Y, pico_args)
        batch_size = cls_out.shape[0]

        loss_fn.pre_update(score_prot, cls_out, index, partial_Y, start_upd_prot)
        loss_cls = loss_fn(cls_out, index, partial_Y)
        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size].unsqueeze(1), pseudo_target_cont.unsqueeze(0)).float()
            loss = loss_cls + pico_args['loss_weight'] * loss_cont_fn(features=features, mask=mask,
                                                                       batch_size=batch_size)
        else:
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=total_loss / (bar.n + 1))
    return total_loss / len(loader)


def train_comco_swap_epoch(comco_args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device):
    """train_comco_weighted_epoch with the cls loss swapped (ComCoWeightedModel
    supplies score_prot for PiCOPartial; L_cont active from epoch 0)."""
    model.train()
    total_loss = 0
    warmup_pos = epoch >= comco_args['warmup_pos']
    warmup_neg = epoch >= comco_args['warmup_neg']
    start_upd_prot = epoch >= comco_args['prot_start']
    loss_fn.set_epoch(epoch, comco_args['conf_ema_range'], comco_args['epochs'])

    bar = tqdm(loader, desc=f"SwapComCo Epoch {epoch + 1}/{comco_args['epochs']}")
    for (images_w, images_s, comp_mask, true_labels, index) in bar:
        images_w, images_s, comp_mask, index = (
            images_w.to(device), images_s.to(device), comp_mask.to(device), index.to(device))
        partial_Y = 1.0 - comp_mask
        cls_out, q, all_feats, all_pseudo, all_comp, score_prot = model(
            images_w, images_s, comp_mask, partial_Y, comco_args)
        pseudo_q = cls_out.argmax(dim=1)

        loss_fn.pre_update(score_prot, cls_out, index, partial_Y, start_upd_prot)
        loss_cls = loss_fn(cls_out, index, partial_Y)
        loss_cont = loss_cont_fn(q, all_feats, all_pseudo, all_comp, pseudo_q, warmup_pos, warmup_neg)
        loss = loss_cls + comco_args['loss_weight'] * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=total_loss / (bar.n + 1))
    return total_loss / len(loader)
