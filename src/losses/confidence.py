"""Confidence-buffer (soft-label) classification losses, moved verbatim 2026-11-22 from
proden_loss.py, pico/utils_loss.py, pico_cls_loss.py and solar/utils_loss.py.

Each keeps a per-sample soft label ("confidence") over the C classes and minimises
weighted_ce(logits, conf). They differ only in how/when conf is updated:
  ProdenLoss    renormalised softmax within Y, every batch (inside forward), optional EMA
  PartialLoss   PiCO: EMA towards one-hot(argmax(score * Y)) via confidence_update(...)
  PiCOCLSLoss   PiCO-CLS / -SC: EMA one-hot of the classifier argmax within Y
  SolarPartialLoss  SoLar's confidence (overwritten by its Sinkhorn step)
conf_buffer(loss) returns the [N, C] buffer of any of them (and of the swap/composite losses).
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pll_init import (biased_all_init, biased_candidates_init, biased_oracle_init,
                           biased_partial_random_init, biased_random_all_init,
                           candidate_masked_init, random_candidate_init, uniform_all_init)


class ProdenLoss(nn.Module):
    """
    PRODEN with cross-epoch confidence accumulation (original paper algorithm).

    Verified against Lv et al., ICML 2020, Algorithm 1 (2026-08-14) -- exact
    match, including the one-step lag between loss computation (uses the
    previously stored confidence) and the confidence refresh (uses the
    current forward pass, under no_grad). This is the class actually used by
    the new pipeline (run_proden). See docs/proden_explanation.md.

    Maintains a persistent confidence matrix conf[N, C] for every training sample:
      - Initialised uniformly over candidate labels (or another init_mode, see below).
      - forward() uses the *stored* confidence as soft-label weights for the loss,
        then updates conf[indices] in-place with the current model's renormalised
        softmax restricted to the candidate set Y -- ready for the next batch. The
        renormalize-every-batch update is UNCHANGED regardless of init_mode (it always
        uses the stored Y, never the support of conf): only the very first batch(es)
        touching a given sample still reflect its initial confidence, after which it's
        overwritten by the model's own prediction -- see
        src/pipeline/algorithms/runners.py's PRODEN-UniformInit / PRODEN-BiasedInit
        variants, which exist specifically to demonstrate this insensitivity as a
        contrast against PiCO's slow EMA-driven confidence update.

    Args:
        partial_targets: list of 1-D LongTensors, one per sample (candidate indices).
        num_classes:     total number of classes C.
        init_mode:       'candidate_masked' (default, existing behavior) | 'uniform_all'
                          | 'biased_oracle' | 'biased_candidates' | 'biased_all'
                          | 'biased_partial_random' | 'biased_random_all' | 'random_candidate'
                          -- see src/pll_init.py.
        orig_targets:    true labels, required for init_mode in
                          {'biased_oracle', 'biased_candidates', 'biased_all', 'biased_partial_random',
                          'biased_random_all'}.
        true_weight:     weight given to the true class; required for init_mode in
                          {'biased_candidates', 'biased_all', 'biased_partial_random',
                          'biased_random_all'} (parametrized sweep, see src/pll_init.py.BIAS_WEIGHTS /
                          BIAS_RAND_WEIGHTS / BIAS_RAND_ALL_WEIGHTS).
        wf:              number of other classes the remaining weight is spread across; required for
                          init_mode in {'biased_partial_random', 'biased_random_all'}. For
                          'biased_partial_random' these are drawn from the sample's own candidate set
                          (see src/pll_init.py.BIAS_RAND_WF_VALUES); for 'biased_random_all' they're
                          drawn from ALL other classes instead (see BIAS_RAND_ALL_N_VALUES) -- same
                          parameter slot, reused rather than adding a separate `n` kwarg since both
                          modes just need "how many others" as a single int.
    """

    def __init__(self, partial_targets: list, num_classes: int,
                 init_mode: str = 'candidate_masked', orig_targets=None, true_weight: float = None,
                 wf: int = None, conf_ema_range=None):
        super().__init__()
        # conf_ema_range: optional [start, end] EMA momentum schedule for the
        # confidence update, mirroring PiCO's PartialLoss.set_conf_ema_m
        # (linear from start at epoch 0 to end at the last epoch). None (the
        # default) keeps the original PRODEN behaviour: conf_ema_m stays 0.0,
        # i.e. every batch hard-overwrites conf with the renormalized softmax
        # -- see forward() and the 2026-09-15 conf_ema sweep in
        # src/pipeline/algorithms/runners.py (CONF_EMA_SWEEP_RUNNERS).
        self.conf_ema_range = list(conf_ema_range) if conf_ema_range is not None else None
        self.conf_ema_m = 0.0
        if init_mode == 'candidate_masked':
            conf = candidate_masked_init(partial_targets, num_classes)
        elif init_mode == 'uniform_all':
            conf = uniform_all_init(len(partial_targets), num_classes)
        elif init_mode == 'biased_oracle':
            if orig_targets is None:
                raise ValueError("init_mode='biased_oracle' requires orig_targets")
            conf = biased_oracle_init(partial_targets, orig_targets, num_classes)
        elif init_mode == 'biased_candidates':
            if orig_targets is None or true_weight is None:
                raise ValueError("init_mode='biased_candidates' requires orig_targets and true_weight")
            conf = biased_candidates_init(partial_targets, orig_targets, num_classes, true_weight)
        elif init_mode == 'biased_all':
            if orig_targets is None or true_weight is None:
                raise ValueError("init_mode='biased_all' requires orig_targets and true_weight")
            conf = biased_all_init(partial_targets, orig_targets, num_classes, true_weight)
        elif init_mode == 'biased_partial_random':
            if orig_targets is None or true_weight is None or wf is None:
                raise ValueError("init_mode='biased_partial_random' requires orig_targets, true_weight, and wf")
            conf = biased_partial_random_init(partial_targets, orig_targets, num_classes, true_weight, wf)
        elif init_mode == 'biased_random_all':
            if orig_targets is None or true_weight is None or wf is None:
                raise ValueError("init_mode='biased_random_all' requires orig_targets, true_weight, and wf")
            conf = biased_random_all_init(orig_targets, num_classes, true_weight, wf)
        elif init_mode == 'random_candidate':
            conf = random_candidate_init(partial_targets, num_classes)
        else:
            raise ValueError(f'Unknown init_mode {init_mode!r}')
        self.register_buffer('conf', conf)   # [N, C], lives on same device as model
        # The candidate set Y is stored separately (2026-11-22 fix): the update used to derive it
        # from `conf > 0`, so any init whose support differs from Y (uniform_all, biased_oracle,
        # biased_all, biased_partial_random, biased_random_all) kept that support for the WHOLE run
        # (oracle leak or no candidate constraint), not just at initialisation.
        self.register_buffer('cand', candidate_masked_init(partial_targets, num_classes).gt(0).float())

    def set_conf_ema_m(self, epoch: int, epochs: int):
        """Same linear schedule as PiCO's PartialLoss.set_conf_ema_m. No-op
        (conf_ema_m stays 0.0 == original hard overwrite) when the loss was
        constructed without conf_ema_range."""
        if self.conf_ema_range is None:
            self.conf_ema_m = 0.0
            return
        start, end = self.conf_ema_range
        self.conf_ema_m = 1. * epoch / epochs * (end - start) + start

    def forward(self, outputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        outputs:  [B, C] logits
        indices:  [B]    integer sample indices into the full training set
        """
        conf = self.conf[indices]                         # [B, C] stored soft labels

        # Loss: weighted CE with last-step confidence
        log_probs = F.log_softmax(outputs, dim=1)         # [B, C]
        loss = -(conf * log_probs).sum(dim=1).mean()

        # Update: renormalise current softmax within candidate mask, then
        # blend into the stored conf with EMA momentum conf_ema_m. With the
        # default conf_ema_m == 0 this is exactly the original PRODEN
        # hard overwrite (conf <- new_conf); conf_ema_m > 0 keeps a
        # PiCO-style memory of the previous confidence instead.
        with torch.no_grad():
            candidate_mask = self.cand[indices]           # [B, C] the true candidate set Y
            new_conf = candidate_mask * torch.softmax(outputs, dim=1)
            new_conf = new_conf / new_conf.sum(dim=1, keepdim=True).clamp(min=1e-8)
            m = self.conf_ema_m
            self.conf[indices] = m * conf + (1.0 - m) * new_conf if m > 0 else new_conf

        return loss


class proden(nn.Module):
    """
    Legacy/simplified PRODEN variant -- only reachable via the OLD pipeline
    (setup_proden in src/model_setup.py), not the new pipeline (which uses
    ProdenLoss above via run_proden).

    KNOWN BUG (2026-08-14, see docs/proden_explanation.md): unlike the
    paper's Algorithm 1, the candidate-restricted softmax weights computed
    here are (a) used to weight the SAME forward pass's own loss with no
    one-step lag, and (b) never detached from the autograd graph, so
    gradients flow through the weighting term itself. The paper treats these
    weights as a latent, non-differentiable EM-style quantity. This makes
    `proden` closer to the paper's own documented failure case
    (PRODEN-sudden, which the paper shows underperforms) than to a valid
    simplification of PRODEN. Left unfixed since it is not on the new
    pipeline's execution path.
    """

    def __init__(self):
        super(proden, self).__init__()
    def forward(self, outputs, partial_labels):
        # Create a mask to ignore padded labels (-1).
        mask = (partial_labels != -1)
        
        predictions = torch.softmax(outputs, dim=1)
        
        # Select only valid labels for gathering.
        masked_labels = partial_labels.clone()
        masked_labels[~mask] = 0 # Replace padding with a valid index to avoid gather errors.
        
        candidate_preds = torch.gather(predictions, 1, masked_labels.long())
        candidate_preds[~mask] = 0 # Zero out predictions for padded labels.
        
        weights = candidate_preds / (torch.sum(candidate_preds, dim=1, keepdim=True) + 1e-8)
        
        log_probs = F.log_softmax(outputs, dim=1)
        individual_losses = -torch.gather(log_probs, 1, masked_labels.long())
        individual_losses[~mask] = 0 # Zero out losses for padded labels.
        
        sample_loss = torch.sum(weights * individual_losses, dim=1)
        return sample_loss.mean()


class PartialLoss(nn.Module):
    """
    Wang et al., ICLR 2022 (PiCO). Verified 2026-08-14 against Eq. 1 (loss)
    and Eq. 6 (confidence_update) -- exact match, including using
    prototype-similarity (score_prot, passed in as temp_un_conf by the
    caller) rather than the classifier's own softmax as the signal driving
    the pseudo-target here, as distinct from the classifier-softmax signal
    used to update prototypes in PiCOModel.forward (Eq. 7). See
    docs/pico_explanation.md. The one confirmed discrepancy is in the
    warm-up scheduling (src/engine.py::train_pico_epoch vs.
    src/fixed_pico_engine.py::train_pico_epoch_fixed), not in this class.
    """

    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence
        self.conf_ema_m = 0.99

    def set_conf_ema_m(self, epoch, args):
        """Sets the EMA momentum for confidence updates based on the current epoch."""
        start = args['conf_ema_range'][0]
        end = args['conf_ema_range'][1]
        self.conf_ema_m = 1. * epoch / args['epochs'] * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = -((final_outputs).sum(dim=1)).mean()
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY, hard: bool = True):
        """Updates sample confidences using EMA.

        hard=True (default, paper Eq. 6): one-hots the argmax of
        (temp_un_conf * batchY) before EMA-blending it into confidence --
        Factor B ('B', hard) of the 2026-09-15 confidence-update-mechanism
        ablation (see scripts/run_ab_sweep.py). hard=False ('B'') keeps the
        full masked/renormalized distribution instead of collapsing it to
        one class -- structurally the same kind of update ProdenLoss.forward
        does (src/proden_loss.py), just still driven by whatever
        temp_un_conf the caller passed in (prototype similarity or
        classifier softmax -- see Factor A / train_pico_epoch_fixed's
        conf_source)."""
        with torch.no_grad():
            masked = temp_un_conf * batchY
            if hard:
                _, prot_pred = masked.max(dim=1)
                new_target = F.one_hot(prot_pred, batchY.shape[1]).float().to(temp_un_conf.device).detach()
            else:
                new_target = masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                             + (1 - self.conf_ema_m) * new_target


class PiCOCLSLoss(nn.Module):
    """
    PiCO classification loss — standalone version without contrastive component.

    Identical to PiCO's PartialLoss in the forward pass:
        L = -sum_j  w_ij * log p_j

    Confidence update replaces prototype scores with model softmax:
        pseudo_label = argmax(softmax(outputs) * PL_mask)   [within candidates]
        w_i ← ema_m * w_i + (1 - ema_m) * one_hot(pseudo_label)

    In full PiCO, prototype similarity drives the pseudo-label selection.
    Here, the current model's own softmax plays that role.

    Args:
        partial_targets:  list of 1-D LongTensors, one per training sample.
        num_classes:      C.
        conf_ema_range:   (start, end) EMA momentum schedule; linearly
                          interpolated from start→end over training.
        epochs:           total training epochs (for EMA schedule).
    """

    def __init__(
        self,
        partial_targets: list,
        num_classes: int,
        conf_ema_range: tuple = (0.95, 0.8),
        epochs: int = 200,
    ):
        super().__init__()
        N = len(partial_targets)

        # confidence matrix: uniform over candidate set initially
        conf    = torch.zeros(N, num_classes)
        # binary PL mask: 1 = candidate label
        pl_mask = torch.zeros(N, num_classes)
        for i, cands in enumerate(partial_targets):
            k = max(len(cands), 1)
            for j in cands:
                conf[i, j.item()]    = 1.0 / k
                pl_mask[i, j.item()] = 1.0

        self.register_buffer('conf',    conf)    # [N, C]
        self.register_buffer('pl_mask', pl_mask) # [N, C]

        self.conf_ema_range = conf_ema_range
        self.epochs         = epochs
        self.conf_ema_m     = conf_ema_range[0]  # current EMA momentum

    # ------------------------------------------------------------------

    def set_conf_ema_m(self, epoch: int):
        """Call once per epoch before training that epoch."""
        start, end = self.conf_ema_range
        self.conf_ema_m = epoch / self.epochs * (end - start) + start

    def forward(self, outputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        outputs: [B, C] logits
        indices: [B]    sample indices into the full training set
        """
        log_probs = F.log_softmax(outputs, dim=1)          # [B, C]
        conf      = self.conf[indices]                      # [B, C]
        return -(conf * log_probs).sum(dim=1).mean()

    @torch.no_grad()
    def update_confidence(self, outputs: torch.Tensor, indices: torch.Tensor):
        """
        Call order differs by caller (kept as-is for reproducibility, 2026-11-22 note):
        run_pico_cls calls it AFTER optimizer.step(); engine.train_pico_sc_epoch
        (PiCO-SC) calls it BEFORE computing the loss (gated by prot_start).

        Selects pseudo-label = argmax of softmax within candidate set,
        then blends into stored confidence via EMA.
        """
        probs    = F.softmax(outputs, dim=1)                # [B, C]
        pl_mask  = self.pl_mask[indices]                    # [B, C]

        # pseudo-label: highest-prob candidate
        _, pseudo = (probs * pl_mask).max(dim=1)            # [B]
        one_hot   = F.one_hot(pseudo, self.conf.shape[1]).float()  # [B, C]

        self.conf[indices] = (
            self.conf_ema_m       * self.conf[indices]
            + (1 - self.conf_ema_m) * one_hot
        )


class SolarPartialLoss(nn.Module):
    def __init__(self, train_givenY, device):
        super().__init__()
        print('Calculating uniform targets...')
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        confidence = train_givenY.float()/tempY
        confidence = confidence.to(device)
        # Initial confidence is based on uniform distribution over candidate labels.
        self.confidence = confidence

    def forward(self, outputs, index, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        if targets is None:
            # Use pre-computed confidence.
            final_outputs = logsm_outputs * self.confidence[index, :].detach()
        else:
            # Use given targets.
            final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        average_loss = loss_vec.mean()
        return average_loss, loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index):
        """Updates sample confidences."""
        self.confidence[batch_index, :] = temp_un_conf
        return None

partial_loss = SolarPartialLoss  # original name (src/solar/utils_loss.py)


def conf_buffer(loss):
    """The [N, C] confidence tensor of any stateful cls loss, or None for stateless ones.
    (Unifies clsloss_swap._conf_buffer and clsloss_solo.conf_buffer.)"""
    for path in ('conf', 'confidence', 'partial.confidence', 'pl.conf', 'partial_loss.confidence'):
        obj = loss
        for a in path.split('.'):
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if torch.is_tensor(obj) and obj.dim() == 2:
            return obj
    return None
