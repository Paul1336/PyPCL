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
        softmax — ready for the next batch. This renormalize-every-batch update is
        UNCHANGED regardless of init_mode: only the very first batch(es) touching a
        given sample still reflect its initial confidence, after which it's
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
                 wf: int = None):
        super().__init__()
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

    def forward(self, outputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        outputs:  [B, C] logits
        indices:  [B]    integer sample indices into the full training set
        """
        conf = self.conf[indices]                         # [B, C] stored soft labels

        # Loss: weighted CE with last-step confidence
        log_probs = F.log_softmax(outputs, dim=1)         # [B, C]
        loss = -(conf * log_probs).sum(dim=1).mean()

        # Update: renormalise current softmax within candidate mask
        with torch.no_grad():
            candidate_mask = (conf > 0).float()           # [B, C]
            new_conf = candidate_mask * torch.softmax(outputs, dim=1)
            new_conf = new_conf / new_conf.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.conf[indices] = new_conf

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
