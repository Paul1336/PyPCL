import torch
import torch.nn as nn
import torch.nn.functional as F


class ProdenLoss(nn.Module):
    """
    PRODEN with cross-epoch confidence accumulation (original paper algorithm).

    Verified against Lv et al., ICML 2020, Algorithm 1 (2026-08-14) -- exact
    match, including the one-step lag between loss computation (uses the
    previously stored confidence) and the confidence refresh (uses the
    current forward pass, under no_grad). This is the class actually used by
    the new pipeline (run_proden). See docs/proden_explanation.md.

    Maintains a persistent confidence matrix conf[N, C] for every training sample:
      - Initialised uniformly over candidate labels.
      - forward() uses the *stored* confidence as soft-label weights for the loss,
        then updates conf[indices] in-place with the current model's renormalised
        softmax — ready for the next batch.

    Args:
        partial_targets: list of 1-D LongTensors, one per sample (candidate indices).
        num_classes:     total number of classes C.
    """

    def __init__(self, partial_targets: list, num_classes: int):
        super().__init__()
        N = len(partial_targets)
        conf = torch.zeros(N, num_classes)
        for i, cands in enumerate(partial_targets):
            k = max(len(cands), 1)
            for j in cands:
                conf[i, j.item()] = 1.0 / k
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
