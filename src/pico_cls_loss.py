import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Call after optimizer.step() each batch.

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
