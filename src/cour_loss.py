import torch
import torch.nn as nn
import torch.nn.functional as F


class CourLoss(nn.Module):
    """
    Uniform-averaging partial label loss from Cour, Sapp & Taskar, JMLR 2011.
    Each candidate label contributes equally (unlike PRODEN which weights by confidence).
    """

    def forward(self, outputs, partial_labels):
        # outputs:       [B, C] logits
        # partial_labels: [B, max_len] class indices, -1 for padding

        B, C = outputs.shape
        log_softmax = F.log_softmax(outputs, dim=1)

        mask = (partial_labels >= 0)                       # [B, max_len]
        count = mask.sum(dim=1).float().clamp(min=1)       # [B]

        # Build binary candidate mask [B, C] via vectorised scatter
        labels_safe = partial_labels.clamp(min=0)          # [B, max_len]
        _, L = labels_safe.shape
        one_hot = torch.zeros(B, L, C, device=outputs.device)
        one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
        one_hot = one_hot * mask.unsqueeze(2).float()      # zero out padding
        candidate_mask = one_hot.sum(dim=1)                # [B, C]

        # Mean log-prob over candidates
        loss_per_sample = -(candidate_mask * log_softmax).sum(dim=1) / count
        return loss_per_sample.mean()
