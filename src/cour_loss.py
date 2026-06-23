import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformCandidateCrossEntropyLoss(nn.Module):
    """
    Uniform-averaging candidate cross-entropy for partial label learning.

    Treats every candidate label as an equally-weighted soft target and
    minimises the mean log-softmax over the candidate set:

        L_i = - (1/|Y_i|) * sum_{a in Y_i} log softmax_a(x_i)

    NOTE: This is NOT the CLPL loss from Cour, Sapp & Taskar (JMLR 2011).
    The paper's CLPL uses raw logits with a squared-hinge objective (see
    CLPLSquaredHingeLoss in src/clpl_loss.py).  This class is kept for
    reference and backward-compatibility with earlier experiment runs.
    """

    def forward(self, outputs, partial_labels):
        # outputs:        [B, C] logits
        # partial_labels: [B, max_len] class indices, -1 for padding

        B, C = outputs.shape
        log_softmax = F.log_softmax(outputs, dim=1)

        mask = (partial_labels >= 0)                        # [B, max_len]
        count = mask.sum(dim=1).float().clamp(min=1)        # [B]

        labels_safe = partial_labels.clamp(min=0)
        _, L = labels_safe.shape
        one_hot = torch.zeros(B, L, C, device=outputs.device)
        one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
        one_hot = one_hot * mask.unsqueeze(2).float()
        candidate_mask = one_hot.sum(dim=1)                 # [B, C]

        loss_per_sample = -(candidate_mask * log_softmax).sum(dim=1) / count
        return loss_per_sample.mean()


# Backward-compat alias (used by earlier experiment runs logged as "Cour2011")
CourLoss = UniformCandidateCrossEntropyLoss
