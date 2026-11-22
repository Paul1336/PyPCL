"""Label-format conversion: the ONE place padded label lists become dense masks.

Padded format (src/collate.py): LongTensor [B, L] of class indices, -1 = padding.
Dense format: float [B, C] 0/1. Replaces 7 hand-written copies (mcl_losses x3,
fixed_mcl_losses, scl_loss, op_loss, cpe_loss, clpl_loss, cour_loss, wu_loss).
"""

import torch


def padded_to_dense(labels, C, dtype=torch.float32):
    """[B, L] padded (-1) index lists -> [B, C] binary mask."""
    B = labels.shape[0]
    valid = labels >= 0
    dense = torch.zeros(B, C, device=labels.device, dtype=dtype)
    rows = torch.arange(B, device=labels.device).unsqueeze(1).expand_as(labels)[valid]
    dense[rows, labels[valid].long()] = 1.0
    return dense


def comp_to_cand(comp):
    return 1.0 - comp


def cand_to_comp(Y):
    return 1.0 - Y
