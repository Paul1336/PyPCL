import torch
import torch.nn as nn
import torch.nn.functional as F


class SCL_NL(nn.Module):
    """
    Surrogate Complementary Loss — Negative Learning (log) variant.

    Reference:
        Chou et al., "Unbiased Risk Estimators Can Mislead: A Case Study of
        Learning with Complementary Labels", ICML 2020.

    Single-CL form (Eq. 11):
        phi_NL(y_bar, g(x)) = -log(1 - p_{y_bar})

    This loss pushes the model to assign LOW probability to the complementary
    label — the mirror image of standard cross-entropy.  It is a surrogate loss
    (not a URE): it does NOT subtract a correction term and does NOT use the
    complementary label as a positive label.

    MCLL extension (average-over-complementary-labels wrapper):
        L(x, Y_bar) = (1 / |Y_bar|) * Σ_{y_bar ∈ Y_bar} phi_NL(y_bar, g(x))

    This wraps the original single-CL setting by treating each element of the
    complementary label set independently and averaging, so that samples with
    different |Y_bar| contribute equally to the batch loss.

    Verified 2026-08-14 (see docs/scl_nl_explanation.md): the single-CL
    formula (Eq. 11) matches the paper exactly. The MCLL averaging wrapper
    above, however, has NO basis in Chou et al. (2020) -- the paper's theory
    and all its experiments assume exactly one complementary label per
    sample; it only *cites* multi-CL as separate related work (Feng et al.
    2020, this repo's MCL-LOG). No fixed_ version was produced since the
    paper provides no alternative multi-CL formula to fix towards -- this is
    a repo-side extrapolation, not a verified-wrong implementation.

    Note:
        - Averaging (not summing) over complementary labels makes the loss scale
          independent of |Y_bar|, which is important for fair comparison across k.
        - This is NOT the MCL unbiased risk estimator (MCL-LOG); it does not
          contain a term that sums over *non*-complementary labels.

    Args:
        outputs:              [B, C] raw logits.
        complementary_labels: [B, max_m] complementary label indices; -1 = padding.
    """

    def forward(self, outputs: torch.Tensor, complementary_labels: torch.Tensor) -> torch.Tensor:
        B, C = outputs.shape
        device = outputs.device

        p = F.softmax(outputs, dim=1)          # [B, C]

        # Build multi-hot complementary mask [B, C]
        valid_mask  = complementary_labels >= 0                         # [B, max_m]
        m           = valid_mask.sum(dim=1).float().clamp(min=1)        # [B]
        labels_safe = complementary_labels.clamp(min=0)                 # remap -1 → 0 before indexing

        # Expand to [B, max_m, C] one-hot, zero out padding rows, then collapse
        max_m   = labels_safe.shape[1]
        one_hot = torch.zeros(B, max_m, C, device=device)
        one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
        one_hot    = one_hot * valid_mask.unsqueeze(2).float()          # zero out padding
        comp_mask  = one_hot.sum(dim=1).clamp(max=1.0)                 # [B, C], binary

        # phi_NL per class: -log(1 - p_j),  numerically stable via log1p
        eps = 1e-7
        per_label_loss = -torch.log1p(-p.clamp(max=1.0 - eps))         # [B, C]

        # Average over complementary labels per sample, then over batch
        per_sample_loss = (per_label_loss * comp_mask).sum(dim=1) / m  # [B]
        return per_sample_loss.mean()
