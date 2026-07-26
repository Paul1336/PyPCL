import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_comp_mask(complementary_labels, B, C, device):
    """Build binary [B, C] CL mask and per-sample count m [B]."""
    valid_mask  = complementary_labels >= 0
    m           = valid_mask.sum(dim=1).float().clamp(min=1)
    labels_safe = complementary_labels.clamp(min=0)
    max_m       = labels_safe.shape[1]
    one_hot     = torch.zeros(B, max_m, C, device=device)
    one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
    one_hot    = one_hot * valid_mask.unsqueeze(2).float()
    comp_mask  = one_hot.sum(dim=1).clamp(max=1.0)   # [B, C]
    return comp_mask, m


class OPLoss(nn.Module):
    """
    Order-Preserving (OP) loss for CLL — Liu et al. AISTATS 2023, Definition 3.1.

    Single-CL form (cross-entropy as the order-preserving loss ℓ):
        L_OP = ℓ(-g(x), y_bar) = -log softmax(-g(x))_{y_bar}

    Key insight: since P(y_bar=k|x) ∝ 1 - P(y=k|x), the complementary label
    should have the LOWEST score in g(x).  Negating logits to -g(x) flips the
    ranking, so a standard CE loss on -g(x) with target y_bar directly pushes
    y_bar to be the argmax of -g(x), i.e. the argmin of g(x).

    The risk estimator  R̄(g;ℓ) = E[ℓ(-g(x), y_bar)]  is naturally non-negative
    (avoids the negative-risk overfitting of URE) and classifier-consistent
    (Theorem 3.1 in the paper).

    MCL extension (wrapper): average over all complementary labels in the set.
        L_OP(x, Y_bar) = (1/|Y_bar|) Σ_{y_bar ∈ Y_bar} -log softmax(-g(x))_{y_bar}

    This is NOT the same as SCL-NL. SCL-NL uses -log(1 - p_{y_bar}) whereas
    OP uses -log softmax(-g)_{y_bar} — a completely different functional form.
    """

    def forward(self, outputs: torch.Tensor, complementary_labels: torch.Tensor) -> torch.Tensor:
        B, C       = outputs.shape
        comp_mask, m = _build_comp_mask(complementary_labels, B, C, outputs.device)

        # CE on negated logits: -log softmax(-g(x))_{y_bar}
        per_label_loss = -F.log_softmax(-outputs, dim=1)   # [B, C]

        per_sample_loss = (per_label_loss * comp_mask).sum(dim=1) / m   # [B]
        return per_sample_loss.mean()


class OPWLoss(nn.Module):
    """
    Weighted Order-Preserving (OP-W) loss — Liu et al. AISTATS 2023, Definition 4.1.

    Improves on naive OP by weighting each complementary label's loss to give
    more importance to examples where the model is over-confident on y_bar:

        L_OP-W = w(g(x), y_bar) · ℓ(-g(x), y_bar)

    Weight function (Appendix D of the paper):
        w(g(x), y) = softmax(u(x) + 1)_y · softmax(g(x))_y + ε
        u_j(x)     = 1 / softmax(-g(x))_j

    Intuition: when the model already ranks y_bar very low in g(x) (small
    softmax(-g)_{y_bar} → large u_{y_bar}), the weight is large, preventing
    the algorithm from ignoring hard complementary labels at the bottom of the
    ranking.

    MCL wrapper: average weighted loss over all complementary labels.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, outputs: torch.Tensor, complementary_labels: torch.Tensor) -> torch.Tensor:
        B, C       = outputs.shape
        comp_mask, m = _build_comp_mask(complementary_labels, B, C, outputs.device)

        # Base OP loss: -log softmax(-g)_{y_bar}
        per_label_loss_base = -F.log_softmax(-outputs, dim=1)   # [B, C]

        # Weight (Appendix D):
        #   u_j = 1 / softmax(-g)_j
        #   w_y = softmax(u + 1)_y * softmax(g)_y + eps
        p_neg  = F.softmax(-outputs, dim=1).clamp(min=1e-7)     # [B, C]
        u      = 1.0 / p_neg                                     # [B, C]
        s_u    = F.softmax(u + 1.0, dim=1)                      # [B, C]
        p      = F.softmax(outputs,  dim=1)                      # [B, C]
        weight = s_u * p + self.eps                              # [B, C]

        per_label_loss  = weight * per_label_loss_base           # [B, C]
        per_sample_loss = (per_label_loss * comp_mask).sum(dim=1) / m
        return per_sample_loss.mean()
