import torch
import torch.nn as nn
import torch.nn.functional as F


class CPELoss(nn.Module):
    """
    CPE-I (Complementary Probability Estimation, no-transition variant)
    — Lin & Lin PAKDD 2023, Section 3.1.

    The CPE framework reduces CLL to probability estimation of complementary
    labels.  CPE-I (the simplest variant) directly uses the model output f(x;θ)
    as an estimator for P(ȳ|x) and trains it with standard CE on the CL:

        L_CPE-I = -log f_{ȳ}(x)   i.e. CE where the complementary label is
                                        treated as the positive target.

    Since the model is trained to maximise f_{ȳ}(x) ≈ P(ȳ|x) and under the
    uniform CL-generation assumption P(ȳ=k|x) ∝ 1-P(y=k|x), the true class
    has the LOWEST predicted "complementary probability".

    *** Inference must use argmin, NOT argmax ***
        ŷ = argmin_k f_k(x)

    This is fundamentally different from SCL-NL (which uses -log(1-f_{ȳ}) and
    argmax) — even though CPE-F with uniform transition is equivalent to SCL-NL
    up to a constant, CPE-I is not.

    MCL extension: average CE over all complementary labels in the set.
        L_CPE-I(x, Ȳ) = (1/|Ȳ|) Σ_{ȳ ∈ Ȳ} -log f_{ȳ}(x)

    Args:
        outputs:              [B, C] raw logits.
        complementary_labels: [B, max_m] CL indices; -1 = padding.
    """

    def forward(self, outputs: torch.Tensor, complementary_labels: torch.Tensor) -> torch.Tensor:
        B, C       = outputs.shape
        device     = outputs.device

        # Build binary CL mask [B, C] and per-sample count m [B]
        valid_mask  = complementary_labels >= 0                         # [B, max_m]
        m           = valid_mask.sum(dim=1).float().clamp(min=1)        # [B]
        labels_safe = complementary_labels.clamp(min=0)
        max_m       = labels_safe.shape[1]

        one_hot = torch.zeros(B, max_m, C, device=device)
        one_hot.scatter_(2, labels_safe.unsqueeze(2), 1.0)
        one_hot   = one_hot * valid_mask.unsqueeze(2).float()
        comp_mask = one_hot.sum(dim=1).clamp(max=1.0)                   # [B, C]

        # CE on complementary labels: train f to predict P(ȳ|x)
        # -log f_{ȳ}(x) = -log softmax(g(x))_{ȳ}
        log_p = F.log_softmax(outputs, dim=1)                           # [B, C]
        per_label_loss = -log_p                                         # [B, C]

        per_sample_loss = (per_label_loss * comp_mask).sum(dim=1) / m   # [B]
        return per_sample_loss.mean()
