"""Fixed version of src/comco/utils_loss.py's ComCoCLSLoss.

See docs/comco_explanation.md ("演算法保真度比對" -> 第 1 項分類損失) for the full
comparison. Jiang, Sun & Tian, Neural Networks 2024 (ComCo), Section 5.1
states explicitly: "For ComCo, we choose SCL-NL as complementary loss L_cls"
-- i.e. the paper's own classification loss is *plain, unscaled* SCL-NL
(Section 3.2, Eq. 1: -log(1-p_ybar)), not a (C-1)/(C-m)-scaled MCL-NL-style
sum over the non-complementary set. The paper's Eq. 3 is a generic wrapper
("L_cls = L-bar(g(x), ybar), where L-bar represents an arbitrary
complementary loss") -- it is not itself the scaled formula the original
ComCoCLSLoss implements. That scaled formula was borrowed from Feng et al.
2020 (MCL-NL), a different paper that ComCo cites only as a baseline.

The original ComCoCLSLoss reduces to plain SCL-NL exactly when a sample has
a single complementary label (m=1, scale=(C-1)/(C-1)=1), so this bug only
changes behavior for samples with multiple complementary labels.

The paper does not give an explicit multi-complementary-label formula for
ComCo's classification loss either (Section 3.3 only says the extension is
"almost identical" without an equation). This fixed version follows the most
paper-faithful reading available: reuse this repo's own SCL_NL multi-CL
wrapper (src/scl_loss.py) -- averaging the *unscaled* single-label SCL-NL
term over the complementary set -- since SCL-NL (unscaled) is what Section
5.1 names as ComCo's actual classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedComCoCLSLoss(nn.Module):
    """Plain (unscaled) SCL-NL, averaged over the complementary label set.

    L = (1/m) * sum_{ybar in Ybar} -log(1 - p_ybar),  m = |Ybar|

    No (C-1)/(C-m) unbiased-risk-estimator scaling, matching the paper's
    Section 5.1 choice of SCL-NL (not MCL-NL) as ComCo's classification loss.
    """

    def forward(self, logits: torch.Tensor, comp_mask: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        m = comp_mask.sum(dim=1).clamp(min=1.0)

        eps = 1e-7
        per_label_loss = -torch.log1p(-probs.clamp(max=1.0 - eps))  # -log(1-p_j) per class j

        per_sample_loss = (per_label_loss * comp_mask).sum(dim=1) / m
        return per_sample_loss.mean()
