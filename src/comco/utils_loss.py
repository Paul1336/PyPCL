import torch
import torch.nn as nn
import torch.nn.functional as F


class ComCoCLSLoss(nn.Module):
    """
    MCL-NL style complementary loss for ComCo (Eq. 3 in paper).

    For each sample with complementary mask Y_bar (binary [B, C]):
        L = -log(sum_{c not in Y_bar} softmax(logits)_c) * (C-1) / (C-m)
    where m = number of complementary labels for that sample.

    Reduces to SCL-NL (-log(1 - g_ybar)) for single complementary label.

    KNOWN BUG (2026-08-14, see docs/comco_explanation.md): Eq. 3 in the paper
    is only a generic wrapper ("L_cls = L-bar(g(x),ybar)"); Section 5.1
    states explicitly "we choose SCL-NL as complementary loss" -- i.e. the
    paper's own classification loss is *plain, unscaled* SCL-NL, not this
    MCL-NL-style (C-1)/(C-m)-scaled formula (borrowed from Feng et al. 2020,
    a different paper ComCo only cites as a baseline). This only matches the
    paper when m=1 (scale=1); for m>1 it diverges. Use FixedComCoCLSLoss
    (src/comco/fixed_utils_loss.py, algorithm ID ComCo-Fixed) for the
    corrected version.
    """

    def forward(self, logits: torch.Tensor, comp_mask: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        non_comp_probs = probs * (1.0 - comp_mask)
        sum_non_comp = non_comp_probs.sum(dim=1).clamp(min=1e-8)
        m = comp_mask.sum(dim=1)
        scale = (C - 1.0) / (C - m).clamp(min=1.0)
        loss = -torch.log(sum_non_comp) * scale
        return loss.mean()


class ComCoContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss with ComCo positive/negative selection strategies.

    Positive set P(x_i):
      - Always includes k_i (the key-view embedding at pool index B+i).
      - After warmup_pos: also includes top-K nearest neighbors in A that share
        the same pseudo-label as x_i (Strategy B from paper).

    Negative set N(x_i) (after warmup_neg):
      - Pool A is partitioned into C subsets: S_c = {z_j | comp_mask_j[c] == 1}.
      - Select the subset most distant from anchor z_i via Dist_min:
          N(x_i) = S_{argmax_c Dist_min(z_i, S_c)}
          Dist_min(z, S_c) = 0.5 * (1 - max_{z_j in S_c} z . z_j)
      - Before warmup_neg: denominator = all pool elements except self (standard).

    Denominator = P(x_i) ∪ N(x_i).
    """

    def __init__(self, temperature: float = 0.17, top_k: int = 1):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k

    def forward(
        self,
        q: torch.Tensor,           # [B, D] query embeddings (L2-normalized)
        all_feats: torch.Tensor,   # [2B+Q, D] pool embeddings (L2-normalized)
        all_pseudo: torch.Tensor,  # [2B+Q] pseudo-labels
        all_comp: torch.Tensor,    # [2B+Q, C] complementary binary masks
        pseudo_q: torch.Tensor,    # [B] pseudo-labels for queries
        warmup_pos: bool,
        warmup_neg: bool,
        return_masks: bool = False,
    ) -> torch.Tensor:
        B = q.shape[0]
        A = all_feats.shape[0]
        C = all_comp.shape[1]
        device = q.device

        sim_matrix = torch.mm(q, all_feats.t())  # [B, A], cosine sim ∈ [-1, 1]

        # --- Positive mask: P(x_i) ---
        pos_mask = torch.zeros(B, A, device=device)

        # k_i is always positive: it sits at pool index B+i
        key_indices = torch.arange(B, device=device) + B  # [B]
        pos_mask.scatter_(1, key_indices.unsqueeze(1), 1.0)

        if warmup_pos and self.top_k > 0:
            # Integrated similarity: Sim(x_i, x_j) = [y~_i == y~_j] * 0.5*(1 + cos_sim)
            label_match = (pseudo_q.unsqueeze(1) == all_pseudo.unsqueeze(0)).float()
            integrated_sim = label_match * 0.5 * (1.0 + sim_matrix)

            # Exclude self (index i) and key view (index B+i) from neighbor search
            self_indices = torch.arange(B, device=device).unsqueeze(1)      # [B, 1]
            integrated_sim.scatter_(1, self_indices, -1e9)
            integrated_sim.scatter_(1, key_indices.unsqueeze(1), -1e9)

            _, top_k_idx = integrated_sim.topk(self.top_k, dim=1)  # [B, K]
            pos_mask.scatter_(1, top_k_idx, 1.0)

        # --- Denominator mask: P(x_i) ∪ N(x_i) ---
        if warmup_neg:
            # For each anchor i and class c, find max similarity to S_c
            # S_c = {z_j in A | all_comp[j, c] == 1}
            # Loop over C to avoid allocating [B, A, C] tensor
            max_sim_per_class = torch.full((B, C), -1e9, device=device)
            for c in range(C):
                mask_c = all_comp[:, c].bool()  # [A]
                if mask_c.any():
                    max_sim_per_class[:, c] = sim_matrix[:, mask_c].max(dim=1).values

            # Dist_min(z_i, S_c) = 0.5 * (1 - max_sim); empty S_c → excluded
            dist_min = 0.5 * (1.0 - max_sim_per_class)
            dist_min[max_sim_per_class <= -1e8] = -1e9

            neg_class = dist_min.argmax(dim=1)  # [B]

            # Build negative mask: for anchor i, all j in S_{neg_class[i]}
            # all_comp[:, neg_class] → [A, B]; transpose → [B, A]
            neg_mask = all_comp[:, neg_class].t()  # [B, A]
            denom_mask = (pos_mask + neg_mask).clamp(max=1.0)
        else:
            # Before neg warmup: all-vs-all InfoNCE (exclude only self)
            denom_mask = torch.ones(B, A, device=device)

        # Always exclude anchor itself (index i) from its own denominator
        self_indices = torch.arange(B, device=device).unsqueeze(1)
        denom_mask.scatter_(1, self_indices, 0.0)

        # --- InfoNCE loss ---
        logits = sim_matrix / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        if return_masks:
            return loss, pos_mask, denom_mask
        return loss
