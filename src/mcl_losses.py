import torch
import torch.nn as nn
import torch.nn.functional as F

class MCL_LOG(nn.Module):
    """
    Feng et al., ICML 2020 ("Learning with Multiple Complementary Labels").

    KNOWN BUG (2026-08-14, see docs/mcl_explanation.md): the loss formula
    itself (-log(sum of non-complementary probs)) matches the paper exactly,
    but `scaling_factor` below should be 2*(C-1)/m per the paper's Eq. 12
    (derived via the MAE-symmetry collapse of the general Theorem 3
    estimator, then substituting L_LOG/L_EXP for L'_MAE), not (C-1)/(C-m).
    The two formulas move in opposite directions as m grows. Use
    FixedMCLLog (src/fixed_mcl_losses.py, algorithm ID MCL-LOG-Fixed) for
    the corrected version.
    """

    def __init__(self, num_classes):
        super(MCL_LOG, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, complementary_labels):
        valid_labels_mask = (complementary_labels != -1)
        num_complementary = valid_labels_mask.sum(dim=1).float()

        batch_size, num_classes = outputs.shape
        probs_all = F.softmax(outputs, dim=1)
        
        mask_complementary = torch.zeros_like(probs_all, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_labels = complementary_labels[i][valid_labels_mask[i]]
            if len(valid_labels) > 0:
                mask_complementary[i].scatter_(0, valid_labels.long(), True)

        mask_non_complementary = ~mask_complementary
        sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary.float()).sum(dim=1)
        
        epsilon = 1e-7
        loss = -torch.log(sum_probs_not_in_complementary_set + epsilon)
        
        # Apply unbiased risk estimator scaling.
        # BUG (docs/mcl_explanation.md): paper's Eq. 12 gives 2*(C-1)/m here,
        # not (C-1)/(C-m) -- see FixedMCLLog/FixedMCLMae/FixedMCLExp in
        # src/fixed_mcl_losses.py for the corrected scaling.
        scaling_factor = (self.num_classes - 1) / (self.num_classes - num_complementary)
        scaled_loss = scaling_factor * loss
        
        return scaled_loss.mean()

class MCL_MAE(nn.Module):
    def __init__(self, num_classes):
        super(MCL_MAE, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, complementary_labels):
        valid_labels_mask = (complementary_labels != -1)
        num_complementary = valid_labels_mask.sum(dim=1).float()

        batch_size, num_classes = outputs.shape
        probs_all = F.softmax(outputs, dim=1)
        
        mask_complementary = torch.zeros_like(probs_all, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_labels = complementary_labels[i][valid_labels_mask[i]]
            if len(valid_labels) > 0:
                mask_complementary[i].scatter_(0, valid_labels.long(), True)

        mask_non_complementary = ~mask_complementary
        sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary.float()).sum(dim=1)
        
        # MAE Loss is 1 - sum of non-complementary probabilities.
        loss = 1.0 - sum_probs_not_in_complementary_set

        # Apply unbiased risk estimator scaling.
        # BUG (docs/mcl_explanation.md): paper's Eq. 12 gives 2*(C-1)/m here,
        # not (C-1)/(C-m) -- see FixedMCLLog/FixedMCLMae/FixedMCLExp in
        # src/fixed_mcl_losses.py for the corrected scaling.
        scaling_factor = (self.num_classes - 1) / (self.num_classes - num_complementary)
        scaled_loss = scaling_factor * loss
        
        return scaled_loss.mean()

class MCL_EXP(nn.Module):
    def __init__(self, num_classes):
        super(MCL_EXP, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, complementary_labels):
        valid_labels_mask = (complementary_labels != -1)
        num_complementary = valid_labels_mask.sum(dim=1).float()

        batch_size, num_classes = outputs.shape
        probs_all = F.softmax(outputs, dim=1)
        
        mask_complementary = torch.zeros_like(probs_all, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_labels = complementary_labels[i][valid_labels_mask[i]]
            if len(valid_labels) > 0:
                mask_complementary[i].scatter_(0, valid_labels.long(), True)

        mask_non_complementary = ~mask_complementary
        sum_probs_not_in_complementary_set = (probs_all * mask_non_complementary.float()).sum(dim=1)
        
        # EXP Loss is exp(-sum of non-complementary probabilities).
        loss = torch.exp(-sum_probs_not_in_complementary_set)

        # Apply unbiased risk estimator scaling.
        # BUG (docs/mcl_explanation.md): paper's Eq. 12 gives 2*(C-1)/m here,
        # not (C-1)/(C-m) -- see FixedMCLLog/FixedMCLMae/FixedMCLExp in
        # src/fixed_mcl_losses.py for the corrected scaling.
        scaling_factor = (self.num_classes - 1) / (self.num_classes - num_complementary)
        scaled_loss = scaling_factor * loss
        
        return scaled_loss.mean()