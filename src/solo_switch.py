"""No-contrastive ("Solo") variant of the bandit / switch experiments (2026-10-14).

Same backbone + cls head as PiCO/ComCo (SupConResNet's classifier output),
no key encoder, no queue, no contrastive loss: only the cls loss trains the
network. Without a contrastive branch there are no meaningful prototypes, so
the PartialLoss confidence s is EMA-updated from the classifier's own
candidate-masked softmax instead (the 'classifier' confidence source, same as
the PiCOCLS loss of the cls-loss solo/swap experiments). Uses the PiCO loader's
weak view only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.pico.resnet import SupConResNet


class SoloModel(nn.Module):
    def __init__(self, num_class: int, low_dim: int):
        super().__init__()
        self.encoder_q = SupConResNet(num_class=num_class, feat_dim=low_dim)
        # unused by training; kept so src/switch_diag.py can run unchanged (proto_acc is not meaningful here)
        self.register_buffer('prototypes', torch.zeros(num_class, low_dim))

    def forward(self, x, eval_only: bool = True):
        logits, _ = self.encoder_q(x)
        return logits


def train_solo_switch_epoch(args, model, loader, loss_fn, loss_cont_fn, optimizer, epoch, device, alpha_fn):
    """One epoch of cls-loss-only training; loss_fn takes (logits, index, partial_Y, alpha=...).
    `loss_cont_fn` is ignored (kept for the generic runner's signature)."""
    model.train()
    total_loss = 0
    start_upd = epoch >= args['prot_start']
    progress_bar = tqdm(loader, desc=f"Solo-Switch Epoch {epoch + 1}/{args['epochs']}")
    for (images_w, _images_s, partial_Y, _true, index) in progress_bar:
        images_w, partial_Y, index = images_w.to(device), partial_Y.to(device), index.to(device)
        logits, _ = model.encoder_q(images_w)
        if start_upd:
            loss_fn.confidence_update(temp_un_conf=F.softmax(logits, dim=1).detach(), batch_index=index,
                                      batchY=partial_Y)
        alpha = alpha_fn(partial_Y)
        loss = loss_fn(logits, index, partial_Y, alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1), alpha=float(alpha))
    return total_loss / len(loader)
