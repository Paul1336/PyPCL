import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pico.resnet import SupConResNet


class ComCoModel(nn.Module):
    """
    Jiang, Sun & Tian, Neural Networks 2024 (ComCo). Verified 2026-08-14 --
    dual momentum encoder + queue architecture, unmasked argmax pseudo-label
    (no candidate set in CLL), and queue_comp bookkeeping all match the
    paper's Section 3.4. See docs/comco_explanation.md. The one confirmed
    discrepancy is in ComCoCLSLoss (src/comco/utils_loss.py), not here.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder_q = SupConResNet(num_class=args['num_class'], feat_dim=args['low_dim'])
        self.encoder_k = SupConResNet(num_class=args['num_class'], feat_dim=args['low_dim'])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        Q = args['moco_queue']
        D = args['low_dim']
        C = args['num_class']

        self.register_buffer("queue", F.normalize(torch.randn(Q, D), dim=1))
        self.register_buffer("queue_pseudo", torch.zeros(Q, dtype=torch.long))
        self.register_buffer("queue_comp", torch.zeros(Q, C))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, moco_m):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * moco_m + param_q.data * (1.0 - moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pseudo_labels, comp_masks, queue_size):
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert queue_size % B == 0, "Queue size must be divisible by batch size"
        self.queue[ptr:ptr + B] = keys
        self.queue_pseudo[ptr:ptr + B] = pseudo_labels
        self.queue_comp[ptr:ptr + B] = comp_masks
        self.queue_ptr[0] = (ptr + B) % queue_size

    def forward(self, img_q, img_k=None, comp_mask=None, args=None, eval_only=False):
        cls_out, q = self.encoder_q(img_q)
        if eval_only:
            return cls_out

        pseudo_q = cls_out.argmax(dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder(args['moco_m'])
            _, k = self.encoder_k(img_k)

        # Build embedding pool A = B_q ∪ B_k ∪ Queue
        all_feats = torch.cat([q, k, self.queue.clone().detach()], dim=0)
        all_pseudo = torch.cat([pseudo_q, pseudo_q, self.queue_pseudo.clone().detach()], dim=0)
        all_comp = torch.cat([comp_mask, comp_mask, self.queue_comp.clone().detach()], dim=0)

        self._dequeue_and_enqueue(k, pseudo_q, comp_mask, args['moco_queue'])

        return cls_out, q, all_feats, all_pseudo, all_comp
