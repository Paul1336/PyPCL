import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import ema_prototypes_loop, init_key_encoder, momentum_update
from src.pico.resnet import SupConResNet


class PiCOModel(nn.Module):
    """Wang et al., ICLR 2022 (PiCO). Verified 2026-08-14 against Eq. 7
    (prototype EMA update, driven by the classifier's own candidate-masked
    softmax argmax) and Eq. 2/3 (MoCo queue / contrastive pool) -- exact
    match. See docs/pico_explanation.md.
    """

    def __init__(self, args):
        super().__init__()
        self.encoder_q = SupConResNet(num_class=args['num_class'], feat_dim=args['low_dim'])
        self.encoder_k = SupConResNet(num_class=args['num_class'], feat_dim=args['low_dim'])
        init_key_encoder(self.encoder_q, self.encoder_k)

        self.register_buffer("queue", torch.randn(args['moco_queue'], args['low_dim']))
        self.register_buffer("queue_pseudo", torch.randn(args['moco_queue']))
        self._register_extra_queues(args)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(args['num_class'], args['low_dim']))
        self.queue = F.normalize(self.queue, dim=0)

    def _register_extra_queues(self, args):
        """Hook for subclasses that track more per-entry fields (registered before queue_ptr)."""

    def predict(self, x):
        return self.encoder_q(x)[0]

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """Momentum update of the key encoder."""
        momentum_update(self.encoder_q, self.encoder_k, args['moco_m'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args, **extra):
        """Update the queue of features and pseudo-labels (+ any extra per-entry fields,
        e.g. queue_true=... in PiCOOracleModel)."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert args['moco_queue'] % batch_size == 0
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        for name, vals in extra.items():
            getattr(self, name)[ptr:ptr + batch_size] = vals
        ptr = (ptr + batch_size) % args['moco_queue']
        self.queue_ptr[0] = ptr

    def _shared_forward(self, img_q, im_k, partial_Y, args):
        """Classifier + candidate-masked pseudo-label + prototype scores/update + key encoder."""
        output, q = self.encoder_q(img_q)
        predicted_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)

        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        self.prototypes = ema_prototypes_loop(self.prototypes, q, pseudo_labels_b, args['proto_m'])

        with torch.no_grad():
            self._momentum_update_key_encoder(args)
            _, k = self.encoder_k(im_k)
        return output, q, k, pseudo_labels_b, score_prot

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
        if eval_only:
            return self.predict(img_q)
        output, q, k, pseudo_labels_b, score_prot = self._shared_forward(img_q, im_k, partial_Y, args)
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)

        self._dequeue_and_enqueue(k, pseudo_labels_b, args)
        return output, features, pseudo_labels, score_prot


class PiCOOracleModel(PiCOModel):
    """PiCO-Oracle: an ablation where the contrastive loss's positive/
    negative pair selection uses the ORACLE (ground-truth) label instead of
    the prototype-derived pseudo-label -- measures an upper bound on how
    much pair-selection noise costs plain PiCO's contrastive objective. Not
    a real algorithm for actual use: true labels aren't available in a
    genuine partial-label setting.
    
    Identical to PiCOModel except: (a) the MoCo queue also tracks each
    entry's ground-truth label (`queue_true`, mirroring `queue_pseudo`),
    and (b) forward() returns BOTH the true-label pool (`true_targets`) and
    the ordinary prototype-derived pseudo-label pool (`pseudo_targets`, same
    construction as PiCOModel.forward's `pseudo_labels`), so the caller can
    compare the model's own pseudo-label-driven pair selection against
    ground truth and graduate between them (see
    src.oracle_pico_engine.train_pico_oracle_graded_epoch). Classification
    loss / confidence tracking / prototype EMA update are all unchanged --
    still driven by PartialLoss and prototype scores exactly as in plain
    PiCO, since those govern candidate-label disambiguation, a separate
    concern from contrastive pair selection.
    
    queue_true is initialized with distinct negative sentinels
    (-moco_queue..-1, one per slot) rather than a single fill value, so
    not-yet-filled slots never spuriously equal each other or a real label
    (which is always in [0, num_class)) under the mask's torch.eq check --
    mirrors PiCOModel's queue_pseudo, which gets the same property "for
    free" from being continuous random floats.
    """

    def _register_extra_queues(self, args):
        self.register_buffer("queue_true", torch.arange(-args['moco_queue'], 0, dtype=torch.long))

    def forward(self, img_q, im_k=None, partial_Y=None, true_labels=None, args=None, eval_only=False):
        if eval_only:
            return self.predict(img_q)
        output, q, k, pseudo_labels_b, score_prot = self._shared_forward(img_q, im_k, partial_Y, args)
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        true_targets = torch.cat((true_labels, true_labels, self.queue_true.clone().detach()), dim=0)
        pseudo_targets = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)

        self._dequeue_and_enqueue(k, pseudo_labels_b, args, queue_true=true_labels)
        return output, features, true_targets, pseudo_targets, score_prot
