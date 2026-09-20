"""Gradient bias-variance diagnostic (`run --biasvariance`), SCL-NL paper
(Chou et al., ICML 2020) Section 4 style, but applied IN-LINE to the real
model/loss state of an actual training run rather than a separate proxy
classifier -- see docs discussion / plan for the full rationale.

For a fixed eval batch (same samples reused across every checkpoint, so
trends are comparable), at each --biasvariance_log_every checkpoint:
  1. Forward the eval batch once through the model's CURRENT (real,
     currently-training) weights -- cache logits/embeddings, no further CNN
     forward passes needed.
  2. Compute an "oracle" gradient: plain cross-entropy against the TRUE
     label (paradigm-agnostic reference point, matching the paper's own
     "only use ordinary-label gradient for comparison, never for updates"
     protocol).
  3. For M counterfactual resamples: redraw a fresh candidate/complementary
     label set per eval sample (independent of whatever candidate set that
     sample was actually assigned during real data generation), recompute
     the loss under test with the algorithm's REAL current loss-function
     state (e.g. PartialLoss's live EMA confidence buffer), backward, and
     collect the gradient.
  4. Bias^2 / Variance / MSE / cosine-similarity of the resampled gradients
     vs. the oracle gradient, appended as one CSV row.

Covers both the classification loss (loss_type='cls') and, for the PiCO/
ComCo dual-encoder families, the contrastive loss (loss_type='cont') --
see log_biasvariance_checkpoint's `cont` argument.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.pico.utils_loss import PartialLoss, SupConLoss
from src.pico.weighted_cls_loss import PiCOWeightedClsLoss

_MEAN = [0.4914, 0.4822, 0.4465]
_STD = [0.247, 0.2435, 0.2616]


def build_fixed_eval_batch(pl_ds, orig_targets, eval_size: int, device):
    """A deterministic (ToTensor+Normalize only, no augmentation), fixed
    subset of `eval_size` training samples -- the first `eval_size` indices
    of the cell's train set -- reused across every checkpoint of this (C, k,
    seed) cell so bias/variance trends are comparable epoch-to-epoch. Returns
    (images [N,3,32,32], true_labels [N] long, index [N] long)."""
    n = min(eval_size, len(pl_ds.data))
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    imgs = torch.stack([tf(Image.fromarray(pl_ds.data[i])) for i in range(n)]).to(device)
    true_labels = torch.as_tensor([int(orig_targets[i]) for i in range(n)], dtype=torch.long, device=device)
    index = torch.arange(n, dtype=torch.long, device=device)
    return imgs, true_labels, index

FIELDS = ['dataset', 'C', 'k', 'algorithm', 'seed', 'epoch', 'loss_type',
          'bias2', 'variance', 'mse', 'cos_sim']


# ─── candidate/complementary-set resampling ────────────────────────────────


def resample_candidate_set(true_label: int, C: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Binary [C] partial_Y with exactly k ones (the true_label plus k-1
    uniformly-random other classes, no replacement) -- mirrors
    ComparisonDataGenerator.generate_pl_dataset's core sampling
    (src/data_utils.py:54-68), but takes a local RandomState instead of
    touching the global np.random state (this runs inside an already-seeded
    training process; reseeding global state here would perturb whatever
    else in that process still draws from it)."""
    partial_Y = np.zeros(C, dtype=np.float32)
    partial_Y[true_label] = 1.0
    if k > 1:
        incorrect = np.delete(np.arange(C), true_label)
        picks = rng.choice(incorrect, size=k - 1, replace=False)
        partial_Y[picks] = 1.0
    return partial_Y


def to_comp_mask(partial_Y: np.ndarray) -> np.ndarray:
    """PLL candidate set -> CLL complementary set: 1 - partial_Y. This repo's
    actual data pipeline (prepare_cifar100_subset, src/cifar100_subset.py)
    derives CL as the exact complement of PL, so one resampled candidate set
    serves both paradigms."""
    return 1.0 - partial_Y


def _resample_batch(true_labels: torch.Tensor, C: int, k: int, seed_val: int, device):
    """Vectorized resample over a batch of true labels. Returns
    (partial_Y [B,C], comp_mask [B,C]) float tensors on `device`."""
    rng = np.random.RandomState(seed_val)
    B = true_labels.shape[0]
    partial_Y = np.zeros((B, C), dtype=np.float32)
    tl = true_labels.cpu().numpy()
    for i in range(B):
        partial_Y[i] = resample_candidate_set(int(tl[i]), C, k, rng)
    partial_Y_t = torch.from_numpy(partial_Y).to(device)
    comp_mask_t = 1.0 - partial_Y_t
    return partial_Y_t, comp_mask_t


def resample_pseudo_target(candidate_set: torch.Tensor, cls_out_softmax: torch.Tensor) -> torch.Tensor:
    """argmax(softmax(cls_out) * candidate_set) -- replicates
    PiCOModel.forward's pseudo_labels_b computation (src/pico/model.py:54-55)
    as a pure function of a cached, label-independent softmax and whatever
    candidate set is being tested, so pseudo_target_cont can be recomputed
    per resample without rerunning the encoder."""
    predicted_scores = cls_out_softmax * candidate_set
    _, pseudo_label = torch.max(predicted_scores, dim=1)
    return pseudo_label


# ─── oracle references ─────────────────────────────────────────────────────


def oracle_cls_grad(logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
    """Gradient of plain cross-entropy (true label) wrt `logits` -- the
    paradigm-agnostic reference gradient every algorithm's resampled
    gradients are compared against."""
    leaf = logits.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(leaf, true_labels)
    loss.backward()
    return leaf.grad.detach()


def oracle_cont_grad(q: torch.Tensor, k_view: torch.Tensor, anchor_true_labels: torch.Tensor,
                      pool_true_labels: torch.Tensor, supcon: SupConLoss) -> torch.Tensor:
    """Gradient of a true-label SupCon mask (positive = same true label)
    wrt the anchor embeddings `q` -- the common, method-agnostic reference
    for both SupConLoss and ComCoContrastiveLoss's L_cont (neither has a
    natural "oracle" form of its own pair-selection logic, so this uses a
    shared true-label-based construction for both). Pool is [q; k_view] for
    this eval batch only, no queue (see log_biasvariance_checkpoint: a full
    model.forward() would mutate the real training queue/prototype state via
    its internal _dequeue_and_enqueue / EMA updates, so this calls encoder_q/
    encoder_k directly instead)."""
    mask = torch.eq(anchor_true_labels.unsqueeze(1), pool_true_labels.unsqueeze(0)).float()
    leaf = q.detach().clone().requires_grad_(True)
    pool = torch.cat([leaf, k_view.detach()], dim=0)
    loss = supcon(pool, mask=mask, batch_size=q.shape[0])
    loss.backward()
    return leaf.grad.detach()


# ─── bias/variance/mse/cos_sim reduction ───────────────────────────────────


def _reduce(grads: list, oracle: torch.Tensor) -> dict:
    stacked = torch.stack(grads, dim=0)          # [M, ...]
    mean_grad = stacked.mean(dim=0)
    bias2 = ((mean_grad - oracle) ** 2).mean().item()
    variance = ((stacked - mean_grad) ** 2).mean().item()
    mse = ((stacked - oracle) ** 2).mean().item()
    cos_sim = F.cosine_similarity(mean_grad.flatten(), oracle.flatten(), dim=0).item()
    return dict(bias2=bias2, variance=variance, mse=mse, cos_sim=cos_sim)


def _append_row(out_dir: str, row: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'biasvariance.csv')
    new_file = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ─── cls-loss adapters: (logits, partial_Y, comp_mask, index) -> scalar loss ──


def _cls_loss_and_grad(algorithm: str, cls_loss_fn, logits: torch.Tensor, partial_Y: torch.Tensor,
                        comp_mask: torch.Tensor, eval_index: torch.Tensor) -> torch.Tensor:
    """Runs the REAL cls_loss_fn currently in use for `algorithm` against a
    resampled (partial_Y, comp_mask) pair, on a fresh leaf cloned from the
    cached `logits`, and returns the leaf's gradient after backward()."""
    leaf = logits.detach().clone().requires_grad_(True)

    if isinstance(cls_loss_fn, PartialLoss):
        # PartialLoss.forward indexes self.confidence[index, :], which was
        # built for the REAL candidate set actually assigned to eval_index --
        # not the resampled one being tested here, and possibly a different
        # cardinality. Reuse that row's REAL, EMA-accumulated confidence
        # (not a fresh uniform init -- see plan) and renormalize it over the
        # resampled candidate mask, matching PartialLoss.forward's own
        # log-softmax * confidence contraction.
        real_conf_row = cls_loss_fn.confidence[eval_index, :].detach()
        masked = real_conf_row * partial_Y
        renorm = masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        logsm = F.log_softmax(leaf, dim=1)
        loss = -((logsm * renorm).sum(dim=1)).mean()
    elif isinstance(cls_loss_fn, PiCOWeightedClsLoss):
        real_conf_row = cls_loss_fn.partial_loss.confidence[eval_index, :].detach()
        masked = real_conf_row * partial_Y
        renorm = masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        logsm = F.log_softmax(leaf, dim=1)
        partial_term = -((logsm * renorm).sum(dim=1)).mean()
        mcl_term = cls_loss_fn.mcl_loss(leaf, partial_Y)
        loss = cls_loss_fn.alpha * partial_term + (1 - cls_loss_fn.alpha) * mcl_term
    elif algorithm == 'ComCo-Fixed':
        loss = cls_loss_fn(leaf, comp_mask)
    else:
        # Stateless: PiCOMCLLoss (PiCO-MCL-Fixed).
        loss = cls_loss_fn(leaf, partial_Y)

    loss.backward()
    return leaf.grad.detach()


def log_biasvariance_checkpoint(model, cls_loss_fn, cont_loss_fn, algorithm: str,
                                 eval_images: torch.Tensor, eval_true_labels: torch.Tensor,
                                 eval_index: torch.Tensor, dataset: str, C: int, k: int,
                                 epoch: int, seed: int, out_dir: str, m_resamples: int, device):
    """Entry point called once per checkpoint from a runner's training loop.
    Logs BOTH loss_type='cls' and (if `cont_loss_fn` is not None) loss_type='cont'
    rows to out_dir/biasvariance.csv.

    Deliberately calls ONLY `model.encoder_q`/`model.encoder_k` directly (the
    plain SupConResNet forward, read-only), never `model.forward()` --
    PiCOModel.forward/ComCoModel.forward both mutate real training state as
    a side effect (MoCo queue enqueue, momentum key-encoder EMA update, and
    for PiCO, prototype EMA update), which a diagnostic forward pass on a
    resampled eval batch must never trigger. Calling encoder_k's forward
    (without the separate _momentum_update_key_encoder step) is safe: it
    only reads the key encoder's current (real, untouched) weights. The
    contrastive pool is therefore [q; k] for this eval batch only -- no
    queue -- which is enough to satisfy SupConLoss/ComCoContrastiveLoss's
    assumption that a same-sample "key view" positive exists in the pool
    (see oracle_cont_grad's docstring)."""
    model.encoder_q.eval()
    model.encoder_k.eval()
    with torch.no_grad():
        cls_out, q = model.encoder_q(eval_images)
        _, k_view = model.encoder_k(eval_images)
    logits = cls_out.detach()
    cls_softmax = F.softmax(logits, dim=1).detach()
    q = q.detach()
    k_view = k_view.detach()
    B = eval_images.shape[0]

    oracle_grad = oracle_cls_grad(logits, eval_true_labels)
    grads = []
    for m in range(m_resamples):
        seed_val = hash((dataset, C, k, epoch, algorithm, m)) & 0xFFFFFFFF
        partial_Y, comp_mask = _resample_batch(eval_true_labels, C, k, seed_val, device)
        g = _cls_loss_and_grad(algorithm, cls_loss_fn, logits, partial_Y, comp_mask, eval_index)
        grads.append(g)
    stats = _reduce(grads, oracle_grad)
    _append_row(out_dir, dict(dataset=dataset, C=C, k=k, algorithm=algorithm, seed=seed,
                               epoch=epoch, loss_type='cls', **{k2: round(v, 8) for k2, v in stats.items()}))

    if cont_loss_fn is None:
        return

    supcon_ref = cont_loss_fn if isinstance(cont_loss_fn, SupConLoss) else SupConLoss()
    pool_true = torch.cat([eval_true_labels, eval_true_labels])
    oc_grad = oracle_cont_grad(q, k_view, eval_true_labels, pool_true, supcon_ref)
    cont_grads = []

    if algorithm == 'ComCo-Fixed':
        pseudo_all = cls_out.argmax(dim=1).detach()
        pool_pseudo = torch.cat([pseudo_all, pseudo_all])
        for m in range(m_resamples):
            seed_val = hash((dataset, C, k, epoch, algorithm, 'cont', m)) & 0xFFFFFFFF
            _partial_Y, comp_mask = _resample_batch(eval_true_labels, C, k, seed_val, device)
            all_comp = torch.cat([comp_mask, comp_mask], dim=0)
            q_leaf = q.clone().requires_grad_(True)
            loss = cont_loss_fn(q_leaf, torch.cat([q_leaf, k_view], dim=0), pool_pseudo, all_comp,
                                 pseudo_all, warmup_pos=True, warmup_neg=True)
            loss.backward()
            cont_grads.append(q_leaf.grad.detach())
    else:
        for m in range(m_resamples):
            seed_val = hash((dataset, C, k, epoch, algorithm, 'cont', m)) & 0xFFFFFFFF
            partial_Y, _comp_mask = _resample_batch(eval_true_labels, C, k, seed_val, device)
            pseudo = resample_pseudo_target(partial_Y, cls_softmax)
            pool_pseudo = torch.cat([pseudo, pseudo])
            mask = torch.eq(pseudo.unsqueeze(1), pool_pseudo.unsqueeze(0)).float()
            q_leaf = q.clone().requires_grad_(True)
            loss = cont_loss_fn(torch.cat([q_leaf, k_view], dim=0), mask=mask, batch_size=B)
            loss.backward()
            cont_grads.append(q_leaf.grad.detach())

    cstats = _reduce(cont_grads, oc_grad)
    _append_row(out_dir, dict(dataset=dataset, C=C, k=k, algorithm=algorithm, seed=seed,
                               epoch=epoch, loss_type='cont', **{k2: round(v, 8) for k2, v in cstats.items()}))
