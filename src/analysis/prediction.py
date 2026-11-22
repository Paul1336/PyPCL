"""Prediction-level and representation-level diagnostics on a fixed evaluation set.

Origins: detail.log_per_class_checkpoint (per-class loss/acc), detail
.log_prediction_concentration_checkpoint (entropy / max-prob, train set), detail.knn_eval
(kNN probe), detail.maybe_plot_tsne, scripts/legacy/plot_{detailed,extended}_summary.py
(prediction-entropy collapse, effective classes, per-class std, confusion matrix).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .labels import entropy


@torch.no_grad()
def collect_outputs(forward, loader, device):
    """Run forward(x) -> logits (or (logits, feat)) over a loader of (x, y, ...) batches.
    Returns (logits [N, C] cpu, labels [N] cpu, feats or None)."""
    L, Ys, Fs = [], [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        out = forward(x)
        if isinstance(out, (tuple, list)):
            L.append(out[0].cpu())
            Fs.append(out[1].cpu())
        else:
            L.append(out.cpu())
        Ys.append(torch.as_tensor(y))
    return torch.cat(L), torch.cat(Ys), (torch.cat(Fs) if Fs else None)


def per_class_loss_acc(logits, y, C, predict='argmax'):
    """Per-class CE loss and accuracy (%) on samples whose TRUE label is c (NaN if absent).
    predict='argmin' for CPE. Returns dict(loss=[C], acc=[C], overall_acc)."""
    pred = logits.argmin(1) if predict == 'argmin' else logits.argmax(1)
    ce = -F.log_softmax(logits, 1)[torch.arange(len(y)), y]
    loss, acc = [], []
    for c in range(C):
        m = y == c
        loss.append(ce[m].mean().item() if m.any() else float('nan'))
        acc.append((pred[m] == c).float().mean().item() * 100 if m.any() else float('nan'))
    return {'loss': loss, 'acc': acc, 'overall_acc': (pred == y).float().mean().item() * 100}


def per_class_row(stats, epoch):
    """Flatten per_class_loss_acc into the per_class_loss.csv row layout."""
    r = {'epoch': epoch, 'overall_acc': stats['overall_acc']}
    for c, (l, a) in enumerate(zip(stats['loss'], stats['acc'])):
        r[f'loss_class_{c}'], r[f'acc_class_{c}'] = l, a
    return r


def concentration(probs):
    """Per-sample entropy (nats) and max probability of the model's own prediction, with
    mean/std/median summaries (concentration_summary.csv). std is torch default (unbiased)."""
    ent = entropy(probs.float())
    mp = probs.max(1).values
    s = {}
    for name, v in (('entropy', ent), ('max_prob', mp)):
        s[f'mean_{name}'], s[f'std_{name}'], s[f'median_{name}'] = v.mean().item(), v.std().item(), v.median().item()
    return s, {'entropy': ent, 'max_prob': mp}


def true_class_confidence_by_class(probs, y, C):
    """Mean p(true class) over samples of each class -> [C] (legacy logit heatmaps)."""
    pt = probs[torch.arange(len(y)), y]
    return [pt[y == c].mean().item() if (y == c).any() else float('nan') for c in range(C)]


def prediction_histogram_entropy(pred, C):
    """Entropy of the predicted-class histogram: low = collapse onto few classes.
    Max = log C."""
    p = np.bincount(np.asarray(pred), minlength=C) / max(len(pred), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def effective_classes(pred, C, share=0.05):
    """Number of classes receiving more than `share` of all predictions."""
    p = np.bincount(np.asarray(pred), minlength=C) / max(len(pred), 1)
    return int((p > share).sum())


def confusion_matrix(y, pred, C, normalise=True):
    """cm[t, p] counts; row-normalised (= per-class recall on the diagonal) if normalise."""
    cm = np.zeros((C, C))
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    if normalise:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    return cm


def per_class_acc_std(acc_per_class):
    """Spread of per-class accuracy (np.nanstd, ddof=0 as in the legacy plots)."""
    return float(np.nanstd(np.asarray(acc_per_class, dtype=float)))


@torch.no_grad()
def knn_top1(bank_feats, bank_labels, query_feats, query_labels, C, k=20, temperature=0.07):
    """Weighted kNN probe on L2-normalised features (MoCo protocol): top-k cosine neighbours,
    weight exp(sim / T), one-hot vote. Returns top-1 accuracy in %."""
    b = F.normalize(bank_feats.float(), dim=1)
    q = F.normalize(query_feats.float(), dim=1)
    correct = 0
    for i in range(0, len(q), 1024):
        sim = q[i:i + 1024] @ b.T
        ts, ti = sim.topk(min(k, b.shape[0]), dim=1)
        scores = (F.one_hot(bank_labels[ti], C).float() * (ts / temperature).exp()[..., None]).sum(1)
        correct += (scores.argmax(1) == query_labels[i:i + 1024]).sum().item()
    return 100.0 * correct / max(len(q), 1)


def tsne_embed(feats, seed=42, max_points=2000):
    """2-D t-SNE of features (same settings as detail.maybe_plot_tsne). Returns (emb, idx)."""
    from sklearn.manifold import TSNE
    n = len(feats)
    idx = np.random.RandomState(seed).permutation(n)[:max_points] if n > max_points else np.arange(n)
    x = np.asarray(feats)[idx]
    emb = TSNE(n_components=2, perplexity=min(30, max(5, len(x) // 100)), init='pca',
               random_state=seed).fit_transform(x)
    return emb, idx
