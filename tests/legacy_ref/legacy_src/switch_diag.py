"""Per-epoch diagnostics for WHY a bandit switches rho 1 -> 0 (2026-10-14).

Measured each epoch on a fixed subsample of TRAINING samples (no augmentation;
true labels used for diagnostics only, never for training or decisions):

  state():
    s_acc        argmax of the confidence s == true label
    s_ent        mean H(s) / log k  (1 = still uniform over the candidates)
    s_sharp      fraction with max s > 0.9
    cand_mass    mean P(S) = sum of model prob on the candidates
    disamb_acc   argmax of the model within S == true label
    p_ent        mean H(p restricted to S, renormalized) / log k
    agree_ps     model's within-S argmax == argmax s
    proto_acc    within-S argmax of prototype similarity (score_prot) == true label
                 (score_prot is what drives s's EMA update)
  grads():  on the first `grad_n` samples, gradient of the encoder w.r.t.
    the rho=0 loss, the rho=1 loss and the true-label cross entropy:
    cos0 = cos(g_rho0, g_true), cos1 = cos(g_rho1, g_true), cos01, and norms.
"""


import numpy as np
import torch
import torch.nn.functional as F

_MEAN = [0.4914, 0.4822, 0.4465]
_STD = [0.247, 0.2435, 0.2616]


class SwitchDiag:
    def __init__(self, pl_ds, orig_targets, train_idx, C, device, n: int = 2000, grad_n: int = 256,
                 seed: int = 0, chunk: int = 500):
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(np.asarray(train_idx), size=min(n, len(train_idx)), replace=False))
        x = torch.tensor(pl_ds.data[idx]).permute(0, 3, 1, 2).float() / 255.0
        self.x = ((x - torch.tensor(_MEAN).view(1, 3, 1, 1)) / torch.tensor(_STD).view(1, 3, 1, 1))
        cand = torch.zeros(len(idx), C)
        for row, i in enumerate(idx):
            cand[row, pl_ds.targets[i]] = 1.0
        self.cand = cand
        self.y = torch.as_tensor(orig_targets)[idx].long()
        self.index = torch.as_tensor(idx).long()
        self.C, self.device, self.grad_n, self.chunk = C, device, grad_n, chunk

    @torch.no_grad()
    def state(self, model, cls_loss) -> dict:
        was_training = model.training
        model.eval()
        dev = self.device
        acc = {k: [] for k in ('s_acc', 's_ent', 's_sharp', 'cand_mass', 'disamb_acc', 'p_ent',
                               'agree_ps', 'proto_acc')}
        protos = model.prototypes.detach()
        for i in range(0, self.x.shape[0], self.chunk):
            x = self.x[i:i + self.chunk].to(dev)
            cand = self.cand[i:i + self.chunk].to(dev)
            y = self.y[i:i + self.chunk].to(dev)
            s = cls_loss.partial_loss.confidence[self.index[i:i + self.chunk].to(dev)]
            logits, q = model.encoder_q(x)
            p = F.softmax(logits, dim=1)
            k = cand.sum(dim=1)
            logk = torch.log(k.clamp(min=2.0))
            s_arg = s.argmax(dim=1)
            pS = p * cand
            P = pS.sum(dim=1)
            pn = pS / P.clamp_min(1e-12).unsqueeze(1)
            p_arg = pS.argmax(dim=1)
            proto_arg = (F.softmax(q @ protos.t(), dim=1) * cand).argmax(dim=1)
            ent = lambda d: -(d * torch.log(d.clamp_min(1e-12))).sum(dim=1)
            acc['s_acc'].append((s_arg == y).float())
            acc['s_ent'].append(ent(s) / logk)
            acc['s_sharp'].append((s.max(dim=1).values > 0.9).float())
            acc['cand_mass'].append(P)
            acc['disamb_acc'].append((p_arg == y).float())
            acc['p_ent'].append(ent(pn) / logk)
            acc['agree_ps'].append((p_arg == s_arg).float())
            acc['proto_acc'].append((proto_arg == y).float())
        if was_training:
            model.train()
        return {k: round(torch.cat(v).mean().item(), 5) for k, v in acc.items()}

    def grads(self, model, loss_at) -> dict:
        """loss_at(logits, index, cand, value) -> scalar cls loss at rho/alpha=value."""
        was_training = model.training
        model.eval()
        dev = self.device
        n = self.grad_n
        x, cand, y, index = (self.x[:n].to(dev), self.cand[:n].to(dev), self.y[:n].to(dev), self.index[:n].to(dev))
        params = [p for p in model.encoder_q.parameters() if p.requires_grad]

        def flat_grad(loss_fn):
            model.zero_grad(set_to_none=True)
            logits, _ = model.encoder_q(x)
            loss_fn(logits).backward()
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
            model.zero_grad(set_to_none=True)
            return g

        g0 = flat_grad(lambda lg: loss_at(lg, index, cand, 0.0))
        g1 = flat_grad(lambda lg: loss_at(lg, index, cand, 1.0))
        gt = flat_grad(lambda lg: F.cross_entropy(lg, y))
        if was_training:
            model.train()
        cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
        return {'cos0': round(cos(g0, gt), 5), 'cos1': round(cos(g1, gt), 5), 'cos01': round(cos(g0, g1), 5),
                'norm0': round(g0.norm().item(), 5), 'norm1': round(g1.norm().item(), 5),
                'norm_true': round(gt.norm().item(), 5)}
