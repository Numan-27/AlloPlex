"""Shared helpers for the step 13 analyses.

train_members(): identical to step6x train_mlp (same MLP, focal loss, AdamW 3e-4 / 1e-4, batch 256,
ReduceLROnPlateau, early stopping on validation AUPRC, patience 40, max 300 epochs, seeds s*42+1),
with two additions: optional per-sample weights in the loss, and it RETURNS the trained networks
(state dicts) instead of only their averaged predictions.
"""
import copy

import numpy as np


def make_mlp(d):
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
                                     nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return MLP(d)


def predict(model, X, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i + 8192], device=device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_members(Xtr, ytr, Xva, yva, seeds, device, weights=None, log=None):
    """-> list of (seed, state_dict, best validation AUPRC). weights: per-training-sample loss weights."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score

    def focal(logits, y, w, gamma=2.0, alpha=0.75):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        at = alpha * y + (1 - alpha) * (1 - y)
        el = at * (1 - pt) ** gamma * ce
        return (el * w).sum() / w.sum() if w is not None else el.mean()

    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr.astype(np.float32), device=device)
    wt = torch.tensor(weights.astype(np.float32), device=device) if weights is not None else None
    out = []
    for s in seeds:
        torch.manual_seed(s * 42 + 1)
        np.random.seed(s * 42 + 1)
        model = make_mlp(Xtr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=10, factor=0.5)
        best, state, bad = -1.0, None, 0
        for epoch in range(300):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(perm) - 1, 256):
                idx = perm[i:i + 256]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = focal(model(Xt[idx]), yt[idx], wt[idx] if wt is not None else None)
                loss.backward()
                opt.step()
            ap = average_precision_score(yva, predict(model, Xva, device))
            sched.step(ap)
            if ap > best:
                best, state, bad = ap, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= 40:
                    break
        if log:
            log(f"      seed {s}: best validation AUPRC {best:.3f} (epoch {epoch + 1 - bad})")
        out.append((s, state, best))
    return out


def load_state(state, d, device):
    m = make_mlp(d).to(device)
    m.load_state_dict(state)
    m.eval()
    return m
