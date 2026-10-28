#!/usr/bin/env python
"""
STEP 8X: TRAIN THE FINAL PRISM MODEL (for release)
==================================================
The benchmark (step 6x) chose the MLP on feature set B (ESM-2 650M pocket mean + 11 fpocket
descriptors). All earlier models were trained inside cross-validation; the released model is
trained on ALL labelled pockets of the stripped holo structures with exactly the same recipe:

  - architecture 1291 -> 512 -> 256 -> 64 -> 1 (BatchNorm, ReLU, dropout 0.3/0.3/0.15),
    focal loss (gamma 2, alpha 0.75), AdamW lr 3e-4 wd 1e-4, batch 256, ReduceLROnPlateau,
    early stopping on validation AUPRC (patience 40, max 300 epochs)
  - fpocket descriptors standardised with the training rows' mean / sd
  - validation set for early stopping: one family fold (folds.csv, repeat 0); this is done for
    each of the 5 folds, so every family is used for training in 4 of the 5 members
  - --seeds models per fold -> ensemble of 5 x seeds networks (default 3 -> 15)

Output: models/prism_v1/
    member_f<fold>_s<seed>.pt   {"state": state_dict, "mu": [11], "sd": [11], "val_auprc": x}
    config.json                 feature definition, recipe, training-set size, members
RUN (from ~/allosite, env allosite, GPU):
    python scripts/step8x_final_model.py --seeds 3
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "models" / "prism_v1"
FP = ["score", "druggability_score", "volume", "polarity_score", "charge_score", "apolar_sasa", "polar_sasa",
      "flexibility", "number_of_alpha_spheres", "mean_local_hydrophobic_density", "n_pocket_residues"]


def build_mlp(d):
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


def train_one(Xtr, ytr, Xva, yva, seed, device):
    import torch
    import torch.nn as nn

    def focal(logits, y, gamma=2.0, alpha=0.75):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        at = alpha * y + (1 - alpha) * (1 - y)
        return (at * (1 - pt) ** gamma * ce).mean()

    def predict(model, X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(torch.sigmoid(model(torch.tensor(X[i:i + 8192], device=device))).cpu().numpy())
        return np.concatenate(out)

    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr.astype(np.float32), device=device)
    torch.manual_seed(seed * 42 + 1)
    np.random.seed(seed * 42 + 1)
    model = build_mlp(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=10, factor=0.5)
    best, state, bad, best_epoch = -1.0, None, 0, 0
    for epoch in range(300):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm) - 1, 256):
            idx = perm[i:i + 256]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = focal(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        ap = average_precision_score(yva, predict(model, Xva))
        sched.step(ap)
        if ap > best:
            best, state, bad, best_epoch = ap, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            bad += 1
            if bad >= 40:
                break
    return {k: v.detach().cpu() for k, v in state.items()}, float(best), best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="networks per validation fold")
    ap.add_argument("--out", default=str(OUTDIR))
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    S = s6.load_set("stripped")
    meta = S["meta"]
    lab = np.where(S["y"] >= 0)[0]
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    f0 = folds[folds.repeat == 0].set_index("family").fold
    fold_of = meta.family.astype(str).map(f0).to_numpy()
    print(f"training pockets: {len(lab)} (positives {int((S['y'][lab] == 1).sum())}), "
          f"families {meta.family.iloc[lab].nunique()}, device {device}")
    members = []
    t0 = time.time()
    for k in range(5):
        va = lab[fold_of[lab] == k]
        tr = lab[(fold_of[lab] != k)]
        parts_tr = s6.matrix(S, "B", tr)
        parts_va = s6.matrix(S, "B", va)
        mu, sd = parts_tr[1].mean(0), parts_tr[1].std(0) + 1e-6
        Xtr, Xva = s6.standardise(parts_tr, parts_va)
        for s in range(args.seeds):
            state, best, ep = train_one(Xtr, S["y"][tr], Xva, S["y"][va], 100 * k + s, device)
            name = f"member_f{k}_s{s}.pt"
            torch.save({"state": state, "mu": mu.tolist(), "sd": sd.tolist(), "val_auprc": best,
                        "best_epoch": ep, "n_in": int(Xtr.shape[1])}, out / name)
            members.append({"file": name, "val_fold": k, "seed": s, "val_auprc": round(best, 4), "best_epoch": ep})
            print(f"  fold {k} seed {s}: validation AUPRC {best:.3f} (best epoch {ep})  ({time.time() - t0:.0f}s)",
                  flush=True)
    cfg = {"name": "PRISM", "version": "1.0", "created": time.strftime("%Y-%m-%d"),
           "features": {"esm": "ESM-2 esm2_t33_650M_UR50D, last layer (33), mean over pocket residues, stored as float16",
                        "fpocket": FP, "fpocket_standardisation": "per member: training mean / sd"},
           "pockets": "fpocket (default parameters) on the ligand-free assembly",
           "architecture": "MLP 1291-512-256-64-1, BatchNorm, ReLU, dropout 0.3/0.3/0.15",
           "training": {"pockets": int(len(lab)), "positives": int((S["y"][lab] == 1).sum()),
                        "families": int(meta.family.iloc[lab].nunique()),
                        "loss": "focal (gamma 2, alpha 0.75)", "optimiser": "AdamW lr 3e-4 wd 1e-4, batch 256",
                        "early_stopping": "validation AUPRC, patience 40, max 300 epochs",
                        "validation": "one family fold (folds.csv repeat 0) per member"},
           "benchmark": "family-split CV (step 6x): PRIMARY Top-1 0.422, Top-3 0.583, site AUROC 0.856",
           "members": members}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    v = [m["val_auprc"] for m in members]
    print(f"done: {len(members)} members, validation AUPRC mean {np.mean(v):.3f} "
          f"(range {min(v):.3f}-{max(v):.3f}); saved in {out}")


if __name__ == "__main__":
    main()
