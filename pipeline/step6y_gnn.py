#!/usr/bin/env python
"""
STEP 6Y: GRAPH NEURAL NETWORK (pre-registered comparator; benchmark part 4)
==========================================================================
Each pocket becomes a graph of its 3D environment:
  nodes  = the pocket's residues + the nearest other residues whose centroid lies within
           12 A of the pocket centre (at most 64 nodes, pocket residues first)
  node features = ESM-2 650M residue embedding (frozen), reduced to 128 numbers by a PCA
           fitted on TRAINING-family residues of each fold; log1p(SASA of the chain alone);
           and, in the interface version, log1p(interface delta-SASA) and an interface flag
           (delta-SASA > 5 A^2); plus a flag "this residue lines the pocket"
  edges  = residue pairs closer than 8 A (centroids); messages use the squared distance,
           so the model is invariant to rotation/translation (E(3)-invariant message passing)
  model  = 3 message-passing layers (hidden 128) -> mean over pocket nodes + mean over all
           nodes -> 256-64-1; focal loss (gamma 2, alpha .75); AdamW 1e-3, wd 1e-4; batch 64
           graphs; each epoch uses all positives + 10x as many randomly drawn negatives;
           early stopping on inner-validation AUPRC (patience 10, max 60 epochs)
Same folds, inner validation, test sets and seeds rule (seed*42+1) as step 6x.
Three versions, saved under feature-set names that step 6x reports:
    gnn  B+I     = with the interface node features (pre-registered)
    gnn  B       = without them (pre-registered)
    gnn  B+I+fp  = B+I plus the 11 fpocket pocket descriptors (z-scored with training
                   rows) joined at the read-out: a fair match for the MLP, which also sees
                   them (added 2026-10-14, see the plan's deviations log)
Seeds per fold: 3 (deviations log, 2026-10-14).
Predictions: processed/dataset_v2/bench/preds/gnn__<B|B+I>__r<rep>__f<fold>.npz

RUN (from ~/allosite, GPU, torch):
    python scripts/step6y_gnn.py --smoke                       # 2-minute check that everything runs
    python scripts/step6y_gnn.py                              # full: 3 versions x 3 seeds (about 9-10 h; tmux; resumable)
    python scripts/step6x_train_models.py --models mlp,lgbm,gnn --report-only   # all models together
The first run builds processed/dataset_v2/bench/gnn_data.npz and gnn_esm.npy (cached).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
B = D / "bench"
ROOTS = {"stripped": (BASE / "processed" / "structures", "holo", "pockets_stripped.csv"),
         "partner": (BASE / "processed" / "structures_partner", "partner", "pockets_partner.csv")}
MAXN, RADIUS, EDGE = 64, 12.0, 8.0


# ----------------------------------------------------------------------------- data preparation
def centroids(pdb_path):
    acc = {}
    for line in open(pdb_path):
        if line.startswith("ATOM  "):
            k = (line[21], int(line[22:26]), line[26].strip())
            s = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
            s[0] += float(line[30:38]); s[1] += float(line[38:46]); s[2] += float(line[46:54]); s[3] += 1
    return {k: (v[0] / v[3], v[1] / v[3], v[2] / v[3]) for k, v in acc.items()}


def prepare():
    out = B / "gnn_data.npz"
    if out.exists() and (B / "gnn_esm.npy").exists():
        return
    idx = pd.read_csv(D / "esm_index.csv", keep_default_na=False)
    hashes = sorted(idx.seq_hash.unique())
    offset, chunks, pos = {}, [], 0
    for h in hashes:
        e = np.load(D / "esm" / f"{h}.npy")
        offset[h] = pos
        pos += len(e)
        chunks.append(e)
    chunks.append(np.zeros((1, 1280), np.float16))  # last row = "no embedding"
    np.save(B / "gnn_esm.npy", np.concatenate(chunks))
    zero_row = pos
    emap = {}
    for _, r in idx.iterrows():
        emap[(r["set"], r.pdb_id, r.chain)] = (r.seq_hash, {(int(n), str(ic)): i for i, (n, ic) in
                                                            enumerate(json.loads(r.residues))})
    xyz, feat, esm_row, graphs, pmask, setcol = [], [], [], [], [], []
    base = 0
    for set_name, (root, esm_set, csv_name) in ROOTS.items():
        if not (B / f"meta_{set_name}.csv").exists():
            continue
        meta = pd.read_csv(B / f"meta_{set_name}.csv", keep_default_na=False)
        pk = pd.read_csv(D / csv_name, usecols=["pdb_id", "pocket", "residues"], low_memory=False)
        assert len(pk) == len(meta) and (pk.pdb_id.values == meta.pdb_id.values).all(), "meta/pocket order mismatch"
        G = np.full((len(meta), MAXN), -1, np.int32)
        M = np.zeros((len(meta), MAXN), bool)
        for pdb, rows in meta.groupby("pdb_id", sort=False).groups.items():
            sdir = root / pdb
            res = pd.read_csv(sdir / "residues.csv", keep_default_na=False)
            cen = centroids(sdir / "stripped.pdb")
            local = {}
            for c, n, ic, ds, sa in zip(res.chain, res.resnum, res.icode, res.delta_sasa, res.sasa_alone):
                key = (str(c), int(n), str(ic))
                if key not in cen:
                    continue
                er = zero_row
                hit = emap.get((esm_set, pdb, str(c)))
                if hit:
                    r = hit[1].get((int(n), str(ic)))
                    if r is not None:
                        er = offset[hit[0]] + r
                local[key] = base + len(local)
                xyz.append(cen[key])
                ds = float(ds or 0)
                feat.append((np.log1p(float(sa or 0)), np.log1p(max(ds, 0)), float(ds > 5)))
                esm_row.append(er)
                setcol.append(set_name)
            keys = list(local)
            P = np.array([cen[k] for k in keys])
            tree = cKDTree(P) if len(P) else None
            for i in rows:
                pres = [local[(c, int(n), str(ic[0]) if ic else "")] for c, n, *ic in json.loads(pk.residues.iloc[i])
                        if (c, int(n), str(ic[0]) if ic else "") in local]
                if not pres or tree is None:
                    continue
                centre = np.array([xyz[j] for j in pres]).mean(0)
                d_p = sorted(pres, key=lambda j: np.linalg.norm(np.array(xyz[j]) - centre))[:MAXN]
                near = tree.query_ball_point(centre, RADIUS)
                others = sorted((base + j for j in near if base + j not in set(d_p)),
                                key=lambda j: np.linalg.norm(np.array(xyz[j]) - centre))
                nodes = (d_p + others)[:MAXN]
                G[i, :len(nodes)] = nodes
                M[i, :len(d_p)] = True
            base += len(local)
        graphs.append(G)
        pmask.append(M)
        print(f"  {set_name}: {len(meta)} pocket graphs, median nodes {int(np.median((G >= 0).sum(1)))}")
    np.savez(out, xyz=np.array(xyz, np.float32), feat=np.array(feat, np.float32),
             esm_row=np.array(esm_row, np.int64), zero_row=zero_row,
             **{f"graphs_{n}": g for n, g in zip([k for k in ROOTS if (B / f"meta_{k}.csv").exists()], graphs)},
             **{f"pmask_{n}": m for n, m in zip([k for k in ROOTS if (B / f"meta_{k}.csv").exists()], pmask)})
    print(f"saved {out} ({len(xyz)} residues)")


# ----------------------------------------------------------------------------- model
def build_model(torch, in_dim, hid=128, fp_dim=0):
    nn = torch.nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge = nn.Sequential(nn.Linear(2 * hid + 1, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
            self.node = nn.Sequential(nn.Linear(2 * hid, hid), nn.SiLU(), nn.Linear(hid, hid))
            self.norm = nn.LayerNorm(hid)

        def forward(self, h, d2, adj, valid):
            n = h.shape[1]
            hi = h[:, :, None, :].expand(-1, -1, n, -1)
            hj = h[:, None, :, :].expand(-1, n, -1, -1)
            m = self.edge(torch.cat([hi, hj, (d2 / EDGE ** 2)[..., None]], -1))
            m = (m * adj[..., None]).sum(2) / adj.sum(2, keepdim=True).clamp(min=1)
            h = self.norm(h + self.node(torch.cat([h, m], -1)))
            return h * valid[..., None]

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(in_dim, hid), nn.SiLU())
            self.layers = nn.ModuleList([Layer() for _ in range(3)])
            self.out = nn.Sequential(nn.Linear(2 * hid + fp_dim, 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 64),
                                     nn.SiLU(), nn.Linear(64, 1))

        def forward(self, x, f, pm, valid, fpv=None):
            h = self.inp(torch.cat([f, pm[..., None].float()], -1)) * valid[..., None]
            d2 = ((x[:, :, None] - x[:, None]) ** 2).sum(-1)
            eye = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)[None]
            adj = (valid[:, :, None] & valid[:, None] & (d2 < EDGE ** 2) & ~eye).float()
            for L in self.layers:
                h = L(h, d2, adj, valid)
            vp = (pm & valid).float()[..., None]
            va = valid.float()[..., None]
            pool = torch.cat([(h * vp).sum(1) / vp.sum(1).clamp(min=1), (h * va).sum(1) / va.sum(1).clamp(min=1)], -1)
            if fpv is not None:
                pool = torch.cat([pool, fpv], -1)
            return self.out(pool).squeeze(-1)

    return GNN()


# ----------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", default="0,1,2")
    ap.add_argument("--variants", default="B+I,B,B+I+fp",
                    help="B+I = with interface node features, B = without, B+I+fp = B+I plus the 11 fpocket descriptors")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    args = ap.parse_args()
    import torch
    from sklearn.metrics import average_precision_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s6.PRED.mkdir(parents=True, exist_ok=True)
    print("preparing graphs (cached after the first run) ...")
    prepare()
    Z = np.load(B / "gnn_data.npz")
    esm_all = np.load(B / "gnn_esm.npy", mmap_mode="r")
    xyz = torch.tensor(Z["xyz"], device=dev)
    feat_all = torch.tensor(Z["feat"], device=dev)
    esm_row = torch.tensor(Z["esm_row"], device=dev)
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if f"graphs_{n}" in Z.files}
    graphs = {n: torch.tensor(Z[f"graphs_{n}"].astype(np.int64), device=dev) for n in sets}
    pmask = {n: torch.tensor(Z[f"pmask_{n}"], device=dev) for n in sets}
    folds = pd.read_csv(B / "folds.csv", dtype={"family": str})
    repeats = [0] if args.smoke else [int(r) for r in args.repeats.split(",")]
    seeds = 1 if args.smoke else args.seeds
    max_ep, patience = (2, 1) if args.smoke else (60, 10)
    t0 = time.time()

    def focal(logits, y, gamma=2.0, alpha=0.75):
        p = torch.sigmoid(logits)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        at = alpha * y + (1 - alpha) * (1 - y)
        return (at * (1 - pt) ** gamma * ce).mean()

    for rep in repeats:
        fr = folds[folds.repeat == rep]
        gbf = {k: set(g.family) for k, g in fr.groupby("fold")}
        for fold in ([0] if args.smoke else range(5)):
            todo = [v for v in args.variants.split(",") if not (s6.PRED / f"gnn__{v}__r{rep}__f{fold}.npz").exists()
                    or args.smoke]
            if not todo:
                continue
            S = sets["stripped"]
            fam = S["meta"].family.astype(str)
            lab = S["y"] >= 0
            is_test = fam.isin(gbf[fold]).to_numpy()
            train_f = sorted(set(fam[~is_test & lab]))
            rng = np.random.default_rng(s6.SEED + 1000 * rep + fold)
            inner = set(rng.choice(train_f, max(1, int(s6.INNER_FRAC * len(train_f))), replace=False))
            is_inner = fam.isin(inner).to_numpy()
            tr = np.where(lab & ~is_test & ~is_inner)[0]
            va = np.where(lab & ~is_test & is_inner)[0]
            if args.smoke:
                tr = np.concatenate([tr[S["y"][tr] == 1][:200], tr[S["y"][tr] == 0][:2000]])
                va = va[:3000]
            # PCA of ESM fitted on residues of training-family graphs only
            g_tr = graphs["stripped"][torch.tensor(tr, device=dev)]
            rows = esm_row[g_tr[g_tr >= 0]].unique()
            rows = rows[torch.randperm(len(rows), device=dev)[:100000]].cpu().numpy()
            X = torch.tensor(np.asarray(esm_all[np.sort(rows)], np.float32), device=dev)
            mu = X.mean(0)
            _, _, V = torch.pca_lowrank(X - mu, q=128, center=False)
            del X
            proj = []
            for i in range(0, esm_all.shape[0], 200000):
                ch = torch.tensor(np.asarray(esm_all[i:i + 200000], np.float32), device=dev)
                proj.append(((ch - mu) @ V).half())
            esm128 = torch.cat(proj)
            esm128[int(Z["zero_row"])] = 0
            sd = esm128.float().std(0) + 1e-6
            esm128 = (esm128.float() / sd).half()

            fp_all = {en: sets[en]["fp"] for en in sets}
            fmu = fp_all["stripped"][tr].mean(0)
            fsd = fp_all["stripped"][tr].std(0) + 1e-6
            fp_t = {en: torch.tensor((fp_all[en] - fmu) / fsd, dtype=torch.float32, device=dev) for en in sets}

            def batch_inputs(set_name, idx, use_if):
                g = graphs[set_name][idx]
                valid = g >= 0
                gi = g.clamp(min=0)
                f = feat_all[gi].clone()
                if not use_if:
                    f[..., 1:] = 0
                e = esm128[esm_row[gi]].float()
                return xyz[gi], torch.cat([e, f], -1) * valid[..., None], pmask[set_name][idx] & valid, valid

            def predict(model, set_name, rows_, use_if, use_fp):
                model.eval()
                out = []
                with torch.no_grad():
                    for i in range(0, len(rows_), 256):
                        idx = torch.tensor(rows_[i:i + 256], device=dev)
                        x, f, pm, v = batch_inputs(set_name, idx, use_if)
                        fpv = fp_t[set_name][idx] if use_fp else None
                        out.append(torch.sigmoid(model(x, f, pm, v, fpv)).cpu().numpy())
                return np.concatenate(out) if out else np.zeros(0)

            tests = {}
            for en in sets:
                fe = sets[en]["meta"].family.astype(str)
                tests[en] = np.where(fe.isin(gbf[fold]).to_numpy() & (sets[en]["y"] >= 0))[0]
                if args.smoke:
                    tests[en] = tests[en][:2000]
            ytr, yva = S["y"][tr], S["y"][va]
            pos, neg = tr[ytr == 1], tr[ytr == 0]
            for variant in todo:
                use_if = variant in ("B+I", "B+I+fp")
                use_fp = variant == "B+I+fp"
                preds = {en: np.zeros(len(tests[en])) for en in sets}
                for s in range(seeds):
                    torch.manual_seed(s * 42 + 1)
                    r2 = np.random.default_rng(s * 42 + 1)
                    model = build_model(torch, 128 + 3 + 1, fp_dim=11 if use_fp else 0).to(dev)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    best, state, bad = -1.0, None, 0
                    for ep in range(max_ep):
                        model.train()
                        ep_rows = np.concatenate([pos, r2.choice(neg, min(len(neg), 10 * len(pos)), replace=False)])
                        r2.shuffle(ep_rows)
                        for i in range(0, len(ep_rows), 64):
                            idx = torch.tensor(ep_rows[i:i + 64], device=dev)
                            x, f, pm, v = batch_inputs("stripped", idx, use_if)
                            yb = torch.tensor(S["y"][ep_rows[i:i + 64]].astype(np.float32), device=dev)
                            opt.zero_grad()
                            fpv = fp_t["stripped"][idx] if use_fp else None
                            loss = focal(model(x, f, pm, v, fpv), yb)
                            loss.backward()
                            opt.step()
                        ap_ = average_precision_score(yva, predict(model, "stripped", va, use_if, use_fp))
                        if ap_ > best:
                            best, bad = ap_, 0
                            state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
                        else:
                            bad += 1
                            if bad >= patience:
                                break
                    model.load_state_dict(state)
                    for en in sets:
                        preds[en] += predict(model, en, tests[en], use_if, use_fp) / seeds
                    print(f"  gnn {variant} repeat {rep} fold {fold} seed {s}: best inner AUPRC {best:.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                if args.smoke:
                    print("SMOKE TEST OK: data, model, training and prediction all ran. Nothing was saved.")
                    return
                np.savez(s6.PRED / f"gnn__{variant}__r{rep}__f{fold}.npz",
                         **{f"{en}_rows": tests[en] for en in sets}, **{f"{en}_pred": preds[en] for en in sets})
            del esm128
            if dev == "cuda":
                torch.cuda.empty_cache()
    print("done; report with: python scripts/step6x_train_models.py --models mlp,lgbm,gnn --report-only")


if __name__ == "__main__":
    main()
