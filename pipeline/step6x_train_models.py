#!/usr/bin/env python
"""
STEP 6X: TRAIN AND EVALUATE THE MLP AND LightGBM (benchmark, part 3) - as pre-registered
=======================================================================================
Data: processed/dataset_v2/bench/ (step 5y). Training set = stripped holo pockets with
y in {0, 1}. Folds = folds.csv (5 family-grouped folds x 3 repeats). For every repeat
and fold: train on the other 4 folds, using 20 % of the TRAINING families (seeded) as
the inner validation set for early stopping / model choice; test folds are never used
for any choice.

Models
  mlp   step-8 recipe unchanged: 512-256-64, BatchNorm, dropout .3/.3/.15, focal loss
        (gamma 2, alpha .75), AdamW 3e-4 / wd 1e-4, batch 256, ReduceLROnPlateau,
        early stopping on inner-validation AUPRC (patience 40); --seeds per fold
        (plan: 10), ensemble = mean; seeds seed*42+1
  lgbm  LightGBM, small grid (num_leaves 15/63 x min_child_samples 20/100) chosen on
        the inner validation AUPRC, early stopping 100 rounds
Feature sets: B, B+I, B+I-d, I (see step 5y). Non-ESM columns are z-scored with the
training rows' statistics; ESM columns are used as they are (as in the old pipeline).

Evaluation (site level, so every set is scored the same way)
  for each site in a test family, with its positive pocket p and the negative pockets N
  of the same structure:
     Top-1 = p scores above every pocket in N; Top-3 = p within the first 3 (ties broken at
     random: expected value)
     site AUROC = share of N scored below p (ties 1/2)
  a site whose pocket was not detected (no positive pocket) counts as a Top-k miss and
  is left out of site AUROC.
  sets: stripped (holo, ligands removed), partner (modulator-free structures), and
  PRIMARY = partner result where the site has one, otherwise stripped (the plan's rule).
  Also pooled AUPRC per set. Metrics are averaged over the 3 repeats.
  Paired differences (B+I minus B etc.) with 95 % FAMILY-bootstrap intervals.
  Decision (plan): interface features help only if, for the MLP on PRIMARY, BOTH the
  Top-1 and the site-AUROC differences (B+I minus B) have intervals above 0.

Inflation table (--inflation, MLP with feature set B, repeat 0 only):
  random pocket-level split | gene-name split | holo inputs (ligands present) versus the
  family split on stripped structures.

Predictions are saved per run in bench/preds/ (resumable: finished runs are skipped).
RUN (from ~/allosite, GPU; needs torch, scikit-learn; pip install lightgbm for lgbm):
    python scripts/step6x_train_models.py --models mlp,lgbm --seeds 2 --repeats 0     # quick trial
    python scripts/step6x_train_models.py --models mlp,lgbm --seeds 10 --inflation    # full (hours; tmux)
    python scripts/step6x_train_models.py --report-only                               # tables from saved runs
ABLATION (loss): plain BCE instead of focal loss, same everything else (writes its own report):
    python scripts/step6x_train_models.py --models mlp,mlpbce --features B --seeds 10 --report-name report_ablation_loss.txt
"""
import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path(__file__).resolve().parent.parent
B = BASE / "processed" / "dataset_v2" / "bench"
D = BASE / "processed" / "dataset_v2"
PRED = B / "preds"
FEATS = ["B", "B+I", "B+I-d", "I", "B+I+fp", "E", "F"]  # B+I+fp: GNN only; E / F: ablation (ESM only / fpocket only)
INNER_FRAC, SEED = 0.2, 20261013
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------- data
def load_set(name):
    f = np.load(B / f"features_{name}.npz")
    meta = pd.read_csv(B / f"meta_{name}.csv", keep_default_na=False)
    return {"esm": f["X_esm"], "fp": f["X_fp"], "if": f["X_if"], "y": f["y"], "meta": meta}


def matrix(S, feat, rows):
    size = S["meta"]["n_pocket_residues"].to_numpy(np.float32)[rows, None]
    if feat == "B":
        return S["esm"][rows].astype(np.float32), S["fp"][rows]
    if feat == "B+I":
        return S["esm"][rows].astype(np.float32), np.c_[S["fp"][rows], S["if"][rows]]
    if feat == "B+I-d":
        return S["esm"][rows].astype(np.float32), np.c_[S["fp"][rows], S["if"][rows][:, :3]]
    if feat == "I":
        return np.zeros((len(rows), 0), np.float32), np.c_[S["if"][rows], size]
    if feat == "E":  # ablation: ESM-2 pocket mean only
        return S["esm"][rows].astype(np.float32), np.zeros((len(rows), 0), np.float32)
    if feat == "F":  # ablation: the 11 fpocket descriptors only
        return np.zeros((len(rows), 0), np.float32), S["fp"][rows]
    raise ValueError(feat)


def standardise(train_parts, *others):
    esm_tr, oth_tr = train_parts
    mu, sd = oth_tr.mean(0), oth_tr.std(0) + 1e-6
    fx = lambda p: np.c_[p[0], (p[1] - mu) / sd].astype(np.float32)
    return [fx(train_parts)] + [fx(o) for o in others]


# ----------------------------------------------------------------------------- models
def train_mlp(Xtr, ytr, Xva, yva, Xtests, seeds, device, loss_name="focal"):
    import torch
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
        return np.concatenate(out) if out else np.zeros(0)

    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr.astype(np.float32), device=device)
    preds = [np.zeros(len(x)) for x in Xtests]
    for s in seeds:
        torch.manual_seed(s * 42 + 1)
        np.random.seed(s * 42 + 1)
        model = MLP(Xtr.shape[1]).to(device)
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
                out_ = model(Xt[idx])
                loss = (focal(out_, yt[idx]) if loss_name == "focal" else
                        nn.functional.binary_cross_entropy_with_logits(out_, yt[idx]))
                loss.backward()
                opt.step()
            ap = average_precision_score(yva, predict(model, Xva))
            sched.step(ap)
            if ap > best:
                best, state, bad = ap, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= 40:
                    break
        model.load_state_dict(state)
        for k, X in enumerate(Xtests):
            preds[k] += predict(model, X) / len(seeds)
    return preds


def train_lgbm(Xtr, ytr, Xva, yva, Xtests, seed):
    import lightgbm as lgb
    best = None
    pos_w = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    for nl in (15, 63):
        for mcs in (20, 100):
            m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=nl, min_child_samples=mcs,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
                                   scale_pos_weight=pos_w, random_state=seed, verbose=-1)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
            if best is None or ap > best[0]:
                best = (ap, m)
    return [best[1].predict_proba(X)[:, 1] for X in Xtests]


def train_fake(Xtr, ytr, Xva, yva, Xtests, seed):
    """testing only: logistic regression"""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=300, class_weight="balanced").fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xtests]


# ----------------------------------------------------------------------------- one run
def run(model, feat, rep, fold, groups, sets, eval_names, train_name, args, device, tag=""):
    out = PRED / f"{model}{tag}__{feat}__r{rep}__f{fold}.npz"
    if out.exists():
        return
    S = sets[train_name]
    g = groups[train_name]
    lab = S["y"] >= 0
    test_g = set(g["groups_by_fold"][fold])
    is_test = g["group"].isin(test_g).to_numpy()
    train_groups = sorted(set(g["group"][~is_test & lab]))
    rng = np.random.default_rng(SEED + 1000 * rep + fold)
    inner = set(rng.choice(train_groups, max(1, int(INNER_FRAC * len(train_groups))), replace=False))
    is_inner = g["group"].isin(inner).to_numpy()
    tr = np.where(lab & ~is_test & ~is_inner)[0]
    va = np.where(lab & ~is_test & is_inner)[0]
    tests = {}
    for en in eval_names:
        E = sets[en]
        ge = groups[en]
        tests[en] = np.where(ge["group"].isin(test_g).to_numpy() & (E["y"] >= 0))[0]
    parts_tr = matrix(S, feat, tr)
    parts = [matrix(S, feat, va)] + [matrix(sets[en], feat, tests[en]) for en in eval_names]
    Xtr, Xva, *Xtests = standardise(parts_tr, *parts)
    ytr, yva = S["y"][tr], S["y"][va]
    if model in ("mlp", "mlpbce"):  # mlpbce = ablation: plain (unweighted) binary cross-entropy instead of focal loss
        preds = train_mlp(Xtr, ytr, Xva, yva, Xtests, range(args.seeds), device,
                          loss_name="bce" if model == "mlpbce" else "focal")
    elif model == "lgbm":
        preds = train_lgbm(Xtr, ytr, Xva, yva, Xtests, SEED + fold)
    else:
        preds = train_fake(Xtr, ytr, Xva, yva, Xtests, SEED + fold)
    np.savez(out, **{f"{en}_rows": tests[en] for en in eval_names}, **{f"{en}_pred": p for en, p in zip(eval_names, preds)})


# ----------------------------------------------------------------------------- evaluation
def site_table(set_name, meta, sites_csv):
    """one row per site of this set: site_id, pdb_id, family, positive pocket row (or -1)."""
    S = pd.read_csv(sites_csv, low_memory=False)
    S = S[S.status == "ok"]
    pos_row = {}
    for i, (s, y) in enumerate(zip(meta.sites.astype(str), meta.y)):
        if y == 1:
            for sid in s.split(","):
                if sid:
                    pos_row[sid] = i
    fam = meta.drop_duplicates("pdb_id").set_index("pdb_id").family
    out = pd.DataFrame({"site_id": S.site_id, "pdb_id": S.pdb_id})
    out["family"] = out.pdb_id.map(fam).astype(str)
    out["pos_row"] = out.site_id.map(pos_row).fillna(-1).astype(int)
    return out


def score_sites(sites, meta, scores):
    """scores: array over meta rows (NaN = not predicted). Returns per-site metrics."""
    neg_by_pdb = defaultdict(list)
    yy = meta.y.to_numpy()
    for i, (p, y) in enumerate(zip(meta.pdb_id, yy)):
        if y == 0 and not np.isnan(scores[i]):
            neg_by_pdb[p].append(scores[i])
    neg_by_pdb = {p: np.array(v) for p, v in neg_by_pdb.items()}
    rows = []
    for s, p, f, r in zip(sites.site_id, sites.pdb_id, sites.family, sites.pos_row):
        if r >= 0 and np.isnan(scores[r]):
            continue  # site not in a predicted (test) structure
        if r < 0:
            if p not in neg_by_pdb:
                continue
            rows.append((s, f, 0.0, 0.0, np.nan))
            continue
        n = neg_by_pdb.get(p, np.array([]))
        if len(n) == 0:
            continue
        sp = scores[r]
        n_gt, n_eq = int((n > sp).sum()), int((n == sp).sum())
        # ties broken at random: expected Top-k = P(rank <= k), rank = n_gt + 1 + U, U uniform on 0..n_eq
        top1 = (1.0 / (n_eq + 1)) if n_gt == 0 else 0.0
        top3 = max(0, min(n_eq + 1, 3 - n_gt)) / (n_eq + 1)
        rows.append((s, f, top1, top3, 1 - (n_gt + 0.5 * n_eq) / len(n)))
    return pd.DataFrame(rows, columns=["site_id", "family", "top1", "top3", "site_auc"])


def collect(model, feat, set_name, meta, repeats, tag=""):
    """average predicted scores per repeat -> list of arrays over meta rows."""
    out = []
    for rep in repeats:
        sc = np.full(len(meta), np.nan)
        files = sorted(PRED.glob(f"{model}{tag}__{feat}__r{rep}__f*.npz"))
        if len(files) < 5:
            return None
        for fpath in files:
            z = np.load(fpath)
            if f"{set_name}_rows" in z:
                sc[z[f"{set_name}_rows"]] = z[f"{set_name}_pred"]
        out.append(sc)
    return out


def boot(fam, vals, B_=2000):
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"f": fam, "v": vals}).dropna()
    g = df.groupby("f").v.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), (B_, len(g)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def evaluate(models, repeats, sets, site_tabs, tag="", label=None):
    res = {}
    for model in models:
        for feat in FEATS:
            per = {}
            for en in ("stripped", "partner"):
                if en not in sets:
                    continue
                sc = collect(model, feat, en, sets[en]["meta"], repeats, tag)
                if sc is None:
                    continue
                tabs = [score_sites(site_tabs[en], sets[en]["meta"], s) for s in sc]
                t = pd.concat(tabs).groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean()
                y = sets[en]["y"]
                aps = []
                for s in sc:
                    m = ~np.isnan(s) & (y >= 0)
                    if len(set(y[m])) == 2:
                        aps.append(average_precision_score(y[m], s[m]))
                per[en] = (t, float(np.mean(aps)) if aps else None)
            if "stripped" not in per:
                continue
            prim = per["stripped"][0].set_index("site_id")
            if "partner" in per:
                part = per["partner"][0].set_index("site_id")
                prim = pd.concat([part, prim[~prim.index.isin(part.index)]])
            per["primary"] = (prim.reset_index(), None)
            res[(model, feat)] = per
    return res


def report(res, title):
    say("")
    say("=" * 108)
    say(title)
    say("=" * 108)
    say(f"  {'model':<6}{'features':<8}{'set':<10}{'sites':>7}{'Top-1':>9}{'Top-3':>9}{'site AUROC':>12}{'AUPRC':>9}")
    for (model, feat), per in res.items():
        for en in ("primary", "partner", "stripped"):
            if en in per:
                t, ap = per[en]
                say(f"  {model:<6}{feat:<8}{en:<10}{len(t):>7}{t.top1.mean():>9.3f}{t.top3.mean():>9.3f}"
                    f"{t.site_auc.mean():>12.3f}{'' if ap is None else f'{ap:9.3f}'}")


def paired(res, model, a, b, sets_=("primary", "partner", "stripped")):
    if (model, a) not in res or (model, b) not in res:
        return None
    out = {}
    say(f"  {model}: {a} minus {b}")
    for en in sets_:
        if en not in res[(model, a)] or en not in res[(model, b)]:
            continue
        ta = res[(model, a)][en][0].set_index("site_id")
        tb = res[(model, b)][en][0].set_index("site_id")
        common = ta.index.intersection(tb.index)
        for m in ("top1", "site_auc"):
            d = (ta.loc[common, m] - tb.loc[common, m])
            lo, hi = boot(ta.loc[common, "family"].to_numpy(), d.to_numpy())
            say(f"     {en:<9}{m:<9} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  (sites {d.notna().sum()})")
            out[(en, m)] = (d.mean(), lo, hi)
    return out


def paired_x(res, A, B_, sets_=("primary", "partner", "stripped")):
    """paired difference between two (model, features) entries, e.g. MLP B vs GNN B+I+fp."""
    if A not in res or B_ not in res:
        return None
    say(f"  {A[0]} {A[1]} minus {B_[0]} {B_[1]}")
    out = {}
    for en in sets_:
        if en not in res[A] or en not in res[B_]:
            continue
        ta = res[A][en][0].set_index("site_id")
        tb = res[B_][en][0].set_index("site_id")
        common = ta.index.intersection(tb.index)
        for m in ("top1", "top3", "site_auc"):
            d = (ta.loc[common, m] - tb.loc[common, m])
            lo, hi = boot(ta.loc[common, "family"].to_numpy(), d.to_numpy())
            say(f"     {en:<9}{m:<9} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  (sites {d.notna().sum()})")
            out[(en, m)] = (d.mean(), lo, hi)
    return out


def add_ensemble(res, sets, site_tabs, repeats, members=(("mlp", "B"), ("gnn", "B+I+fp")), name=("ens", "MLP+GNN")):
    """EXPLORATORY (not pre-registered): average the predicted probabilities of two models."""
    per = {}
    for en in ("stripped", "partner"):
        if en not in sets:
            continue
        scs = [collect(m, f, en, sets[en]["meta"], repeats) for m, f in members]
        if any(x is None for x in scs):
            return
        sc = [np.nanmean(np.vstack([x[r] for x in scs]), axis=0) for r in range(len(repeats))]
        tabs = [score_sites(site_tabs[en], sets[en]["meta"], v) for v in sc]
        t = pd.concat(tabs).groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean()
        y = sets[en]["y"]
        aps = [average_precision_score(y[m_], v[m_]) for v in sc
               for m_ in [~np.isnan(v) & (y >= 0)] if len(set(y[m_])) == 2]
        per[en] = (t, float(np.mean(aps)) if aps else None)
    if "stripped" not in per:
        return
    prim = per["stripped"][0].set_index("site_id")
    if "partner" in per:
        part = per["partner"][0].set_index("site_id")
        prim = pd.concat([part, prim[~prim.index.isin(part.index)]])
    per["primary"] = (prim.reset_index(), None)
    res[name] = per


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,lgbm")
    ap.add_argument("--features", default="B,B+I,B+I-d,I", help="B+I+fp exists for the GNN only")
    ap.add_argument("--repeats", default="0,1,2")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--inflation", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--fake-model", action="store_true", help="testing only (logistic regression)")
    ap.add_argument("--report-name", default="report_step6x.txt", help="report file name in bench/")
    args = ap.parse_args()
    PRED.mkdir(parents=True, exist_ok=True)
    models = ["fake"] if args.fake_model else [m for m in args.models.split(",") if m]
    feats = [f for f in args.features.split(",") if f]
    repeats = [int(r) for r in args.repeats.split(",")]
    device = "cpu"
    if any(m.startswith("mlp") for m in models):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sets = {n: load_set(n) for n in ("stripped", "partner", "holo") if (B / f"features_{n}.npz").exists()}
    folds = pd.read_csv(B / "folds.csv", dtype={"family": str})
    master = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", low_memory=False,
                         usecols=["pdb_id", "gene", "target_id"])
    gene = master.drop_duplicates("pdb_id").set_index("pdb_id")
    groups = {}
    for n, S in sets.items():
        groups[n] = {"group": S["meta"].family.astype(str)}
    desc = ", ".join("%s (%d pockets)" % (n, len(S["y"])) for n, S in sets.items())
    say(f"sets: {desc}; device {device}")

    if not args.report_only:
        t0 = time.time()
        for rep in repeats:
            fr = folds[folds.repeat == rep]
            gbf = {k: list(g.family) for k, g in fr.groupby("fold")}
            for n in sets:
                groups[n]["groups_by_fold"] = gbf
            for model in models:
                for feat in feats:
                    for fold in range(5):
                        run(model, feat, rep, fold, groups, sets, [e for e in ("stripped", "partner") if e in sets],
                            "stripped", args, device)
                        say(f"  done {model} {feat} repeat {rep} fold {fold}  ({time.time() - t0:.0f}s)")
        if args.inflation:
            m0 = "fake" if args.fake_model else "mlp"
            fr = folds[folds.repeat == 0]
            # (a) random pocket split, (b) gene split, (c) holo inputs; feature set B, repeat 0
            S = sets["stripped"]
            rng = np.random.default_rng(SEED)
            variants = {"_random": pd.Series(rng.integers(0, 5, len(S["y"])).astype(str)),
                        "_gene": S["meta"].pdb_id.map(gene.target_id).fillna(S["meta"].pdb_id).astype(str)}
            for tag, grp in variants.items():
                if tag == "_random":
                    gbf = {k: [str(k)] for k in range(5)}
                else:
                    ug = sorted(grp.unique())
                    perm = np.random.default_rng(SEED).permutation(len(ug))
                    gbf = {k: [ug[i] for i in perm[k::5]] for k in range(5)}
                gr = {"stripped": {"group": grp, "groups_by_fold": gbf}}
                for fold in range(5):
                    run(m0, "B", 0, fold, gr, {"stripped": S}, ["stripped"], "stripped", args, device, tag=tag)
                say(f"  done inflation {tag}")
            if "holo" in sets:
                gbf = {k: list(g.family) for k, g in fr.groupby("fold")}
                gr = {"holo": {"group": sets["holo"]["meta"].family.astype(str), "groups_by_fold": gbf}}
                for fold in range(5):
                    run(m0, "B", 0, fold, gr, {"holo": sets["holo"]}, ["holo"], "holo", args, device, tag="_holo")
                say("  done inflation _holo")

    # ------------------------------------------------------------------ report
    site_tabs = {"stripped": site_table("stripped", sets["stripped"]["meta"], D / "sites_stripped.csv")}
    if "partner" in sets:
        site_tabs["partner"] = site_table("partner", sets["partner"]["meta"], D / "sites_partner.csv")
    res = evaluate(models, repeats, sets, site_tabs)
    report(res, f"RESULTS (mean over repeats {repeats}; PRIMARY = partner where available, else stripped)")
    say("")
    say("PAIRED DIFFERENCES (95 % family-bootstrap intervals)")
    for model in models:
        for a, b in (("B+I", "B"), ("B+I-d", "B"), ("B+I", "B+I-d"), ("I", "B"), ("B+I+fp", "B+I"), ("E", "B"), ("F", "B")):
            paired(res, model, a, b)
    if ("mlp", "B") in res and ("gnn", "B+I+fp") in res:
        say("")
        say("MODEL COMPARISON (which model becomes the tool?)")
        paired_x(res, ("mlp", "B"), ("gnn", "B+I+fp"))
        if ("lgbm", "B+I") in res:
            paired_x(res, ("mlp", "B"), ("lgbm", "B+I"), ("primary",))
        add_ensemble(res, sets, site_tabs, repeats)
        if ("ens", "MLP+GNN") in res:
            say("")
            say("EXPLORATORY (not pre-registered): ensemble = mean of MLP (B) and GNN (B+I+fp) probabilities")
            for en in ("primary", "partner", "stripped"):
                if en in res[("ens", "MLP+GNN")]:
                    t, ap = res[("ens", "MLP+GNN")][en]
                    say(f"  ens   MLP+GNN {en:<9}{len(t):>6}{t.top1.mean():>9.3f}{t.top3.mean():>9.3f}{t.site_auc.mean():>12.3f}"
                        + (f"{ap:>9.3f}" if ap is not None else ""))
            paired_x(res, ("ens", "MLP+GNN"), ("mlp", "B"))
    if ("mlp", "B") in res and ("mlpbce", "B") in res:
        say("")
        say("ABLATION: loss function (MLP, feature set B): focal loss minus plain binary cross-entropy")
        paired_x(res, ("mlp", "B"), ("mlpbce", "B"))
    mp = "fake" if args.fake_model else "mlp"
    if (mp, "B+I") in res and (mp, "B") in res:
        d = paired(res, mp, "B+I", "B", ("primary",))
        ok = d and d[("primary", "top1")][1] > 0 and d[("primary", "site_auc")][1] > 0
        say(f"  -> DECISION (plan): interface features improve prediction: {'SUPPORTED' if ok else 'NOT SUPPORTED'}")

    if args.inflation or args.report_only:
        m0 = "fake" if args.fake_model else "mlp"
        ref = evaluate([m0], [0], {"stripped": sets["stripped"]}, {"stripped": site_tabs["stripped"]})
        if (m0, "B") in ref:
            say("")
            say("INFLATION TABLE (feature set B, repeat 0; reference = family split, stripped structures)")
            rt = ref[(m0, "B")]["stripped"][0].set_index("site_id")
            say(f"  {'split / input':<22}{'Top-1':>8}{'site AUROC':>12}   difference vs reference (Top-1, 95 % family CI)")
            say(f"  {'family (reference)':<22}{rt.top1.mean():>8.3f}{rt.site_auc.mean():>12.3f}")
            for tag, name, setn in (("_random", "random pocket split", "stripped"), ("_gene", "gene-name split", "stripped"),
                                    ("_holo", "holo inputs", "holo")):
                if setn not in sets:
                    continue
                sc = collect(m0, "B", setn, sets[setn]["meta"], [0], tag)
                if sc is None:
                    continue
                st = site_tabs["stripped"] if setn == "stripped" else site_table("holo", sets["holo"]["meta"],
                                                                                   D / "sites_holo.csv")
                t = score_sites(st, sets[setn]["meta"], sc[0]).set_index("site_id")
                common = t.index.intersection(rt.index)
                dd = t.loc[common, "top1"] - rt.loc[common, "top1"]
                lo, hi = boot(rt.loc[common, "family"].to_numpy(), dd.to_numpy())
                say(f"  {name:<22}{t.top1.mean():>8.3f}{t.site_auc.mean():>12.3f}   {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (B / args.report_name).write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
