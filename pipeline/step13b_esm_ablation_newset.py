#!/usr/bin/env python
"""
STEP 13B: IS ESM-2 WORTH IT ON NEW DATA? (supervisor point 5)
=============================================================
Trains three released-style ensembles under IDENTICAL settings and scores them on the independent
2024+ complexes (step 10e), so the only difference is the input features:
    B  ESM-2 pocket mean + 11 fpocket descriptors   (= AlloPlex)
    F  11 fpocket descriptors only
    E  ESM-2 pocket mean only
Released-style ensemble = for each of the 5 family folds (folds.csv, repeat 0), 3 networks (seeds 0-2)
trained on the other 4 folds with that fold used for early stopping -> 15 networks per feature set.
A 2024+ complex similar to a training family is scored with the 3 networks that never trained on that
family; a complex from a new family with all 15 (the same rule as the paper, taken from step 10e).

RUN (from ~/allosite, env allosite; GPU recommended):
    python scripts/step13b_esm_ablation_newset.py train      # 45 networks, ~1-3 h on a GPU
    python scripts/step13b_esm_ablation_newset.py score      # featurize the 2024+ set once (cached), score
Needs: processed/newset/built.csv + structures/ (step 10e build) and processed/newset/results.csv
       (step 10e score, for the family assignment); fpocket on PATH; ESM-2 weights as for step 10e.
Output: models/ablation_{B,F,E}/member_f<k>_s<s>.pt, processed/newset/features/<pdb>.pkl,
        figures/data/s13b_newset_ablation.csv (per complex), figures/data/report_step13b.txt
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402
import step13_common as c13  # noqa: E402

BASE = s6.BASE
NEW = BASE / "processed" / "newset"
MOD = BASE / "models"
DATA = BASE / "figures" / "data"
FEATS = ("B", "F", "E")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def device_of(args):
    import torch
    return "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"


def train(args):
    import torch
    device = device_of(args)
    S = s6.load_set("stripped")
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    fold = folds[folds.repeat == 0].set_index("family").fold
    fam = S["meta"].family.astype(str)
    k_of = fam.map(fold).fillna(-1).astype(int).to_numpy()
    lab = S["y"] >= 0
    say(f"stripped set: {len(S['y']):,} pockets; device {device}; seeds per fold {args.seeds}")
    for feat in args.feats.split(","):
        out = MOD / f"ablation_{feat}"
        out.mkdir(parents=True, exist_ok=True)
        for k in range(5):
            if all((out / f"member_f{k}_s{s}.pt").exists() for s in range(args.seeds)):
                say(f"  {feat} fold {k}: done already")
                continue
            tr = np.where(lab & (k_of != k) & (k_of >= 0))[0]
            va = np.where(lab & (k_of == k))[0]
            esm_tr, oth_tr = s6.matrix(S, feat, tr)
            esm_va, oth_va = s6.matrix(S, feat, va)
            mu, sd = oth_tr.mean(0), oth_tr.std(0) + 1e-6
            Xtr = np.c_[esm_tr, (oth_tr - mu) / sd].astype(np.float32)
            Xva = np.c_[esm_va, (oth_va - mu) / sd].astype(np.float32)
            say(f"  {feat} fold {k}: train {len(tr):,} pockets, early-stopping fold {len(va):,}, {Xtr.shape[1]} inputs")
            for s, state, best in c13.train_members(Xtr, S["y"][tr], Xva, S["y"][va], range(args.seeds), device,
                                                    log=say):
                torch.save({"state": state, "n_in": Xtr.shape[1], "mu": mu.tolist(), "sd": sd.tolist(),
                            "feat": feat, "fold": k, "seed": s, "val_auprc": best}, out / f"member_f{k}_s{s}.pt")
    (DATA / "report_step13b_train.txt").write_text("\n".join(LOG) + "\n")


def featurize(path, workdir, fpocket, esm, pp):
    """step 10e / alloplex_predict.score_structure without the scoring: -> pocket table, X_esm, X_fp"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    clean = workdir / "clean.pdb"
    pp.clean_structure(path, clean)
    P = pp.run_fpocket(clean, fpocket)
    if P.empty:
        raise RuntimeError("fpocket found no pockets")
    emb = {}
    for ch, res in pp.chains_of(clean).items():
        emb[ch] = (esm.embed_chain("".join(r[2] for r in res)), {(r[0], r[1]): i for i, r in enumerate(res)})
    X_esm = np.zeros((len(P), 1280), np.float32)
    for i, res in enumerate(P.residues):
        vecs = [emb[c][0][emb[c][1][(n, ic)]] for c, n, ic in res if c in emb and (n, ic) in emb[c][1]]
        if vecs:
            X_esm[i] = np.mean(vecs, 0)
    X_esm = X_esm.astype(np.float16).astype(np.float32)
    return P, X_esm, P[pp.FP].to_numpy(np.float32)


def load_feat_members(feat, key, device):
    import torch
    pat = f"member_{key}_s*.pt" if key != "all" else "member_*.pt"
    mem = []
    for f in sorted((MOD / f"ablation_{feat}").glob(pat)):
        ck = torch.load(f, map_location=device)
        mem.append((c13.load_state(ck["state"], ck["n_in"], device), np.array(ck["mu"], np.float32),
                    np.array(ck["sd"], np.float32)))
    if not mem:
        raise SystemExit(f"no networks for {feat} {key} - run 'train' first")
    return mem


def score_feat(mem, feat, X_esm, X_fp, device):
    import torch
    e = X_esm if feat in ("B", "E") else np.zeros((len(X_esm), 0), np.float32)
    o = X_fp if feat in ("B", "F") else np.zeros((len(X_fp), 0), np.float32)
    out = np.zeros(len(X_esm))
    with torch.no_grad():
        for m, mu, sd in mem:
            X = np.c_[e, (o - mu) / sd].astype(np.float32)
            out += torch.sigmoid(m(torch.tensor(X, device=device))).cpu().numpy() / len(mem)
    return out


def score(args):
    import alloplex_predict as pp
    import step10e_new_test_set as s10e
    device = device_of(args)
    b = pd.read_csv(NEW / "built.csv")
    b = b[b.used]
    res10e = pd.read_csv(NEW / "results.csv").set_index("pdb_id")
    root = NEW / "structures"
    cache = NEW / "features"
    cache.mkdir(exist_ok=True)
    esm, fpocket = None, shutil.which("fpocket")
    members = {}
    rows = []
    for r in b.itertuples():
        if r.pdb_id not in res10e.index:
            continue                                    # failed in step 10e as well
        key = res10e.loc[r.pdb_id, "networks"]           # 'f<k>' (held-out networks) or 'all' (new family)
        cf = cache / f"{r.pdb_id}.pkl"
        if cf.exists():
            P, X_esm, X_fp = pd.read_pickle(cf)
        else:
            if esm is None:
                esm = pp.ESM(device)
            try:
                P, X_esm, X_fp = featurize(root / r.pdb_id / "stripped.pdb", NEW / "work13" / r.pdb_id, fpocket, esm, pp)
            except Exception as e:
                say(f"  {r.pdb_id}: failed - {type(e).__name__}: {e}")
                continue
            pd.to_pickle((P, X_esm, X_fp), cf)
        site = {(c, int(n)) for c, n, *_ in json.loads(r.residues_assembly)}
        cm = json.load(open(root / r.pdb_id / "chainmap.json"))
        out = {"pdb_id": r.pdb_id, "family_status": res10e.loc[r.pdb_id, "family_status"],
               "group": res10e.loc[r.pdb_id, "nearest_training_family"] if isinstance(
                   res10e.loc[r.pdb_id, "nearest_training_family"], str) else r.pdb_id,
               "n_pockets": len(P), "rank_alloplex_released": res10e.loc[r.pdb_id, "rank_alloplex"]}
        for feat in FEATS:
            if (feat, key) not in members:
                members[(feat, key)] = load_feat_members(feat, key, device)
            P[f"s_{feat}"] = score_feat(members[(feat, key)], feat, X_esm, X_fp, device)
            out[f"rank_{feat}"] = s10e.ranks_with_copies(P, site, cm, f"s_{feat}")[0]
        out["rank_druggability"] = s10e.ranks_with_copies(P, site, cm, "druggability_score")[0]
        rows.append(out)
    d = pd.DataFrame(rows)
    d.to_csv(DATA / "s13b_newset_ablation.csv", index=False)
    g = d.group.astype(str).to_numpy()
    say(f"\n2024+ complexes scored: {len(d)} (new families {int((d.family_status == 'new family').sum())})")
    say("Top-1 / Top-3 [95% family-bootstrap CI for Top-1]:")
    for name, col in (("B  ESM-2 + fpocket (retrained)", "rank_B"), ("F  fpocket only", "rank_F"),
                      ("E  ESM-2 only", "rank_E"), ("AlloPlex released (step 10e)", "rank_alloplex_released"),
                      ("fpocket druggability", "rank_druggability")):
        t1, t3 = (d[col] <= 1).astype(float), (d[col] <= 3).astype(float)
        lo, hi = s6.boot(g, t1.to_numpy())
        say(f"  {name:<32} Top-1 {t1.mean():.3f} [{lo:.3f}, {hi:.3f}]   Top-3 {t3.mean():.3f}")
    say("Paired differences in Top-1 (B minus other):")
    for oth in ("F", "E"):
        x = (d.rank_B <= 1).astype(float) - (d[f"rank_{oth}"] <= 1).astype(float)
        lo, hi = s6.boot(g, x.to_numpy())
        x3 = (d.rank_B <= 3).astype(float) - (d[f"rank_{oth}"] <= 3).astype(float)
        say(f"  B - {oth}: Top-1 {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   Top-3 {x3.mean():+.3f}")
    for st, gg in d.groupby("family_status"):
        say(f"  {st:<32} n={len(gg):>3}: Top-1 B {(gg.rank_B <= 1).mean():.3f}, F {(gg.rank_F <= 1).mean():.3f}, "
            f"E {(gg.rank_E <= 1).mean():.3f}")
    say("(check: 'B retrained' should be close to 'AlloPlex released'; if not, report both)")
    (DATA / "report_step13b.txt").write_text("\n".join(LOG) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "score"])
    ap.add_argument("--feats", default="B,F,E")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    train(a) if a.stage == "train" else score(a)


if __name__ == "__main__":
    main()
