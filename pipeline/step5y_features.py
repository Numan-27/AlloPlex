#!/usr/bin/env python
"""
STEP 5Y: POCKET FEATURES AND CROSS-VALIDATION FOLDS (benchmark, part 2)
======================================================================
Builds, for each pocket set that exists,
    stripped : holo structures with ligands removed (training + secondary evaluation)
    partner  : modulator-free partner structures (primary evaluation where available)
    holo     : holo structures WITH ligands (only for the inflation table)
the files processed/dataset_v2/bench/features_<set>.npz and meta_<set>.csv:

  y (benchmark label, all sites incl. monomers and PPI-annotated sites):
     1 = best pocket of a site (label_primary)
     0 = touches no site and is not a symmetric copy of one
    -1 = excluded (extra pockets overlapping a site, symmetric copies)
  X_esm : ESM-2 650M mean over the pocket's residues (1280, float16)
  X_fp  : 11 fpocket descriptors (corrected names, as in the plan)
  X_if  : interface features: frac_iface_5, log1p(mean delta-SASA), log1p(max delta-SASA),
          log1p(distance to nearest interface residue, capped at 60 A; 60 for monomers)
  meta  : pdb_id, pocket, family, sites (site ids the pocket is the best pocket of),
          complex kind, n_chains, n_residues_found_in_esm

Feature sets used by the models (columns are combined at training time):
    B     = X_esm + X_fp
    B+I   = B + all 4 X_if columns
    B+I-d = B + X_if without the distance column (declared in the plan's deviations log)
    I     = X_if + n_pocket_residues

Folds (fixed once, used by every model): processed/dataset_v2/bench/folds.csv
  5 folds x 3 repeats; whole FAMILIES are assigned to folds (a family is never in two
  folds), balanced for positives and for the families' main complex type. Seeds fixed.

RUN (from ~/allosite):  python scripts/step5y_features.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
OUT = D / "bench"
FP = ["score", "druggability_score", "volume", "polarity_score", "charge_score", "apolar_sasa", "polar_sasa",
      "flexibility", "number_of_alpha_spheres", "mean_local_hydrophobic_density", "n_pocket_residues"]
IF = ["frac_iface_5", "log_mean_delta_sasa", "log_max_delta_sasa", "log_dist_iface_5"]
DIST_CAP = 60.0
N_FOLDS, REPEATS, SEED = 5, 3, 20261013
TF = {True: True, False: False, "True": True, "False": False}


def load_esm_index():
    idx = pd.read_csv(D / "esm_index.csv")
    table = {}
    for _, r in idx.iterrows():
        rows = {(int(n), str(ic) if isinstance(ic, str) else ""): i for i, (n, ic) in enumerate(json.loads(r.residues))}
        table[(r["set"], r.pdb_id, r.chain)] = (r.seq_hash, rows)
    return table


def build(set_name, pockets_csv, esm_set, esm_table, kinds):
    P = pd.read_csv(pockets_csv, low_memory=False)
    for c in ("label_primary", "label_any", "symmetric_copy"):
        P[c] = P[c].map(TF).fillna(False).astype(bool)
    y = np.where(P.label_primary, 1, np.where(~P.label_any & ~P.symmetric_copy, 0, -1)).astype(np.int8)
    missing = [c for c in FP if c not in P.columns]
    if missing:
        raise SystemExit(f"{pockets_csv.name}: fpocket columns not found: {missing}")
    X_fp = P[FP].to_numpy(np.float32)
    dist = P["dist_iface_5"].astype(float).fillna(DIST_CAP).clip(upper=DIST_CAP).to_numpy()
    X_if = np.c_[P.frac_iface_5.fillna(0).to_numpy(), np.log1p(P.mean_delta_sasa.fillna(0).clip(lower=0)),
                 np.log1p(P.max_delta_sasa.fillna(0).clip(lower=0)), np.log1p(dist)].astype(np.float32)
    X_esm = np.zeros((len(P), 1280), np.float16)
    found = np.zeros(len(P), np.int32)
    cache = {}
    for pdb, g in P.groupby("pdb_id", sort=False):
        for i, res in zip(g.index, g.residues):
            vecs = []
            for c, n, *ic in json.loads(res):
                key = (esm_set, pdb, c)
                if key not in esm_table:
                    continue
                h, rows = esm_table[key]
                r = rows.get((int(n), ic[0] if ic else ""))
                if r is None:
                    continue
                if h not in cache:
                    if len(cache) > 400:
                        cache.clear()
                    cache[h] = np.load(D / "esm" / f"{h}.npy")
                vecs.append(cache[h][r].astype(np.float32))
            if vecs:
                X_esm[i] = np.mean(vecs, 0).astype(np.float16)
                found[i] = len(vecs)
    meta = pd.DataFrame({"pdb_id": P.pdb_id, "pocket": P.pocket, "family": P.family.astype(str),
                         "sites": P.sites_primary.fillna(""), "n_chains": P.n_chains,
                         "kind": P.pdb_id.map(kinds).fillna(""), "y": y, "n_res_esm": found,
                         "n_pocket_residues": P.n_pocket_residues})
    np.savez_compressed(OUT / f"features_{set_name}.npz", X_esm=X_esm, X_fp=X_fp, X_if=X_if, y=y)
    meta.to_csv(OUT / f"meta_{set_name}.csv", index=False)
    no_esm = int((found == 0).sum())
    print(f"{set_name:<9} pockets {len(P):>7}  positives {int((y == 1).sum()):>5}  negatives {int((y == 0).sum()):>7}  "
          f"excluded {int((y == -1).sum()):>6}  structures {P.pdb_id.nunique():>5}  families {meta.family.nunique():>4}  "
          f"pockets without ESM residues {no_esm}")
    return meta


def make_folds(meta):
    fam = meta.groupby("family").agg(pos=("y", lambda s: int((s == 1).sum())),
                                     kind=("kind", lambda s: Counter(s).most_common(1)[0][0]))
    rows = []
    for rep in range(REPEATS):
        rng = np.random.default_rng(SEED + rep)
        load = np.zeros(N_FOLDS)
        kind_load = defaultdict(lambda: np.zeros(N_FOLDS))
        order = fam.sample(frac=1, random_state=int(rng.integers(1e9))).sort_values("pos", ascending=False, kind="stable")
        for f, r in order.iterrows():
            score = load + 0.5 * kind_load[r.kind] + rng.random(N_FOLDS) * 1e-3
            k = int(np.argmin(score))
            load[k] += max(r.pos, 0.2)
            kind_load[r.kind][k] += 1
            rows.append({"family": f, "repeat": rep, "fold": k})
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "folds.csv", index=False)
    m = meta.merge(folds[folds.repeat == 0], on="family")
    print("\nfolds (repeat 0): positives / families per fold:",
          [(int((g.y == 1).sum()), g.family.nunique()) for _, g in m.groupby("fold")])
    return folds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    esm_table = load_esm_index()
    master = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", low_memory=False,
                         usecols=["pdb_id", "complex_kind"])
    kinds = master.drop_duplicates("pdb_id").set_index("pdb_id").complex_kind
    metas = {}
    for set_name, csv_name, esm_set in (("stripped", "pockets_stripped.csv", "holo"),
                                        ("partner", "pockets_partner.csv", "partner"),
                                        ("holo", "pockets_holo.csv", "holo")):
        path = D / csv_name
        if not path.exists():
            print(f"{set_name:<9} skipped ({csv_name} not found)")
            continue
        metas[set_name] = build(set_name, path, esm_set, esm_table, kinds if set_name != "partner" else
                                pd.Series(dtype=str))
    if "partner" in metas:  # partner structures take the complex type of their family's holo structures
        pm = pd.read_csv(OUT / "meta_partner.csv")
        st = metas["stripped"].drop_duplicates("family").set_index("family").kind
        pm["kind"] = pm.family.map(st).fillna("")
        pm.to_csv(OUT / "meta_partner.csv", index=False)
    make_folds(metas["stripped"])
    if "partner" in metas:
        f0 = pd.read_csv(OUT / "folds.csv")
        miss = set(pd.read_csv(OUT / "meta_partner.csv").family) - set(f0.family)
        print(f"partner families without a fold (not in the training set): {len(miss)}")
    print(f"\nSaved features_*.npz, meta_*.csv and folds.csv in {OUT}")


if __name__ == "__main__":
    main()
