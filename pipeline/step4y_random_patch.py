#!/usr/bin/env python
"""
STEP 4Y: RANDOM-PATCH CONTROL (pre-registered control)
======================================================
Question: does the REAL subunit interface behave differently from a random patch
of protein surface of the same size?

For every structure of the primary analysis set that holds both a positive and a
negative pocket:
  - real interface = residues with delta-SASA > 5 A^2 (as in the primary test)
  - 100 random patches: on each chain, a seed residue is drawn at random from the
    chain's surface (SASA of the chain alone > 5 A^2) and the patch is that seed
    plus its nearest surface neighbours on the same chain (residue centroids),
    with exactly as many residues as the real interface has on that chain
  - two scores are computed for the real interface and for every random patch:
      A  fraction of pocket residues in the interface / patch  (primary score)
      B  pockets touching no interface / patch residue, scored by minus the
         distance (residue centroids) to the nearest interface / patch residue
         (the S3 "beyond contact" question)
  - matched AUC with the SAME cells as the primary test (structure x size tercile);
    for B the size terciles of the primary set are reused, so real and random are
    compared on equal terms (B here can differ slightly from step 4x S3a, which
    used atom distances and its own terciles)

Report per score: real AUC, random-patch mean and 2.5-97.5 % range, the share of
random replicates at least as extreme as the real interface, and the difference
real - random with a 95 % FAMILY-bootstrap interval.

RUN (from ~/allosite):  python scripts/step4y_random_patch.py --workers 8
Outputs: processed/dataset_v2/results_step4y.csv, report_step4y.txt
"""
import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
STRUCT = BASE / "processed" / "structures"
SEED = 20261011
CUT = 5.0
SURF = 5.0
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def centroids(pdb_path):
    acc = {}
    for line in open(pdb_path):
        if line.startswith("ATOM  "):
            k = (line[21], int(line[22:26]))
            x = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            s = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
            s[0] += x[0]; s[1] += x[1]; s[2] += x[2]; s[3] += 1
    return {k: (v[0] / v[3], v[1] / v[3], v[2] / v[3]) for k, v in acc.items()}


def u_pairs(y, score, cells):
    """Sum of U and of positive-negative pairs over informative cells."""
    U = P = 0.0
    for c in np.unique(cells):
        m = cells == c
        yy = y[m]
        n1 = yy.sum()
        n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(score[m])
        U += r[yy == 1].sum() - n1 * (n1 + 1) / 2
        P += n1 * n0
    return U, P


def process(args):
    pdb, pockets, n_rep = args
    sdir = STRUCT / pdb
    res = pd.read_csv(sdir / "residues.csv", keep_default_na=False)
    cen = centroids(sdir / "stripped.pdb")
    res["key"] = list(zip(res.chain, res.resnum))
    res = res[res.key.isin(cen)].reset_index(drop=True)
    keys = list(res.key)
    idx = {k: i for i, k in enumerate(keys)}
    xyz = np.array([cen[k] for k in keys])
    chain = res.chain.to_numpy()
    real = (res.delta_sasa.astype(float) > CUT).to_numpy()
    surface = (res.sasa_alone.astype(float) > SURF).to_numpy()

    y = np.array([p["y"] for p in pockets])
    cells = np.array([p["cell"] for p in pockets])
    pidx = [np.array([idx[(c, int(n))] for c, n, *_ in json.loads(p["residues"]) if (c, int(n)) in idx], dtype=int)
            for p in pockets]

    def scores(mask):
        frac = np.array([mask[i].mean() if len(i) else 0.0 for i in pidx])
        dist = np.full(len(pidx), np.nan)
        if mask.any():
            tree = cKDTree(xyz[mask])
            for j, i in enumerate(pidx):
                if len(i) and not mask[i].any():
                    dist[j] = tree.query(xyz[i])[0].min()
        return frac, dist

    def both(mask):
        frac, dist = scores(mask)
        ua, pa = u_pairs(y, frac, cells)
        ok = ~np.isnan(dist)
        ub, pb = u_pairs(y[ok], -dist[ok], cells[ok]) if ok.any() else (0.0, 0.0)
        return ua, pa, ub, pb

    out = {"pdb_id": pdb, "family": pockets[0]["family"], "real": both(real), "rand": []}
    rng = np.random.default_rng(SEED + zlib.crc32(pdb.encode()))
    per_chain = {c: int(real[chain == c].sum()) for c in np.unique(chain)}
    surf_idx = {c: np.where((chain == c) & surface)[0] for c in per_chain}
    for _ in range(n_rep):
        mask = np.zeros(len(keys), bool)
        for c, k in per_chain.items():
            cand = surf_idx[c]
            if k == 0 or len(cand) == 0:
                continue
            seed = cand[rng.integers(len(cand))]
            d = np.linalg.norm(xyz[cand] - xyz[seed], axis=1)
            mask[cand[np.argsort(d)[:k]]] = True
        out["rand"].append(both(mask))
    return out


def boot_diff(fam_real, fam_rand, B=2000):
    """fam_real, fam_rand: arrays (n_families, 2) of [U, pairs]; difference of pooled AUCs."""
    rng = np.random.default_rng(SEED)
    k = len(fam_real)
    idx = rng.integers(0, k, (B, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        a = fam_real[idx, 0].sum(1) / fam_real[idx, 1].sum(1)
        b = fam_rand[idx, 0].sum(1) / fam_rand[idx, 1].sum(1)
    d = a - b
    d = d[np.isfinite(d)]
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    P = pd.read_csv(D / "pockets_stripped.csv", usecols=["pdb_id", "pocket", "n_pocket_residues", "residues",
                                                          "role_primary", "family"], low_memory=False)
    P = P[P.role_primary.isin(["positive", "negative"])].copy()
    P["y"] = (P.role_primary == "positive").astype(int)
    P["tercile"] = pd.qcut(P.n_pocket_residues.rank(method="first"), 3, labels=False)
    P["cell"] = P.pdb_id.astype(str) + "|" + P.tercile.astype(str)
    informative = P.groupby("pdb_id").y.agg(["min", "max"])
    ids = informative[(informative["min"] == 0) & (informative["max"] == 1)].index
    P = P[P.pdb_id.isin(ids)]
    jobs = [(pdb, g[["y", "cell", "residues", "family"]].to_dict("records"), args.reps)
            for pdb, g in P.groupby("pdb_id")]
    if args.limit:
        jobs = jobs[:args.limit]
    say(f"{len(jobs)} structures with both positive and negative pockets; {args.reps} random patches each")
    outs = []
    with Pool(args.workers) as pool:
        for i, o in enumerate(pool.imap_unordered(process, jobs), 1):
            outs.append(o)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    rows = []
    say("")
    say("=" * 96)
    say("RANDOM-PATCH CONTROL: real interface vs random surface patches of the same size per chain")
    say("=" * 96)
    for name, ui, pi in (("A  fraction of pocket in interface/patch (primary score)", 0, 1),
                         ("B  no-contact pockets, -distance to interface/patch (S3 question)", 2, 3)):
        real_u = sum(o["real"][ui] for o in outs)
        real_p = sum(o["real"][pi] for o in outs)
        rep = np.array([[o["rand"][r][ui] for o in outs] for r in range(args.reps)])
        rep_p = np.array([[o["rand"][r][pi] for o in outs] for r in range(args.reps)])
        with np.errstate(invalid="ignore", divide="ignore"):
            rep_auc = rep.sum(1) / rep_p.sum(1)
        rep_auc = rep_auc[np.isfinite(rep_auc)]
        real_auc = real_u / real_p if real_p else np.nan
        frac_ge = (np.sum(np.abs(rep_auc - 0.5) >= abs(real_auc - 0.5)) + 1) / (len(rep_auc) + 1)
        fams = sorted({o["family"] for o in outs})
        fr = np.zeros((len(fams), 2))
        fn = np.zeros((len(fams), 2))
        fi = {f: i for i, f in enumerate(fams)}
        for j, o in enumerate(outs):
            i = fi[o["family"]]
            fr[i] += [o["real"][ui], o["real"][pi]]
            fn[i] += [rep[:, j].mean(), rep_p[:, j].mean()]
        rand_pooled = fn[:, 0].sum() / fn[:, 1].sum()
        lo, hi = boot_diff(fr, fn)
        say(f"{name}")
        say(f"   real interface AUC {real_auc:.3f} | random patches mean {rep_auc.mean():.3f} "
            f"[{np.percentile(rep_auc, 2.5):.3f}, {np.percentile(rep_auc, 97.5):.3f}] (range over {len(rep_auc)} replicates)")
        say(f"   share of random replicates at least as far from 0.5 as the real interface: {frac_ge:.3f}")
        say(f"   real - random: {real_auc - rand_pooled:+.3f}, 95 % family interval [{lo:+.3f}, {hi:+.3f}]")
        rows.append({"score": name, "real_auc": real_auc, "random_mean": rep_auc.mean(),
                     "random_lo": np.percentile(rep_auc, 2.5), "random_hi": np.percentile(rep_auc, 97.5),
                     "share_as_extreme": frac_ge, "diff": real_auc - rand_pooled, "diff_lo": lo, "diff_hi": hi})
    say("")
    say("How to read: if the real interface sits inside the random-patch range and the family interval")
    say("of the difference includes 0, the interface is not special: any surface patch of that size")
    say("relates to allosteric pockets in the same way.")
    pd.DataFrame(rows).to_csv(D / "results_step4y.csv", index=False)
    (D / "report_step4y.txt").write_text("\n".join(LOG) + "\n")
    say(f"Saved {D / 'results_step4y.csv'} and report_step4y.txt")


if __name__ == "__main__":
    main()
