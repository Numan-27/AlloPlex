#!/usr/bin/env python
"""
STEP 3X: LABEL POCKETS AND BUILD THE NEW DATASET (follows the analysis plan)
===========================================================================
Inputs (from steps 1x, 2x, 2y):
  processed/master/allosteric_master.csv          site annotations (position, family ...)
  processed/structures/sites_mapped.csv           site residues on the assembly, modulator status
  processed/structures/<PDB>/stripped.pdb, residues.csv, pockets_<which>.csv, chainmap.json, modulators.json

For every pocket:
  - interface features at 1, 5, 10, 20 A^2: number and fraction of lining residues
    at an interface, mean/max delta-SASA, distance to the nearest interface residue
    (0 if the pocket touches one), share of interface residues facing a DIFFERENT
    protein (hetero) vs a copy of the same protein (homo)
  - labels (per the plan):
      label_primary   the ONE best-matching pocket of a site (highest residue Jaccard, >= 0.1)
      label_any       shares >= 1 residue with a site (secondary rule)
      label_nearest   pocket whose centre is nearest the modulator (secondary rule)
      symmetric_copy  matches a site only on an identical copy of the protein (same entity,
                      same residue numbers on another chain) -> set aside, neither + nor -
      touches_ppi_site  overlaps a site annotated "PPI interface" (or its copies)
  - role_primary (primary analysis, key set): positive / negative / excluded
      positive = label_primary of a NON-PPI site in a structure with >= 2 protein chains
      negative = overlaps no site, no copy of a site, in such a structure
      excluded = everything else (PPI-site pockets, copies, extra overlapping pockets ...)

For every site: detected?, best pocket and Jaccard, number of overlapping pockets,
share of site residues at the interface, distance modulator -> interface.

RUN (from ~/allosite):
    python scripts/step3x_label_pockets.py                      # stripped pockets (primary)
    python scripts/step3x_label_pockets.py --which holo         # holo pockets (leakage checks)
Outputs: processed/dataset_v2/pockets_<which>.csv, sites_<which>.csv, report_<which>.txt
"""
import argparse
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
STRUCT = BASE / "processed" / "structures"
OUT = BASE / "processed" / "dataset_v2"
CUTOFFS = [1, 5, 10, 20]
PRIMARY_CUT = 5
J_MIN = 0.1
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def read_atoms(path):
    rows = []
    for line in open(path):
        if line.startswith("ATOM  "):
            rows.append((line[21], int(line[22:26]), line[26].strip(),
                         float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return pd.DataFrame(rows, columns=["chain", "resnum", "icode", "x", "y", "z"])


def jacc(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def process(args):
    pdb, which, sites = args
    sdir = STRUCT / pdb
    pk_path = sdir / f"pockets_{which}.csv"
    if not pk_path.exists() or not (sdir / "residues.csv").exists():
        return [], [{"site_id": s["site_id"], "pdb_id": pdb, "status": "no_pockets"} for s in sites]
    try:
        pockets = pd.read_csv(pk_path)
    except pd.errors.EmptyDataError:  # fpocket found no pocket
        pockets = pd.DataFrame()
    if pockets.empty or "residues" not in pockets.columns:
        return [], [{"site_id": s["site_id"], "pdb_id": pdb, "status": "no_pockets"} for s in sites]
    res = pd.read_csv(sdir / "residues.csv", keep_default_na=False)
    cmap = json.load(open(sdir / "chainmap.json"))
    mods = json.load(open(sdir / "modulators.json")) if (sdir / "modulators.json").exists() else {}
    ent = {c: str(v["entity"]) for c, v in cmap.items()}
    n_chains = len(cmap)
    atoms = read_atoms(sdir / "stripped.pdb")
    atoms["key"] = list(zip(atoms.chain, atoms.resnum))
    xyz = atoms[["x", "y", "z"]].to_numpy()

    res["key"] = list(zip(res.chain, res.resnum))
    delta = dict(zip(res.key, res.delta_sasa.astype(float)))
    partners = dict(zip(res.key, res.partner_chains.astype(str)))
    iface = {c: {k for k, d in delta.items() if d > c} for c in CUTOFFS}
    trees = {}
    for c in CUTOFFS:
        m = atoms.key.isin(iface[c]).to_numpy()
        trees[c] = cKDTree(xyz[m]) if m.any() else None

    def hetero(k):
        p = [x for x in partners.get(k, "").split(",") if x]
        return any(ent.get(x) != ent.get(k[0]) for x in p)

    # pockets: residue sets (chain, resnum), collapsed sets (entity, resnum)
    P = []
    for _, r in pockets.iterrows():
        rs = {(c, int(n)) for c, n, *_ in json.loads(r["residues"])}
        P.append({"set": rs, "col": {(ent.get(c, c), n) for c, n in rs},
                  "center": np.array([r.get("center_x", np.nan), r.get("center_y", np.nan), r.get("center_z", np.nan)],
                                     dtype=float)})

    # sites
    lab = [defaultdict(list) for _ in P]
    site_rows = []
    for s in sites:
        S = {(c, int(n)) for c, n, *_ in json.loads(s["residues_assembly"])} if isinstance(s["residues_assembly"], str) else set()
        row = {"site_id": s["site_id"], "pdb_id": pdb, "position_class": s["position_class"], "status": "ok",
               "n_site_residues_mapped": len(S), "n_pockets": len(P)}
        if not S or not P:
            row["status"] = "no_site_residues" if not S else "no_pockets"
            site_rows.append(row)
            continue
        Sc = {(ent.get(c, c), n) for c, n in S}
        js = [jacc(p["set"], S) for p in P]
        ov = [len(p["set"] & S) for p in P]
        best = max(range(len(P)), key=lambda i: (js[i], ov[i]))
        detected = js[best] >= J_MIN
        is_ppi = s["position_class"] == "PPI interface"
        for i, p in enumerate(P):
            if detected and i == best:
                lab[i]["primary"].append(s["site_id"])
            if ov[i] > 0:
                lab[i]["any"].append(s["site_id"])
            elif jacc(p["col"], Sc) >= J_MIN:
                lab[i]["copy"].append(s["site_id"])
            if is_ppi and (ov[i] > 0 or jacc(p["col"], Sc) >= J_MIN):
                lab[i]["ppi"].append(s["site_id"])
            if not is_ppi and detected and i == best:
                lab[i]["primary_nonppi"].append(s["site_id"])
        # nearest pocket to the modulator
        mdist = None
        if s["site_id"] in mods:
            mx = np.array(mods[s["site_id"]]["xyz"])
            cen = mx.mean(0)
            d = [np.linalg.norm(p["center"] - cen) if np.isfinite(p["center"]).all() else np.inf for p in P]
            near = int(np.argmin(d))
            if np.isfinite(d[near]):
                lab[near]["nearest"].append(s["site_id"])
                row["nearest_pocket"] = int(pockets.pocket.iloc[near])
                row["nearest_pocket_dist"] = round(float(d[near]), 2)
            if trees[PRIMARY_CUT] is not None:
                mdist = float(trees[PRIMARY_CUT].query(mx)[0].min())
        site_iface = sum(1 for k in S if k in iface[PRIMARY_CUT]) / len(S)
        row.update(detected=detected, best_pocket=int(pockets.pocket.iloc[best]), best_jaccard=round(js[best], 3),
                   n_overlapping_pockets=int(sum(o > 0 for o in ov)), site_frac_interface_5=round(site_iface, 3),
                   modulator_dist_to_interface_5=None if mdist is None else round(mdist, 2),
                   modulator_contacts_interface=None if mdist is None else mdist <= 4.5)
        site_rows.append(row)

    # pocket features
    rows = []
    for i, (p, (_, r)) in enumerate(zip(P, pockets.iterrows())):
        keys = p["set"]
        d = [delta.get(k, 0.0) for k in keys]
        f = {"pdb_id": pdb, "pocket": int(r["pocket"]), "n_chains": n_chains}
        for c in r.index:
            if c not in ("pocket", "residues"):
                f[c] = r[c]
        f["residues"] = r["residues"]
        mask = atoms.key.isin(keys).to_numpy()
        for c in CUTOFFS:
            n_if = sum(1 for k in keys if k in iface[c])
            f[f"n_iface_{c}"] = n_if
            f[f"frac_iface_{c}"] = round(n_if / len(keys), 4) if keys else 0.0
            if n_if:
                f[f"dist_iface_{c}"] = 0.0
            elif trees[c] is not None and mask.any():
                f[f"dist_iface_{c}"] = round(float(trees[c].query(xyz[mask])[0].min()), 2)
            else:
                f[f"dist_iface_{c}"] = np.nan
        f["mean_delta_sasa"] = round(float(np.mean(d)), 3) if d else 0.0
        f["max_delta_sasa"] = round(float(np.max(d)), 3) if d else 0.0
        het = [k for k in keys if k in iface[PRIMARY_CUT] and hetero(k)]
        f["n_iface_hetero_5"] = len(het)
        f["n_iface_homo_5"] = f["n_iface_5"] - len(het)
        L = lab[i]
        f["label_primary"] = bool(L["primary"])
        f["label_any"] = bool(L["any"])
        f["label_nearest"] = bool(L["nearest"])
        f["symmetric_copy"] = bool(L["copy"]) and not L["any"]
        f["touches_ppi_site"] = bool(L["ppi"])
        f["sites_primary"] = ",".join(L["primary"])
        f["sites_any"] = ",".join(L["any"])
        if n_chains >= 2 and L["primary_nonppi"] and not L["ppi"]:
            role = "positive"
        elif n_chains >= 2 and not L["any"] and not L["copy"] and not L["ppi"]:
            role = "negative"
        else:
            role = "excluded"
        f["role_primary"] = role
        rows.append(f)
    return rows, site_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="stripped", choices=["stripped", "holo"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", dtype={"modulator_alias": str})
    mapped = pd.read_csv(STRUCT / "sites_mapped.csv")
    sites = mapped[mapped.status == "ok"].merge(
        master[["site_id", "position_class", "family", "source", "complex_kind", "target_id", "gene"]],
        on="site_id", how="left")
    jobs = [(pdb, args.which, g.to_dict("records")) for pdb, g in sites.groupby("pdb_id")]
    say(f"{len(sites)} mapped sites in {len(jobs)} structures; pockets: {args.which}")
    prow, srow = [], []
    with Pool(args.workers) as pool:
        for i, (p, s) in enumerate(pool.imap_unordered(process, jobs), 1):
            prow += p
            srow += s
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    P = pd.DataFrame(prow)
    S = pd.DataFrame(srow)
    fam = master.drop_duplicates("pdb_id").set_index("pdb_id")
    P["family"] = P.pdb_id.map(fam["family"])
    S = S.merge(master[["site_id", "family", "source", "complex_kind", "target_id"]], on="site_id", how="left")
    P.to_csv(OUT / f"pockets_{args.which}.csv", index=False)
    S.to_csv(OUT / f"sites_{args.which}.csv", index=False)

    say("")
    say("=" * 72)
    say(f"REPORT ({args.which} pockets) - data quality only; the tests are the next step")
    say("=" * 72)
    ok = S[S.status == "ok"]
    say(f"pockets: {len(P)} in {P.pdb_id.nunique()} structures (median {P.groupby('pdb_id').size().median():.0f} per structure)")
    say(f"sites: {len(S)}; with residues and pockets: {len(ok)}; status: {dict(Counter(S.status))}")
    if len(ok):
        say(f"detected (best pocket Jaccard >= {J_MIN}): {ok.detected.mean():.1%}; median best Jaccard "
            f"{ok.best_jaccard.median():.2f}; median overlapping pockets per site {ok.n_overlapping_pockets.median():.0f}")
        for pc, g in ok.groupby("position_class"):
            say(f"   {pc:<36} n={len(g):>5}  detected {g.detected.mean():.1%}")
        m = ok.modulator_contacts_interface.dropna()
        say(f"sites with a located modulator and an interface: {len(m)}")
    say(f"pockets set aside as symmetric copies of a site: {int(P.symmetric_copy.sum())}; "
        f"touching a PPI-annotated site: {int(P.touches_ppi_site.sum())}")
    k = P[P.role_primary != "excluded"]
    say(f"PRIMARY ANALYSIS SET: {int((P.role_primary == 'positive').sum())} positive and "
        f"{int((P.role_primary == 'negative').sum())} negative pockets in {k.pdb_id.nunique()} structures, "
        f"{k[k.role_primary == 'positive'].family.nunique()} families with a positive")
    say(f"saved {OUT / f'pockets_{args.which}.csv'} and sites_{args.which}.csv")
    (OUT / f"report_{args.which}.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
