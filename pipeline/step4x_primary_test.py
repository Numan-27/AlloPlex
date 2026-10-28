#!/usr/bin/env python
"""
STEP 4X: PRIMARY TEST, SECONDARY ANALYSES AND SENSITIVITY CHECKS (as pre-registered)
===================================================================================
Reads processed/dataset_v2/pockets_stripped.csv and sites_stripped.csv (step 3x).

Statistic everywhere: MATCHED AUC = probability that a positive pocket scores higher
than a negative pocket of the same cell (ties = 1/2); cell = one structure x one
pocket-size tercile (terciles computed once within each analysis set); cells pooled
by number of positive-negative pairs. No effect = 0.5.
  95 % interval: 2,000 bootstrap resamples of FAMILIES.
  p-value: label permutations within cells (10,000 for the primary test, 2,000 elsewhere).

PRIMARY (plan): key set = structures with >= 2 protein chains; positives = best pocket
of a non-PPI site; negatives = pockets touching no site and no copy of a site;
score = fraction of pocket residues at the interface (> 5 A^2).
  Supported only if the family interval lies above 0.5.
  Companion: Mantel-Haenszel odds ratio for "pocket has >= 1 interface residue".

SECONDARY S1-S6 (Holm-adjusted p-values), SENSITIVITY checks (direction must hold).
Not run here (need extra data, separate scripts): random-patch control,
modulator-free partner pockets, asymmetric-unit structures.

RUN (from ~/allosite):  python scripts/step4x_primary_test.py
Outputs: processed/dataset_v2/results_step4x.csv, report_step4x.txt
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
LOG, RESULTS = [], []
SEED = 20261011


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------------------- core statistics
def add_cells(df, size_col="n_pocket_residues"):
    df = df.copy()
    df["tercile"] = pd.qcut(df[size_col].rank(method="first"), 3, labels=False)
    df["cell"] = df.pdb_id.astype(str) + "|" + df.tercile.astype(str)
    return df


def cell_table(df, score):
    """Per informative cell: family, U statistic, pairs, ranks array, n_pos."""
    out = []
    for c, g in df.groupby("cell", sort=False):
        y = g.y.to_numpy()
        n1 = int(y.sum())
        n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(g[score].to_numpy())
        u = r[y == 1].sum() - n1 * (n1 + 1) / 2
        out.append({"cell": c, "family": g.family.iloc[0], "U": u, "pairs": n1 * n0, "ranks": r, "n1": n1})
    return out


def boot_ratio(fam_num, fam_den, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    num, den = np.array(fam_num), np.array(fam_den)
    k = len(num)
    idx = rng.integers(0, k, (B, k))
    vals = num[idx].sum(1) / den[idx].sum(1)
    return np.percentile(vals, [2.5, 97.5])


def perm_p(cells, obs, B, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    tot_pairs = sum(c["pairs"] for c in cells)
    perm_u = np.zeros(B)
    for c in cells:
        n, n1 = len(c["ranks"]), c["n1"]
        for s in range(0, B, 2000):
            b = min(2000, B - s)
            pick = np.argpartition(rng.random((b, n)), n1 - 1, axis=1)[:, :n1] if n1 < n else np.tile(np.arange(n), (b, 1))
            perm_u[s:s + b] += c["ranks"][pick].sum(1) - n1 * (n1 + 1) / 2
    auc = perm_u / tot_pairs
    return (np.sum(np.abs(auc - 0.5) >= abs(obs - 0.5)) + 1) / (B + 1)


def matched_auc(df, score, label, n_perm=2000, equal_family=False, record=True, kind="secondary"):
    """df needs columns: y, score, cell, family."""
    cells = cell_table(df, score)
    if not cells:
        say(f"  {label:<62} no informative cells")
        return None
    fam = {}
    for c in cells:
        f = fam.setdefault(c["family"], [0.0, 0.0])
        f[0] += c["U"]
        f[1] += c["pairs"]
    if equal_family:
        per = np.array([u / p for u, p in fam.values()])
        auc = per.mean()
        rng = np.random.default_rng(SEED)
        lo, hi = np.percentile([per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)], [2.5, 97.5])
        p = None
    else:
        auc = sum(c["U"] for c in cells) / sum(c["pairs"] for c in cells)
        lo, hi = boot_ratio([v[0] for v in fam.values()], [v[1] for v in fam.values()])
        p = perm_p(cells, auc, n_perm) if n_perm else None
    npos = int(df.y.sum())
    nfam = df[df.y == 1].family.nunique()
    ptxt = f"p={p:.4f}" if p is not None else ""
    say(f"  {label:<62} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  {ptxt:<9} pos {npos}, neg {int((df.y == 0).sum())}, "
        f"families {nfam}, cells {len(cells)}")
    if record:
        RESULTS.append({"kind": kind, "analysis": label, "auc": auc, "lo": lo, "hi": hi, "p": p,
                        "n_pos": npos, "n_neg": int((df.y == 0).sum()), "n_families_pos": nfam,
                        "n_cells": len(cells)})
    return auc, lo, hi, p


def mantel_haenszel(df, exposure):
    fam = {}
    for _, g in df.groupby("cell", sort=False):
        if g.y.nunique() < 2:
            continue
        e = g[exposure].to_numpy() > 0
        y = g.y.to_numpy() == 1
        a, b = np.sum(e & y), np.sum(~e & y)
        c, d = np.sum(e & ~y), np.sum(~e & ~y)
        n = len(g)
        f = fam.setdefault(g.family.iloc[0], [0.0, 0.0])
        f[0] += a * d / n
        f[1] += b * c / n
    num, den = [v[0] for v in fam.values()], [v[1] for v in fam.values()]
    if not den or sum(den) == 0:
        return None, None, None
    orr = sum(num) / sum(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo, hi = boot_ratio(num, den)
    return orr, lo, hi


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def two_prop_p(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 2 * (1 - norm.cdf(abs(k1 / n1 - k2 / n2) / se)) if se > 0 else 1.0


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i])
    m, adj, running = len(order), [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


# ----------------------------------------------------------------------------- analysis sets
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-primary", type=int, default=10000)
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()

    P = pd.read_csv(D / "pockets_stripped.csv", low_memory=False)
    S = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    S = S[S.status == "ok"].copy()
    tf = {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
    S["modulator_contacts_interface"] = S.modulator_contacts_interface.map(tf)
    for col in ("label_primary", "label_any", "label_nearest", "symmetric_copy", "touches_ppi_site"):
        P[col] = P[col].map(tf).fillna(False).astype(bool)
    site_pos = dict(zip(S.site_id, S.position_class))
    site_src = dict(zip(S.site_id, S.source))
    site_contact = dict(zip(S.site_id, S.modulator_contacts_interface))
    kind = S.drop_duplicates("pdb_id").set_index("pdb_id").complex_kind
    for col in ("sites_primary", "sites_any"):
        P[col] = P[col].fillna("").astype(str)
    P["prim_sites"] = P.sites_primary.map(lambda s: [x for x in s.split(",") if x])
    P["any_sites"] = P.sites_any.map(lambda s: [x for x in s.split(",") if x])
    multi = P.n_chains >= 2
    neg = multi & ~P.label_any & ~P.symmetric_copy & ~P.touches_ppi_site
    neg_all = multi & ~P.label_any & ~P.symmetric_copy  # S1/S2: PPI sites are part of the question

    def nonppi(sites):
        return any(site_pos.get(s) != "PPI interface" for s in sites)

    def build(pos_mask, neg_mask, size_col="n_pocket_residues"):
        d = P[pos_mask | neg_mask].copy()
        d["y"] = pos_mask[pos_mask | neg_mask].astype(int).to_numpy()
        return add_cells(d, size_col)

    say("=" * 100)
    say("PRIMARY TEST (pre-registered): interface enrichment of allosteric sites NOT annotated as PPI")
    say("=" * 100)
    pos = (P.role_primary == "positive")
    prim = build(pos, P.role_primary == "negative")
    res = matched_auc(prim, "frac_iface_5", "PRIMARY: key set, frac_iface_5, structure x size tercile",
                      n_perm=args.perm_primary, kind="primary")
    orr, olo, ohi = mantel_haenszel(prim, "n_iface_5")
    if orr is None:
        say("  Mantel-Haenszel odds ratio, pocket has >= 1 interface residue: not estimable (no discordant cells)")
    else:
        say(f"  Mantel-Haenszel odds ratio, pocket has >= 1 interface residue: {orr:.2f} [{olo:.2f}, {ohi:.2f}]")
    RESULTS.append({"kind": "primary", "analysis": "MH odds ratio (n_iface_5 > 0)", "auc": orr, "lo": olo, "hi": ohi})
    raw_p = (prim[prim.y == 1].n_iface_5 > 0).mean()
    raw_n = (prim[prim.y == 0].n_iface_5 > 0).mean()
    say(f"  descriptive: pockets with >= 1 interface residue: positives {raw_p:.1%}, negatives {raw_n:.1%}")
    if res:
        verdict = "SUPPORTED" if res[1] > 0.5 else "NOT SUPPORTED"
        say(f"  -> decision rule (lower family bound > 0.5): {verdict}")

    say("")
    say("=" * 100)
    say("SECONDARY ANALYSES (Holm-adjusted p-values at the end)")
    say("=" * 100)
    sec_start = len(RESULTS)
    # S1 all multi-chain sites
    pos_s1 = multi & P.label_primary
    matched_auc(build(pos_s1, neg_all), "frac_iface_5", "S1 all multi-chain sites (PPI + non-PPI)", args.perm)
    # S2 PPI-annotated only
    pos_s2 = multi & P.prim_sites.map(lambda s: any(site_pos.get(x) == "PPI interface" for x in s))
    matched_auc(build(pos_s2, neg_all), "frac_iface_5", "S2 PPI-annotated sites only (size of the circular effect)", args.perm)
    # S3 beyond contact: pockets with no interface residue, score = - distance to the interface
    nocontact = P.n_iface_5 == 0
    d3 = build(pos & nocontact, (P.role_primary == "negative") & nocontact)
    d3["neg_dist"] = -d3.dist_iface_5
    d3 = d3[d3.neg_dist.notna()]
    matched_auc(d3, "neg_dist", "S3a beyond contact: no-interface pockets, -distance to interface", args.perm)
    far = P.prim_sites.map(lambda s: any(site_contact.get(x) is False for x in s if site_pos.get(x) != "PPI interface"))
    d3b = build(pos & nocontact & far, (P.role_primary == "negative") & nocontact)
    d3b["neg_dist"] = -d3b.dist_iface_5
    d3b = d3b[d3b.neg_dist.notna()]
    matched_auc(d3b, "neg_dist", "S3b as S3a, sites whose modulator is > 4.5 A from the interface", args.perm)
    d3c = d3[d3.n_chains == 2]
    d3c = add_cells(d3c)
    matched_auc(d3c, "neg_dist", "S3c as S3a, 2-chain structures only", args.perm)
    # S4 strata
    P["kind_group"] = P.pdb_id.map(kind).map(lambda k: "homo-oligomer" if str(k).startswith("homo") else
                                              ("hetero or protein-peptide" if k in ("hetero-complex", "protein-peptide")
                                               else ("same-protein multi-entity" if k == "pseudo-hetero" else "other")))
    for kg in ["homo-oligomer", "hetero or protein-peptide", "same-protein multi-entity"]:
        m = P.kind_group == kg
        matched_auc(build(pos & m, (P.role_primary == "negative") & m), "frac_iface_5", f"S4 {kg}", args.perm)
    for lab, m in (("2 chains", P.n_chains == 2), (">= 3 chains", P.n_chains >= 3)):
        matched_auc(build(pos & m, (P.role_primary == "negative") & m), "frac_iface_5", f"S4 {lab}", args.perm)
    # S5 annotation agreement, S6 ligand contact (proportions)
    nch = P.groupby("pdb_id").n_chains.first()
    S["n_chains"] = S.pdb_id.map(nch)
    Sm = S[S.n_chains >= 2]
    say("")
    say("  S5 share of sites with >= 1 interface residue (multi-chain structures), Wilson 95 % interval:")
    k = {}
    for pc in ["PPI interface", "inner protein", "allosteric position (unspecified)"]:
        g = Sm[Sm.position_class == pc]
        kk, nn = int((g.site_frac_interface_5 > 0).sum()), len(g)
        p, lo, hi = wilson(kk, nn)
        k[pc] = (kk, nn)
        say(f"     {pc:<36} {kk}/{nn} = {p:.1%} [{lo:.1%}, {hi:.1%}]")
    p5 = two_prop_p(*k["PPI interface"], *k["inner protein"])
    say(f"     PPI vs inner protein: two-proportion p = {p5:.2g}")
    RESULTS.append({"kind": "secondary", "analysis": "S5 PPI vs inner protein, share with interface residue",
                    "auc": None, "p": p5})
    mono_ppi = S[(S.n_chains == 1) & (S.position_class == "PPI interface")]
    say(f"     PPI-annotated sites in single-chain structures: {len(mono_ppi)} "
        f"({', '.join(mono_ppi.site_id.head(10))}{' ...' if len(mono_ppi) > 10 else ''})")
    say("  S6 share of located modulators within 4.5 A of the interface (multi-chain):")
    k6 = {}
    for pc in ["PPI interface", "inner protein", "allosteric position (unspecified)"]:
        g = Sm[(Sm.position_class == pc) & Sm.modulator_contacts_interface.notna()]
        kk, nn = int(g.modulator_contacts_interface.astype(bool).sum()), len(g)
        p, lo, hi = wilson(kk, nn)
        k6[pc] = (kk, nn)
        say(f"     {pc:<36} {kk}/{nn} = {p:.1%} [{lo:.1%}, {hi:.1%}]")
    p6 = two_prop_p(*k6["PPI interface"], *k6["inner protein"])
    say(f"     PPI vs inner protein: two-proportion p = {p6:.2g}")
    RESULTS.append({"kind": "secondary", "analysis": "S6 PPI vs inner protein, modulator contacts interface",
                    "auc": None, "p": p6})
    sec = RESULTS[sec_start:]
    adj = holm([r.get("p") for r in sec])
    say("")
    say("  Holm-adjusted p-values:")
    for r, a in zip(sec, adj):
        r["p_holm"] = a
        if a is not None:
            say(f"     {r['analysis']:<66} p_holm = {a:.4f}")

    say("")
    say("=" * 100)
    say("SENSITIVITY CHECKS (primary result should keep its direction)")
    say("=" * 100)
    for c in (1, 10, 20):
        matched_auc(prim, f"frac_iface_{c}", f"cutoff {c} A^2", args.perm, kind="sensitivity")
    pos_any = multi & P.any_sites.map(nonppi) & ~P.touches_ppi_site
    matched_auc(build(pos_any, P.role_primary == "negative"), "frac_iface_5", "label rule: any overlap", args.perm,
                kind="sensitivity")
    near = set()
    for _, r in S[S.nearest_pocket.notna()].iterrows():
        if r.position_class != "PPI interface":
            near.add((r.pdb_id, int(r.nearest_pocket)))
    pos_near = multi & pd.Series([(a, b) in near for a, b in zip(P.pdb_id, P.pocket)], index=P.index) & ~P.touches_ppi_site
    matched_auc(build(pos_near, P.role_primary == "negative"), "frac_iface_5", "label rule: nearest to modulator",
                args.perm, kind="sensitivity")
    if "volume" in P:
        matched_auc(build(pos, P.role_primary == "negative", "volume"), "frac_iface_5", "size = volume terciles",
                    args.perm, kind="sensitivity")
    matched_auc(prim, "frac_iface_5", "families weighted equally", 0, equal_family=True, kind="sensitivity")
    rng = np.random.default_rng(SEED)
    keep = []
    for f, g in prim.groupby("family"):
        ids = g.pdb_id.unique()
        keep += list(rng.choice(ids, min(5, len(ids)), replace=False))
    capped = add_cells(prim[prim.pdb_id.isin(keep)].drop(columns=["tercile", "cell"]))
    matched_auc(capped, "frac_iface_5", "at most 5 structures per family", args.perm, kind="sensitivity")
    asd = P.prim_sites.map(lambda s: any(site_src.get(x) == "ASD" and site_pos.get(x) != "PPI interface" for x in s))
    matched_auc(build(pos & asd, P.role_primary == "negative"), "frac_iface_5", "ASD sites only", args.perm,
                kind="sensitivity")
    say("  not run here: random-patch control, modulator-free partner pockets, asymmetric unit (separate scripts)")

    pd.DataFrame(RESULTS).to_csv(D / "results_step4x.csv", index=False)
    (D / "report_step4x.txt").write_text("\n".join(LOG) + "\n")
    say(f"\nSaved {D / 'results_step4x.csv'} and report_step4x.txt")


if __name__ == "__main__":
    main()
