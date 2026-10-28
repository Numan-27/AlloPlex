#!/usr/bin/env python
"""
STEP 11: ANALYSES REQUESTED BY THE REVIEWER (no new training; minutes on a CPU)
=============================================================================
Uses the saved cross-validation predictions (AlloPlex = MLP, feature set B, 3 repeats), the saved
tool results (step 7) and the matched evaluation (all methods rank the same fpocket pockets).

  1. SUBSETS (concern 9)      Top-1 of every method on partner-only sites, on stripped-only sites
                              (the PRIMARY mixture) and on ALL sites scored on stripped structures,
                              with paired differences to AlloPlex.
  2. LABEL THRESHOLD (4)      Jaccard cut-off 0.1 / 0.2 / 0.3 / 0.5 for the site pocket:
                              strict  = a site whose best pocket is below the cut-off counts as a miss
                              subset  = only sites whose best pocket reaches the cut-off are scored
  3. POCKET SIZE (5)          Top-1 per volume tercile of the site pocket, and SIZE-MATCHED Top-1
                              (the site pocket competes only with pockets of the same volume tercile
                              in its structure); pocket volume itself as a baseline.
  4. NEGATIVES (moderate)     number of pockets set aside (overlap an annotated site or a symmetric
                              copy) and Top-1 when ALL of them count as negatives (strict lower bound).
  5. FAMILY WEIGHTING (10)    family sizes; Top-1 weighted by site (paper) and by family.
  6. COUNTS (consistency)     2,375 vs 2,349 sites; modulator categories (2,216 vs 2,349);
                              fpocket detection in cross-validation vs the 2024+ set.
  7. FAILURE CASES (moderate) candidate case studies where AlloPlex misses a well-detected site.
  8. SCORE GUIDE ON 2024+ SET (moderate) - if the per-pocket files of step 10e are present.

RUN (from ~/allosite, env allosite):
    python scripts/step11_reviewer_analyses.py
Output: figures/data/report_step11.txt and figures/data/s11_*.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402
import step7y_score_tools as s7  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
DATA = BASE / "figures" / "data"
OURS = "AlloPlex"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def ci(fam, v):
    v = np.asarray(v, float)
    lo, hi = s6.boot(np.asarray(fam), v)
    return float(np.nanmean(v)), float(lo), float(hi)


def fmt(m, lo, hi, sign=False):
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"


# ============================================================================= data
def load():
    man = pd.read_csv(s7.T / "manifest.csv")
    man = man[man.roles.str.contains("primary")]
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {n: s6.site_table(n, sets[n]["meta"], D / f"sites_{n}.csv") for n in sets}
    for n in sets:
        tabs[n]["site_id"] = tabs[n].site_id.astype(str)
        st = pd.read_csv(D / f"sites_{n}.csv", low_memory=False)
        st["site_id"] = st.site_id.astype(str)
        bj = st.drop_duplicates("site_id").set_index("site_id").get("best_jaccard")
        tabs[n]["best_j"] = tabs[n].site_id.map(bj).astype(float) if bj is not None else np.nan
    pocket_res = {}
    for setn, csv in s7.SETMAP.values():
        if setn in sets:
            pk = pd.read_csv(D / csv, usecols=["pdb_id", "residues"], low_memory=False)
            pocket_res[setn] = [[(c, int(n)) for c, n, *_ in json.loads(r)] for r in pk.residues]
    site_key = {s: k for k, ss in zip(man.key, man.sites) for s in str(ss).split(",")}
    methods = {}  # name -> (list of {set: score array}, failed site ids)
    mlp = {n: s6.collect("mlp", "B", n, sets[n]["meta"], [0, 1, 2]) for n in sets}
    if any(v is None for v in mlp.values()):
        raise SystemExit("saved predictions for mlp B are missing (run step 6x)")
    methods[OURS] = ([{n: mlp[n][r] for n in sets} for r in range(3)], set())
    for name, fn in s7.TOOLS.items():
        path = s7.T / "results" / fn
        if path.exists():
            sc, failed = s7.scores_for_tool(json.load(open(path)), man, sets, pocket_res)
            methods[name] = ([sc], {s for s, k in site_key.items() if k in failed})
    methods["Druggability"] = ([{n: sets[n]["fp"][:, 1].astype(float) for n in sets}], set())
    methods["Pocket volume"] = ([{n: sets[n]["fp"][:, 2].astype(float) for n in sets}], set())
    desc = ", ".join("%s (%d pockets)" % (n, len(sets[n]["y"])) for n in sets)
    say("sets: " + desc + "; methods: " + ", ".join(methods))
    return sets, tabs, methods


# ============================================================================= site-level evaluation
def eval_set(tab, S, scores, neg="standard", band=None, jcut=None, jmode=None):
    """mirrors step 6x score_sites; returns per-site top1, top3, rank.
    neg='standard': negatives = y == 0;  neg='all': y in {0, -1} (set-aside pockets count as negatives)
    band: array of size-band labels per pocket -> the site pocket competes only within its band
    jcut/jmode: label threshold; 'strict' = below cut-off is a miss; 'subset' = skip such sites"""
    meta, y = S["meta"], S["y"]
    pdb = meta.pdb_id.to_numpy()
    ok = ~np.isnan(scores)
    isneg = (y == 0) if neg == "standard" else (y != 1)
    idx_by_pdb = {}
    for i, p in enumerate(pdb):
        idx_by_pdb.setdefault(p, []).append(i)
    rows = []
    for s, p, f, r, bj in zip(tab.site_id, tab.pdb_id, tab.family, tab.pos_row, tab.best_j):
        if jcut is not None and jmode == "subset" and not (bj >= jcut):
            continue
        cand = np.array(idx_by_pdb.get(p, []), int)
        if r >= 0 and not ok[r]:
            continue
        nmask = isneg[cand] & ok[cand] & (cand != r)
        if band is not None and r >= 0:
            nmask &= band[cand] == band[r]
        n = scores[cand[nmask]]
        miss = r < 0 or (jcut is not None and jmode == "strict" and not (bj >= jcut))
        if miss:
            if (isneg[cand] & ok[cand]).any():
                rows.append((s, f, 0.0, 0.0, np.inf))
            continue
        if len(n) == 0:
            continue  # no competing pocket (in its size band): not informative, skipped for every method
        sp = scores[r]
        n_gt, n_eq = int((n > sp).sum()), int((n == sp).sum())
        top1 = (1.0 / (n_eq + 1)) if n_gt == 0 else 0.0
        top3 = max(0, min(n_eq + 1, 3 - n_gt)) / (n_eq + 1)
        rows.append((s, f, top1, top3, 1 + n_gt + 0.5 * n_eq))
    return pd.DataFrame(rows, columns=["site_id", "family", "top1", "top3", "rank"])


def per_site(sets, tabs, runs, failed, which="primary", **kw):
    """-> per-site table (mean over runs) with column 'set'. which: primary | partner | stripped_all"""
    out = []
    for sc in runs:
        per = {n: eval_set(tabs[n], sets[n], sc[n], **kw).assign(set=n) for n in sets}
        if which == "stripped_all":
            t = per["stripped"]
        elif which == "partner":
            t = per["partner"]
        else:
            t = per["stripped"]
            if "partner" in per:
                t = pd.concat([per["partner"], t[~t.site_id.isin(per["partner"].site_id)]])
        miss = [s for s in failed if s not in set(t.site_id)]
        if miss and which == "primary":
            fam = pd.concat([tabs[n][["site_id", "family"]] for n in tabs]).drop_duplicates("site_id").set_index("site_id").family
            t = pd.concat([t, pd.DataFrame({"site_id": miss, "family": [fam.get(s, "") for s in miss], "top1": 0.0,
                                            "top3": 0.0, "rank": np.inf, "set": "failed"})])
        out.append(t)
    t = pd.concat(out)
    return t.groupby(["site_id", "family", "set"], as_index=False)[["top1", "top3", "rank"]].mean()


def compare(tables, label, csv_rows, extra=None):
    """tables: {method: per-site table}; prints mean [CI] and paired difference to AlloPlex."""
    ref = tables[OURS].set_index("site_id")
    for m, t in tables.items():
        if t.empty:
            continue
        mm = ci(t.family, t.top1)
        row = {"analysis": label, "method": m, "sites": len(t), "top1": mm[0], "lo": mm[1], "hi": mm[2],
               "top3": t.top3.mean(), **(extra or {})}
        txt = f"    {m:<14} n={len(t):>5}  Top-1 {fmt(*mm)}  Top-3 {t.top3.mean():.3f}"
        if m != OURS:
            j = t.set_index("site_id").join(ref[["top1"]], rsuffix="_ref", how="inner")
            d = ci(j.family, j.top1_ref - j.top1)
            row.update(diff=d[0], diff_lo=d[1], diff_hi=d[2], paired_sites=len(j))
            txt += f"   AlloPlex minus {m}: {fmt(*d, sign=True)} (n={len(j)})"
        say(txt)
        csv_rows.append(row)


# ============================================================================= analyses
def subsets(sets, tabs, methods, rows):
    say("\n1. SUBSETS (concern 9): does the ordering hold on partner-only and stripped-only sites?")
    say("  (check: the PRIMARY block must reproduce Table 5 - AlloPlex 0.422, DeepAllo 0.387, APOP 0.379 ...)")
    prim = {m: per_site(sets, tabs, r, f) for m, (r, f) in methods.items()}
    for lab, sel in (("PRIMARY (paper)", None), ("partner structures only", "partner"),
                     ("stripped only (sites without a partner)", "stripped")):
        say(f"  {lab}")
        tb = {m: (t if sel is None else t[t.set == sel]) for m, t in prim.items()}
        compare(tb, f"subset: {lab}", rows)
    say("  ALL sites scored on stripped (holo-derived) structures")
    compare({m: per_site(sets, tabs, r, f, which="stripped_all") for m, (r, f) in methods.items()},
            "subset: all sites, stripped", rows)
    return prim


def jaccard(sets, tabs, methods, rows):
    say("\n2. LABEL THRESHOLD (concern 4): Jaccard cut-off for the site pocket")
    for mode, txt in (("strict", "sites below the cut-off count as misses"),
                      ("subset", "only sites whose best pocket reaches the cut-off")):
        say(f"  mode '{mode}': {txt}")
        for j in (0.1, 0.2, 0.3, 0.5):
            say(f"   cut-off {j}")
            tb = {m: per_site(sets, tabs, r, f, jcut=j, jmode=mode) for m, (r, f) in methods.items()
                  if m != "Pocket volume"}
            compare(tb, f"jaccard {mode} {j}", rows, {"mode": mode, "cutoff": j})


def size(sets, tabs, methods, prim, rows):
    say("\n3. POCKET SIZE (concern 5)")
    allv = np.concatenate([sets[n]["fp"][:, 2] for n in sets])
    q = np.quantile(allv, [1 / 3, 2 / 3])
    say(f"  volume terciles over all pockets: < {q[0]:.0f}, {q[0]:.0f}-{q[1]:.0f}, >= {q[1]:.0f} A^3")
    band = {n: np.digitize(sets[n]["fp"][:, 2], q) for n in sets}
    # tercile of the SITE pocket
    vol = {}
    for n in sets:
        for s, r in zip(tabs[n].site_id, tabs[n].pos_row):
            if r >= 0:
                vol[(n, s)] = int(band[n][r])
    labs = ["small", "medium", "large"]
    say("  (a) Top-1 by volume tercile of the site pocket (standard evaluation)")
    for k, lab in enumerate(labs):
        say(f"   {lab} site pockets")
        tb = {m: t[[vol.get((st, s), -1) == k for st, s in zip(t.set, t.site_id)]] for m, t in prim.items()}
        compare(tb, f"size tercile {lab}", rows)
    say("  (b) SIZE-MATCHED Top-1: the site pocket competes only with pockets of its own volume tercile")
    tb = {}
    for m, (r, f) in methods.items():
        out = []
        for sc in r:
            per = {n: eval_set(tabs[n], sets[n], sc[n], band=band[n]).assign(set=n) for n in sets}
            t = pd.concat([per["partner"], per["stripped"][~per["stripped"].site_id.isin(per["partner"].site_id)]]) \
                if "partner" in per else per["stripped"]
            out.append(t)
        tb[m] = pd.concat(out).groupby(["site_id", "family", "set"], as_index=False)[["top1", "top3", "rank"]].mean()
    compare(tb, "size-matched", rows)
    say("   (pocket volume should fall towards chance here: it can no longer use size; sites whose pocket has no"
        " competitor in its size band are skipped for every method)")


def negatives(sets, tabs, methods, rows):
    say("\n4. NEGATIVE SET (moderate concern): pockets set aside from the negatives")
    for n in sets:
        y = sets[n]["y"]
        say(f"  {n}: {len(y)} pockets; positive {int((y == 1).sum())}, negative {int((y == 0).sum())}, "
            f"set aside {int((y == -1).sum())} ({(y == -1).mean():.1%}; overlap an annotated site or a symmetric copy)")
    say("  Top-1 when ALL set-aside pockets count as negatives (strict lower bound: includes other annotated")
    say("  sites and identical copies of the same site on other chains, which a perfect method would also rank high)")
    tb = {m: per_site(sets, tabs, r, f, neg="all") for m, (r, f) in methods.items()}
    compare(tb, "negatives: set-aside counted as negatives", rows)


def family_weighting(prim):
    say("\n5. FAMILY WEIGHTING (concern 10)")
    t = prim[OURS]
    fs = t.groupby("family").size().sort_values(ascending=False)
    say(f"  {len(t)} sites in {len(fs)} families; sites per family: median {fs.median():.0f}, max {fs.max()}; "
        f"10 largest families hold {fs.head(10).sum() / fs.sum():.1%} of sites")
    say("  The bootstrap resamples whole families (clustered bootstrap); the estimate is the site-weighted mean.")
    rng = np.random.default_rng(s6.SEED)
    for m in (OURS, "DeepAllo", "APOP", "Druggability"):
        if m not in prim:
            continue
        g = prim[m].groupby("family").top1.mean()
        bs = [g.to_numpy()[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        say(f"  {m:<14} site-weighted Top-1 {prim[m].top1.mean():.3f};  family-weighted {g.mean():.3f} [{lo:.3f}, {hi:.3f}]")


def counts(sets, tabs):
    say("\n6. COUNTS (internal consistency)")
    st = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    say(f"  sites in the stripped site table: {st.site_id.nunique()}; status: "
        + ", ".join(f"{k} {v}" for k, v in st.drop_duplicates("site_id").status.value_counts().items()))
    say(f"  sites evaluated (status ok): {tabs['stripped'].site_id.nunique()}")
    try:
        import step10a_extra_analyses as s10a
        A = s10a.site_attributes(sets, tabs)
        Bn = s10a.bins(A)
        ids = set(tabs["stripped"].site_id)
        mod = Bn.loc[Bn.index.isin(ids), "Modulator"]
        say("  modulator categories among evaluated sites: "
            + ", ".join(f"{k} {v}" for k, v in mod.fillna("no size information").value_counts().items()))
    except Exception as e:  # noqa: BLE001
        say(f"  (modulator categories not computed: {type(e).__name__}: {e})")
    for n in sets:
        t = tabs[n]
        say(f"  fpocket detection, {n}: {(t.pos_row >= 0).mean():.1%} of {len(t)} sites (best pocket Jaccard >= 0.1 "
            f"with the annotated site residues)")
    f = DATA / "s10e_newset.csv"
    if f.exists():
        d = pd.read_csv(f)
        say(f"  2024+ set: site found in {d.detected.mean():.1%} of {len(d)} complexes (site = residues within 4.5 A of a"
            f" single drug-like ligand, i.e. compact, well-defined pockets)")


def failures(sets, tabs, methods, prim):
    say("\n7. CANDIDATE FAILURE CASES (AlloPlex rank >= 10 although fpocket found the site well, J >= 0.5)")
    a = prim[OURS].set_index("site_id")
    others = [m for m in methods if m not in (OURS, "Pocket volume")]
    bj = pd.concat([tabs[n].set_index("site_id").best_j for n in tabs])
    bj = bj[~bj.index.duplicated()]
    pdb = pd.concat([tabs[n].set_index("site_id").pdb_id for n in tabs])
    pdb = pdb[~pdb.index.duplicated()]
    rows = []
    for s, r in a.iterrows():
        if not np.isfinite(r["rank"]) or r["rank"] < 10 or not (bj.get(s, 0) >= 0.5):
            continue
        ranks = {m: prim[m].set_index("site_id")["rank"].get(s, np.nan) for m in others}
        best = min((v for v in ranks.values() if np.isfinite(v)), default=np.nan)
        rows.append({"site_id": s, "pdb_id": pdb.get(s), "set": r["set"], "best_jaccard": bj.get(s),
                     "rank_alloplex": r["rank"], **{f"rank_{m}": v for m, v in ranks.items()}, "best_other": best})
    d = pd.DataFrame(rows)
    if d.empty:
        say("  none")
        return
    d = d.sort_values(["best_other", "rank_alloplex"], ascending=[True, False])
    d.to_csv(DATA / "s11_failure_candidates.csv", index=False)
    say(f"  {len(d)} candidates; the 10 where another method does best:")
    for r in d.head(10).itertuples():
        oth = ", ".join(f"{m} {getattr(r, 'rank_' + m):.0f}" for m in others if np.isfinite(getattr(r, "rank_" + m)))
        say(f"   {r.pdb_id} site {r.site_id} ({r.set}, J={r.best_jaccard:.2f}): AlloPlex rank {r.rank_alloplex:.0f}; {oth}")
    say("  (full list: figures/data/s11_failure_candidates.csv; pick one with a clear modulator and a"
        " competitor ranking it first)")


def newset_guide():
    say("\n8. SCORE GUIDE ON THE 2024+ SET (independent check of the cross-validation thresholds)")
    f = DATA / "s10e_newset.csv"
    if not f.exists():
        say("  s10e_newset.csv not found - skipped")
        return
    d = pd.read_csv(f)
    try:
        import step10e_new_test_set as s10e
        work = s10e.N / "work"
    except Exception:  # noqa: BLE001
        work = D / "newset" / "work"
    top = []
    for p in d.pdb_id:
        cands = list((work / p).rglob("alloplex_pockets.csv")) if (work / p).exists() else []
        if not cands:
            top.append(np.nan)
            continue
        P = pd.read_csv(cands[0])
        col = "alloplex_score" if "alloplex_score" in P.columns else [c for c in P.columns if "score" in c][0]
        top.append(float(P[col].max()))
    d["top_score"] = top
    if d.top_score.isna().all():
        say(f"  per-pocket files not found under {work} - skipped (the guide can be re-derived by re-running step 10e)")
        return
    d["correct"] = (d.rank_alloplex.astype(float) <= 1).astype(float)
    for t in (0.0, 0.4, 0.5, 0.6, 0.7):
        g = d[d.top_score >= t]
        say(f"  top-pocket score >= {t:.1f}: {len(g):>3} complexes ({len(g) / len(d):.0%}); top pocket is the site in "
            f"{g.correct.mean():.1%}" if len(g) else f"  top-pocket score >= {t:.1f}: none")
    d[["pdb_id", "top_score", "rank_alloplex", "correct"]].to_csv(DATA / "s11_newset_scoreguide.csv", index=False)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    sets, tabs, methods = load()
    rows = []
    prim = subsets(sets, tabs, methods, rows)
    jaccard(sets, tabs, methods, rows)
    size(sets, tabs, methods, prim, rows)
    negatives(sets, tabs, methods, rows)
    family_weighting(prim)
    counts(sets, tabs)
    failures(sets, tabs, methods, prim)
    newset_guide()
    pd.DataFrame(rows).to_csv(DATA / "s11_results.csv", index=False)
    (DATA / "report_step11.txt").write_text("\n".join(LOG) + "\n")
    say("\nsaved figures/data/report_step11.txt and s11_*.csv")


if __name__ == "__main__":
    main()
