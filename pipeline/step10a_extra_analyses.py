#!/usr/bin/env python
"""
STEP 10A: EXTRA ANALYSES FROM EXISTING CROSS-VALIDATION RESULTS (no new training)
================================================================================
Uses the saved cross-validation predictions of AlloPlex (MLP, feature set B, 3 repeats) and the
PRIMARY evaluation rule (modulator-free partner structure where available, else stripped holo).

  1. ERROR ANALYSIS - Top-1 by site property (family-bootstrap 95 % CI):
       modulator type (peptide / fragment < 15 heavy atoms / small molecule 15-40 / large > 40),
       site position (monomer / away from interface / partly / mostly at interface),
       size of the allosteric pocket (volume terciles), structure size (number of pockets, terciles),
       how well fpocket's best pocket matches the site (Jaccard), data source (ASD / AlloBench)
  2. FPOCKET CEILING - share of sites that fpocket finds as a pocket at all (AlloPlex can only rank
       detected pockets), and Top-1 / Top-3 among the detected sites
  4. CALIBRATION - reliability of the score (10 quantile bins, Brier score, expected calibration
       error) and a practical table: when the top pocket of a structure scores >= t, how often is it
       the allosteric site?
(item 3, feature importance: step10b; item 5, runtime: step10c; 6, AlphaFold: step10d; 7, new set: step10e)

RUN (from ~/allosite, env allosite; a few minutes, CPU):
    python scripts/step10a_extra_analyses.py
Output: figures/data/s10a_*.csv, figures/FigS3_error_calibration.pdf/.png, figures/data/report_step10a.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
FIG = BASE / "figures"
DATA = FIG / "data"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def ci(fam, v):
    lo, hi = s6.boot(np.asarray(fam), np.asarray(v, float))
    return float(np.nanmean(v)), float(lo), float(hi)


# ----------------------------------------------------------------------------- data
def load():
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {n: s6.site_table(n, sets[n]["meta"], D / f"sites_{n}.csv") for n in sets}
    scores, per = {}, {}
    for n in sets:
        sc = s6.collect("mlp", "B", n, sets[n]["meta"], [0, 1, 2])
        if sc is None:
            raise SystemExit(f"no saved predictions for mlp B on the {n} set (run step 6x first)")
        scores[n] = sc[0]  # calibration: the scores of one cross-validation run (repeat 0)
        # per-site metrics: computed per repeat, then averaged (exactly as in step 6x / the paper tables)
        t = pd.concat([s6.score_sites(tabs[n], sets[n]["meta"], s) for s in sc])
        per[n] = t.groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean().assign(set=n)
    # PRIMARY rule
    prim = per["stripped"]
    if "partner" in per:
        prim = pd.concat([per["partner"], prim[~prim.site_id.isin(per["partner"].site_id)]])
    prim = prim.reset_index(drop=True)
    prim["site_id"] = prim.site_id.astype(str)
    # detection (does the site have a positive pocket in the set used?)
    det = {}
    for n in sets:
        for sid, r in zip(tabs[n].site_id.astype(str), tabs[n].pos_row):
            det[(n, sid)] = r >= 0
    prim["detected"] = [det.get((n, s), False) for n, s in zip(prim.set, prim.site_id)]
    return sets, tabs, scores, prim


def site_attributes(sets, tabs):
    st = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    st["site_id"] = st.site_id.astype(str)
    keep = [c for c in ("site_id", "pdb_id", "best_jaccard", "n_pockets", "site_frac_interface_5", "complex_kind",
                        "source", "position_class") if c in st.columns]
    A = st[keep].drop_duplicates("site_id").set_index("site_id")
    mp = BASE / "processed" / "structures" / "sites_mapped.csv"
    if mp.exists():
        m = pd.read_csv(mp, low_memory=False)
        m["site_id"] = m.site_id.astype(str)
        m = m.drop_duplicates("site_id").set_index("site_id")
        for c in ("modulator_n_atoms", "modulator_status", "modulator_code_used"):
            if c in m.columns:
                A[c] = m[c]
    ms = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", low_memory=False)
    ms["site_id"] = ms.site_id.astype(str)
    ms = ms.drop_duplicates("site_id").set_index("site_id")
    for c in ("modulator_class", "modulator_is_peptide"):
        if c in ms.columns:
            A[c] = ms[c]
    # volume of the positive pocket (stripped set)
    S = sets["stripped"]
    vol = {}
    for sid, r in zip(tabs["stripped"].site_id.astype(str), tabs["stripped"].pos_row):
        if r >= 0:
            vol[sid] = float(S["fp"][r, 2])
    A["pocket_volume"] = pd.Series(vol)
    return A


def bins(A):
    """site property -> category label (None = not assignable)."""
    out = pd.DataFrame(index=A.index)
    pep = A.get("modulator_is_peptide", pd.Series(False, index=A.index)).astype(str).isin(["True", "true", "1"])
    na = pd.to_numeric(A.get("modulator_n_atoms", pd.Series(np.nan, index=A.index)), errors="coerce")
    na = na.where(na > 0)
    lig = np.where(pep, "peptide", np.where(na < 15, "fragment (< 15 atoms)",
                   np.where(na <= 40, "small molecule (15-40)", np.where(na > 40, "large (> 40 atoms)", None))))
    out["Modulator"] = lig
    fi = pd.to_numeric(A.get("site_frac_interface_5"), errors="coerce")
    mono = A.get("complex_kind", pd.Series("", index=A.index)).astype(str).eq("monomer")
    out["Site position"] = np.where(mono, "monomer", np.where(fi == 0, "away from interface",
                                    np.where(fi < 0.5, "partly at interface",
                                             np.where(fi >= 0.5, "mostly at interface", None))))
    v = A["pocket_volume"]
    q = v.quantile([1 / 3, 2 / 3]).to_numpy()
    labs = [f"small (< {q[0]:.0f} \u00c5\u00b3)", "medium", f"large (>= {q[1]:.0f} \u00c5\u00b3)"]
    out["Pocket volume"] = np.where(v.isna(), None, np.where(v < q[0], labs[0], np.where(v < q[1], labs[1], labs[2])))
    ORDER["Pocket volume"] = labs
    npk = pd.to_numeric(A.get("n_pockets"), errors="coerce")
    q = npk.quantile([1 / 3, 2 / 3]).to_numpy()
    labs = [f"< {q[0]:.0f} pockets", f"{q[0]:.0f}-{q[1]:.0f} pockets", f">= {q[1]:.0f} pockets"]
    out["Structure size"] = np.where(npk.isna(), None, np.where(npk < q[0], labs[0], np.where(npk < q[1], labs[1], labs[2])))
    ORDER["Structure size"] = labs
    bj = pd.to_numeric(A.get("best_jaccard"), errors="coerce")
    out["fpocket match"] = np.where(bj.isna(), None, np.where(bj < 0.25, "poor (J < 0.25)",
                                    np.where(bj < 0.5, "fair (0.25-0.5)", "good (J >= 0.5)")))
    out["Source"] = A.get("source")
    return out


ORDER = {"Modulator": ["peptide", "fragment (< 15 atoms)", "small molecule (15-40)", "large (> 40 atoms)"],
         "Site position": ["monomer", "away from interface", "partly at interface", "mostly at interface"],
         "fpocket match": ["poor (J < 0.25)", "fair (0.25-0.5)", "good (J >= 0.5)"],
         "Source": ["ASD", "AlloBench"]}


# ----------------------------------------------------------------------------- analyses
def error_analysis(prim, A):
    B = bins(A)
    d = prim.join(B, on="site_id")
    rows = []
    say("\n1. ERROR ANALYSIS (AlloPlex Top-1 by site property, PRIMARY sites, 95 % family-bootstrap CI)")
    say(f"   all sites: n = {len(d)}, Top-1 {d.top1.mean():.3f}")
    for prop in B.columns:
        cats = [c for c in d[prop].dropna().unique()]
        cats = [c for c in ORDER.get(prop, []) if c in cats] + sorted(c for c in cats if c not in ORDER.get(prop, []))
        say(f"   {prop}")
        for c in cats:
            t = d[d[prop] == c]
            if len(t) < 10:
                continue
            m, lo, hi = ci(t.family, t.top1)
            m3 = t.top3.mean()
            rows.append({"property": prop, "category": c, "sites": len(t), "families": t.family.nunique(),
                         "top1": m, "lo": lo, "hi": hi, "top3": m3, "detected": t.detected.mean()})
            say(f"     {c:<28} n={len(t):>5}  Top-1 {m:.3f} [{lo:.3f}, {hi:.3f}]  Top-3 {m3:.3f}  "
                f"detected by fpocket {t.detected.mean():.0%}")
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "s10a_error_analysis.csv", index=False)
    return out


def ceiling(prim):
    say("\n2. FPOCKET CEILING (AlloPlex can only rank pockets that fpocket detects)")
    m, lo, hi = ci(prim.family, prim.detected.astype(float))
    say(f"   sites detected as a pocket (best-pocket Jaccard >= 0.1): {prim.detected.sum()}/{len(prim)} = "
        f"{m:.1%} [{lo:.1%}, {hi:.1%}]")
    t = prim[prim.detected]
    r = {"sites": len(prim), "detected": int(prim.detected.sum()), "detected_share": m, "lo": lo, "hi": hi}
    for k in ("top1", "top3"):
        a = ci(prim.family, prim[k])
        b = ci(t.family, t[k])
        r[f"{k}_all"], r[f"{k}_detected"] = a[0], b[0]
        r[f"{k}_detected_lo"], r[f"{k}_detected_hi"] = b[1], b[2]
        say(f"   {k.replace('top', 'Top-')}: all sites {a[0]:.3f}; detected sites only {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]"
            f"  (highest possible with fpocket pockets: {m:.3f})")
    pd.DataFrame([r]).to_csv(DATA / "s10a_ceiling.csv", index=False)
    return r


def calibration(sets, scores, prim):
    say("\n4. CALIBRATION (cross-validation scores, repeat 0; labelled pockets of the PRIMARY structures)")
    ys, ss = [], []
    used = {(n, p) for n, p in zip(prim.set, prim.site_id)}
    pdbs = {n: set() for n in sets}
    tabs_pdb = {}
    for n in sets:
        st = pd.read_csv(D / f"sites_{n}.csv", low_memory=False, usecols=["site_id", "pdb_id"])
        tabs_pdb[n] = dict(zip(st.site_id.astype(str), st.pdb_id))
    for n, sid in used:
        if sid in tabs_pdb[n]:
            pdbs[n].add(tabs_pdb[n][sid])
    top_rows = []
    for n in sets:
        meta, y = sets[n]["meta"], sets[n]["y"]
        sc = scores[n]
        m = np.isin(meta.pdb_id.to_numpy(), list(pdbs[n])) & (y >= 0) & np.isfinite(sc)
        ys.append(y[m])
        ss.append(sc[m])
        # top pocket of each structure: its score and whether it is an allosteric (positive) pocket
        df = pd.DataFrame({"pdb": meta.pdb_id.to_numpy()[m], "s": sc[m], "y": y[m],
                           "family": meta.family.astype(str).to_numpy()[m]})
        top = df.sort_values("s", ascending=False).drop_duplicates("pdb")
        top_rows.append(top.assign(set=n))
    y, s = np.concatenate(ys), np.concatenate(ss)
    edges = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    rel = pd.DataFrame({"bin": idx, "s": s, "y": y}).groupby("bin").agg(mean_score=("s", "mean"),
                                                                         observed=("y", "mean"), n=("y", "size"))
    brier = float(np.mean((s - y) ** 2))
    ece = float((rel.n * (rel.mean_score - rel.observed).abs()).sum() / rel.n.sum())
    say(f"   pockets: {len(y)} (allosteric {int(y.sum())}, {y.mean():.1%}); Brier score {brier:.4f} "
        f"(always-'no' baseline {y.mean():.4f}); expected calibration error {ece:.3f}")
    for _, r in rel.iterrows():
        say(f"     mean score {r.mean_score:.3f}  observed share allosteric {r.observed:.3f}  (n={int(r.n)})")
    rel.to_csv(DATA / "s10a_reliability.csv", index=False)
    top = pd.concat(top_rows)
    rows = []
    say("   top pocket of a structure: how often it is an allosteric pocket, by its score")
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        g = top[top.s >= t]
        if len(g) < 10:
            continue
        m, lo, hi = ci(g.family, g.y.astype(float))
        rows.append({"threshold": t, "structures": len(g), "share_of_structures": len(g) / len(top),
                     "hit_rate": m, "lo": lo, "hi": hi})
        say(f"     top score >= {t:.1f}: {len(g):>5} structures ({len(g) / len(top):.0%}); top pocket allosteric "
            f"{m:.1%} [{lo:.1%}, {hi:.1%}]")
    th = pd.DataFrame(rows)
    th.to_csv(DATA / "s10a_threshold.csv", index=False)
    pd.DataFrame([{"brier": brier, "brier_baseline": float(y.mean()), "ece": ece, "pockets": len(y)}]).to_csv(
        DATA / "s10a_calibration_summary.csv", index=False)
    return rel, th


# ----------------------------------------------------------------------------- figure
def figure(err, rel, th, allm=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import step9x_figures as f9
    plt = f9.style()
    OI = f9.OI
    fig = plt.figure(figsize=(f9.W2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    props = [p for p in ("Modulator", "Site position", "Pocket volume", "fpocket match") if p in set(err.property)]
    labels, ms, los, his, ypos, y0, heads = [], [], [], [], [], 0, []
    for p in props:
        t = err[err.property == p]
        heads.append((y0, p))
        y0 += 1
        for _, r in t.iterrows():
            labels.append(f"{r.category} ({r.sites})")
            ms.append(r.top1)
            los.append(r.lo)
            his.append(r.hi)
            ypos.append(y0)
            y0 += 1
        y0 += 0.4
    ypos = np.array(ypos)
    if allm is None:
        allm = float(np.average(err[err.property == props[0]].top1, weights=err[err.property == props[0]].sites))
    for yy, m, lo, hi in zip(ypos, ms, los, his):
        ax.plot([lo, hi], [yy, yy], color=OI["blue"], lw=1.0)
        ax.plot(m, yy, "o", color=OI["blue"], ms=3.6, mec="white", mew=0.5)
    ax.axvline(allm, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5.8)
    for yy, p in heads:
        ax.text(-0.02, yy, p, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.2,
                fontweight="bold")
    ax.set_ylim(y0 - 0.4, -0.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AlloPlex Top-1")
    f9.letter(ax, "a", x=-0.95, y=1.0)
    ax = fig.add_subplot(gs[1])
    mx = max(rel.mean_score.max(), rel.observed.max()) * 1.05
    ax.plot([0, mx], [0, mx], color="#BBBBBB", lw=0.6, ls=(0, (2, 2)))
    ax.plot(rel.mean_score, rel.observed, "o-", color=OI["blue"], ms=3.2, lw=1.0, mec="white", mew=0.5)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Mean AlloPlex score")
    ax.set_ylabel("Observed share allosteric")
    f9.letter(ax, "b", x=-0.42, y=1.0)
    ax = fig.add_subplot(gs[2])
    ax.plot(th.threshold, th.hit_rate * 100, "o-", color=OI["blue"], ms=3.2, lw=1.0, mec="white", mew=0.5,
            label="Top pocket correct")
    ax.fill_between(th.threshold, th.lo * 100, th.hi * 100, color=OI["blue"], alpha=0.15, lw=0)
    ax.plot(th.threshold, th.share_of_structures * 100, "s--", color=OI["grey"], ms=2.8, lw=0.9,
            label="Structures above")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Top-pocket score ≥")
    ax.set_ylabel("%")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=5.8)
    f9.letter(ax, "c", x=-0.42, y=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"FigS3_error_calibration.{ext}", bbox_inches="tight")
    print("  saved figures/FigS3_error_calibration.pdf/.png")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    sets, tabs, scores, prim = load()
    say(f"PRIMARY sites: {len(prim)} (partner {int((prim.set == 'partner').sum())}, stripped "
        f"{int((prim.set == 'stripped').sum())}); AlloPlex Top-1 {prim.top1.mean():.3f}")
    A = site_attributes(sets, tabs)
    err = error_analysis(prim, A)
    ceiling(prim)
    rel, th = calibration(sets, scores, prim)
    figure(err, rel, th, float(prim.top1.mean()))
    (DATA / "report_step10a.txt").write_text("\n".join(LOG) + "\n")
    print(f"\nreport: {DATA / 'report_step10a.txt'}")


if __name__ == "__main__":
    main()
