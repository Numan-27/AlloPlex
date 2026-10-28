#!/usr/bin/env python
"""
STEP 13A: WHERE ALLOPLEX AND DEEPALLO DISAGREE (supervisor point 7; no training, minutes on a CPU)
===================================================================================================
Per-site ranks of AlloPlex (cross-validation, mean of 3 repeats) and DeepAllo on the PRIMARY sites,
exactly as in the paper (step 11 loaders). Writes the full per-site table for the supplement and
counts every category, including how many sites meet the two case-study selection rules.

  disagreement categories (Top-1 = the site pocket ranked first):
    AlloPlex only   AlloPlex first (in at least 2 of 3 repeats), DeepAllo not first
    DeepAllo only   DeepAllo first, AlloPlex not first
    both / neither
  case-study rules (as stated in the manuscript):
    success rule    AlloPlex first while EVERY other method ranks the site outside the top ten (5OLK)
    failure rule    AlloPlex outside the top ten and at least one other method ranks it first (2VD4)

RUN (from ~/allosite, env allosite):
    python scripts/step13a_disagreement.py
Output: figures/data/s13a_disagreement.csv (one row per site), figures/data/report_step13a.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step11_reviewer_analyses as s11  # noqa: E402
import step10a_extra_analyses as s10a  # noqa: E402

OUT = s11.DATA
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def main():
    sets, tabs, methods = s11.load()
    if "DeepAllo" not in methods:
        raise SystemExit("DeepAllo results not found (step 7 tool results)")
    prim = {m: s11.per_site(sets, tabs, runs, failed) for m, (runs, failed) in methods.items()
            if m != "Pocket volume"}
    A = prim[s11.OURS].set_index("site_id")
    D = prim["DeepAllo"].set_index("site_id")
    others = [m for m in prim if m != s11.OURS]

    # site attributes (volume tercile, modulator, interface position, fpocket match) as in the error analysis
    attrs = s10a.site_attributes(sets, tabs)
    cats = s10a.bins(attrs)
    master = s10a.BASE / "processed" / "master" / "allosteric_master.csv"
    names = {}
    if master.exists():
        ms = pd.read_csv(master, low_memory=False)
        ms["site_id"] = ms.site_id.astype(str)
        ms = ms.drop_duplicates("site_id").set_index("site_id")
        for c in ("protein_name", "gene", "organism", "modulator_name", "modulator_code"):
            if c in ms.columns:
                names[c] = ms[c]

    pdb = pd.concat([tabs[n].set_index("site_id").pdb_id for n in tabs])
    pdb = pdb[~pdb.index.duplicated()]
    rows = []
    for s in A.index.intersection(D.index):
        a, d = A.loc[s], D.loc[s]
        r = {"site_id": s, "pdb_id": pdb.get(s), "family": a["family"], "structure_set": a["set"],
             "alloplex_top1": round(float(a["top1"]), 3), "alloplex_rank": float(a["rank"]),
             "deepallo_top1": round(float(d["top1"]), 3), "deepallo_rank": float(d["rank"])}
        for m in others:
            if m != "DeepAllo":
                r[f"rank_{m}"] = float(prim[m].set_index("site_id")["rank"].get(s, np.nan))
        for c in ("Pocket volume", "Modulator", "Site position", "fpocket match"):
            r[c.lower().replace(" ", "_")] = cats[c].get(s) if c in cats else None
        r["pocket_volume_A3"] = attrs["pocket_volume"].get(s, np.nan)
        for c, ser in names.items():
            r[c] = ser.get(s)
        rows.append(r)
    t = pd.DataFrame(rows)
    a1, d1 = t.alloplex_top1 >= 0.5, t.deepallo_top1 >= 0.5
    t["category"] = np.select([a1 & ~d1, d1 & ~a1, a1 & d1], ["AlloPlex only", "DeepAllo only", "both first"],
                              "neither first")
    other_cols = [c for c in t.columns if c.startswith("rank_")] + ["deepallo_rank"]
    t["best_other_rank"] = t[other_cols].min(axis=1)
    t["success_rule"] = (t.alloplex_top1 >= 0.999) & (t[other_cols] > 10).all(axis=1)
    t["failure_rule"] = (t.alloplex_rank > 10) & (t[other_cols] <= 1).any(axis=1)
    t = t.sort_values(["category", "family", "site_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "s13a_disagreement.csv", index=False)

    n = len(t)
    say(f"PRIMARY sites scored by both AlloPlex and DeepAllo: {n} ({t.family.nunique()} families)")
    for c, g in t.groupby("category"):
        say(f"  {c:<14} {len(g):>5} sites ({100 * len(g) / n:4.1f}%), {g.family.nunique():>3} families")
    dis = t[t.category.isin(["AlloPlex only", "DeepAllo only"])]
    say(f"  sites where exactly one of the two ranks the site first: {len(dis)}")
    say("\nDisagreements by site-pocket volume tercile (AlloPlex only / DeepAllo only):")
    for v, g in dis.groupby("pocket_volume"):
        say(f"  {str(v):<24} {int((g.category == 'AlloPlex only').sum()):>4} / {int((g.category == 'DeepAllo only').sum()):>4}")
    say("\nDisagreements by modulator type (AlloPlex only / DeepAllo only):")
    for v, g in dis.groupby("modulator"):
        say(f"  {str(v):<24} {int((g.category == 'AlloPlex only').sum()):>4} / {int((g.category == 'DeepAllo only').sum()):>4}")
    say("\nCase-study selection rules:")
    say(f"  success rule (AlloPlex first in all repeats, every other method outside the top ten): "
        f"{int(t.success_rule.sum())} sites; e.g. {', '.join(t[t.success_rule].pdb_id.astype(str).head(12))}")
    say(f"  failure rule (AlloPlex outside the top ten, at least one other method first): "
        f"{int(t.failure_rule.sum())} sites")
    say("  (5OLK must appear in the success list and 2VD4 in the failure list - check below)")
    for code in ("5OLK", "2VD4"):
        g = t[t.pdb_id.astype(str).str.upper() == code]
        for r in g.itertuples():
            say(f"   {code}: site {r.site_id} AlloPlex rank {r.alloplex_rank:.1f}, DeepAllo rank {r.deepallo_rank:.1f}, "
                f"success_rule {r.success_rule}, failure_rule {r.failure_rule}")
    (OUT / "report_step13a.txt").write_text("\n".join(LOG) + "\n")
    say(f"\nwrote {OUT / 's13a_disagreement.csv'} and report_step13a.txt")


if __name__ == "__main__":
    main()
