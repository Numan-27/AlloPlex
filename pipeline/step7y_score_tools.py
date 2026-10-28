#!/usr/bin/env python
"""
STEP 7Y: SCORE THE EXTERNAL TOOLS ON THE SAME SITES AS OUR MODELS
================================================================
Reads the tools' JSON results (from the other server) placed in
    processed/dataset_v2/tools/results/deepallo.json, apop.json, p2rank.json (+ passer.json, alloses.json if present)
    {"<key>": {"status": "ok", "pockets": [{"score": s, "residues": [...]}, ...]}}
    key = set__pdb__chain from tools_input/manifest.csv

Mapping to OUR pockets (the plan): each of our pockets in that structure gets the score
of the best tool pocket whose residues overlap it with Jaccard >= 0.5, computed on our
pocket's residues in the exported chain (the tool sees only that chain); our pockets
that match no tool pocket rank below every matched one.
Scoring: exactly the step-6x site metrics on the PRIMARY set (partner where available,
else stripped): Top-1, Top-3, site AUROC; undetected sites = misses. APOP-style results flagged "all_chains" (whole assembly, residues with chains)
are compared with our full pockets. A chain on which
the tool failed counts its sites as misses ("all sites"); results are also shown for
the sites where the tool ran ("tool ran"). Then paired differences (family bootstrap)
of our MLP (feature set B) and the druggability baseline against each tool.

TRAINING-OVERLAP CHECK: the tools learned from ASD-derived data; DeepAllo's own copy is
ASD_Release_201909_AS.txt (ASD 2019). Each test site is classed by its protein (ASD target ID /
UniProt) as seen in ASD 2019 or NEW since 2019, and its family (>=30 % identity) as NEW if no member
protein is in ASD 2019. Both comparisons are repeated on the NEW-protein and NEW-family sites.

RUN (from ~/allosite):  python scripts/step7y_score_tools.py [--asd2019 PATH]
"""
import json
from collections import defaultdict
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
T = D / "tools"
TOOLS = {"DeepAllo": "deepallo.json", "APOP": "apop.json", "P2Rank": "p2rank.json",
         "PASSer": "passer.json", "AllosES": "alloses.json"}  # missing files are skipped
SETMAP = {"partner": ("partner", "pockets_partner.csv"), "holo": ("stripped", "pockets_stripped.csv")}
SUBSET_TOOLS = {"PASSer"}  # web-only tools run on a random subset (step 7v)
J_MIN = 0.5
LOW = -1e9


def tool_residues(res, chain):
    if isinstance(res, str):
        res = [x for x in re.split(r"[,\s]+", res) if x]
    out = set()
    for r in res:
        if isinstance(r, (list, tuple)):  # [chain, num] or [num, chain]
            a, b = r[0], r[1]
            c, n = (a, b) if not str(a).lstrip("-").isdigit() else (b, a)
        elif ":" in str(r):
            a, b = str(r).split(":")[:2]
            c, n = (b, a) if a.lstrip("-").isdigit() else (a, b)
        else:
            c, n = chain, r
        m = re.match(r"-?\d+", str(n))
        if m and str(c) == chain:
            out.add(int(m.group()))
    return out


def scores_for_tool(result, manifest, sets, pocket_res):
    """-> {set_name: score array over meta rows (NaN = not scored)}, set of failed keys"""
    sc = {n: np.full(len(sets[n]["meta"]), np.nan) for n in sets}
    failed = set()
    for _, m in manifest.iterrows():
        setn, _ = SETMAP[m["set"]]
        r = result.get(m.key)
        if not r or r.get("status") != "ok" or not r.get("pockets"):
            failed.add(m.key)
            continue
        whole = bool(r.get("all_chains"))  # e.g. APOP: run on the whole assembly, residues carry chains
        tp = [(float(p["score"]), tool_residues(p.get("residues", []), m.chain)) for p in r["pockets"]]
        rows = np.where(sets[setn]["meta"].pdb_id.to_numpy() == m.pdb_id)[0]
        chains_here = {c for i in rows for c, _ in pocket_res[setn][i]}
        if whole:
            tp = [(s, {(c, n) for c in chains_here for n in tool_residues(raw, c)})
                  for s, raw in [(float(p["score"]), p.get("residues", [])) for p in r["pockets"]]]
        for i in rows:
            mine = set(pocket_res[setn][i]) if whole else {n for c, n in pocket_res[setn][i] if c == m.chain}
            best = LOW
            for s, t in tp:
                if mine and t and len(mine & t) / len(mine | t) >= J_MIN:
                    best = max(best, s)
            sc[setn][i] = best
    return sc, failed


def primary_table(sets, tabs, sc, failed_sites):
    per = {}
    for n in ("stripped", "partner"):
        if n in sets:
            per[n] = s6.score_sites(tabs[n], sets[n]["meta"], sc[n]).set_index("site_id")
    prim = per["stripped"]
    if "partner" in per:
        prim = pd.concat([per["partner"], prim[~prim.index.isin(per["partner"].index)]])
    prim = prim.reset_index()
    fam = pd.concat([tabs[n][["site_id", "family"]] for n in tabs]).drop_duplicates("site_id").set_index("site_id").family
    miss = [s for s in failed_sites if s not in set(prim.site_id)]
    if miss:
        prim = pd.concat([prim, pd.DataFrame({"site_id": miss, "family": [fam.get(s, "") for s in miss],
                                              "top1": 0.0, "top3": 0.0, "site_auc": np.nan})])
    return prim


# ---------------------------------------------------------------------------------------------
# NATIVE evaluation: every method ranks ITS OWN pockets; a site counts as found at rank k if one of
# the top-k pockets satisfies the hit rule. No mapping to our fpocket pockets, so tools whose pockets
# differ from fpocket's (P2Rank) are not penalised by the matching step.
# Hit rule (primary): pocket covers >= 50 % of the site residues AND >= 25 % of the pocket lies on
# the site.  Sensitivity: Jaccard(pocket, site) >= 0.2.
# ---------------------------------------------------------------------------------------------
def hit_main(P, S):
    i = len(P & S)
    return bool(S) and bool(P) and i / len(S) >= 0.5 and i / len(P) >= 0.25


def hit_jac(P, S):
    return bool(S) and bool(P) and len(P & S) / len(P | S) >= 0.2


def native_ranks(ranked, S, rule):
    """ranked: list of residue sets, best first -> (top1, top3) hit indicators."""
    hits = [rule(P, S) for P in ranked[:3]]
    return float(bool(hits[:1] and hits[0])), float(any(hits))


def site_residues():
    """(set, site_id) -> {(chain, resnum)} on the evaluated assembly. Stripped/holo: sites_mapped.csv.
    Partner structures: step 2z does not write sites_mapped.csv, so read each partner's info.json."""
    out = {}
    for setn, root in (("stripped", BASE / "processed" / "structures"), ("partner", BASE / "processed" / "structures_partner")):
        f = root / "sites_mapped.csv"
        recs = []
        if f.exists():
            m = pd.read_csv(f, low_memory=False)
            recs = [r for r in m.to_dict("records") if r.get("status") == "ok"]
        else:
            for info in root.glob("*/info.json"):
                try:
                    recs += [r for r in json.load(open(info)).get("sites", []) if r.get("status") == "ok"]
                except Exception:
                    pass
        for r in recs:
            res = r.get("residues_assembly")
            if isinstance(res, str):
                res = json.loads(res)
            if res:
                out[(setn, str(r["site_id"]))] = {(c, int(n)) for c, n, *_ in res}
    return out


def native_eval(man, sets, tabs, pocket_res, tool_results, sc_mlp):
    """-> {method: DataFrame(site_id, family, ran, top1, top3, top1_j, top3_j)}, match-rate lines"""
    SR = site_residues()
    # which set each primary site is evaluated in (partner where available, else stripped)
    psites = set(tabs["partner"].site_id.astype(str)) if "partner" in tabs else set()
    fam = pd.concat([tabs[n][["site_id", "family"]] for n in tabs]).drop_duplicates("site_id")
    fam = dict(zip(fam.site_id.astype(str), fam.family))
    sites = []
    for n in tabs:
        for sid, pdb in zip(tabs[n].site_id.astype(str), tabs[n].pdb_id):
            if (n == "partner") or (sid not in psites):
                sites.append((sid, n, pdb))
    keys = {}
    for k, setn, pdb, ss in zip(man.key, man["set"], man.pdb_id, man.sites):
        for sid in ss.split(","):
            keys[(SETMAP[setn][0], pdb, sid)] = k
    rows_of = {n: defaultdict(list) for n in sets}
    for n in sets:
        for i, p in enumerate(sets[n]["meta"].pdb_id):
            rows_of[n][p].append(i)

    def rank_ours(n, pdb, score):
        idx = [i for i in rows_of[n][pdb] if not np.isnan(score[i])]
        idx.sort(key=lambda i: -score[i])
        return [set(pocket_res[n][i]) for i in idx]

    out, match_lines = {}, []
    methods = {}
    if sc_mlp is not None:
        methods["MLP (B, ours)"] = ("ours", sc_mlp)
    methods["fpocket druggability"] = ("ours", {n: [sets[n]["fp"][:, 1].astype(float)] for n in sets})
    for name, r in tool_results.items():
        methods[name] = ("tool", r)
    for name, (kind, obj) in methods.items():
        recs = []
        n_tool_p = n_matched = 0
        for sid, n, pdb in sites:
            S = SR.get((n, sid))
            if not S:
                continue
            if kind == "ours":
                if not rows_of[n].get(pdb):
                    continue
                vals = []
                for sc in obj[n]:
                    ranked = rank_ours(n, pdb, sc)
                    if not ranked:
                        continue
                    vals.append(native_ranks(ranked, S, hit_main) + native_ranks(ranked, S, hit_jac))
                if not vals:
                    continue
                v = np.mean(vals, axis=0)
                recs.append((sid, fam.get(sid, ""), True, *v))
            else:
                k = keys.get((n, pdb, sid))
                if k is None:
                    continue
                r = obj.get(k)
                ok = bool(r) and r.get("status") == "ok" and bool(r.get("pockets"))
                if not ok:
                    recs.append((sid, fam.get(sid, ""), False, 0.0, 0.0, 0.0, 0.0))
                    continue
                chain = k.split("__")[-1]
                whole = bool(r.get("all_chains"))
                ranked = []
                for p in sorted(r["pockets"], key=lambda q: -float(q["score"])):
                    raw = p.get("residues", [])
                    if whole:
                        chs = {c for c, _ in S} | {c for i in rows_of[n].get(pdb, []) for c, _ in pocket_res[n][i]}
                        ranked.append({(c, x) for c in chs for x in tool_residues(raw, c)})
                    else:
                        ranked.append({(chain, x) for x in tool_residues(raw, chain)})
                # match rate: share of the tool's top-3 pockets that match one of our pockets (J >= 0.5)
                ours_sets = [set(pocket_res[n][i]) for i in rows_of[n].get(pdb, [])]
                for P in ranked[:3]:
                    n_tool_p += 1
                    if any(P and Q and len(P & Q) / len(P | Q) >= J_MIN for Q in ours_sets):
                        n_matched += 1
                recs.append((sid, fam.get(sid, ""), True, *native_ranks(ranked, S, hit_main), *native_ranks(ranked, S, hit_jac)))
        out[name] = pd.DataFrame(recs, columns=["site_id", "family", "ran", "top1", "top3", "top1_j", "top3_j"])
        if kind == "tool" and n_tool_p:
            match_lines.append(f"     {name:<12} {n_matched / n_tool_p:6.1%} of its top-3 pockets match an fpocket pocket (J >= {J_MIN})")
    return out, match_lines


def novelty_classes(path):
    """site_id -> (protein_new, family_new) relative to an older ASD release (AS file)."""
    old = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, on_bad_lines="skip")
    tid = old.iloc[:, 0].fillna("").str.split("_").str[0].str.upper()
    seen_t = set(tid) - {""}
    seen_u = set(old.get("pdb_uniprot", pd.Series(dtype=str)).dropna().str.upper())
    seen_p = set(old.get("allosteric_pdb", pd.Series(dtype=str)).dropna().str.upper())
    master = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", dtype=str, low_memory=False)
    t = master.get("target_id", pd.Series("", index=master.index)).fillna("").str.split("_").str[0].str.upper()
    ucol = next((c for c in ("uniprot", "uniprot_id", "pdb_uniprot") if c in master.columns), None)
    u = master[ucol].fillna("").str.upper() if ucol else pd.Series("", index=master.index)
    seen = t.isin(seen_t) | (u != "") & u.isin(seen_u)
    fam_seen = set(master.family[seen].dropna())
    out = {sid: (not sv, f not in fam_seen) for sid, sv, f in zip(master.site_id.astype(str), seen, master.family)}
    stats = (f"ASD 2019: {len(seen_t)} targets, {len(seen_p)} PDB entries; our sites seen-protein "
             f"{int(seen.sum())}/{len(master)}, used UniProt column: {ucol or 'none'}")
    return out, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asd2019", default=str(BASE / "tools" / "DeepAllo" / "source_data" / "ASD_Release_201909_AS.txt"))
    args = ap.parse_args()
    man = pd.read_csv(T / "manifest.csv")
    man = man[man.roles.str.contains("primary")]
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {n: s6.site_table(n, sets[n]["meta"], D / f"sites_{n}.csv") for n in sets}
    pocket_res = {}
    for setn, csv in SETMAP.values():
        if setn in sets:
            pk = pd.read_csv(D / csv, usecols=["pdb_id", "residues"], low_memory=False)
            pocket_res[setn] = [[(c, int(n)) for c, n, *_ in json.loads(r)] for r in pk.residues]
    site_key = {s: k for k, ss in zip(man.key, man.sites) for s in ss.split(",")}
    results, raw_results = {}, {}
    for name, fn in TOOLS.items():
        path = T / "results" / fn
        if not path.exists():
            s6.say(f"{name}: {path} not found - skipped")
            continue
        raw = json.load(open(path))
        raw_results[name] = raw
        sc, failed = scores_for_tool(raw, man, sets, pocket_res)
        failed_sites = [s for s, k in site_key.items() if k in failed]
        prim = primary_table(sets, tabs, sc, failed_sites)
        results[name] = (prim, set(failed_sites), len(failed))

    # our MLP (B) and the druggability baseline on the same primary sites
    ours = {}
    sc_mlp = {n: s6.collect("mlp", "B", n, sets[n]["meta"], [0, 1, 2]) for n in sets}
    if all(v is not None for v in sc_mlp.values()):
        tabs_r = []
        for r in range(3):
            tabs_r.append(primary_table(sets, tabs, {n: sc_mlp[n][r] for n in sets}, []))
        ours["MLP (B, ours)"] = pd.concat(tabs_r).groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean()
    ours["fpocket druggability"] = primary_table(sets, tabs, {n: sets[n]["fp"][:, 1].astype(float) for n in sets}, [])

    s6.say("=" * 100)
    s6.say("EXTERNAL TOOLS vs OURS (PRIMARY set: modulator-free partner where available, else stripped)")
    s6.say("=" * 100)
    s6.say(f"  {'method':<24}{'sites':>7}{'Top-1':>9}{'Top-3':>9}{'site AUROC':>12}   tool failed on")
    for name, (prim, fs, nk) in results.items():
        ran = prim[~prim.site_id.isin(fs)]
        if name not in SUBSET_TOOLS:  # PASSer was run on a random subset only: "all sites" is meaningless
          s6.say(f"  {name + ' (all sites)':<24}{len(prim):>7}{prim.top1.mean():>9.3f}{prim.top3.mean():>9.3f}"
               f"{prim.site_auc.mean():>12.3f}   {nk} chains / {len(fs)} sites")
        s6.say(f"  {name + ' (tool ran)':<24}{len(ran):>7}{ran.top1.mean():>9.3f}{ran.top3.mean():>9.3f}{ran.site_auc.mean():>12.3f}")
    for name, t in ours.items():
        s6.say(f"  {name:<24}{len(t):>7}{t.top1.mean():>9.3f}{t.top3.mean():>9.3f}{t.site_auc.mean():>12.3f}")
    s6.say("")
    s6.say("  Paired differences on the sites where the tool ran (95 % family-bootstrap interval):")
    for oname, ot in ours.items():
        o = ot.set_index("site_id")
        for name, (prim, fs, _) in results.items():
            t = prim[~prim.site_id.isin(fs)].set_index("site_id")
            common = o.index.intersection(t.index)
            for m in ("top1", "site_auc"):
                d = (o.loc[common, m] - t.loc[common, m])
                if d.notna().sum() == 0:
                    continue
                lo, hi = s6.boot(o.loc[common, "family"].to_numpy(), d.to_numpy())
                s6.say(f"     {oname} minus {name:<9} {m:<9} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  (sites {d.notna().sum()})")
    s6.say("")
    s6.say("=" * 100)
    s6.say("NATIVE EVALUATION: each method ranks its OWN pockets (no matching to our fpocket pockets)")
    s6.say("  hit = pocket covers >= 50 % of the site and >= 25 % of the pocket is site;  [J] = Jaccard >= 0.2")
    s6.say("=" * 100)
    nat, match_lines = native_eval(man, sets, tabs, pocket_res, raw_results,
                                   {n: sc_mlp[n] for n in sets} if all(v is not None for v in sc_mlp.values()) else None)
    s6.say(f"  {'method':<24}{'sites':>7}{'Top-1':>9}{'Top-3':>9}{'Top-1 [J]':>11}{'Top-3 [J]':>11}")
    for name, t in nat.items():
        if name in ours:
            rows = [(name, t)]
        else:
            rows = ([] if name in SUBSET_TOOLS else [(name + " (all sites)", t)]) + [(name + " (tool ran)", t[t.ran])]
        for lab, tt in rows:
            s6.say(f"  {lab:<24}{len(tt):>7}{tt.top1.mean():>9.3f}{tt.top3.mean():>9.3f}"
                   f"{tt.top1_j.mean():>11.3f}{tt.top3_j.mean():>11.3f}")
    s6.say("")
    s6.say("  Paired native differences on the sites where the tool ran (95 % family-bootstrap interval):")
    for oname in [o for o in ours if o in nat]:
        o = nat[oname].set_index("site_id")
        for name, t in nat.items():
            if name in ours:
                continue
            t = t[t.ran].set_index("site_id")
            common = o.index.intersection(t.index)
            for m in ("top1", "top3"):
                d = o.loc[common, m] - t.loc[common, m]
                if len(d):
                    lo, hi = s6.boot(o.loc[common, "family"].to_numpy(), d.to_numpy())
                    s6.say(f"     {oname} minus {name:<9} {m:<5} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  (sites {len(d)})")
    s6.say("")
    s6.say("  How often the tools' pockets can be matched to our fpocket pockets (low = the matched")
    s6.say("  evaluation above is unfair to that tool; then trust the native evaluation):")
    for line in match_lines:
        s6.say(line)
    s6.say("")
    if Path(args.asd2019).exists():
        nov, stats = novelty_classes(args.asd2019)
        s6.say("=" * 100)
        s6.say("TRAINING-OVERLAP CHECK (tools may have trained on ASD 2019 proteins; our MLP never sees a test family)")
        s6.say("  " + stats)
        s6.say("=" * 100)
        for label, idx in (("NEW proteins (not in ASD 2019)", 0), ("NEW families (no member in ASD 2019)", 1)):
            keep = {sid for sid, v in nov.items() if v[idx]}
            s6.say(f"  --- {label} ---")
            s6.say(f"  {'method':<24}{'sites':>7}{'Top-1':>9}{'Top-3':>9}{'AUROC':>8}  |{'native T1':>10}{'native T3':>10}")
            allm = {**{k: v[0][~v[0].site_id.isin(v[1])] for k, v in results.items()}, **ours}
            for name, t in allm.items():
                tt = t[t.site_id.astype(str).isin(keep)]
                nt = nat.get(name)
                nt = nt[nt.ran & nt.site_id.astype(str).isin(keep)] if nt is not None else None
                s6.say(f"  {name:<24}{len(tt):>7}{tt.top1.mean():>9.3f}{tt.top3.mean():>9.3f}{tt.site_auc.mean():>8.3f}  |"
                       f"{(nt.top1.mean() if nt is not None and len(nt) else float('nan')):>10.3f}"
                       f"{(nt.top3.mean() if nt is not None and len(nt) else float('nan')):>10.3f}")
            if "MLP (B, ours)" in ours:
                o = ours["MLP (B, ours)"].set_index("site_id")
                on = nat["MLP (B, ours)"].set_index("site_id") if "MLP (B, ours)" in nat else None
                for name, (prim, fs, _) in results.items():
                    t = prim[~prim.site_id.isin(fs)].set_index("site_id")
                    common = [c for c in o.index.intersection(t.index) if str(c) in keep]
                    if len(common) < 10:
                        continue
                    d = o.loc[common, "top1"] - t.loc[common, "top1"]
                    lo, hi = s6.boot(o.loc[common, "family"].to_numpy(), d.to_numpy())
                    line = f"     MLP minus {name:<9} Top-1 matched {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (sites {len(d)})"
                    if on is not None and name in nat:
                        tn = nat[name][nat[name].ran].set_index("site_id")
                        cn = [c for c in on.index.intersection(tn.index) if str(c) in keep]
                        if len(cn) >= 10:
                            dn = on.loc[cn, "top1"] - tn.loc[cn, "top1"]
                            lo2, hi2 = s6.boot(on.loc[cn, "family"].to_numpy(), dn.to_numpy())
                            line += f";  native {dn.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}] (sites {len(dn)})"
                    s6.say(line)
            s6.say("")
    else:
        s6.say(f"  (training-overlap check skipped: {args.asd2019} not found)")
    s6.say("  Caution: these tools may have been trained on some of these proteins (their training data are")
    s6.say("  ASD-derived), which would favour them; our MLP never saw a test family in training.")
    (T / "report_step7y.txt").write_text("\n".join(s6.LOG) + "\n")


if __name__ == "__main__":
    main()
