#!/usr/bin/env python
"""
STEP 12: WHOLE ASSEMBLY VERSUS ISOLATED CHAINS (reviewer concern 7)
===================================================================
Question: does running the pipeline on the whole protein-protein interaction complex help, compared
with treating every chain separately (as single-chain predictors do)?

For every multi-chain structure of the benchmark (stripped, ligand-free; each scored with the released
networks that never trained on its family):
  ASSEMBLY      fpocket + ESM-2 + AlloPlex on the whole assembly (as in the paper)
  CHAINS        the same pipeline on each chain alone; the pockets of all chains are pooled
                (same protein material, but pockets cannot span chains and fpocket sees no partner)
  OWN CHAIN     as CHAINS, but only the pockets of the chain(s) that carry the site are ranked
                (what a single-chain tool given the right chain would see; fewer competitors)
Same label rule in every arm: positive = best-Jaccard pocket (>= 0.1); pockets touching the site or the
same residues on identical chains are not negatives. The fpocket druggability score is ranked too.

RUN (from ~/allosite, env allosite, GPU; resumable - finished structures are skipped):
    nohup python scripts/step12_isolated_chains.py run > logs_chains.txt 2>&1 &
    python scripts/step12_isolated_chains.py run --per-family 3     # quicker: at most 3 structures per family
    python scripts/step12_isolated_chains.py report                 # tables (can be run any time)
Output: processed/dataset_v2/isolated_chains/ (one json per structure), figures/data/s12_*.csv,
        figures/data/report_step12.txt
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
STRUCT = BASE / "processed" / "structures"
OUT = D / "isolated_chains"
DATA = BASE / "figures" / "data"
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rank_rule(P, site, ent, col, keep=None):
    """benchmark rule (as step 10e): -> (rank or inf, best Jaccard). keep: optional row mask"""
    if keep is not None:
        P = P[keep]
    if P.empty:
        return np.inf, 0.0
    sc_ent = {(ent.get(c, c), n) for c, n in site}
    pres = [{(c, int(n)) for c, n, *_ in r} for r in P.residues]
    js = [len(p & site) / len(p | site) if p else 0 for p in pres]
    b = int(np.argmax(js))
    touched = {i for i, p in enumerate(pres) if (p & site) or ({(ent.get(c, c), n) for c, n in p} & sc_ent)}
    neg = P[col].to_numpy()[[i for i in range(len(P)) if i not in touched]]
    if js[b] < 0.1:
        return np.inf, js[b]
    s = P[col].iloc[b]
    return float(1 + (neg > s).sum() + 0.5 * (neg == s).sum()), js[b]


def split_chains(clean_pdb, outdir):
    lines = open(clean_pdb).read().splitlines()
    chains = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and l[21] not in chains:
            chains.append(l[21])
    files = {}
    for c in chains:
        f = outdir / f"chain_{c if c.strip() else '_'}.pdb"
        f.write_text("\n".join(l for l in lines if l.startswith(("ATOM", "HETATM")) and l[21] == c) + "\nEND\n")
        files[c] = f
    return files


def to_json(P):
    return [{"residues": [[c, int(n)] for c, n, *_ in r], "alloplex": float(a), "drug": float(d)}
            for r, a, d in zip(P.residues, P.alloplex_score, P.druggability_score)]


def run(args):
    import torch
    import alloplex_predict as pp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    st = st[st.status == "ok"]
    st["site_id"] = st.site_id.astype(str)
    sm = pd.read_csv(STRUCT / "sites_mapped.csv", low_memory=False)
    sm["site_id"] = sm.site_id.astype(str)
    site_res = {r.site_id: [[c, int(n)] for c, n, *_ in json.loads(r.residues_assembly)]
                for r in sm.itertuples() if isinstance(r.residues_assembly, str)}
    meta = pd.read_csv(s6.B / "meta_stripped.csv", usecols=["pdb_id", "family"], keep_default_na=False)
    fam = meta.drop_duplicates("pdb_id").set_index("pdb_id").family.astype(str)
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    fold = folds[folds.repeat == 0].set_index("family").fold
    pdbs = st.pdb_id.drop_duplicates().tolist()
    pdbs = [p for p in pdbs if (STRUCT / p / "stripped.pdb").exists()]
    multi = []
    for p in pdbs:
        ch = {l[21] for l in open(STRUCT / p / "stripped.pdb") if l.startswith("ATOM")}
        if len(ch) >= 2:
            multi.append(p)
    if args.per_family:
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"p": multi, "f": [fam.get(p, "") for p in multi]})
        multi = [p for _, g in df.groupby("f") for p in g.p.iloc[rng.permutation(len(g))[:args.per_family]]]
    say(f"{len(multi)} multi-chain structures to process (of {len(pdbs)} structures with sites)")
    esm = pp.ESM(device)
    fpocket = shutil.which("fpocket")
    mdir = next((BASE / "models" / m for m in ("alloplex_v1", "prism_v1") if (BASE / "models" / m).exists()))
    members = {}
    t0 = time.time()
    done = 0
    for i, p in enumerate(multi):
        f = OUT / f"{p}.json"
        if f.exists():
            continue
        k = int(fold.get(fam.get(p, ""), -1))
        if k < 0:
            continue
        if k not in members:
            members[k] = pp.load_members(mdir, device, f"member_f{k}_s*.pt")
        work = OUT / "work" / p
        try:
            P_a = pp.score_structure(STRUCT / p / "stripped.pdb", work / "assembly", fpocket, esm, members[k], device)
            parts = []
            for c, cf in split_chains(work / "assembly" / "clean.pdb", work).items():
                try:
                    Pc = pp.score_structure(cf, work / f"ch_{c}", fpocket, esm, members[k], device)
                    parts.append(Pc)
                except Exception:  # noqa: BLE001  (very short chains: no pockets)
                    continue
            P_c = pd.concat(parts, ignore_index=True) if parts else P_a.iloc[0:0]
        except Exception as e:  # noqa: BLE001
            say(f"  {p}: failed - {type(e).__name__}: {e}")
            continue
        sites = {s: site_res[s] for s in st[st.pdb_id == p].site_id if s in site_res}
        json.dump({"pdb_id": p, "family": fam.get(p, ""), "fold": k, "sites": sites,
                   "assembly": to_json(P_a), "chains": to_json(P_c)}, open(f, "w"))
        shutil.rmtree(work, ignore_errors=True)
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            say(f"  {i + 1}/{len(multi)} structures; {el / done:.1f} s each; ~{el / done * (len(multi) - i - 1) / 3600:.1f} h left")
    say("run finished; now: python scripts/step12_isolated_chains.py report")


def report(args):
    files = sorted(OUT.glob("*.json"))
    if not files:
        raise SystemExit("nothing yet - run first")
    st = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    st["site_id"] = st.site_id.astype(str)
    st = st.drop_duplicates("site_id").set_index("site_id")
    rows = []
    for f in files:
        r = json.load(open(f))
        cm_f = STRUCT / r["pdb_id"] / "chainmap.json"
        cm = json.load(open(cm_f)) if cm_f.exists() else {}
        ent = {c: v.get("entity") if isinstance(v, dict) else v for c, v in cm.items()}
        arms = {}
        for arm in ("assembly", "chains"):
            P = pd.DataFrame(r[arm])
            if P.empty:
                P = pd.DataFrame(columns=["residues", "alloplex", "drug"])
            arms[arm] = P
        for sid, res in r["sites"].items():
            site = {(c, int(n)) for c, n in res}
            site_chains = {c for c, _ in site}
            row = {"site_id": sid, "pdb_id": r["pdb_id"], "family": r["family"], "n_site_chains": len(site_chains),
                   "complex_kind": st.complex_kind.get(sid, "") if "complex_kind" in st else "",
                   "frac_interface": st.site_frac_interface_5.get(sid, np.nan) if "site_frac_interface_5" in st else np.nan}
            for arm, P in arms.items():
                row[f"n_pockets_{arm}"] = len(P)
                for col in ("alloplex", "drug"):
                    rk, j = rank_rule(P, site, ent, col)
                    row[f"rank_{col}_{arm}"], row[f"j_{arm}"] = rk, j
            P = arms["chains"]
            if len(P):
                own = np.array([any(c in site_chains for c, _ in rr) for rr in P.residues])
                row["n_pockets_own"] = int(own.sum())
                for col in ("alloplex", "drug"):
                    row[f"rank_{col}_own"], _ = rank_rule(P, site, ent, col, keep=own)
            rows.append(row)
    d = pd.DataFrame(rows)
    for c in [c for c in d.columns if c.startswith("rank_")]:
        d["top1" + c[4:]] = (d[c] <= 1).astype(float)
        d["top3" + c[4:]] = (d[c] <= 3).astype(float)
    DATA.mkdir(parents=True, exist_ok=True)
    d.to_csv(DATA / "s12_isolated_chains.csv", index=False)
    say(f"STEP 12: {d.pdb_id.nunique()} multi-chain structures, {len(d)} sites, {d.family.nunique()} families")
    say(f"  pockets per structure: assembly {d.groupby('pdb_id').n_pockets_assembly.first().median():.0f}, "
        f"pooled chains {d.groupby('pdb_id').n_pockets_chains.first().median():.0f} (median)")
    say(f"  site found by fpocket (J >= 0.1): assembly {np.isfinite(d.rank_alloplex_assembly).mean():.1%}, "
        f"isolated chains {np.isfinite(d.rank_alloplex_chains).mean():.1%}; median best Jaccard "
        f"{d.j_assembly.median():.2f} vs {d.j_chains.median():.2f}")
    groups = [("all multi-chain sites", d), ("sites on one chain", d[d.n_site_chains == 1]),
              ("sites spanning >= 2 chains", d[d.n_site_chains >= 2])]
    fi = pd.to_numeric(d.frac_interface, errors="coerce")
    groups += [("site away from the interface", d[fi == 0]), ("site partly/mostly at the interface", d[fi > 0])]
    for kind in sorted(k for k in d.complex_kind.dropna().unique() if k):
        groups.append((f"complex type: {kind}", d[d.complex_kind == kind]))
    out = []
    for lab, g in groups:
        if len(g) < 10:
            continue
        say(f"\n  {lab}: n={len(g)} sites, {g.family.nunique()} families")
        for col, name in (("alloplex", "AlloPlex"), ("drug", "Druggability")):
            for k in ("top1", "top3"):
                a = g[f"{k}_{col}_assembly"]
                c = g[f"{k}_{col}_chains"]
                ma, la, ha = mean_ci(g.family, a)
                mc, lc, hc = mean_ci(g.family, c)
                md, ld, hd = mean_ci(g.family, a - c)
                txt = (f"    {name:<12} {k.replace('top', 'Top-')}: assembly {ma:.3f} [{la:.3f}, {ha:.3f}]; "
                       f"pooled chains {mc:.3f} [{lc:.3f}, {hc:.3f}]; assembly minus chains {md:+.3f} [{ld:+.3f}, {hd:+.3f}]")
                if f"{k}_{col}_own" in g:
                    o = g[f"{k}_{col}_own"]
                    mo = o.mean()
                    txt += f"; own chain only {mo:.3f}"
                say(txt)
                out.append({"group": lab, "method": name, "metric": k, "sites": len(g), "assembly": ma,
                            "assembly_lo": la, "assembly_hi": ha, "chains": mc, "chains_lo": lc, "chains_hi": hc,
                            "diff": md, "diff_lo": ld, "diff_hi": hd,
                            "own_chain": g[f"{k}_{col}_own"].mean() if f"{k}_{col}_own" in g else np.nan})
    pd.DataFrame(out).to_csv(DATA / "s12_summary.csv", index=False)
    (DATA / "report_step12.txt").write_text("\n".join(LOG) + "\n")
    say("\nsaved figures/data/s12_isolated_chains.csv, s12_summary.csv, report_step12.txt")


def mean_ci(fam, v):
    v = np.asarray(v, float)
    lo, hi = s6.boot(np.asarray(fam), v)
    return float(np.nanmean(v)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("run")
    a.add_argument("--per-family", type=int, default=0, help="at most this many structures per family (0 = all)")
    sp.add_parser("report")
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
