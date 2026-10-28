#!/usr/bin/env python
"""
STEP 13D: A LARGER ALPHAFOLD 3 EVALUATION (supervisor point 3)
==============================================================
Extends the 60-complex AlphaFold 3 comparison (step 10d) with more complexes chosen by the SAME rules
(multi-chain, 2-6 chains, <= 1,500 residues, fpocket detected the site, one complex per family,
proportional to complex type), excluding every family already used, so the combined set stays one
complex per family. Scoring is step 10d unchanged (held-out released networks, same labels).

1) PICK and write the AlphaFold Server jobs (30 per file = one day of the server quota):
       python scripts/step13d_alphafold_more.py pick --n 140
   -> processed/alphafold_ext/inputs/{complexes.csv, mapping.json, af3_day_*.json}
2) PREDICT on alphafoldserver.com: upload one af3_day_*.json per day, download each job and unzip it into
   processed/alphafold_ext/models/
3) SCORE the new complexes and COMBINE with the original 60:
       python scripts/step13d_alphafold_more.py score
       python scripts/step13d_alphafold_more.py combine
   -> processed/alphafold_ext/results.csv, figures/data/s13d_alphafold_combined.csv, report_step13d.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step10d_alphafold as s10d  # noqa: E402

BASE = s10d.BASE
OLD = BASE / "processed" / "alphafold"
EXT = BASE / "processed" / "alphafold_ext"
DATA = BASE / "figures" / "data"


def pick(args):
    import alloplex_predict as pp
    used = pd.read_csv(OLD / "inputs" / "complexes.csv", dtype={"family": str})
    used_fam = set(used.family.astype(str))
    st = pd.read_csv(s10d.D / "sites_stripped.csv", low_memory=False)
    st = st[(st.status == "ok") & (st.detected.astype(str).isin(["True", "true", "1"]))]
    st = st[~st.complex_kind.astype(str).isin(["monomer", "no protein chain >= 50 aa"])]
    st = st[~st.family.astype(str).isin(used_fam)]
    rows = []
    for pdb, g in st.groupby("pdb_id"):
        path = s10d.STRUCT / pdb / "stripped.pdb"
        if not path.exists():
            continue
        ch = pp.chains_of(path)
        n = sum(len(v) for v in ch.values())
        if 2 <= len(ch) <= args.max_chains and n <= args.max_residues:
            rows.append({"pdb_id": pdb, "n_chains": len(ch), "n_res": n, "kind": g.complex_kind.iloc[0],
                         "family": str(g.family.iloc[0]), "sites": ",".join(g.site_id.astype(str))})
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise SystemExit("no eligible complexes left")
    pool = pool.sample(frac=1, random_state=args.seed).drop_duplicates("family")
    take = []
    for k, g in pool.groupby("kind"):
        take.append(g.head(max(3, int(round(args.n * len(g) / len(pool))))))
    take = pd.concat(take).sample(frac=1, random_state=args.seed).head(args.n).reset_index(drop=True)
    inp = EXT / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    mapping, jobs = {}, []
    for r in take.itertuples():
        raw = pp.chains_of(s10d.STRUCT / r.pdb_id / "stripped.pdb")
        ch = {c: [x for x in res if x[2] != "X"] for c, res in raw.items()}
        ch = {c: res for c, res in ch.items() if res}
        order = list(ch)
        mapping[r.pdb_id] = {"chains": order, "residues": {c: [[n, ic] for n, ic, _ in ch[c]] for c in order}}
        jobs.append({"name": r.pdb_id, "modelSeeds": [], "dialect": "alphafoldserver", "version": 1,
                     "sequences": [{"proteinChain": {"sequence": "".join(aa for _, _, aa in ch[c]), "count": 1}}
                                   for c in order]})
    json.dump(mapping, open(inp / "mapping.json", "w"))
    for i in range(0, len(jobs), args.per_day):
        json.dump(jobs[i:i + args.per_day], open(inp / f"af3_day_{i // args.per_day + 1}.json", "w"), indent=1)
    take.to_csv(inp / "complexes.csv", index=False)
    print(f"families already used: {len(used_fam)}; eligible new complexes (one per family): {len(pool)}; "
          f"picked {len(take)}: " + ", ".join(f"{k} {v}" for k, v in take.kind.value_counts().items()))
    print(f"{int(np.ceil(len(jobs) / args.per_day))} job files of up to {args.per_day} jobs in {inp} "
          f"(upload one per day; unzip results into {EXT / 'models'})")


def score(args):
    import shutil
    # step 10d also writes figures/data/s10d_alphafold.csv (the original 60, used by Figure S5):
    # keep a copy and restore it afterwards so the original results are never overwritten
    orig = DATA / "s10d_alphafold.csv"
    keep = DATA / "s10d_alphafold.original.csv"
    if orig.exists() and not keep.exists():
        shutil.copy(orig, keep)
    s10d.A = EXT                                       # step 10d scoring, unchanged, on the new inputs
    args.models_dir = str(EXT / "models") if args.models_dir is None else args.models_dir
    try:
        s10d.score(args)
    finally:
        if keep.exists():
            shutil.copy(keep, orig)
            print(f"restored the original {orig.name} (extension results are in {EXT / 'results.csv'})")


def combine(args):
    import step6x_train_models as s6
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)
    d0 = pd.read_csv(OLD / "results.csv", dtype={"family": str}).assign(batch="original")
    d1 = pd.read_csv(EXT / "results.csv", dtype={"family": str}).assign(batch="extension")
    d = pd.concat([d0, d1], ignore_index=True)
    d.to_csv(DATA / "s13d_alphafold_combined.csv", index=False)
    say(f"combined: {len(d)} sites, {d.pdb_id.nunique()} complexes, {d.family.nunique()} families "
        f"(original {d0.pdb_id.nunique()}, extension {d1.pdb_id.nunique()})")
    for lab, g in (("all", d), ("original 60", d0), ("extension", d1)):
        for m in ("top1", "top3"):
            e, a = g[f"{m}_exp"], g[f"{m}_af"]
            lo, hi = s6.boot(g.family.to_numpy(), (a - e).to_numpy())
            say(f"  {lab:<12} {m.replace('top', 'Top-')}: experimental {e.mean():.3f}, AlphaFold {a.mean():.3f}, "
                f"difference {(a - e).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(g)})")
    say("Stratified (combined; still post hoc, but larger):")
    for lab, g in (("site pLDDT >= 70", d[d.site_plddt >= 70]), ("site pLDDT < 70", d[d.site_plddt < 70]),
                   ("ipTM >= 0.6", d[d.iptm >= 0.6]), ("ipTM < 0.6", d[d.iptm < 0.6])):
        if len(g):
            lo, hi = s6.boot(g.family.to_numpy(), (g.top1_af - g.top1_exp).to_numpy())
            say(f"  {lab:<18} n={len(g):>4}: Top-1 experimental {g.top1_exp.mean():.3f}, AlphaFold {g.top1_af.mean():.3f}, "
                f"difference {(g.top1_af - g.top1_exp).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (DATA / "report_step13d.txt").write_text("\n".join(log) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pick")
    a.add_argument("--n", type=int, default=140)
    a.add_argument("--seed", type=int, default=12)
    a.add_argument("--max-residues", type=int, default=1500)
    a.add_argument("--max-chains", type=int, default=6)
    a.add_argument("--per-day", type=int, default=30)
    b = sub.add_parser("score")
    b.add_argument("--models-dir", default=None)
    b.add_argument("--cpu", action="store_true")
    sub.add_parser("combine")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    {"pick": pick, "score": score, "combine": combine}[args.cmd](args)


if __name__ == "__main__":
    main()
