#!/usr/bin/env python
"""
STEP 10D: DOES AlloPlex WORK ON PREDICTED (AlphaFold) COMPLEX STRUCTURES?
=========================================================================
For a random set of multi-chain evaluation complexes, the same protein assembly is
(i) taken from the PDB (experimental, ligand-free = what the benchmark used) and
(ii) predicted with AlphaFold-Multimer (ColabFold) or the AlphaFold 3 server from the sequence
observed in that structure. AlloPlex scores both with the 3 released networks that never trained
on the complex's family (member_f<fold>_s*.pt, fold from folds.csv repeat 0), so the comparison is
held-out and paired: same sites, same model, only the structure source differs.

1) PICK the complexes and write the prediction inputs:
       python scripts/step10d_alphafold.py pick --n 60
   -> processed/alphafold/inputs/colabfold.csv            (ColabFold:  id,sequence  chains joined by ':')
      processed/alphafold/inputs/af3_day_*.json           (AlphaFold 3 server, 30 jobs per file = 1 day)
      processed/alphafold/inputs/mapping.json              (model position -> original chain / residue)
2) PREDICT, either
     ColabFold (GPU; needs access to the ColabFold MSA server or a local database):
       colabfold_batch --num-models 1 --model-type alphafold2_multimer_v3 \\
           processed/alphafold/inputs/colabfold.csv processed/alphafold/models
     or the AlphaFold 3 server (alphafoldserver.com, upload the json files, download the results
       and unzip every job folder into processed/alphafold/models/)
3) SCORE (GPU or CPU):
       python scripts/step10d_alphafold.py score
   -> processed/alphafold/results.csv, figures/data/s10d_alphafold.csv, report printed
Site labels on both structures use the same rule: positive pocket = the pocket with the highest
residue Jaccard to the site (>= 0.1); pockets touching any annotated site but not chosen are left out;
a site whose pocket is not found counts as a miss.
"""
import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
A = BASE / "processed" / "alphafold"
STRUCT = BASE / "processed" / "structures"
J_MIN = 0.1


def pick(args):
    import alloplex_predict as pp
    st = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    st = st[(st.status == "ok") & (st.detected.astype(str).isin(["True", "true", "1"]))]
    st = st[~st.complex_kind.astype(str).isin(["monomer", "no protein chain >= 50 aa"])]
    rows = []
    for pdb, g in st.groupby("pdb_id"):
        path = STRUCT / pdb / "stripped.pdb"
        if not path.exists():
            continue
        ch = pp.chains_of(path)
        n = sum(len(v) for v in ch.values())
        if 2 <= len(ch) <= args.max_chains and n <= args.max_residues:
            rows.append({"pdb_id": pdb, "n_chains": len(ch), "n_res": n, "kind": g.complex_kind.iloc[0],
                         "family": str(g.family.iloc[0]), "sites": ",".join(g.site_id.astype(str))})
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise SystemExit("no eligible complexes")
    pool = pool.sample(frac=1, random_state=args.seed).drop_duplicates("family")  # one complex per family
    take = []
    for k, g in pool.groupby("kind"):  # proportional to complex type, at least 3 of each
        take.append(g.head(max(3, int(round(args.n * len(g) / len(pool))))))
    take = pd.concat(take).sample(frac=1, random_state=args.seed).head(args.n).reset_index(drop=True)
    inp = A / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    mapping, cf, jobs, fixed = {}, [], [], []
    for r in take.itertuples():
        raw = pp.chains_of(STRUCT / r.pdb_id / "stripped.pdb")
        # non-standard residues (X) are not accepted by AlphaFold: left out of sequence AND mapping
        ch = {c: [x for x in res if x[2] != "X"] for c, res in raw.items()}
        ch = {c: res for c, res in ch.items() if res}
        if sum(len(v) for v in ch.values()) != sum(len(v) for v in raw.values()):
            fixed.append(r.pdb_id)
        order = list(ch)
        mapping[r.pdb_id] = {"chains": order, "residues": {c: [[n, ic] for n, ic, _ in ch[c]] for c in order}}
        seqs = ["".join(aa for _, _, aa in ch[c]) for c in order]
        cf.append({"id": r.pdb_id, "sequence": ":".join(seqs)})
        jobs.append({"name": r.pdb_id, "modelSeeds": [], "dialect": "alphafoldserver", "version": 1,
                     "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in seqs]})
    json.dump(mapping, open(inp / "mapping.json", "w"))
    pd.DataFrame(cf).to_csv(inp / "colabfold.csv", index=False)
    for i in range(0, len(jobs), 30):  # AlphaFold Server: 30 jobs per day
        json.dump(jobs[i:i + 30], open(inp / f"af3_day_{i // 30 + 1}.json", "w"), indent=1)
    if fixed:
        json.dump([j for j in jobs if j["name"] in fixed], open(inp / "af3_fixed_jobs.json", "w"), indent=1)
        print(f"non-standard residues (X) removed in {len(fixed)} complexes: {', '.join(fixed)} "
              f"-> af3_fixed_jobs.json")
    take.to_csv(inp / "complexes.csv", index=False)
    print(f"eligible complexes (2-{args.max_chains} chains, <= {args.max_residues} residues, one per family): "
          f"{len(pool)}; picked {len(take)}: " + ", ".join(f"{k} {v}" for k, v in take.kind.value_counts().items()))
    print(f"residues per complex: median {int(take.n_res.median())}, max {take.n_res.max()}; sites {take.sites.str.count(',').add(1).sum()}")
    print(f"inputs in {inp}  (sequence = residues observed in the PDB structure; missing loops are not modelled)")


def find_model(pdb, mdir):
    cands = [p for p in glob.glob(str(mdir / "**" / "*"), recursive=True)
             if pdb.lower() in Path(p).name.lower() and p.lower().endswith((".pdb", ".cif"))]
    for pat in (r"rank_001", r"rank_1\b", r"model_0\.cif$", r"ranked_0"):
        hit = [p for p in cands if re.search(pat, Path(p).name.lower())]
        if hit:
            return Path(sorted(hit)[0])
    return Path(cands[0]) if cands else None


def iptm_of(model):
    """ipTM from the ColabFold scores json or the AF3 summary_confidences json next to the model."""
    for pat in (str(model.parent / "*scores_rank_001*.json"), str(model.parent / "*summary_confidences_0.json")):
        for f in glob.glob(pat):
            try:
                j = json.load(open(f))
                v = j.get("iptm")
                if v is not None:
                    return float(v)
            except Exception:
                pass
    return np.nan


def renumber(model, mapping, out):
    """AlphaFold model (chains A, B, ... in input order; residues 1..L) -> original chain IDs and numbers."""
    import alloplex_predict as pp
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.pdb")
    pp.clean_structure(model, tmp)
    lines, order = [], []
    for l in open(tmp):
        if l.startswith("ATOM"):
            if l[21] not in order:
                order.append(l[21])
            lines.append(l)
    if len(order) != len(mapping["chains"]):
        raise RuntimeError(f"model has {len(order)} chains, expected {len(mapping['chains'])}")
    out_lines = []
    for l in lines:
        k = order.index(l[21])
        c = mapping["chains"][k]
        res = mapping["residues"][c]
        i = int(l[22:26]) - 1
        if not 0 <= i < len(res):
            raise RuntimeError(f"residue {i + 1} beyond the mapped sequence of chain {c}")
        n, ic = res[i]
        out_lines.append(f"{l[:21]}{c}{int(n):>4}{(ic or ' ')[:1]}{l[27:]}")
    out.write_text("".join(out_lines) + "END\n")
    tmp.unlink()
    return out


def site_ranks(P, sites):
    """sites: {site_id: set((chain, resnum))} -> {site_id: (rank or inf, n_candidates)}"""
    pres = [{(c, int(n)) for c, n, _ in r} for r in P.residues]
    sc = P.alloplex_score.to_numpy()
    pos, touched = {}, set()
    for sid, S in sites.items():
        js = [len(p & S) / len(p | S) if p else 0 for p in pres]
        for i, p in enumerate(pres):
            if p & S:
                touched.add(i)
        b = int(np.argmax(js))
        if js[b] >= J_MIN:
            pos[sid] = b
    keep_neg = [i for i in range(len(P)) if i not in touched]
    out = {}
    for sid in sites:
        if sid not in pos:
            out[sid] = (np.inf, len(keep_neg) + 1)
            continue
        s = sc[pos[sid]]
        neg = sc[keep_neg]
        out[sid] = (1 + (neg > s).sum() + 0.5 * (neg == s).sum(), len(keep_neg) + 1)
    return out


def score(args):
    import torch
    import alloplex_predict as pp
    import step6x_train_models as s6
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    inp = A / "inputs"
    take = pd.read_csv(inp / "complexes.csv", dtype={"family": str})
    mapping = json.load(open(inp / "mapping.json"))
    mdir = Path(args.models_dir)
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    fold = folds[folds.repeat == 0].set_index("family").fold
    sm = pd.read_csv(STRUCT / "sites_mapped.csv", low_memory=False)
    sm["site_id"] = sm.site_id.astype(str)
    site_res = {r.site_id: {(c, int(n)) for c, n, *_ in json.loads(r.residues_assembly)}
                for r in sm.itertuples() if isinstance(r.residues_assembly, str)}
    model_root = next((BASE / "models" / m for m in ("alloplex_v1", "prism_v1") if (BASE / "models" / m).exists()))
    esm = pp.ESM(device)
    fpocket = shutil.which("fpocket")
    members = {}
    rows = []
    work = A / "work"
    for r in take.itertuples():
        model = find_model(r.pdb_id, mdir)
        if model is None:
            print(f"  {r.pdb_id}: no model found in {mdir} - skipped")
            continue
        k = int(fold.get(str(r.family), -1))
        if k < 0:
            print(f"  {r.pdb_id}: family not in folds.csv - skipped")
            continue
        if k not in members:
            members[k] = pp.load_members(model_root, device, f"member_f{k}_s*.pt")
        sites = {s: site_res[s] for s in r.sites.split(",") if s in site_res}
        try:
            ren = renumber(model, mapping[r.pdb_id], work / r.pdb_id / "af_renumbered.pdb")
            P_af = pp.score_structure(ren, work / r.pdb_id / "af", fpocket, esm, members[k], device)
            P_ex = pp.score_structure(STRUCT / r.pdb_id / "stripped.pdb", work / r.pdb_id / "exp", fpocket, esm,
                                      members[k], device)
        except Exception as e:
            print(f"  {r.pdb_id}: failed - {type(e).__name__}: {e}")
            continue
        plddt = {}
        for l in open(ren):
            if l.startswith("ATOM") and l[12:16].strip() == "CA":
                plddt[(l[21], int(l[22:26]))] = float(l[60:66])
        iptm = iptm_of(model)
        ra, re_ = site_ranks(P_af, sites), site_ranks(P_ex, sites)
        for sid in sites:
            pl = [plddt[x] for x in sites[sid] if x in plddt]
            rows.append({"pdb_id": r.pdb_id, "site_id": sid, "family": r.family, "kind": r.kind, "fold": k,
                         "rank_exp": re_[sid][0], "rank_af": ra[sid][0], "n_pockets_exp": len(P_ex),
                         "n_pockets_af": len(P_af), "site_plddt": np.mean(pl) if pl else np.nan, "iptm": iptm,
                         "model_file": model.name})
        print(f"  {r.pdb_id}: {len(sites)} sites; best rank experimental {min(v[0] for v in re_.values())}, "
              f"AlphaFold {min(v[0] for v in ra.values())}; ipTM {iptm:.2f}", flush=True)
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("nothing scored")
    for s in ("exp", "af"):
        d[f"top1_{s}"] = (d[f"rank_{s}"] <= 1).astype(float)
        d[f"top3_{s}"] = (d[f"rank_{s}"] <= 3).astype(float)
    d.to_csv(A / "results.csv", index=False)
    (BASE / "figures" / "data").mkdir(parents=True, exist_ok=True)
    d.to_csv(BASE / "figures" / "data" / "s10d_alphafold.csv", index=False)
    print(f"\nRESULT ({len(d)} sites, {d.pdb_id.nunique()} complexes, {d.family.nunique()} families; held-out members)")
    for m in ("top1", "top3"):
        e, a = d[f"{m}_exp"], d[f"{m}_af"]
        lo, hi = s6.boot(d.family.to_numpy(), (a - e).to_numpy())
        print(f"  {m.replace('top', 'Top-')}: experimental {e.mean():.3f}, AlphaFold {a.mean():.3f}, "
              f"difference {(a - e).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for lab, g in (("site pLDDT >= 70", d[d.site_plddt >= 70]), ("site pLDDT < 70", d[d.site_plddt < 70]),
                   ("ipTM >= 0.6", d[d.iptm >= 0.6]), ("ipTM < 0.6", d[d.iptm < 0.6])):
        if len(g):
            print(f"  {lab:<18} n={len(g):>4}: Top-1 experimental {g.top1_exp.mean():.3f}, AlphaFold {g.top1_af.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("pick")
    a.add_argument("--n", type=int, default=60)
    a.add_argument("--seed", type=int, default=11)
    a.add_argument("--max-residues", type=int, default=1500)
    a.add_argument("--max-chains", type=int, default=6)
    b = sp.add_parser("score")
    b.add_argument("--models-dir", default=str(A / "models"))
    b.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    {"pick": pick, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
