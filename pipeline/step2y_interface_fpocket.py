#!/usr/bin/env python
"""
STEP 2Y: INTERFACES AND POCKETS FOR THE NEW DATASET
===================================================
For every structure built by step 2x (status ok in processed/structures/structures.csv):

  1. SOLVENT ACCESSIBILITY (Shrake-Rupley, probe 1.4 A, 100 points per atom,
     radii C 1.70, N 1.55, O 1.52, S 1.80, Se 1.90, other 1.80) of every residue
       - in the full assembly            -> sasa_complex
       - with its own chain alone        -> sasa_alone
       - delta = sasa_alone - sasa_complex  (the surface buried by OTHER chains)
     and which other chains bury it (partner_chains). All chain pairs are
     covered at once. A residue is "interface" at cutoff c if delta > c (the
     plan's primary cutoff is 5 A^2; 1, 10, 20 are sensitivity checks), so
     nothing is fixed here: every cutoff can be applied later.
     -> processed/structures/<PDB>/residues.csv

  2. FPOCKET (default settings) on stripped.pdb (ligand-free), and with --holo
     also on holo.pdb (ligands present, for the leakage comparison).
     -> processed/structures/<PDB>/pockets_stripped.csv (and pockets_holo.csv):
        pocket number, every fpocket descriptor, lining residues, centre

RUN (from ~/allosite; needs fpocket on PATH, numpy, scipy, pandas):
    python scripts/step2y_interface_fpocket.py --limit 20              # trial
    python scripts/step2y_interface_fpocket.py --workers 8 --holo      # everything (resumes if stopped)
Summary: processed/structures/step2y_summary.csv
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "processed" / "structures"
PROBE = 1.4
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90}
DEFAULT_R = 1.80


def sphere_points(n=100):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


UNIT = sphere_points()


def read_pdb(path, hetatm=False):
    rows = []
    for line in open(path):
        rec = line[:6]
        if rec == "ATOM  " or (hetatm and rec == "HETATM"):
            el = line[76:78].strip().upper() or line[12:14].strip().upper()
            rows.append((line[21], int(line[22:26]), line[26].strip(), line[17:20].strip(), line[12:16].strip(), el,
                         float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return pd.DataFrame(rows, columns=["chain", "resnum", "icode", "resname", "atom", "el", "x", "y", "z"])


def sasa(atoms):
    """Per-atom SASA in the complex and with the atom's own chain alone, plus burying chains."""
    xyz = atoms[["x", "y", "z"]].to_numpy()
    r = atoms["el"].map(RADII).fillna(DEFAULT_R).to_numpy() + PROBE
    chains = atoms["chain"].to_numpy()
    n = len(xyz)
    tree = cKDTree(xyz)
    neigh = tree.query_ball_point(xyz, r + r.max())
    area_unit = 4 * np.pi * r ** 2 / len(UNIT)
    s_cplx, s_alone = np.zeros(n), np.zeros(n)
    partners = [set() for _ in range(n)]
    for i in range(n):
        nb = np.array([j for j in neigh[i] if j != i], dtype=int)
        pts = xyz[i] + r[i] * UNIT
        if len(nb) == 0:
            s_cplx[i] = s_alone[i] = len(UNIT) * area_unit[i]
            continue
        d2 = ((pts[:, None, :] - xyz[nb][None, :, :]) ** 2).sum(-1)
        inside = d2 < (r[nb] ** 2)[None, :]          # point x neighbour
        same = chains[nb] == chains[i]
        bur_all = inside.any(1)
        bur_same = inside[:, same].any(1) if same.any() else np.zeros(len(UNIT), bool)
        s_cplx[i] = (~bur_all).sum() * area_unit[i]
        s_alone[i] = (~bur_same).sum() * area_unit[i]
        other_only = bur_all & ~bur_same
        if other_only.any():
            hit = inside[other_only][:, ~same].any(0)
            partners[i] = set(chains[nb][~same][hit])
    return s_cplx, s_alone, partners


def residue_table(atoms):
    sc, sa, partners = sasa(atoms)
    a = atoms.assign(sasa_complex=sc, sasa_alone=sa)
    a["partners"] = [",".join(sorted(p)) for p in partners]
    g = a.groupby(["chain", "resnum", "icode"], sort=False)
    res = g.agg(resname=("resname", "first"), sasa_complex=("sasa_complex", "sum"),
                sasa_alone=("sasa_alone", "sum")).reset_index()
    res["delta_sasa"] = (res.sasa_alone - res.sasa_complex).clip(lower=0).round(2)
    res["sasa_complex"] = res.sasa_complex.round(2)
    res["sasa_alone"] = res.sasa_alone.round(2)
    part = g["partners"].agg(lambda s: ",".join(sorted({c for x in s for c in x.split(",") if c})))
    res["partner_chains"] = part.to_numpy()
    return res


# ----------------------------------------------------------------------------- fpocket
def norm_key(k):
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


def parse_fpocket(out_dir, stem):
    info = out_dir / f"{stem}_info.txt"
    pockets = {}
    cur = None
    if info.exists():
        for line in open(info):
            m = re.match(r"\s*Pocket\s+(\d+)\s*:", line)
            if m:
                cur = int(m.group(1))
                pockets[cur] = {"pocket": cur}
                continue
            if cur is not None and ":" in line:
                k, v = line.split(":", 1)
                try:
                    pockets[cur][norm_key(k)] = float(v.strip())
                except ValueError:
                    pass
    rows = []
    for p, d in sorted(pockets.items()):
        atm = out_dir / "pockets" / f"pocket{p}_atm.pdb"
        res = set()
        if atm.exists():
            for line in open(atm):
                if line.startswith(("ATOM", "HETATM")):
                    res.add((line[21], int(line[22:26]), line[26].strip()))
        vert = out_dir / "pockets" / f"pocket{p}_vert.pqr"
        cen = [np.nan] * 3
        if vert.exists():
            pts = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in open(vert) if l.startswith("ATOM")]
            if pts:
                cen = list(np.mean(pts, 0).round(3))
        d.update(n_pocket_residues=len(res), residues=json.dumps(sorted(res)),
                 center_x=cen[0], center_y=cen[1], center_z=cen[2])
        rows.append(d)
    return pd.DataFrame(rows)


def run_fpocket(sdir, which):
    src = sdir / f"{which}.pdb"
    wdir = sdir / f"fpocket_{which}"
    if wdir.exists():
        shutil.rmtree(wdir)
    wdir.mkdir()
    shutil.copy(src, wdir / f"{which}.pdb")
    p = subprocess.run(["fpocket", "-f", f"{which}.pdb"], cwd=wdir, capture_output=True, text=True)
    out = wdir / f"{which}_out"
    if p.returncode != 0 or not out.exists():
        return None, (p.stderr or p.stdout)[-200:]
    df = parse_fpocket(out, which)
    # keep only what later steps need: drop the bulky per-pocket files
    shutil.rmtree(out / "pockets", ignore_errors=True)
    return df, ""


# ----------------------------------------------------------------------------- one structure
def process(args):
    pdb, holo, force = args
    sdir = OUT / pdb
    done = sdir / "step2y_done.json"
    if done.exists() and not force:
        return json.load(open(done))
    row = {"pdb_id": pdb, "status": "ok", "n_residues": 0, "n_chains": 0, "n_interface_5A2": 0,
           "n_pockets_stripped": None, "n_pockets_holo": None, "seconds": 0, "note": ""}
    t0 = time.time()
    try:
        atoms = read_pdb(sdir / "stripped.pdb")
        res = residue_table(atoms)
        res.to_csv(sdir / "residues.csv", index=False)
        row.update(n_residues=len(res), n_chains=atoms.chain.nunique(),
                   n_interface_5A2=int((res.delta_sasa > 5).sum()))
        for which in (["stripped", "holo"] if holo else ["stripped"]):
            df, err = run_fpocket(sdir, which)
            if df is None:
                row["status"], row["note"] = f"fpocket_failed_{which}", err
                continue
            df.to_csv(sdir / f"pockets_{which}.csv", index=False)
            row[f"n_pockets_{which}"] = len(df)
    except Exception as e:
        row["status"], row["note"] = "error", f"{type(e).__name__}: {e}"[:150]
    row["seconds"] = round(time.time() - t0, 1)
    if row["status"] == "ok":
        json.dump(row, open(done, "w"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--holo", action="store_true", help="also run fpocket on holo.pdb")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not shutil.which("fpocket"):
        raise SystemExit("fpocket not found on PATH (activate the allosite environment)")
    S = pd.read_csv(OUT / "structures.csv")
    ids = sorted(S[S.status == "ok"].pdb_id)
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} structures; workers {args.workers}; holo fpocket: {args.holo}")
    rows, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, [(p, args.holo, args.force) for p in ids]), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)}  ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(rows).sort_values("pdb_id")
    R.to_csv(OUT / "step2y_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print("status:", dict(Counter(R.status)))
    ok = R[R.status == "ok"]
    if len(ok):
        multi = ok[ok.n_chains > 1]
        print(f"residues per structure: median {ok.n_residues.median():.0f}")
        print(f"interface residues (> 5 A^2), multi-chain structures: median {multi.n_interface_5A2.median():.0f} "
              f"(n = {len(multi)}); single-chain structures with any 'interface': "
              f"{int((ok[ok.n_chains == 1].n_interface_5A2 > 0).sum())} (should be 0)")
        print(f"pockets per structure (stripped): median {ok.n_pockets_stripped.median():.0f}, "
              f"min {ok.n_pockets_stripped.min():.0f}, max {ok.n_pockets_stripped.max():.0f}")
        if args.holo:
            print(f"pockets per structure (holo): median {ok.n_pockets_holo.median():.0f}")
        print(f"time per structure: median {ok.seconds.median():.0f}s, max {ok.seconds.max():.0f}s")
    bad = R[R.status != "ok"]
    for _, r in bad.head(15).iterrows():
        print(f"   {r.pdb_id}  {r.status}  {r.note}")
    print(f"\nSaved {OUT / 'step2y_summary.csv'}; per-structure residues.csv and pockets_*.csv")


if __name__ == "__main__":
    main()
