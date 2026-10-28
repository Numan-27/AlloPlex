#!/usr/bin/env python
"""
STEP 7Z: RUN APOP ON OUR EVALUATION STRUCTURES (this server)
===========================================================
APOP (Bioinformatics 2023, github.com/Ambuj-UF/APOP) is built for multi-chain biological
assemblies, so it gets the WHOLE ligand-free structure (stripped.pdb of the holo or partner
structure), not a single chain. It is run once per structure; its ranked fpocket pockets
(APOP score + the pocket's residues with chain IDs, read from the pocket files APOP returns)
are written for every manifest key of that structure, flagged "all_chains": true so that
step 7y compares them with our full pockets.

Needs: tools/APOP (git clone); APOP's own environment (it pins numpy 1.21, so it gets
a separate conda env "apop" with python 3.9); fpocket from the allosite environment
(this script puts it on APOP's PATH).
RUN (from ~/allosite):
    APY=$(conda run -n apop which python)
    python scripts/step7z_run_apop.py --python $APY --limit 3        # check on 3 structures
    python scripts/step7z_run_apop.py --python $APY --workers 8      # all (resumable)
    python scripts/step7z_run_apop.py --python $APY --sizes          # only print structure sizes
Structures are run smallest first; --max-residues skips larger ones (recorded as "skipped").
Each APOP process gets cpu_count/workers BLAS threads, so parallel jobs do not fight over cores.
Output: processed/dataset_v2/tools/results/apop.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
T = BASE / "processed" / "dataset_v2" / "tools"
APOP = BASE / "tools" / "APOP" / "apop.py"
ROOT = {"holo": BASE / "processed" / "structures", "partner": BASE / "processed" / "structures_partner"}


def parse_zip(zpath):
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    txt = next((n for n in names if n.endswith("apop_output.txt")), None)
    if txt is None:
        raise RuntimeError("apop_output.txt missing")
    pockets = []
    for block in z.read(txt).decode(errors="replace").split("Pocket name:")[1:]:
        pname = block.split("\n")[0].strip()
        m = re.search(r"APOP score:\s*(-?[0-9.eE+-]+)", block)
        if not m:
            continue
        member = next((n for n in names if n.endswith(pname)), None)
        res = set()
        if member:
            for line in z.read(member).decode(errors="replace").splitlines():
                if line.startswith(("ATOM", "HETATM")):
                    res.add(f"{int(line[22:26])}:{line[21]}")
        pockets.append({"score": float(m.group(1)), "residues": sorted(res), "pocket_file": pname})
    return sorted(pockets, key=lambda p: -p["score"])


def n_residues(setn, pdb):
    seen = set()
    for line in open(ROOT[setn] / pdb / "stripped.pdb"):
        if line.startswith("ATOM"):
            seen.add((line[21], line[22:27]))
    return len(seen)


def run_structure(args):
    setn, pdb, timeout, python, env = args
    src = ROOT[setn] / pdb / "stripped.pdb"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(src, Path(tmp) / "s.pdb")
        try:
            p = subprocess.run([python, str(APOP), "s.pdb", "--output", "res.zip"], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=timeout)
            z = Path(tmp) / "res.zip"
            if not z.exists():
                return setn, pdb, {"status": f"failed: {(p.stderr or p.stdout)[-160:]}", "pockets": []}
            return setn, pdb, {"status": "ok", "all_chains": True, "pockets": parse_zip(z),
                               "seconds": round(time.time() - t0, 1)}
        except subprocess.TimeoutExpired:
            return setn, pdb, {"status": f"failed: timeout {timeout}s", "pockets": []}
        except Exception as e:
            return setn, pdb, {"status": f"failed: {type(e).__name__}: {e}"[:200], "pockets": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--max-residues", type=int, help="skip structures with more residues than this")
    ap.add_argument("--retry-failed", action="store_true", help="run again structures that failed before")
    ap.add_argument("--sizes", action="store_true", help="only print the size distribution and exit")
    ap.add_argument("--python", default=sys.executable,
                    help="python of the environment where APOP's requirements are installed")
    args = ap.parse_args()
    if not APOP.exists():
        raise SystemExit(f"{APOP} not found: git clone https://github.com/Ambuj-UF/APOP tools/APOP")
    man = pd.read_csv(T / "manifest.csv")
    out = T / "results" / "apop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(out)) if out.exists() else {}
    def done(k):  # "skipped" is never final; "failed" is retried only with --retry-failed
        st = res.get(k, {}).get("status", "")
        return st == "ok" or (st.startswith("failed") and not args.retry_failed)
    structs = sorted({(r.set, r.pdb_id) for r in man.itertuples() if not done(r.key)})
    size = {sp: n_residues(*sp) for sp in structs}
    if args.sizes:
        ser = pd.Series(size)
        print(f"{len(ser)} structures to run; residues per structure:")
        print(ser.describe(percentiles=[.25, .5, .75, .9]).round(0).to_string())
        for cut in (1000, 1500, 2000, 3000, 5000):
            print(f"  <= {cut} residues: {(ser <= cut).sum()} structures")
        return
    keys_of = man.groupby(["set", "pdb_id"]).key.apply(list).to_dict()
    if args.max_residues:
        big = [sp for sp in structs if size[sp] > args.max_residues]
        for sp in big:
            for k in keys_of[sp]:
                res[k] = {"status": f"skipped: {size[sp]} residues > {args.max_residues}", "pockets": [],
                          "n_residues": size[sp]}
        json.dump(res, open(out, "w"))
        structs = [sp for sp in structs if size[sp] <= args.max_residues]
        print(f"skipped {len(big)} structures larger than {args.max_residues} residues")
    structs.sort(key=lambda sp: size[sp])  # smallest first
    if args.limit:
        structs = structs[:args.limit]
    print(f"{len(man)} manifest chains; {len(structs)} structures to run with APOP ({args.workers} workers)")
    env = dict(os.environ)
    nthr = str(max(1, (os.cpu_count() or 8) // args.workers))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = nthr
    fp = shutil.which("fpocket")
    if fp:  # make fpocket of this environment visible to APOP's own environment
        env["PATH"] = os.path.dirname(fp) + os.pathsep + env.get("PATH", "")
    else:
        print("WARNING: fpocket not found on PATH; APOP needs it")
    print(f"APOP python: {args.python}; fpocket: {fp}; BLAS threads per job: {nthr}")
    t0 = time.time()
    with Pool(args.workers) as pool:
        jobs = [(s, p, args.timeout, args.python, env) for s, p in structs]
        for i, (setn, pdb, r) in enumerate(pool.imap_unordered(run_structure, jobs), 1):
            r["n_residues"] = size[(setn, pdb)]
            for k in keys_of[(setn, pdb)]:
                res[k] = r
            json.dump(res, open(out, "w"))
            print(f"  [{i}/{len(structs)}] {setn} {pdb} ({size[(setn, pdb)]} res): {r['status'][:60]} "
                  f"({len(r['pockets'])} pockets, {r.get('seconds', '-')}s)  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    ok = sum(v["status"] == "ok" for v in res.values())
    print(f"done: {ok}/{len(res)} manifest chains have APOP results; saved {out}")


if __name__ == "__main__":
    main()
