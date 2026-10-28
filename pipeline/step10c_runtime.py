#!/usr/bin/env python
"""
STEP 10C: RUNTIME OF AlloPlex AND THE OTHER TOOLS (seconds per structure)
========================================================================
Same 20 evaluation structures for every tool (random, fixed seed, spread over small / medium /
large assemblies).

  1) pick the structures and write the inputs for each tool:
        python scripts/step10c_runtime.py pick
  2) AlloPlex, timed stage by stage (cleaning, fpocket, ESM-2, model), GPU and CPU:
        python scripts/step10c_runtime.py alloplex
        python scripts/step10c_runtime.py alloplex --cpu
  3) DeepAllo and P2Rank on the same structures (commands are printed by 'pick'; they write
     their wall time into processed/dataset_v2/tools/runtime/*.time)
  4) table:
        python scripts/step10c_runtime.py report
     APOP and PASSer: the seconds recorded during the benchmark runs for these structures
     (APOP: 4 threads per structure; PASSer: web server, includes upload and queue time).
Output: processed/dataset_v2/tools/runtime/runtime.csv and figures/data/s10c_runtime.csv
"""
import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = Path(__file__).resolve().parent.parent
T = BASE / "processed" / "dataset_v2" / "tools"
R = T / "runtime"
ROOT = {"holo": BASE / "processed" / "structures", "partner": BASE / "processed" / "structures_partner"}


def n_res(path):
    return len({(l[21], l[22:27]) for l in open(path) if l.startswith("ATOM")})


def pick(args):
    man = pd.read_csv(T / "manifest.csv")
    man = man[man.roles.str.contains("primary")]
    st = man.drop_duplicates(["set", "pdb_id"])[["set", "pdb_id"]].copy()
    st["path"] = [str(ROOT[s] / p / "stripped.pdb") for s, p in zip(st.set, st.pdb_id)]
    st["n_res"] = [n_res(p) for p in st.path]
    st = st[st.n_res <= 2000]  # APOP and PASSer were only run up to 2000 residues
    st["size"] = pd.qcut(st.n_res, 3, labels=["small", "medium", "large"])
    rng = np.random.default_rng(args.seed)
    take = pd.concat([g.iloc[rng.permutation(len(g))[:int(np.ceil(args.n / 3))]] for _, g in st.groupby("size")])
    take = take.head(args.n).reset_index(drop=True)
    R.mkdir(parents=True, exist_ok=True)
    take.to_csv(R / "structures.csv", index=False)
    sub = man[man.set.astype(str).add("|").add(man.pdb_id).isin(take.set.astype(str) + "|" + take.pdb_id)].copy()
    sub["file"] = [str((T / f).resolve()) for f in sub.file]  # absolute: the manifest sits in another folder
    sub.to_csv(R / "manifest_runtime.csv", index=False)
    (R / "p2rank.ds").write_text("\n".join(take.path) + "\n")
    print(f"picked {len(take)} structures (median {int(take.n_res.median())} residues; "
          f"{', '.join(f'{k} {v}' for k, v in take['size'].value_counts().sort_index().items())}); "
          f"DeepAllo chains: {len(sub)}")
    print("\nDeepAllo (in tools/DeepAllo, deepallo environment):")
    print(f"  cd ~/allosite/tools/DeepAllo && /usr/bin/time -f %e -o {R}/deepallo.time \\\n"
          f"    conda run --no-capture-output -n deepallo python deepallo_batch.py \\\n"
          f"    --manifest {R}/manifest_runtime.csv --out {R}/deepallo_runtime.json")
    print("\nP2Rank (1 thread, java17 environment):")
    print(f"  cd ~/allosite && J17=$(conda info --base)/envs/java17 && \\\n"
          f"  JAVA_HOME=$J17 PATH=$J17/bin:$PATH /usr/bin/time -f %e -o {R}/p2rank.time \\\n"
          f"    tools/p2rank*/prank predict -threads 1 {R}/p2rank.ds -o {R}/p2rank_out")


def alloplex(args):
    import torch
    import alloplex_predict as pp
    take = pd.read_csv(R / "structures.csv")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device == "cpu":
        torch.set_num_threads(args.threads)
    t0 = time.time()
    mdir = next((BASE / "models" / m for m in ("alloplex_v1", "prism_v1") if (BASE / "models" / m).exists()))
    members = pp.load_members(mdir, device)
    esm = pp.ESM(device)
    load = time.time() - t0
    fpocket = shutil.which("fpocket")
    rows = []
    for r in take.itertuples():
        tm = {}
        pp.score_structure(r.path, R / f"alloplex_{device}" / f"{r.set}__{r.pdb_id}", fpocket, esm, members,
                           device, tm)
        rows.append({"set": r.set, "pdb_id": r.pdb_id, "n_res": r.n_res, **tm})
        print(f"  {r.pdb_id} ({r.n_res} res): {tm['total']:.1f}s (fpocket {tm['fpocket']:.1f}, ESM-2 {tm['esm']:.1f}, "
              f"model {tm['model']:.2f})", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(R / f"alloplex_{device}.csv", index=False)
    json.dump({"load_seconds": load, "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "",
               "cpu": platform.processor() or platform.machine(), "threads": args.threads if device == "cpu" else None},
              open(R / f"alloplex_{device}.json", "w"))
    print(f"AlloPlex ({device}): median {d.total.median():.1f}s per structure; model + ESM-2 loading {load:.0f}s once")


def report(args):
    take = pd.read_csv(R / "structures.csv")
    rows = []
    for dev in ("cuda", "cpu"):
        f = R / f"alloplex_{dev}.csv"
        if f.exists():
            d = pd.read_csv(f)
            info = json.load(open(R / f"alloplex_{dev}.json"))
            rows.append({"tool": f"AlloPlex ({'GPU' if dev == 'cuda' else 'CPU'})", "structures": len(d),
                         "median_s": d.total.median(), "mean_s": d.total.mean(), "max_s": d.total.max(),
                         "note": (info.get("gpu") or f"{info.get('threads')} CPU threads")
                         + f"; one-off model loading {info['load_seconds']:.0f}s"})
    keys = set(take.set + "|" + take.pdb_id)
    for tool, fn, note in (("APOP", "apop.json", "recorded in benchmark run; 4 threads"),
                           ("PASSer", "passer.json", "web server; includes upload and queue")):
        f = T / "results" / fn
        if not f.exists():
            continue
        res = json.load(open(f))
        secs = {}
        for k, v in res.items():
            p = k.split("__")
            if len(p) >= 2 and f"{p[0]}|{p[1]}" in keys and v.get("seconds") is not None and v.get("status") == "ok":
                secs[f"{p[0]}|{p[1]}"] = float(v["seconds"])
        if secs:
            s = pd.Series(secs)
            rows.append({"tool": tool, "structures": len(s), "median_s": s.median(), "mean_s": s.mean(),
                         "max_s": s.max(), "note": note})
    for tool, f, n, note in (("DeepAllo", R / "deepallo.time", len(pd.read_csv(R / "manifest_runtime.csv")),
                              "batch run incl. model loading; per chain"),
                             ("P2Rank", R / "p2rank.time", len(take), "1 thread; batch run incl. JVM start")):
        if f.exists():
            tot = float(f.read_text().strip().splitlines()[-1])
            rows.append({"tool": tool, "structures": n, "median_s": np.nan, "mean_s": tot / n, "max_s": np.nan,
                         "note": note + f"; total {tot:.0f}s"})
    out = pd.DataFrame(rows)
    out.to_csv(R / "runtime.csv", index=False)
    (BASE / "figures" / "data").mkdir(parents=True, exist_ok=True)
    out.to_csv(BASE / "figures" / "data" / "s10c_runtime.csv", index=False)
    print(f"{len(take)} structures, median {int(take.n_res.median())} residues")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("pick")
    a.add_argument("--n", type=int, default=20)
    a.add_argument("--seed", type=int, default=7)
    b = sp.add_parser("alloplex")
    b.add_argument("--cpu", action="store_true")
    b.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    sp.add_parser("report")
    args = ap.parse_args()
    {"pick": pick, "alloplex": alloplex, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
