#!/usr/bin/env python
"""
STEP 7X: EXPORT THE EVALUATION STRUCTURES FOR EXTERNAL TOOLS (DeepAllo, AllosES, APOP)
====================================================================================
The three tools work on ONE chain and find their own pockets, and they run on another
server. This script writes, for the PRIMARY evaluation set of the benchmark,
    for each site: the modulator-free partner structure if the site has one,
                   otherwise the stripped holo structure;
                   the chain = the chain holding most of the site's residues
one single-chain PDB file per (set, structure, chain), with a CRYST1 line (newer DSSP
refuses files without it), plus a manifest, packed into tools_input.tar.gz.

  --all-stripped   also export the stripped holo chain of EVERY site (secondary set)

Expected results (bring back one JSON per tool, same format as your earlier runs):
    {"<key>": {"status": "ok", "pockets": [{"score": 0.81, "residues": [...]}, ...]}, ...}
    key = the manifest's "key" column (e.g. partner__2BBB__A); residues may be numbers
    ("45", 45) or "45:A" strings; status other than "ok" = tool failed on that chain.

RUN (from ~/allosite):  python scripts/step7x_export_for_tools.py
Output: processed/dataset_v2/tools/tools_input.tar.gz  (copy it to the other server)
"""
import argparse
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
OUT = D / "tools"
ROOT = {"holo": BASE / "processed" / "structures", "partner": BASE / "processed" / "structures_partner"}
CRYST1 = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"


def site_chains(set_name):
    """site_id -> (pdb_id, main chain) from the assembly-numbered site residues."""
    out = {}
    if set_name == "holo":
        m = pd.read_csv(BASE / "processed" / "structures" / "sites_mapped.csv")
        rows = m[m.status == "ok"][["site_id", "pdb_id", "residues_assembly"]].itertuples(index=False)
    else:
        rows = []
        for d in sorted(ROOT["partner"].iterdir()):
            info = d / "info.json"
            if info.exists():
                for s in json.load(open(info)).get("sites", []):
                    if s.get("status") == "ok":
                        rows.append((s["site_id"], d.name, s["residues_assembly"]))
    for sid, pdb, res in rows:
        r = json.loads(res) if isinstance(res, str) else []
        if r:
            out[sid] = (pdb, Counter(x[0] for x in r).most_common(1)[0][0])
    return out


def write_chain(src, chain, dst):
    lines = [l for l in open(src) if l.startswith("ATOM  ") and l[21] == chain]
    if not lines:
        return 0
    dst.write_text(CRYST1 + "".join(lines) + "TER\nEND\n")
    return len({(l[22:27]) for l in lines})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-stripped", action="store_true")
    args = ap.parse_args()
    (OUT / "pdb").mkdir(parents=True, exist_ok=True)
    holo = site_chains("holo")
    part = site_chains("partner")
    sites_ok = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    sites_ok = set(sites_ok[sites_ok.status == "ok"].site_id)
    sp = D / "sites_partner.csv"
    part_ok = set()
    if sp.exists():
        t = pd.read_csv(sp, low_memory=False)
        part_ok = set(t[t.status == "ok"].site_id)

    jobs = defaultdict(list)  # (set, pdb, chain) -> [(site_id, role)]
    for sid in sorted(sites_ok):
        if sid in part_ok and sid in part:
            p, c = part[sid]
            jobs[("partner", p, c)].append((sid, "primary"))
            if args.all_stripped and sid in holo:
                p, c = holo[sid]
                jobs[("holo", p, c)].append((sid, "stripped"))
        elif sid in holo:
            p, c = holo[sid]
            jobs[("holo", p, c)].append((sid, "primary+stripped"))
    rows = []
    for (setn, pdb, chain), sl in sorted(jobs.items()):
        key = f"{setn}__{pdb}__{chain}"
        fn = OUT / "pdb" / f"{key}.pdb"
        n = write_chain(ROOT[setn] / pdb / "stripped.pdb", chain, fn)
        if n == 0:
            continue
        rows.append({"key": key, "set": setn, "pdb_id": pdb, "chain": chain, "file": f"pdb/{key}.pdb",
                     "n_residues": n, "sites": ",".join(s for s, _ in sl), "roles": ",".join(sorted({r for _, r in sl}))})
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    (OUT / "README_tools.txt").write_text(
        "Run DeepAllo, AllosES and APOP on every file in pdb/ (one chain per file; the chain ID is the\n"
        "last part of the key). Write one JSON per tool: deepallo.json, alloses.json, apop.json, each\n"
        '{"<key>": {"status": "ok", "pockets": [{"score": 0.81, "residues": ["45", "46"]}, ...]}}\n'
        "key = manifest.csv 'key' column = file name without .pdb. Keep ALL pockets the tool ranks,\n"
        "not only the top 3. Failed runs: status = short error message, pockets = [].\n")
    with tarfile.open(OUT / "tools_input.tar.gz", "w:gz") as tar:
        for f in ["manifest.csv", "README_tools.txt"]:
            tar.add(OUT / f, arcname=f"tools_input/{f}")
        for r in rows:
            tar.add(OUT / r["file"], arcname=f"tools_input/{r['file']}")
    prim = man[man.roles.str.contains("primary")]
    print(f"chains exported: {len(man)} (primary set {len(prim)}: partner {int((prim.set == 'partner').sum())}, "
          f"holo {int((prim.set == 'holo').sum())}); sites covered: {len({s for x in man.sites for s in x.split(',')})}")
    print(f"chain length: median {int(man.n_residues.median())}, max {int(man.n_residues.max())}")
    print(f"saved {OUT / 'tools_input.tar.gz'} - copy it to the other server, run the tools, bring back the JSONs")


if __name__ == "__main__":
    main()
