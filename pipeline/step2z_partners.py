#!/usr/bin/env python
"""
STEP 2Z: MODULATOR-FREE PARTNER STRUCTURES (ligand-free evaluation set of the plan)
==================================================================================
For every labelled site (processed/dataset_v2/sites_stripped.csv, status ok) whose
protein has another PDB entry solved WITHOUT the site's modulator (master column
best_modulator_free, from step 1x):

  1. transfer the site residues onto a partner entry, trying up to 3 candidates per
     site (the best modulator-free entry, then strict-apo entries), first success wins
       - same numbering if >= 90 % of site residues are found there with the same
         residue name ("direct")
       - otherwise a semi-global sequence alignment (unmatched ends free, so domain
         constructs align) of the holo chain with each partner chain; identity >= 90 %
         over aligned residues required ("aligned"); >= 80 % of site residues must transfer
  2. build the partner exactly like the holo structures (step 2x: assembly,
     stripped.pdb ...), then interfaces + fpocket (step 2y), then pocket labels
     (step 3x), into processed/structures_partner/<PDB>/
  3. write processed/dataset_v2/pockets_partner.csv and sites_partner.csv
     (site_id = the holo site it stands for; partner_pdb added)

Needs step2x_build_structures.py, step2y_interface_fpocket.py, step3x_label_pockets.py
in the same scripts/ folder (their functions are reused unchanged), internet, fpocket.

RUN (from ~/allosite):
    python scripts/step2z_partners.py --limit 20      # trial
    python scripts/step2z_partners.py --workers 8     # everything (resumes if stopped)
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2x_build_structures as s2x  # noqa: E402
import step2y_interface_fpocket as s2y  # noqa: E402
import step3x_label_pockets as s3x  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
PART = BASE / "processed" / "structures_partner"
MIN_PARTNER_CHAIN = 20  # residues; shorter partner chains are not used for alignment
D = BASE / "processed" / "dataset_v2"
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
         "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
         "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}


def chain_residues(cif):
    """auth chain -> ordered list of (resnum, resname) for protein chains."""
    data = s2x.parse_cif(cif)
    atoms = s2x.read_atoms(data) or []
    poly = {r["entity_id"]: r.get("type", "") for r in s2x.table(data, "_entity_poly")}
    prot = {e for e, t in poly.items() if "polypeptide" in t}
    out, seen = defaultdict(list), set()
    for a in atoms:
        if a["ent"] not in prot:
            continue
        k = (a["chain"], a["seq"], a["ins"])
        if k not in seen:
            seen.add(k)
            out[a["chain"]].append((a["seq"], a["comp"]))
    return out


def nw(a, b, match=2.0, mismatch=-1.0, gap=-2.0):
    """Semi-global alignment: unmatched ends of either sequence are free (a domain construct
    aligns cleanly to a full-length chain). Returns list of (i, j) aligned index pairs."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    A = np.frombuffer(a.encode(), dtype=np.uint8)
    B = np.frombuffer(b.encode(), dtype=np.uint8)
    H = np.zeros((n + 1, m + 1))
    ptr = np.zeros((n + 1, m + 1), np.int8)  # 0 diagonal, 1 up, 2 left
    jj = np.arange(m + 1)
    for i in range(1, n + 1):
        s = np.where(B == A[i - 1], match, mismatch)
        diag = H[i - 1, :-1] + s
        up = H[i - 1, 1:] + gap
        T = np.maximum(diag, up)
        tp = np.where(diag >= up, 0, 1)
        full = np.concatenate([[0.0], T])
        Hi = np.maximum.accumulate(full - gap * jj) + gap * jj
        H[i] = Hi
        ptr[i, 1:] = np.where(Hi[1:] > T + 1e-9, 2, tp)
    jr = int(np.argmax(H[n]))
    ic = int(np.argmax(H[:, m]))
    i, j = (n, jr) if H[n, jr] >= H[ic, m] else (ic, m)
    pairs = []
    while i > 0 and j > 0:
        if ptr[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif ptr[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def transfer(site_res, holo_chains, part_chains):
    """site_res [(chain, name, num)] in holo numbering -> (list in partner numbering, method, note)."""
    by_chain = defaultdict(list)
    for c, nme, num in site_res:
        by_chain[c].append((nme, num))
    used, out, methods = set(), [], []
    for c, items in by_chain.items():
        best = None
        for pc, plist in part_chains.items():
            if pc in used:
                continue
            pmap = {num: nme for num, nme in plist}
            hit = sum(1 for nme, num in items if pmap.get(num) == nme)
            score = hit / len(items)
            if best is None or score > best[0]:
                best = (score, pc)
        if best and best[0] >= 0.9:
            used.add(best[1])
            pmap = {num: nme for num, nme in part_chains[best[1]]}
            out += [(best[1], nme, num) for nme, num in items if pmap.get(num) == nme]
            methods.append("direct")
            continue
        hl = holo_chains.get(c)
        if not hl:
            return [], "failed", f"holo chain {c} not found"
        hseq = "".join(THREE.get(n, "X") for _, n in hl)
        cand = None
        for pc, plist in part_chains.items():
            if pc in used or len(plist) < MIN_PARTNER_CHAIN:
                continue
            pseq = "".join(THREE.get(n, "X") for _, n in plist)
            pairs = nw(hseq, pseq)
            if len(pairs) < min(30, 0.5 * min(len(hl), len(plist))):
                continue
            ident = sum(hseq[i] == pseq[j] for i, j in pairs) / len(pairs)
            if cand is None or ident > cand[0]:
                cand = (ident, pc, pairs, plist)
        if not cand or cand[0] < 0.9:
            return [], "failed", f"no partner chain >= 90 % identical to holo chain {c}"
        ident, pc, pairs, plist = cand
        used.add(pc)
        h2p = {hl[i][0]: plist[j] for i, j in pairs}
        for nme, num in items:
            if num in h2p:
                out.append((pc, h2p[num][1], h2p[num][0]))
        methods.append("aligned")
    if len(out) < 0.8 * len(site_res):
        return [], "failed", f"only {len(out)}/{len(site_res)} site residues transferred"
    return out, ("aligned" if "aligned" in methods else "direct"), ""


def ensure_cif(pdb):
    path = s2x.CIF_DIR / f"{pdb}.cif.gz"
    if not path.exists():
        import urllib.request
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(s2x.URL.format(pdb), path)
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(3 * (attempt + 1))
    return path


@lru_cache(maxsize=64)
def chains_of(pdb):
    path = ensure_cif(pdb)
    return chain_residues(path) if path is not None else None


def transfer_site(r):
    """Try the candidate partners of one site in order; first success wins."""
    res = [tuple(x) for x in json.loads(r["residues"])]
    base = {"site_id": r["site_id"], "holo_pdb": r["pdb_id"], "n_site_residues": len(res)}
    try:
        holo_chains = chains_of(r["pdb_id"]) or {}
        notes = []
        for k, cand in enumerate(r["candidates"], 1):
            pc = chains_of(cand)
            if pc is None:
                notes.append(f"{cand}: download failed")
                continue
            moved, how, note = transfer(res, holo_chains, pc)
            if moved:
                return dict(base, partner_pdb=cand, transfer=how, candidate_rank=k, n_transferred=len(moved),
                            note="; ".join(notes), residues=json.dumps([list(x) for x in moved]),
                            position_class=r["position_class"])
            notes.append(f"{cand}: {note}")
        return dict(base, partner_pdb="", transfer="failed", candidate_rank=None, n_transferred=0,
                    note="; ".join(notes)[:300])
    except Exception as e:
        return dict(base, partner_pdb="", transfer="error", n_transferred=0, note=f"{type(e).__name__}: {e}"[:200])


def build_partner(args):
    partner, rows = args
    t0 = time.time()
    try:
        s2x.OUT, s2y.OUT, s3x.STRUCT = PART, PART, PART
        site_rows = [{"site_id": r["site_id"], "residues": r["residues"], "modulator_alias": "",
                      "modulator_class": "", "position_class": r["position_class"]} for r in rows]
        sdir = PART / partner
        marker = sdir / "sites.json"
        ids = sorted(r["site_id"] for r in site_rows)
        force = not marker.exists() or json.load(open(marker)) != ids
        srow, mapped = s2x.build(partner, site_rows, force=force)
        if srow["status"] != "ok":
            return [], [], {"partner_pdb": partner, "status": "build_" + srow["status"]}
        json.dump(ids, open(marker, "w"))
        yrow = s2y.process((partner, False, False))
        if yrow["status"] != "ok":
            return [], [], {"partner_pdb": partner, "status": "step2y_" + yrow["status"]}
        pos = {r["site_id"]: r["position_class"] for r in site_rows}
        sites3 = [dict(m, position_class=pos[m["site_id"]]) for m in mapped if m.get("status") == "ok"]
        prow, srow3 = s3x.process((partner, "stripped", sites3))
        return prow, srow3, {"partner_pdb": partner, "status": "ok", "n_sites": len(ids),
                             "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        return [], [], {"partner_pdb": partner, "status": "error", "note": f"{type(e).__name__}: {e}"[:150]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N sites (trial)")
    ap.add_argument("--max-candidates", type=int, default=3, help="partner structures tried per site")
    args = ap.parse_args()
    PART.mkdir(parents=True, exist_ok=True)
    s2x.CIF_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", dtype={"modulator_alias": str},
                         low_memory=False)
    sites = pd.read_csv(D / "sites_stripped.csv", low_memory=False)
    sites = sites[sites.status == "ok"][["site_id"]]
    m = master.merge(sites, on="site_id")
    def cands(r):
        out = []
        for x in [r.get("best_modulator_free")] + str(r.get("apo_candidates_top10") or "").split(","):
            x = str(x).strip().upper()
            if len(x) == 4 and x != r["pdb_id"] and x not in out and x != "NAN":
                out.append(x)
        return out[:args.max_candidates]
    m["candidates"] = [cands(r) for _, r in m.iterrows()]
    m = m[m.candidates.map(len) > 0]
    print(f"{len(sites)} labelled sites; {len(m)} have at least one modulator-free candidate "
          f"(up to {args.max_candidates} tried per site)")
    rows = m[["site_id", "pdb_id", "residues", "position_class", "candidates"]].to_dict("records")
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    info = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(transfer_site, rows, chunksize=4), 1):
            info.append(r)
            if i % 200 == 0 or i == len(rows):
                print(f"  transfer {i}/{len(rows)}  ({time.time() - t0:.0f}s)")
    I = pd.DataFrame(info)
    good = I[I.partner_pdb.astype(str).str.len() == 4]
    jobs = [(p, g.to_dict("records")) for p, g in good.groupby("partner_pdb")]
    print(f"building {len(jobs)} partner structures ...")
    prows, srows, stat = [], [], []
    with Pool(args.workers) as pool:
        for i, (b, c, d) in enumerate(pool.imap_unordered(build_partner, jobs), 1):
            prows += b
            srows += c
            stat.append(d)
            if i % 50 == 0 or i == len(jobs):
                print(f"  build {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
    Pp, Sp, St = pd.DataFrame(prows), pd.DataFrame(srows), pd.DataFrame(stat)
    if len(Sp):
        Sp = Sp.merge(I[["site_id", "holo_pdb", "partner_pdb", "transfer", "candidate_rank"]], on="site_id", how="left")
    if len(Pp):
        fam = master.drop_duplicates("pdb_id").set_index("pdb_id").family
        first_holo = good.drop_duplicates("partner_pdb").set_index("partner_pdb").holo_pdb
        Pp["family"] = Pp.pdb_id.map(first_holo).map(fam)
    I = I.drop(columns=["residues", "position_class"], errors="ignore")
    I.to_csv(D / "partner_transfer.csv", index=False)
    Pp.to_csv(D / "pockets_partner.csv", index=False)
    Sp.to_csv(D / "sites_partner.csv", index=False)
    St.to_csv(D / "partner_structures.csv", index=False)

    print("\n" + "=" * 76)
    print("REPORT")
    print("=" * 76)
    print("partner structures:", dict(Counter(St.status)))
    print("site transfer:", dict(Counter(I.transfer)))
    if "candidate_rank" in I:
        print("successful transfers by candidate rank (1 = best modulator-free entry):",
              dict(Counter(I.candidate_rank.dropna().astype(int))))
    if len(Sp):
        ok = Sp[Sp.status == "ok"]
        holo = pd.read_csv(D / "sites_stripped.csv", low_memory=False).set_index("site_id")
        both = ok[ok.site_id.isin(holo.index)]
        hdet = holo.loc[both.site_id, "detected"].astype(bool).to_numpy()
        pdet = both.detected.astype(bool).to_numpy()
        print(f"sites evaluated on a partner: {len(ok)}")
        print(f"detected by fpocket (best pocket Jaccard >= 0.1): partner {pdet.mean():.1%} vs same sites in the "
              f"stripped holo structure {hdet.mean():.1%}")
        print(f"median best Jaccard: partner {both.best_jaccard.median():.2f} vs holo "
              f"{holo.loc[both.site_id, 'best_jaccard'].median():.2f}")
        print("  (a site missing in the partner but present in the stripped holo structure suggests the pocket")
        print("   forms or opens when the modulator binds)")
        print("detection by annotation (partner):")
        for pc, g in ok.groupby("position_class"):
            print(f"   {pc:<36} n={len(g):>5}  detected {g.detected.astype(bool).mean():.1%}")
    bad = St[St.status != "ok"]
    for _, r in bad.head(10).iterrows():
        print(f"   {r.partner_pdb} {r.status} {r.get('note', '')}")
    print(f"\nSaved {D}/pockets_partner.csv, sites_partner.csv, partner_transfer.csv, partner_structures.csv")


if __name__ == "__main__":
    main()
