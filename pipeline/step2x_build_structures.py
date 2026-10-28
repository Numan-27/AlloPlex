#!/usr/bin/env python
"""
STEP 2X: BUILD THE STRUCTURES FOR THE NEW DATASET
=================================================
For every quality-filtered site in processed/master/allosteric_master.csv:

  1. downloads the PDB entry (mmCIF) once per structure
  2. builds the BIOLOGICAL ASSEMBLY (the first one that contains the site chains;
     usually assembly 1) from the file's own assembly instructions
     (falls back to the asymmetric unit if there are none, or if the assembly
     is too large for PDB format: > 62 protein chains or > 99,999 atoms)
  3. writes two files per structure, protein chains only (>= 5 residues):
        stripped.pdb  no ligands, no waters, no ions (modified residues kept)
        holo.pdb      the same plus the ligands (for the holo-vs-stripped checks)
  4. maps every site's residues onto the assembly (the "original" copy of each
     chain = the one placed by the identity operator)
  5. finds the MODULATOR by its PDB ligand code (modulator_alias) within 5 A of
     the site; falls back to the nearest copy of that code (<= 12 A), then to
     the nearest other drug-like ligand (>= 6 heavy atoms, <= 5 A)

Nothing is labelled or computed on pockets here: interfaces and fpocket are the
next step (step2y), labels after that, following the analysis plan.

RUN (from ~/allosite; needs internet; numpy + pandas only):
    python scripts/step2x_build_structures.py --limit 20       # trial run first
    python scripts/step2x_build_structures.py --workers 8      # everything (resumes if stopped)
Outputs:
    data/structures/cif/<PDB>.cif.gz                downloaded files
    processed/structures/<PDB>/stripped.pdb, holo.pdb, chainmap.json, modulators.json, info.json
    processed/structures/structures.csv             one row per structure (status, assembly used ...)
    processed/structures/sites_mapped.csv           one row per site (residues on the assembly, modulator)
"""
import argparse
import gzip
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "processed" / "master" / "allosteric_master.csv"
CIF_DIR = BASE / "data" / "structures" / "cif"
OUT = BASE / "processed" / "structures"
URL = "https://files.rcsb.org/download/{}.cif.gz"
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MAX_ATOMS = 99999
MIN_CHAIN_RES = 5
ADDITIVES = set("""HOH DOD GOL EDO PEG PG4 PGE 1PE P6G 2PE PE4 12P 15P XPE PEU HEZ MPD MRD IPA EOH MOH
SO4 PO4 PI ACT ACY FMT CL NA MG ZN CA K MN FE FE2 CU NI CO CD IOD BR DMS MES EPE TRS CIT FLC
IMD BME DTT NO3 SCN NH4 MLI TAR BOG LDA CXS BTB B3P TLA SIN MLA AZI CAC UNX UNL UNK
NAG NDG MAN BMA FUC GAL GLC BGC SUC TRE LMT LMU HTG C8E SPM SPD""".split())
WATER = {"HOH", "DOD", "WAT", "H2O"}
WANTED = {"_atom_site", "_entity", "_entity_poly", "_pdbx_struct_assembly_gen", "_pdbx_struct_oper_list"}


# ----------------------------------------------------------------------------- mmCIF reader
TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'|"(?:[^"]|"(?=\S))*"|\S+""")


def cif_tokens(path):
    """Yield (value, quoted) tokens of an mmCIF file (plain or .gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        text = None
        for line in f:
            if text is not None:
                if line.startswith(";"):
                    yield "".join(text).strip(), True
                    text = None
                else:
                    text.append(line)
                continue
            if line.startswith(";"):
                text = [line[1:]]
                continue
            if line.startswith("#"):
                continue
            for t in TOKEN.findall(line):
                if len(t) >= 2 and t[0] in "'\"" and t[-1] == t[0]:
                    yield t[1:-1], True
                else:
                    yield t, False


def parse_cif(path, wanted=WANTED):
    """Return {category: (keys, rows)} for the wanted categories (loops and single-value form)."""
    data, single = {}, defaultdict(dict)
    it = cif_tokens(path)
    tok = next(it, None)
    while tok is not None:
        v, q = tok
        if not q and v == "loop_":
            heads = []
            tok = next(it, None)
            while tok is not None and not tok[1] and tok[0].startswith("_"):
                heads.append(tok[0])
                tok = next(it, None)
            cat = heads[0].split(".")[0] if heads else ""
            keys = [h.split(".", 1)[1] if "." in h else h for h in heads]
            keep = cat in wanted
            rows, row = [], []
            while tok is not None and (tok[1] or not (tok[0] == "loop_" or tok[0].startswith("_")
                                                      or tok[0].startswith("data_"))):
                if keep:
                    row.append(tok[0])
                    if len(row) == len(keys):
                        rows.append(row)
                        row = []
                tok = next(it, None)
            if keep:
                data[cat] = (keys, rows)
            continue
        if not q and v.startswith("_"):
            cat, key = v.split(".", 1) if "." in v else (v, "")
            val = next(it, None)
            if cat in wanted and val is not None:
                single[cat][key] = val[0]
            tok = next(it, None)
            continue
        tok = next(it, None)
    for cat, d in single.items():
        if cat not in data:
            data[cat] = (list(d), [list(d.values())])
    return data


def table(data, cat):
    if cat not in data:
        return []
    keys, rows = data[cat]
    return [dict(zip(keys, r)) for r in rows]


# ----------------------------------------------------------------------------- assembly operators
def expand_ops(expr):
    """'(1-3)(5,6)' -> [['1','5'],['1','6'],...]  (applied right to left)."""
    groups = re.findall(r"\(([^)]*)\)", expr) or [expr]
    lists = []
    for g in groups:
        ids = []
        for part in g.split(","):
            part = part.strip()
            m = re.fullmatch(r"(\d+)-(\d+)", part)
            if m:
                ids += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
            elif part:
                ids.append(part)
        lists.append(ids)
    combos = [[]]
    for lst in lists:
        combos = [c + [x] for c in combos for x in lst]
    return combos


def oper_matrices(data):
    ops = {}
    for r in table(data, "_pdbx_struct_oper_list"):
        try:
            m = np.array([[float(r[f"matrix[{i}][{j}]"]) for j in (1, 2, 3)] for i in (1, 2, 3)])
            v = np.array([float(r[f"vector[{i}]"]) for i in (1, 2, 3)])
        except (KeyError, ValueError):
            continue
        ops[r["id"]] = (m, v)
    return ops


def compose(combo, ops):
    m, v = np.eye(3), np.zeros(3)
    for oid in reversed(combo):  # rightmost is applied first
        om, ov = ops[oid]
        m, v = om @ m, om @ v + ov
    return m, v


def is_identity(mv):
    return np.allclose(mv[0], np.eye(3), atol=1e-4) and np.allclose(mv[1], 0, atol=1e-3)


# ----------------------------------------------------------------------------- one structure
def read_atoms(data):
    keys, rows = data.get("_atom_site", ([], []))
    if not rows:
        return None
    k = {name: i for i, name in enumerate(keys)}

    def col(*names):
        for n in names:
            if n in k:
                return k[n]
        return None
    c = {"group": col("group_PDB"), "el": col("type_symbol"), "name": col("auth_atom_id", "label_atom_id"),
         "alt": col("label_alt_id"), "comp": col("auth_comp_id", "label_comp_id"), "asym": col("label_asym_id"),
         "ent": col("label_entity_id"), "seq": col("auth_seq_id", "label_seq_id"), "ins": col("pdbx_PDB_ins_code"),
         "x": col("Cartn_x"), "y": col("Cartn_y"), "z": col("Cartn_z"), "chain": col("auth_asym_id", "label_asym_id"),
         "model": col("pdbx_PDB_model_num"), "occ": col("occupancy"), "b": col("B_iso_or_equiv")}
    first_model = rows[0][c["model"]] if c["model"] is not None else None
    out = []
    for r in rows:
        if first_model is not None and r[c["model"]] != first_model:
            continue
        el = r[c["el"]].upper() if c["el"] is not None else ""
        if el in ("H", "D"):
            continue
        alt = r[c["alt"]] if c["alt"] is not None else "."
        if alt not in (".", "?", "A", "1"):
            continue
        ins = r[c["ins"]] if c["ins"] is not None else "?"
        try:
            seq = int(r[c["seq"]])
        except ValueError:
            seq = 0
        out.append({"el": el, "name": r[c["name"]], "comp": r[c["comp"]], "asym": r[c["asym"]],
                    "ent": r[c["ent"]] if c["ent"] is not None else "", "seq": seq,
                    "ins": "" if ins in ("?", ".") else ins, "chain": r[c["chain"]],
                    "xyz": (float(r[c["x"]]), float(r[c["y"]]), float(r[c["z"]])),
                    "occ": float(r[c["occ"]]) if c["occ"] is not None and r[c["occ"]] not in ("?", ".") else 1.0,
                    "b": float(r[c["b"]]) if c["b"] is not None and r[c["b"]] not in ("?", ".") else 0.0})
    return out


def pdb_atom_line(serial, rec, a, chain, xyz):
    name = a["name"]
    name4 = name if len(name) >= 4 else ((" " + name).ljust(4) if len(a["el"]) <= 1 else name.ljust(4))
    return (f"{rec:<6}{serial:>5} {name4[:4]} {a['comp'][:3]:>3} {chain}{a['seq']:>4}{(a['ins'] or ' ')[:1]}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{a['occ']:6.2f}{min(a['b'], 999.99):6.2f}          "
            f"{a['el'][:2]:>2}\n")


def build(pdb, site_rows, force=False):
    sdir = OUT / pdb
    info_path = sdir / "info.json"
    if info_path.exists() and not force:
        info = json.load(open(info_path))
        return info["structure"], info["sites"]
    sdir.mkdir(parents=True, exist_ok=True)
    srow = {"pdb_id": pdb, "status": "ok", "assembly_used": "", "n_protein_chains": 0, "n_atoms": 0,
            "n_ligand_groups": 0, "note": ""}
    site_out = []

    def finish(status, note=""):
        srow["status"], srow["note"] = status, note
        for s in site_rows:
            if not any(x["site_id"] == s["site_id"] for x in site_out):
                site_out.append({"site_id": s["site_id"], "pdb_id": pdb, "status": status})
        json.dump({"structure": srow, "sites": site_out}, open(info_path, "w"))
        return srow, site_out

    # 1. download
    cif = CIF_DIR / f"{pdb}.cif.gz"
    if not cif.exists():
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(URL.format(pdb), cif)
                break
            except Exception as e:
                if attempt == 2:
                    return finish("download_failed", str(e)[:100])
                time.sleep(3 * (attempt + 1))
    try:
        data = parse_cif(cif)
        atoms = read_atoms(data)
    except Exception as e:
        return finish("parse_error", str(e)[:100])
    if not atoms:
        return finish("parse_error", "no atoms")

    ent_type = {r["id"]: r.get("type", "") for r in table(data, "_entity")}
    poly_type = {r["entity_id"]: r.get("type", "") for r in table(data, "_entity_poly")}
    protein_ents = {e for e, t in poly_type.items() if "polypeptide" in t}
    by_asym = defaultdict(list)
    for a in atoms:
        by_asym[a["asym"]].append(a)
    asym_chain = {s: lst[0]["chain"] for s, lst in by_asym.items()}
    asym_ent = {s: lst[0]["ent"] for s, lst in by_asym.items()}
    protein_asyms = {s for s in by_asym if asym_ent[s] in protein_ents
                     and len({(a["seq"], a["ins"]) for a in by_asym[s]}) >= MIN_CHAIN_RES}
    ligand_asyms = {s for s in by_asym if s not in protein_asyms and ent_type.get(asym_ent[s], "") == "non-polymer"
                    and by_asym[s][0]["comp"] not in WATER}
    if not protein_asyms:
        return finish("no_protein")

    # 2. assembly
    ops = oper_matrices(data)
    gens = table(data, "_pdbx_struct_assembly_gen")
    first_asm = None
    if gens and ops:
        # choose the first assembly that contains the main chain of every site
        # (some entries split the asymmetric unit into several assemblies)
        ids = list(dict.fromkeys(g.get("assembly_id", "1") for g in gens))
        mains = set()
        for s in site_rows:
            res = json.loads(s["residues"])
            if res:
                mains.add(Counter(r[0] for r in res).most_common(1)[0][0])
        best, best_n = ids[0], -1
        for aid in ids:
            covered = {asym_chain[x.strip()] for g in gens if g.get("assembly_id", "1") == aid
                       for x in g.get("asym_id_list", "").split(",") if x.strip() in protein_asyms}
            n = len(mains & covered)
            if n > best_n:
                best, best_n = aid, n
            if n == len(mains):
                break
        first_asm = best
    copies = []  # (auth_chain, op_label, matrix, vector, [asyms])
    if first_asm:
        try:
            for g in gens:
                if g.get("assembly_id", "1") != first_asm:
                    continue
                asyms = [x.strip() for x in g.get("asym_id_list", "").split(",") if x.strip()]
                for combo in expand_ops(g.get("oper_expression", "1")):
                    m, v = compose(combo, ops)
                    copies.append(("x".join(combo), m, v, asyms))
        except KeyError:
            copies = []
    assembly_used = first_asm if copies else "asym"
    if not copies:
        copies = [("1", np.eye(3), np.zeros(3), sorted(by_asym))]

    def chains_from(copy_list):
        prot, lig = {}, defaultdict(list)
        for op, m, v, asyms in copy_list:
            for s in asyms:
                if s in protein_asyms:
                    prot.setdefault((asym_chain[s], op), (m, v, []))[2].append(s)
                elif s in ligand_asyms:
                    lig[(asym_chain[s], op)].append((s, m, v))
        return prot, lig

    prot, lig = chains_from(copies)
    n_atoms = sum(len(by_asym[s]) for (_, _), (_, _, ss) in prot.items() for s in ss)
    if len(prot) > len(CHAIN_IDS) or n_atoms > MAX_ATOMS:
        if assembly_used != "asym":
            assembly_used = "asym (assembly too large)"
            copies = [("1", np.eye(3), np.zeros(3), sorted(by_asym))]
            prot, lig = chains_from(copies)
            n_atoms = sum(len(by_asym[s]) for (_, _), (_, _, ss) in prot.items() for s in ss)
        if len(prot) > len(CHAIN_IDS) or n_atoms > MAX_ATOMS:
            return finish("too_large", f"{len(prot)} chains, {n_atoms} atoms")

    # original copy of each author chain: identity operator if present, else first
    orig = {}
    for (ch, op), (m, v, _) in prot.items():
        if ch not in orig or (is_identity((m, v)) and not is_identity(orig[ch][1])):
            orig[ch] = (op, (m, v))
    keys = sorted(prot, key=lambda k: (k[1] != orig[k[0]][0], k[0], k[1]))  # original copies first
    new_id = {k: CHAIN_IDS[i] for i, k in enumerate(keys)}
    chainmap = {new_id[k]: {"orig_chain": k[0], "operator": k[1], "entity": asym_ent[prot[k][2][0]],
                            "original_copy": orig[k[0]][0] == k[1]} for k in keys}

    # 3. write stripped + holo
    serial, strip_lines, holo_lig = 0, [], []
    for k in keys:
        m, v, ss = prot[k]
        for s in ss:
            for a in by_asym[s]:
                serial += 1
                strip_lines.append(pdb_atom_line(serial, "ATOM", a, new_id[k], m @ np.array(a["xyz"]) + v))
        strip_lines.append("TER\n")
    n_lig = 0
    spare = iter([c for c in CHAIN_IDS if c not in new_id.values()])
    lig_chain = {}
    for key, items in lig.items():
        ch = new_id.get(key)
        if ch is None:
            ch = lig_chain.get(key) or next(spare, None)
            lig_chain[key] = ch
        if ch is None:
            continue
        for s, m, v in items:
            n_lig += 1
            for a in by_asym[s]:
                serial += 1
                holo_lig.append(pdb_atom_line(serial, "HETATM", a, ch, m @ np.array(a["xyz"]) + v))
    (sdir / "stripped.pdb").write_text("".join(strip_lines) + "END\n")
    (sdir / "holo.pdb").write_text("".join(strip_lines) + "".join(holo_lig) + "END\n")
    json.dump(chainmap, open(sdir / "chainmap.json", "w"), indent=1)
    srow.update(assembly_used=assembly_used, n_protein_chains=len(keys), n_atoms=n_atoms, n_ligand_groups=n_lig)

    # 4-5. sites and modulators (searched in the asymmetric unit, then moved with the site chain's operator)
    res_atoms = defaultdict(list)  # (chain, seq, ins) -> coords, protein only
    res_name = {}
    for s in protein_asyms:
        for a in by_asym[s]:
            res_atoms[(a["chain"], a["seq"], a["ins"])].append(a["xyz"])
            res_name[(a["chain"], a["seq"])] = a["comp"]
    ligs = []
    for s in ligand_asyms:
        groups = defaultdict(list)
        for a in by_asym[s]:
            groups[(a["chain"], a["seq"], a["comp"])].append(a["xyz"])
        for (ch, seq, comp), xyz in groups.items():
            ligs.append({"chain": ch, "seq": seq, "comp": comp, "xyz": np.array(xyz)})
    orig_new = {k[0]: new_id[k] for k in keys if chainmap[new_id[k]]["original_copy"]}
    modulators = {}
    for s in site_rows:
        res = [tuple(x) for x in json.loads(s["residues"])]
        mapped, missing, name_mismatch, site_xyz = [], 0, 0, []
        for ch, rname, num in res:
            hits = [k for k in res_atoms if k[0] == ch and k[1] == num]
            if not hits or ch not in orig_new:
                missing += 1
                continue
            mapped.append([orig_new[ch], num, hits[0][2]])
            site_xyz += res_atoms[hits[0]]
            if res_name.get((ch, num), rname) != rname:
                name_mismatch += 1
        row = {"site_id": s["site_id"], "pdb_id": pdb, "status": "ok" if mapped else "site_not_found",
               "n_site_residues": len(res), "n_mapped": len(mapped), "n_missing": missing,
               "n_resname_mismatch": name_mismatch, "residues_assembly": json.dumps(mapped),
               "modulator_status": "", "modulator_code_used": "", "modulator_n_atoms": 0,
               "modulator_min_dist_to_site": None}
        alias = str(s.get("modulator_alias") or "").strip().upper()
        if alias in ("NAN", "NONE"):
            alias = ""
        if site_xyz:
            sx = np.array(site_xyz)

            def dmin(xyz):
                return float(np.sqrt(((xyz[:, None, :] - sx[None, :, :]) ** 2).sum(-1)).min())
            cands = sorted(((dmin(l["xyz"]), l) for l in ligs), key=lambda t: t[0])
            pick, how = None, ""
            if re.fullmatch(r"[A-Z0-9]{1,5}", alias or ""):
                same = [(d, l) for d, l in cands if l["comp"].upper() == alias]
                if same and same[0][0] <= 5.0:
                    pick, how = same[0], "code_within_5A"
                elif same and same[0][0] <= 12.0:
                    pick, how = same[0], "code_nearest_12A"
            if pick is None:
                other = [(d, l) for d, l in cands if l["comp"].upper() not in ADDITIVES and len(l["xyz"]) >= 6]
                if other and other[0][0] <= 5.0:
                    pick, how = other[0], "other_ligand_5A"
            if str(s.get("modulator_class", "")).lower().startswith("pep") and pick is None:
                how = "peptide_modulator"
            if pick is not None:
                d, l = pick
                main = Counter(ch for ch, _, _ in res).most_common(1)[0][0]
                m, v = orig.get(main, ("1", (np.eye(3), np.zeros(3))))[1]
                xyz = (m @ l["xyz"].T).T + v
                modulators[s["site_id"]] = {"code": l["comp"], "orig_chain": l["chain"], "seq": l["seq"],
                                            "xyz": np.round(xyz, 3).tolist()}
                row.update(modulator_code_used=l["comp"], modulator_n_atoms=len(l["xyz"]),
                           modulator_min_dist_to_site=round(d, 2))
            row["modulator_status"] = how or "not_found"
        site_out.append(row)
    json.dump(modulators, open(sdir / "modulators.json", "w"))
    return finish("ok")


def work(args):
    pdb, rows, force = args
    try:
        return build(pdb, rows, force)
    except Exception as e:  # never let one structure stop the run
        return ({"pdb_id": pdb, "status": "error", "note": f"{type(e).__name__}: {e}"[:150]},
                [{"site_id": r["site_id"], "pdb_id": pdb, "status": "error"} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N structures (trial run)")
    ap.add_argument("--force", action="store_true", help="rebuild structures already done")
    args = ap.parse_args()
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(MASTER, dtype={"modulator_alias": str})
    keep = m[(m.quality_ok == True) & (m.n_residues > 0)  # noqa: E712
             & ~m.complex_kind.isin(["not in PDB", "no protein chain >= 50 aa"])
             & (m.position_class != "DNA/RNA")]
    print(f"master table: {len(m)} sites -> {len(keep)} selected sites in {keep.pdb_id.nunique()} structures")
    jobs = [(pdb, g.to_dict("records"), args.force) for pdb, g in keep.groupby("pdb_id")]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"building {len(jobs)} structures with {args.workers} workers (finished ones are skipped) ...")

    srows, siterows, t0 = [], [], time.time()
    with Pool(args.workers) as pool:
        for i, (sr, si) in enumerate(pool.imap_unordered(work, jobs), 1):
            srows.append(sr)
            siterows += si
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
    S = pd.DataFrame(srows).sort_values("pdb_id")
    T = pd.DataFrame(siterows).sort_values("site_id")
    S.to_csv(OUT / "structures.csv", index=False)
    T.to_csv(OUT / "sites_mapped.csv", index=False)

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print("structures by status:", dict(Counter(S.status)))
    ok = S[S.status == "ok"]
    if len(ok):
        print("assembly used:", dict(Counter(ok.assembly_used)))
        print(f"protein chains per structure: median {ok.n_protein_chains.median():.0f}, "
              f"max {ok.n_protein_chains.max():.0f}; atoms median {ok.n_atoms.median():.0f}")
    bad = S[S.status != "ok"]
    if len(bad):
        print("not built (first 15):")
        for _, r in bad.head(15).iterrows():
            print(f"   {r.pdb_id}  {r.status}  {r.get('note', '')}")
    if "n_mapped" in T:
        t = T[T.status == "ok"]
        full = (t.n_missing == 0).mean() if len(t) else 0
        print(f"\nsites: {len(T)}; mapped onto the assembly: {len(t)}; all residues found: {full:.1%}; "
              f"residue-name mismatches in {int((t.n_resname_mismatch > 0).sum())} sites")
        print("modulator located:", dict(Counter(t.modulator_status)))
        d = t.modulator_min_dist_to_site.dropna()
        if len(d):
            print(f"modulator-to-site distance: median {d.median():.1f} A, 90th percentile {d.quantile(0.9):.1f} A")
    print(f"\nSaved {OUT / 'structures.csv'} and {OUT / 'sites_mapped.csv'}")


if __name__ == "__main__":
    main()
