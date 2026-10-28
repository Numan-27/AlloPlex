#!/usr/bin/env python
"""
STEP 1X: COMPILE ONE MASTER TABLE OF ALLOSTERIC SITES
=====================================================
Builds ONE file, processed/master/allosteric_master.csv, with one row per
allosteric site, combining:

  1. ASD (all sites, ALL 'position' values, several sites per PDB kept)
  2. RCSB PDB facts for each structure: method, resolution, protein/peptide
     chains, UniProt ids, biological-assembly size, complex type
     (homo-oligomer / hetero-complex / protein-peptide / monomer ...)
  3. ligands present in the structure (PDB chemical ids, >= 150 Da, common
     buffers/ions/sugars removed): helps to identify the modulator later
  4. protein family (MMseqs2, >= 30 % identity, >= 50 % coverage)
  5. APO PARTNERS: other PDB entries of the same protein (same UniProt) with
     no drug-like ligand at all, good quality (X-ray/cryo-EM <= 3.5 A)
  6. AlloBench (same columns as ASD, residues like 'B-THR-7'): sites matching an
     ASD site (same modulator record, or >= 50 % residue overlap) are flagged
     in_allobench_site; AlloBench sites not in our ASD file are ADDED as rows
     (source = AlloBench, with residues). CASBench: PDB-id flag only.
  7. modulator PDB ligand code (modulator_alias) -> modulator_in_entry, and
     "modulator-free" partners: same protein solved without that modulator
     (cofactors allowed), in addition to strict apo partners.
  8. flag: in the current PRISM 228 dataset (and its split).

Needs: internet; scripts/feasibility_count.py in the same folder (it reuses its
cache processed/feasibility/rcsb_cache.json, so the 1412 ASD entries are not
downloaded again).

RUN (from ~/allosite):
    python scripts/step1x_compile_master.py
    (data/AlloBench.csv is used automatically if it exists; --no-allobench to skip it)
    python scripts/step1x_compile_master.py --casbench data/CASBench.csv
    python scripts/step1x_compile_master.py --no-apo          (skip the apo search: much faster)
First full run: roughly 15-40 min (apo search). Re-runs use caches and take seconds.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feasibility_count as fc  # noqa: E402

BASE = fc.BASE
OUT = BASE / "processed" / "master"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
MIN_LIGAND_DA = 150
# buffers, cryo-protectants, detergents, ions, common sugars: not modulators
ADDITIVES = set("""HOH DOD GOL EDO PEG PG4 PGE 1PE P6G 2PE PE4 12P 15P XPE PEU HEZ MPD MRD IPA EOH MOH
SO4 PO4 PI ACT ACY FMT CL NA MG ZN CA K MN FE FE2 CU NI CO CD IOD BR DMS MES EPE TRS CIT FLC
IMD BME DTT NO3 SCN NH4 MLI TAR BOG LDA CXS BTB B3P TLA SIN MLA AZI CAC UNX UNL UNK
NAG NDG MAN BMA FUC GAL GLC BGC SUC TRE LMT LMU HTG C8E SPM SPD""".split())

LIGAND_QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id
 exptl{method} rcsb_entry_info{resolution_combined}
 nonpolymer_entities{nonpolymer_comp{chem_comp{id formula_weight}}}}}"""

POSITION_CLASS = [("protein-protein", "PPI interface"), ("inner protein", "inner protein"),
                  ("allosteric position", "allosteric position (unspecified)"),
                  ("dna/rna", "DNA/RNA"), ("", "blank")]


def say(*a):
    fc.say(*a)


def position_class(p):
    p = (p or "").strip().lower()
    for key, label in POSITION_CLASS:
        if key and key in p:
            return label
    return "blank" if not p else "other: " + p


# ----------------------------------------------------------------------------- web
def gql_fetch(query, ids, cache_path, label):
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    todo = sorted({i for i in ids if i not in cache})
    if todo:
        say(f"RCSB {label}: {len(todo)} entries to query ({len(set(ids)) - len(todo)} cached)")
    for k in range(0, len(todo), 100):
        batch = todo[k:k + 100]
        for attempt in range(4):
            try:
                ans = fc.post(query, batch)
                if ans.get("errors") and not (ans.get("data") or {}).get("entries"):
                    raise RuntimeError(str(ans["errors"])[:300])
                break
            except Exception as e:
                if attempt == 3:
                    sys.exit(f"RCSB API failed ({label}, batch starting {batch[0]}): {e}\n"
                             f"Re-run later; finished batches are cached.")
                time.sleep(5 * (attempt + 1))
        got = {(e.get("rcsb_id") or "").upper(): e for e in (ans.get("data") or {}).get("entries") or [] if e}
        for i in batch:
            cache[i] = got.get(i)
        json.dump(cache, open(cache_path, "w"))
        if (k // 100) % 10 == 9 or k + 100 >= len(todo):
            say(f"  {min(k + 100, len(todo))}/{len(todo)}")
    return cache


def search_post(query):
    req = urllib.request.Request(SEARCH_URL, data=json.dumps(query).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        txt = r.read()
    return json.loads(txt) if txt else {}


def entries_for_uniprot(uniprots, cache_path):
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    todo = sorted(u for u in uniprots if u not in cache)
    if todo:
        say(f"RCSB search: other PDB entries for {len(todo)} UniProt ids ({len(uniprots) - len(todo)} cached)")
    for n, u in enumerate(todo, 1):
        q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match", "value": u}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                "operator": "exact_match", "value": "UniProt"}}]},
             "return_type": "entry", "request_options": {"return_all_hits": True}}
        for attempt in range(4):
            try:
                ans = search_post(q)
                break
            except Exception as e:
                if attempt == 3:
                    sys.exit(f"RCSB search failed for {u}: {e}\nRe-run later; finished ids are cached.")
                time.sleep(5 * (attempt + 1))
        cache[u] = sorted({(x.get("identifier") or "").upper() for x in ans.get("result_set") or []} - {""})
        if n % 25 == 0 or n == len(todo):
            json.dump(cache, open(cache_path, "w"))
            say(f"  {n}/{len(todo)}")
    return cache


def ligands(entry):
    out = set()
    for n in (entry or {}).get("nonpolymer_entities") or []:
        cc = ((n or {}).get("nonpolymer_comp") or {}).get("chem_comp") or {}
        cid = (cc.get("id") or "").upper()
        if cid and cid not in ADDITIVES and (cc.get("formula_weight") or 0) >= MIN_LIGAND_DA:
            out.add(cid)
    return sorted(out)


def entry_quality(entry, max_res):
    if not entry:
        return False, None
    m = " ".join((x or {}).get("method") or "" for x in entry.get("exptl") or []).upper()
    res = ((entry.get("rcsb_entry_info") or {}).get("resolution_combined") or [None])[0]
    ok = ("X-RAY" in m or "ELECTRON MICROSCOPY" in m) and res is not None and res <= max_res
    return ok, res


# ----------------------------------------------------------------------------- external lists
def load_external(path):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, dtype=str).fillna("")
    else:
        sep = "\t" if p.suffix.lower() in (".tsv", ".txt") else ","
        df = pd.read_csv(p, sep=sep, dtype=str).fillna("")
    cols = [c for c in df.columns if "pdb" in str(c).lower()] or list(df.columns)
    ids = set()
    for c in cols:
        for v in df[c]:
            for t in re.findall(r"\b([0-9][A-Za-z0-9]{3})\b", str(v)):
                if not t.isdigit():
                    ids.add(t.upper())
    return ids, cols


def parse_list_residues(text):
    """AlloBench format: "['B-THR-7', 'A-ALA--3', ...]" -> [(chain, resname, resnum)]"""
    return [(c, n, int(i)) for c, n, i in re.findall(r"([A-Za-z0-9]+)-([A-Z0-9]{3})-(-?\d+)", str(text))]


def load_site_table(path, source):
    """ASD-style table (ASD AS file or AlloBench csv) -> list of site dicts."""
    p = Path(path)
    if p.suffix.lower() in (".txt", ".tsv"):
        df = fc.read_tab(p)
    else:
        df = pd.read_csv(p, sep=None, engine="python", dtype=str).fillna("")
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].str.strip()
    rows, seen = [], set()
    for _, r in df.iterrows():
        pdb = r.get("allosteric_pdb", "").upper()
        raw = r.get("allosteric_site_residue", "")
        res = fc.parse_residues(raw) if "Chain" in raw else parse_list_residues(raw)
        if len(pdb) != 4 or not pdb.isalnum() or not res:
            continue
        key = (pdb, tuple(sorted(set(res))), r.get("modulator_name", "").lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"pdb_id": pdb, "target_id": r.get("target_id", ""), "gene": r.get("target_gene", ""),
                     "organism_asd": r.get("organism", ""), "position": r.get("position", ""),
                     "modulator_class": r.get("modulator_class", ""), "modulator_name": r.get("modulator_name", ""),
                     "modulator_chain": r.get("modulator_chain", ""), "modulator_serial": r.get("modulator_serial", ""),
                     "modulator_alias": r.get("modulator_alias", "").upper(),
                     "active_site_residue": r.get("active_site_residue", ""),
                     "residues": res, "source": source})
    say(f"{source}: {len(df)} rows -> {len(rows)} distinct sites with a PDB id and residues, "
        f"{len({x['pdb_id'] for x in rows})} PDB entries")
    return rows


def jaccard(a, b):
    a, b = {(c, i) for c, _, i in a}, {(c, i) for c, _, i in b}
    return len(a & b) / len(a | b) if a | b else 0.0


def merge_allobench(asd, ab):
    """Mark ASD sites also in AlloBench; return AlloBench sites not in ASD."""
    by_pdb = {}
    for s in asd:
        s.update(in_allobench_site=False, allobench_jaccard=None)
        by_pdb.setdefault(s["pdb_id"], []).append(s)
    new = []
    for t in ab:
        best, bj = None, 0.0
        for s in by_pdb.get(t["pdb_id"], []):
            same_serial = s["modulator_serial"] and s["modulator_serial"] == t["modulator_serial"]
            j = jaccard(s["residues"], t["residues"])
            if same_serial:
                j = max(j, 0.5)  # same ASD modulator record: count as the same site
            if j > bj:
                best, bj = s, j
        if best is not None and bj >= 0.5:
            best["in_allobench_site"] = True
            best["allobench_jaccard"] = round(jaccard(best["residues"], t["residues"]), 3)
        else:
            t.update(in_allobench_site=True, allobench_jaccard=None)
            new.append(t)
    return new


def raw_ligand_ids(entry):
    return {((((n or {}).get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id") or "").upper()
            for n in (entry or {}).get("nonpolymer_entities") or []} - {""}


def is_ccd(alias):
    return bool(re.fullmatch(r"[A-Z0-9]{1,5}", alias or ""))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allobench", help="AlloBench csv (default: data/AlloBench.csv if it exists)")
    ap.add_argument("--no-allobench", action="store_true", help="ignore AlloBench even if the file exists")
    ap.add_argument("--casbench")
    ap.add_argument("--no-apo", action="store_true")
    ap.add_argument("--max-res", type=float, default=3.5)
    ap.add_argument("--max-per-uniprot", type=int, default=300,
                    help="at most this many other PDB entries checked per UniProt id")
    args = ap.parse_args()
    if args.no_allobench:
        args.allobench = None
    elif not args.allobench and (BASE / "data" / "AlloBench.csv").exists():
        args.allobench = str(BASE / "data" / "AlloBench.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    (BASE / "processed" / "feasibility").mkdir(parents=True, exist_ok=True)

    say("=" * 78)
    say("PART 1: SOURCES")
    say("=" * 78)
    sites = load_site_table(fc.AS_FILE, "ASD")
    pp = fc.load_pp()
    asd_ids = {s["pdb_id"] for s in sites}
    ab_pdbs = set()
    if args.allobench:
        ab = load_site_table(args.allobench, "AlloBench")
        ab_pdbs = {t["pdb_id"] for t in ab}
        new = merge_allobench(sites, ab)
        matched = [s for s in sites if s["in_allobench_site"]]
        js = [s["allobench_jaccard"] for s in matched]
        say(f"  ASD sites also in AlloBench: {len(matched)}/{len(sites)}"
            + (f" (median residue overlap, Jaccard {pd.Series(js).median():.2f})" if js else ""))
        say(f"  AlloBench sites NOT in our ASD file: {len(new)} ({len({t['pdb_id'] for t in new})} PDB entries) -> added")
        sites += new
    else:
        for s in sites:
            s.update(in_allobench_site=None, allobench_jaccard=None)
    external = {}
    if args.casbench:
        ids, cols = load_external(args.casbench)
        external["CASBench"] = ids
        say(f"CASBench: {len(ids)} PDB ids (from columns {cols[:3]}); {len(ids & asd_ids)} also in ASD, "
            f"{len(ids - asd_ids - ab_pdbs)} new")
    site_ids = {s["pdb_id"] for s in sites}
    ext_ids = set().union(*external.values()) if external else set()
    all_ids = sorted(site_ids | ext_ids)

    cache = fc.fetch(all_ids, BASE / "processed" / "feasibility" / "rcsb_cache.json")
    info = {i: fc.summarise(cache.get(i)) for i in all_ids}
    lig = gql_fetch(LIGAND_QUERY, all_ids, OUT / "ligand_cache.json", "ligands")

    prism = {}
    cur = BASE / "processed" / "final_dataset_with_splits.csv"
    if cur.exists():
        head = pd.read_csv(cur, nrows=0).columns
        use = ["pdb_id"] + (["split"] if "split" in head else [])
        d = pd.read_csv(cur, usecols=use).drop_duplicates("pdb_id")
        prism = dict(zip(d.pdb_id.astype(str).str.upper(), d["split"] if "split" in d else ["yes"] * len(d)))

    # --- one row per site (ASD, then AlloBench-only)
    rows, per_pdb = [], Counter()
    for s in sites:
        pdb = s["pdb_id"]
        per_pdb[pdb] += 1
        inf = info.get(pdb)
        r = {"site_id": f"{pdb}_{per_pdb[pdb]}", "source": s["source"], "pdb_id": pdb,
             "target_id": s["target_id"], "gene": s["gene"], "organism_asd": s["organism_asd"],
             "asd_position": s["position"], "position_class": position_class(s["position"]),
             "modulator_class": s["modulator_class"], "modulator_name": s["modulator_name"],
             "modulator_chain": s["modulator_chain"], "modulator_serial": s["modulator_serial"],
             "modulator_alias": s["modulator_alias"],
             "residues": json.dumps(s["residues"]), "n_residues": len(s["residues"]),
             "site_chains": ",".join(sorted({c for c, _, _ in s["residues"]})),
             "in_allobench_site": s["in_allobench_site"], "allobench_jaccard": s["allobench_jaccard"],
             "active_site_residue": s["active_site_residue"]}
        rows.append((r, s["residues"], inf))
    for name, ids in external.items():
        for pdb in sorted(ids - site_ids):
            r = {"site_id": f"{pdb}_{name}", "source": name, "pdb_id": pdb, "residues": "", "n_residues": 0}
            rows.append((r, [], info.get(pdb)))

    out = []
    for r, res, inf in rows:
        pdb = r["pdb_id"]
        r["complex_kind"] = fc.classify(inf)
        r["quality_ok"] = fc.quality_ok(inf, args.max_res)
        r["method"] = inf["methods"] if inf else ""
        r["resolution"] = inf["resolution"] if inf else None
        r["assembly1_protein_chains"] = inf["assembly1_chains"] if inf else None
        r["n_protein_entities"] = sum(e["kind"] == "protein" for e in inf["entities"]) if inf else 0
        r["n_peptide_entities"] = sum(e["kind"] == "peptide" for e in inf["entities"]) if inf else 0
        r["n_protein_chains"] = sum(len(e["chains"]) for e in inf["entities"] if e["kind"] == "protein") if inf else 0
        r["ligands_in_entry"] = ",".join(ligands(lig.get(pdb)))
        target, partners, groups, mapped, unmapped = None, [], set(), [], 0
        if inf:
            c2e = {c: e for e in inf["entities"] for c in e["chains"]}
            if res:
                cnt = Counter(c2e[c]["entity"] for c, _, _ in res if c in c2e)
                if cnt:
                    tid = cnt.most_common(1)[0][0]
                    target = next(e for e in inf["entities"] if e["entity"] == tid)
                chains = {c for c, _, _ in res}
                mapped = [c2e[c] for c in chains if c in c2e]
                unmapped = len(chains) - len(mapped)
            else:  # external rows: largest protein entity as the target
                prot = sorted((e for e in inf["entities"] if e["kind"] == "protein"), key=lambda e: -e["length"])
                target = prot[0] if prot else None
            groups = {e["group"] for e in mapped if e["kind"] == "protein"}
            if target:
                partners = [e for e in inf["entities"] if e["group"] != target["group"]
                            and e["kind"] in ("protein", "peptide")]
            all_uni = {u for e in inf["entities"] for u in e["uniprot"]}
        else:
            all_uni = set()
        r["target_uniprot"] = ";".join(target["uniprot"]) if target else ""
        r["target_description"] = target["desc"] if target else ""
        r["target_organism"] = target["organism"] if target else ""
        r["target_length"] = target["length"] if target else None
        r["site_chains_unmapped"] = unmapped
        r["site_spans_two_proteins"] = len(groups) >= 2
        r["site_on_peptide"] = any(e["kind"] == "peptide" for e in mapped)
        r["partner_in_structure"] = bool(partners)
        r["partner_description"] = "; ".join(sorted({p["desc"] for p in partners}))[:200]
        r["modulator_is_peptide"] = str(r.get("modulator_class", "")).lower().startswith("pep")
        r["gold_ppi"] = bool(pp.get(r.get("target_id", ""), set()) & all_uni)
        r["in_prism228"] = pdb in prism
        r["prism_split"] = prism.get(pdb, "")
        r["in_allobench_pdb"] = (pdb in ab_pdbs) if args.allobench else None
        r["in_casbench"] = (pdb in external["CASBench"]) if "CASBench" in external else None
        alias = r.get("modulator_alias") or ""
        r["modulator_in_entry"] = (alias in raw_ligand_ids(lig.get(pdb))) if is_ccd(alias) else None
        r["_seq"] = target["seq"] if target else ""
        out.append(r)
    df = pd.DataFrame(out)

    fallback = {i: "T:" + (t or p) for i, t, p in zip(df.index, df["target_id"].fillna(""), df["pdb_id"])}
    fam, how = fc.families(dict(zip(df.index, df["_seq"])), fallback)
    df["family"] = [fam.get(i) or fallback[i] for i in df.index]
    say(f"families: {how}")

    # --- apo partners
    if not args.no_apo:
        unis = sorted({u for x in df["target_uniprot"] for u in str(x).split(";") if u})
        hits = entries_for_uniprot(unis, OUT / "uniprot_search.json")
        holo = set(df["pdb_id"])
        cand = set()
        for u in unis:
            cand |= set(sorted(set(hits.get(u, [])) - holo)[:args.max_per_uniprot])
        lig2 = gql_fetch(LIGAND_QUERY, sorted(cand), OUT / "ligand_cache.json", "other entries of the same proteins")
        good, apo, raw = {}, {}, {}
        for c in cand:
            ok, res = entry_quality(lig2.get(c), args.max_res)
            if ok:
                good[c] = res
                raw[c] = raw_ligand_ids(lig2.get(c))
                if not ligands(lig2.get(c)):
                    apo[c] = res
        n_same, n_apo, best, best_res, top, n_mf, best_mf, best_mf_res = [], [], [], [], [], [], [], []
        for x, pdb, alias in zip(df["target_uniprot"], df["pdb_id"], df["modulator_alias"].fillna("")):
            us = [u for u in str(x).split(";") if u]
            same = {c for u in us for c in hits.get(u, []) if c in good and c != pdb}
            ap_ = sorted((c for c in same if c in apo), key=lambda c: apo[c])
            n_same.append(len(same))
            n_apo.append(len(ap_))
            best.append(ap_[0] if ap_ else "")
            best_res.append(apo[ap_[0]] if ap_ else None)
            top.append(",".join(ap_[:10]))
            if is_ccd(alias):  # same protein, solved WITHOUT this modulator (cofactors allowed)
                mf = sorted((c for c in same if alias not in raw[c]), key=lambda c: good[c])
                n_mf.append(len(mf))
                best_mf.append(mf[0] if mf else "")
                best_mf_res.append(good[mf[0]] if mf else None)
            else:
                n_mf.append(None)
                best_mf.append("")
                best_mf_res.append(None)
        df["n_other_structures_same_protein"] = n_same
        df["n_apo_candidates"] = n_apo
        df["best_apo"] = best
        df["best_apo_resolution"] = best_res
        df["apo_candidates_top10"] = top
        df["n_modulator_free"] = n_mf
        df["best_modulator_free"] = best_mf
        df["best_modulator_free_resolution"] = best_mf_res

    df = df.drop(columns="_seq")
    df.to_csv(OUT / "allosteric_master.csv", index=False)

    # --- report
    q = df[df.quality_ok]
    say("")
    say("=" * 78)
    say("PART 2: THE MASTER TABLE")
    say("=" * 78)
    say(f"rows (sites): {len(df)}  |  structures: {df.pdb_id.nunique()}  |  families: {df.family.nunique()}")
    say(f"by source: {dict(Counter(df.source))}")
    say(f"quality-filtered (X-ray/cryo-EM <= {args.max_res} A): {len(q)} sites, {q.pdb_id.nunique()} structures, "
        f"{q.family.nunique()} families")
    say("")
    say("Quality-filtered labelled sites (ASD + AlloBench): position annotation x complex type (sites / families)")
    qa = q[q.source.isin(["ASD", "AlloBench"])]
    kinds = [k for k in fc.KIND_ORDER if k in set(qa.complex_kind)]
    say(f"  {'position':<36}" + "".join(f"{k[:14]:>16}" for k in kinds))
    for pc in sorted(set(qa.position_class)):
        sub = qa[qa.position_class == pc]
        say(f"  {pc[:35]:<36}" + "".join(
            f"{(sub.complex_kind == k).sum():>9} / {sub[sub.complex_kind == k].family.nunique():<4}" for k in kinds))
    multi = qa[qa.complex_kind.isin(["homo-oligomer", "homo (assembly only)", "hetero-complex",
                                     "pseudo-hetero", "protein-peptide"])]
    nonppi = multi[multi.position_class != "PPI interface"]
    say("")
    say(f"KEY SET for the non-circular test: sites in multi-chain assemblies NOT annotated as PPI: "
        f"{len(nonppi)} sites, {nonppi.pdb_id.nunique()} structures, {nonppi.family.nunique()} families")
    na = nonppi[nonppi.source == "ASD"]
    say(f"  ... of which from ASD: {len(na)} sites, {na.family.nunique()} families; "
        f"added by AlloBench: {len(nonppi) - len(na)} sites, "
        f"{len(set(nonppi.family) - set(na.family))} new families")
    say(f"control set: sites in monomers: {(qa.complex_kind == 'monomer').sum()} sites, "
        f"{qa[qa.complex_kind == 'monomer'].family.nunique()} families")
    if "n_apo_candidates" in df:
        has = qa[qa.n_apo_candidates > 0]
        say("")
        say(f"apo partners (same UniProt, no drug-like ligand, quality-filtered): available for "
            f"{len(has)}/{len(qa)} sites, {has.family.nunique()}/{qa.family.nunique()} families; "
            f"median {int(qa.n_apo_candidates.median())} candidates per site")
        say("  note: strict 'apo' = no ligand >= 150 Da other than buffers/ions/sugars (a bound cofactor such")
        say("  as NAD, ATP or heme disqualifies the structure).")
        withc = qa[qa.n_modulator_free.notna()]
        mf = withc[withc.n_modulator_free > 0]
        say(f"modulator-free partners (same UniProt, solved WITHOUT this site's modulator; cofactors allowed):")
        say(f"  sites with a PDB ligand code for the modulator: {len(withc)}/{len(qa)}; of these, "
            f"{len(mf)} have >= 1 modulator-free structure ({mf.family.nunique()}/{withc.family.nunique()} families), "
            f"median {int(withc.n_modulator_free.median()) if len(withc) else 0} per site")
    mie = qa[qa.modulator_in_entry.notna()]
    if len(mie):
        say(f"modulator ligand code found in its own PDB entry: {int(mie.modulator_in_entry.sum())}/{len(mie)} sites "
            f"(the rest: alias is not a PDB ligand code, e.g. a peptide or ion name, or the entry differs)")
    if prism:
        say("")
        say(f"current PRISM dataset: {df[df.in_prism228].pdb_id.nunique()} of {len(prism)} structures found in the table")
    if args.allobench:
        say(f"AlloBench: {df[df.in_allobench_pdb == True].pdb_id.nunique()} structures in the table; "  # noqa: E712
            f"{(df.source == 'AlloBench').sum()} sites came from AlloBench only")
    if "CASBench" in external:
        say(f"CASBench: {df[df.in_casbench == True].pdb_id.nunique()} structures in the table "  # noqa: E712
            f"({(df.source == 'CASBench').sum()} rows from CASBench only, residues not parsed)")
    say("")
    say(f"Saved: {OUT / 'allosteric_master.csv'}")
    (OUT / "master_report.txt").write_text("\n".join(fc.LOG) + "\n")


if __name__ == "__main__":
    main()
