#!/usr/bin/env python
"""
AlloPlex: prediction of allosteric pockets in protein assemblies
=============================================================
Input : one or more protein structures (.pdb / .ent / .cif / .mmcif, optionally .gz).
        Give the biological assembly you are interested in (e.g. the RCSB "assembly 1" file);
        multi-chain complexes and protein-peptide complexes are fine.
Steps : 1. keep protein chains only (ligands, waters, ions and nucleic acids removed;
           hydrogens dropped; first model; alternate locations ' '/A/1; chains < 5 residues dropped)
        2. fpocket (default parameters) finds the pockets
        3. ESM-2 650M embeddings of every chain (sequence = residues present, windows of 1022)
        4. each pocket: mean ESM-2 vector of its residues + 11 fpocket descriptors
        5. the AlloPlex ensemble (MLP, models/alloplex_v1) scores each pocket
Output: <out>/<name>/alloplex_pockets.csv   all pockets ranked by AlloPlex score
        <out>/<name>/clean.pdb           the ligand-free structure that was analysed
        <out>/<name>/alloplex_view.pml      PyMOL script: top pockets coloured on the structure
        and a short table on the screen.

Benchmark (family-level cross-validation, unseen protein families, ligand-free partner
structures): the top-ranked pocket is the known allosteric site in ~42 % of cases, one of the
top 3 in ~58 %. Treat the ranking as a shortlist for experiments, not as a certainty.

RUN (env with torch, fair-esm, numpy, pandas; fpocket on PATH; gemmi optional):
    python alloplex_predict.py my_complex.pdb --out alloplex_results
    python alloplex_predict.py 1abc-assembly1.cif.gz 2xyz.pdb --top 5
"""
import argparse
import gzip
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "models" / "alloplex_v1"
if not DEFAULT_MODEL.exists() and (HERE.parent / "models" / "prism_v1").exists():  # trained before the rename
    DEFAULT_MODEL = HERE.parent / "models" / "prism_v1"
FP = ["score", "druggability_score", "volume", "polarity_score", "charge_score", "apolar_sasa", "polar_sasa",
      "flexibility", "number_of_alpha_spheres", "mean_local_hydrophobic_density", "n_pocket_residues"]
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
         "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
         "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}
# modified amino acids that are part of the protein chain (kept, as in the training structures)
MODIFIED = {"MSE", "SEP", "TPO", "PTR", "CSO", "CME", "CSD", "OCS", "KCX", "LLP", "MLY", "M3L", "MLZ", "ALY",
            "HYP", "PCA", "SEC", "PYL", "CSS", "NEP", "HIC", "TYS", "FME", "CAS", "CSX", "SCH", "SMC", "OCY",
            "DAL", "DLE", "DVA", "AIB", "ABA", "NLE", "ORN", "CGU", "TRQ", "HTR", "TYI", "PHD", "SNN", "CXM"}
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
MIN_CHAIN_RES = 5
WINDOW, STEP = 1022, 822


# ============================================================================ structure input
def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_pdb(path):
    atoms, model_seen = [], False
    for line in _open(path):
        rec = line[:6]
        if rec.startswith("MODEL"):
            if model_seen:
                break
            model_seen = True
        if rec.startswith("ENDMDL"):
            break
        if rec not in ("ATOM  ", "HETATM"):
            continue
        try:
            atoms.append({"rec": rec.strip(), "name": line[12:16].strip(), "alt": line[16].strip(),
                          "comp": line[17:20].strip(), "chain": line[21].strip() or "A",
                          "seq": int(line[22:26]), "ins": line[26].strip(),
                          "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                          "occ": float(line[54:60] or 1), "b": float(line[60:66] or 0),
                          "el": (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16])[:1]).upper()})
        except ValueError:
            continue
    return atoms


def _cif_tokens(line):
    return shlex.split(line, posix=True) if ("'" in line or '"' in line) else line.split()


def read_cif(path):
    """minimal mmCIF reader for the _atom_site loop (first model)."""
    lines = list(_open(path))
    i, cols, rows = 0, [], []
    while i < len(lines):
        if lines[i].strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].startswith("_atom_site."):
            i += 1
            while i < len(lines) and lines[i].startswith("_atom_site."):
                cols.append(lines[i].strip().split(".", 1)[1])
                i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith(("loop_", "_", "#")):
                buf += _cif_tokens(lines[i])
                while len(buf) >= len(cols):
                    rows.append(buf[:len(cols)])
                    buf = buf[len(cols):]
                i += 1
            break
        i += 1
    if not rows:
        raise ValueError("no _atom_site loop found")
    c = {k: j for j, k in enumerate(cols)}

    def g(r, *names, default=None):
        for n in names:
            if n in c and r[c[n]] not in ("?", "."):
                return r[c[n]]
        return default
    first = g(rows[0], "pdbx_PDB_model_num")
    atoms = []
    for r in rows:
        if first is not None and g(r, "pdbx_PDB_model_num") != first:
            continue
        try:
            atoms.append({"rec": g(r, "group_PDB", default="ATOM"), "name": g(r, "auth_atom_id", "label_atom_id"),
                          "alt": g(r, "label_alt_id", default=""), "comp": g(r, "auth_comp_id", "label_comp_id"),
                          "chain": g(r, "auth_asym_id", "label_asym_id"),
                          "seq": int(float(g(r, "auth_seq_id", "label_seq_id", default=0))),
                          "ins": g(r, "pdbx_PDB_ins_code", default=""),
                          "xyz": (float(g(r, "Cartn_x")), float(g(r, "Cartn_y")), float(g(r, "Cartn_z"))),
                          "occ": float(g(r, "occupancy", default=1)), "b": float(g(r, "B_iso_or_equiv", default=0)),
                          "el": (g(r, "type_symbol", default="") or "").upper()})
        except (TypeError, ValueError):
            continue
    return atoms


def clean_structure(path, out_pdb):
    """protein chains only -> PDB file; returns {new_chain: original_chain}, report dict."""
    p = str(path).lower()
    atoms = read_cif(path) if re.search(r"\.(cif|mmcif)(\.gz)?$", p) else read_pdb(path)
    keep = [a for a in atoms if a["el"] not in ("H", "D") and a["alt"] in ("", "A", "1")
            and (a["rec"] == "ATOM" and a["comp"] not in ("HOH", "DOD", "WAT")
                 or a["rec"] == "HETATM" and a["comp"] in MODIFIED)
            and (a["comp"] in THREE or a["comp"] in MODIFIED or a["rec"] == "ATOM")]
    # drop nucleic acids written as ATOM records (DA, DC, A, U, ...)
    keep = [a for a in keep if not re.fullmatch(r"D?[ACGTUI]", a["comp"])]
    by_chain = {}
    for a in keep:
        by_chain.setdefault(a["chain"], []).append(a)
    good = [ch for ch, lst in by_chain.items() if len({(a["seq"], a["ins"]) for a in lst}) >= MIN_CHAIN_RES]
    if not good:
        raise ValueError("no protein chain with >= 5 residues")
    if len(good) > len(CHAIN_IDS):
        raise ValueError(f"{len(good)} protein chains: more than PDB format allows ({len(CHAIN_IDS)})")
    used, cmap = set(), {}
    for ch in good:  # keep one-character chain IDs where possible
        if len(ch) == 1 and ch not in used:
            cmap[ch] = ch
            used.add(ch)
    spare = iter(c for c in CHAIN_IDS if c not in used)
    for ch in good:
        if ch not in cmap:
            cmap[ch] = next(spare)
    lines, serial = [], 0
    for ch in good:
        for a in by_chain[ch]:
            serial += 1
            name = a["name"]
            name4 = name if len(name) >= 4 else ((" " + name).ljust(4) if len(a["el"]) <= 1 else name.ljust(4))
            lines.append(f"ATOM  {serial % 100000:>5} {name4[:4]} {a['comp'][:3]:>3} {cmap[ch]}{a['seq']:>4}"
                         f"{(a['ins'] or ' ')[:1]}   {a['xyz'][0]:8.3f}{a['xyz'][1]:8.3f}{a['xyz'][2]:8.3f}"
                         f"{a['occ']:6.2f}{min(a['b'], 999.99):6.2f}          {a['el'][:2]:>2}\n")
        lines.append("TER\n")
    Path(out_pdb).write_text("".join(lines) + "END\n")
    removed = len(atoms) - len(keep)
    return {cmap[ch]: ch for ch in good}, {"chains": len(good), "atoms": serial, "removed_atoms": removed}


# ============================================================================ fpocket
def norm_key(k):
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


def parse_fpocket(out_dir, stem):
    info = out_dir / f"{stem}_info.txt"
    pockets, cur = {}, None
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
        res = set()
        atm = out_dir / "pockets" / f"pocket{p}_atm.pdb"
        if atm.exists():
            for line in open(atm):
                if line.startswith(("ATOM", "HETATM")):
                    res.add((line[21], int(line[22:26]), line[26].strip()))
        cen = [np.nan] * 3
        vert = out_dir / "pockets" / f"pocket{p}_vert.pqr"
        if vert.exists():
            pts = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in open(vert) if l.startswith("ATOM")]
            if pts:
                cen = list(np.mean(pts, 0).round(3))
        d.update(n_pocket_residues=len(res), residues=sorted(res), center_x=cen[0], center_y=cen[1], center_z=cen[2])
        rows.append(d)
    return pd.DataFrame(rows)


def run_fpocket(clean_pdb, fpocket):
    wdir = Path(tempfile.mkdtemp(prefix="alloplex_fp_"))
    shutil.copy(clean_pdb, wdir / "clean.pdb")
    p = subprocess.run([fpocket, "-f", "clean.pdb"], cwd=wdir, capture_output=True, text=True)
    out = wdir / "clean_out"
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"fpocket failed: {(p.stderr or p.stdout)[-300:]}")
    df = parse_fpocket(out, "clean")
    shutil.rmtree(wdir, ignore_errors=True)
    missing = [c for c in FP if c not in df.columns]
    if missing:
        raise RuntimeError(f"fpocket output lacks descriptors {missing} (different fpocket version?)")
    return df


# ============================================================================ ESM-2
def chains_of(pdb_path):
    out, seen = {}, set()
    for line in open(pdb_path):
        if line.startswith("ATOM  "):
            ch, num, ic, nm = line[21], int(line[22:26]), line[26].strip(), line[17:20].strip()
            if (ch, num, ic) in seen:
                continue
            seen.add((ch, num, ic))
            out.setdefault(ch, []).append((num, ic, THREE.get(nm, "X")))
    return out


def windows(n):
    if n <= WINDOW:
        return [(0, n)]
    starts = list(range(0, n - WINDOW, STEP)) + [n - WINDOW]
    return [(s, s + WINDOW) for s in sorted(set(starts))]


class ESM:
    def __init__(self, device, fake=False):
        self.fake = fake
        if fake:
            return
        import torch
        import esm
        self.torch = torch
        self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.eval().to(device)
        if device == "cuda":
            self.model = self.model.half()
        self.device = device
        self.bc = alphabet.get_batch_converter()

    def embed_chain(self, seq):
        """whole chain -> (len x 1280) float32, windows averaged (as in training)."""
        if self.fake:
            return np.stack([np.full(1280, (ord(c) % 7) / 7.0, np.float32) for c in seq])
        acc = np.zeros((len(seq), 1280), np.float32)
        cnt = np.zeros(len(seq), np.float32)
        for a, b in windows(len(seq)):
            _, _, toks = self.bc([("x", seq[a:b])])
            with self.torch.no_grad():
                rep = self.model(toks.to(self.device), repr_layers=[33])["representations"][33]
            acc[a:b] += rep.float().cpu().numpy()[0, 1:b - a + 1]
            cnt[a:b] += 1
        return acc / cnt[:, None]


# ============================================================================ model
def load_members(model_dir, device, pattern="member_*.pt"):
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                     nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
                                     nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    members = []
    for f in sorted(Path(model_dir).glob(pattern)):
        ck = torch.load(f, map_location=device)
        m = MLP(ck["n_in"]).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        members.append((m, np.array(ck["mu"], np.float32), np.array(ck["sd"], np.float32)))
    if not members:
        raise SystemExit(f"no model files ({pattern}) in {model_dir}")
    return members


def score(members, X_esm, X_fp, device, fake=False):
    if fake:
        return X_fp[:, 1].astype(float)  # testing only: druggability
    import torch
    out = np.zeros(len(X_esm))
    with torch.no_grad():
        for m, mu, sd in members:
            X = np.c_[X_esm, (X_fp - mu) / sd].astype(np.float32)
            out += torch.sigmoid(m(torch.tensor(X, device=device))).cpu().numpy() / len(members)
    return out


# ============================================================================ helper for analyses
def score_structure(path, workdir, fpocket, esm, members, device, timings=None):
    """clean -> fpocket -> ESM-2 -> model for one structure (used by the step 10 analyses).
    Returns the pocket table (residues as [(chain, resnum, icode)], 'alloplex_score', fpocket
    descriptors), not sorted. timings: optional dict that receives seconds per stage."""
    t = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    clean = workdir / "clean.pdb"
    clean_structure(path, clean)
    t1 = time.time()
    P = run_fpocket(clean, fpocket)
    t2 = time.time()
    if P.empty:
        raise RuntimeError("fpocket found no pockets")
    emb = {}
    for ch, res in chains_of(clean).items():
        emb[ch] = (esm.embed_chain("".join(r[2] for r in res)), {(r[0], r[1]): i for i, r in enumerate(res)})
    X_esm = np.zeros((len(P), 1280), np.float32)
    for i, res in enumerate(P.residues):
        vecs = [emb[c][0][emb[c][1][(n, ic)]] for c, n, ic in res if c in emb and (n, ic) in emb[c][1]]
        if vecs:
            X_esm[i] = np.mean(vecs, 0)
    X_esm = X_esm.astype(np.float16).astype(np.float32)
    t3 = time.time()
    P["alloplex_score"] = score(members, X_esm, P[FP].to_numpy(np.float32), device, fake=members is None)
    t4 = time.time()
    if timings is not None:
        timings.update(clean=t1 - t, fpocket=t2 - t1, esm=t3 - t2, model=t4 - t3, total=t4 - t)
    return P


# ============================================================================ one structure
def predict_one(path, args, esm, members, device, used):
    name = re.sub(r"\.(pdb|ent|cif|mmcif)(\.gz)?$", "", Path(path).name, flags=re.I)
    base, k = name, 2
    while name in used:  # two inputs with the same name (e.g. x.pdb and x.cif)
        name, k = f"{base}_{k}", k + 1
    used.add(name)
    od = Path(args.out) / name
    od.mkdir(parents=True, exist_ok=True)
    clean = od / "clean.pdb"
    cmap, rep = clean_structure(path, clean)
    P = run_fpocket(clean, args.fpocket)
    if P.empty:
        raise RuntimeError("fpocket found no pockets")
    # ESM per chain
    emb, seq_of = {}, {}
    for ch, res in chains_of(clean).items():
        seq_of[ch] = "".join(r[2] for r in res)
        e = esm.embed_chain(seq_of[ch])
        emb[ch] = (e, {(r[0], r[1]): i for i, r in enumerate(res)})
    X_esm = np.zeros((len(P), 1280), np.float32)
    n_found = np.zeros(len(P), int)
    for i, res in enumerate(P.residues):
        vecs = [emb[c][0][emb[c][1][(n, ic)]] for c, n, ic in res if c in emb and (n, ic) in emb[c][1]]
        if vecs:
            X_esm[i] = np.mean(vecs, 0)
            n_found[i] = len(vecs)
    X_esm = X_esm.astype(np.float16).astype(np.float32)  # training features were stored as float16
    X_fp = P[FP].to_numpy(np.float32)
    P["alloplex_score"] = score(members, X_esm, X_fp, device, fake=args.fake_model)
    P = P.sort_values("alloplex_score", ascending=False).reset_index(drop=True)
    P.insert(0, "rank", np.arange(1, len(P) + 1))
    # symmetric copies: the same pocket on identical chains (homo-oligomers) -> one site group
    # chains are copies of the same protein if their residues agree where both are resolved
    # (>= 80 % of the shorter chain shared, >= 95 % identical there; missing loops may differ)
    resmap = {ch: {(n, ic): aa for n, ic, aa in res} for ch, res in chains_of(clean).items()}
    ent, reps_e = {}, []
    for ch in resmap:
        for e, r in enumerate(reps_e):
            shared = resmap[ch].keys() & resmap[r].keys()
            if len(shared) >= 0.8 * min(len(resmap[ch]), len(resmap[r])) and \
                    sum(resmap[ch][k] == resmap[r][k] for k in shared) >= 0.95 * max(1, len(shared)):
                ent[ch] = e
                break
        else:
            ent[ch] = len(reps_e)
            reps_e.append(ch)
    sig = [{(ent.get(c, c), n, ic) for c, n, ic in r} for r in P.residues]
    group, reps = [], []
    for i, sg in enumerate(sig):
        g = next((k for k, j in enumerate(reps) if sg and len(sg & sig[j]) / len(sg | sig[j]) >= 0.6), None)
        if g is None:
            reps.append(i)
            g = len(reps) - 1
        group.append(g + 1)
    P["site_group"] = group
    P["copies_in_group"] = P.site_group.map(P.site_group.value_counts())
    P["chains"] = [",".join(sorted({c for c, _, _ in r})) for r in P.residues]
    P["original_chains"] = [",".join(sorted({cmap.get(c, c) for c, _, _ in r})) for r in P.residues]
    P["residues"] = [" ".join(f"{c}:{n}{ic}" for c, n, ic in r) for r in P.residues]
    cols = ["rank", "site_group", "copies_in_group", "pocket", "alloplex_score", "druggability_score", "score", "volume", "n_pocket_residues", "chains",
            "original_chains", "center_x", "center_y", "center_z", "residues"]
    P[cols].rename(columns={"pocket": "fpocket_pocket", "score": "fpocket_score",
                            "druggability_score": "fpocket_druggability"}).to_csv(od / "alloplex_pockets.csv",
                                                                                 index=False, float_format="%.4f")
    # PyMOL view of the top pockets
    colours = ["red", "orange", "yellow", "green", "cyan"]
    pml = [f"load {clean.name}, prot", "hide everything", "show cartoon, prot", "color grey80, prot"]
    for k in range(min(len(colours), len(P))):
        sel = " or ".join(f"(chain {t.split(':')[0]} and resi {t.split(':')[1]})" for t in P.residues[k].split())
        pml += [f"select pocket_rank{k + 1}, prot and ({sel})", f"show sticks, pocket_rank{k + 1}",
                f"color {colours[k]}, pocket_rank{k + 1}"]
    pml += ["orient pocket_rank1", "set cartoon_transparency, 0.3"]
    (od / "alloplex_view.pml").write_text("\n".join(pml) + "\n")
    return name, P, cmap, rep, int((n_found == 0).sum()), od


def main():
    ap = argparse.ArgumentParser(description="AlloPlex: rank pockets of a protein assembly by allosteric likelihood")
    ap.add_argument("structures", nargs="+", help=".pdb/.ent/.cif/.mmcif files (optionally .gz)")
    ap.add_argument("--out", default="alloplex_results")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--fpocket", default=shutil.which("fpocket") or "fpocket")
    ap.add_argument("--top", type=int, default=5, help="pockets shown on the screen")
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    ap.add_argument("--members", default="member_*.pt", help="which model files to use (default: all 15)")
    ap.add_argument("--fake-model", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fake-esm", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    device = "cpu"
    if not args.fake_model or not args.fake_esm:
        import torch
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    members = None if args.fake_model else load_members(args.model_dir, device, args.members)
    esm = ESM(device, fake=args.fake_esm)
    print(f"AlloPlex v1: {0 if members is None else len(members)} ensemble members; device {device}")
    used = set()
    for path in args.structures:
        t0 = time.time()
        try:
            name, P, cmap, rep, n_noesm, od = predict_one(path, args, esm, members, device, used)
        except Exception as e:
            print(f"\n{path}: FAILED - {type(e).__name__}: {e}")
            continue
        ren = {k: v for k, v in cmap.items() if k != v}
        print(f"\n{name}: {rep['chains']} protein chains, {len(P)} pockets ({time.time() - t0:.0f}s)"
              + (f"; chains renamed {ren}" if ren else ""))
        if n_noesm:
            print(f"  note: {n_noesm} pockets had no residue with an ESM vector")
        print(f"  top {args.top} distinct sites (symmetric copies on identical chains are grouped):")
        print(f"  {'site':>4} {'score':>6} {'drugg.':>6} {'res':>4}  {'copies':<16} first residues")
        best = P.drop_duplicates("site_group").head(args.top)
        for _, r in best.iterrows():
            g = P[P.site_group == r.site_group]
            copies = f"{len(g)} (chains {'/'.join(g.chains)})" if len(g) > 1 else f"1 (chain {r.chains})"
            first = " ".join(r.residues.split()[:6]) + (" ..." if len(r.residues.split()) > 6 else "")
            print(f"  {r.site_group:>4} {r.alloplex_score:>6.3f} {r.druggability_score:>6.2f} {int(r.n_pocket_residues):>4}  "
                  f"{copies:<16} {first}")
        print(f"  -> {od / 'alloplex_pockets.csv'}  (all pockets, residues)   PyMOL: cd {od} && pymol alloplex_view.pml")


if __name__ == "__main__":
    main()
