#!/usr/bin/env python
"""
STEP 5X: ESM-2 RESIDUE EMBEDDINGS FOR ALL STRUCTURES (benchmark, part 1)
=======================================================================
For every chain of every built structure (holo: processed/structures/<PDB>/stripped.pdb,
partners: processed/structures_partner/<PDB>/stripped.pdb):
  - sequence = residues present in the structure, in file order (as in the old step 7)
  - each UNIQUE sequence is embedded once with ESM-2 650M (esm2_t33_650M_UR50D, last
    layer, 1280 numbers per residue); chains longer than 1022 residues are embedded in
    overlapping windows (1022 long, step 822) and the overlaps averaged
  - saved as float16: processed/dataset_v2/esm/<sequence hash>.npy  (length x 1280)
  - index: processed/dataset_v2/esm_index.csv
      set (holo/partner), pdb_id, chain, seq_hash, length, residues (JSON list of
      [resnum, icode] in the same order as the embedding rows)

Pocket features (next script) average these rows over each pocket's residues; the
GNN uses them per residue.

RUN (from ~/allosite, GPU, conda env with torch + fair-esm, as in the old step 7):
    python scripts/step5x_esm_embeddings.py              # resumes: finished sequences are skipped
    python scripts/step5x_esm_embeddings.py --max-tokens 4000   # if the GPU runs out of memory
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
EMB = D / "esm"
SETS = {"holo": BASE / "processed" / "structures", "partner": BASE / "processed" / "structures_partner"}
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
         "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
         "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}
WINDOW, STEP = 1022, 822


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


def seq_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:20]


def windows(n):
    if n <= WINDOW:
        return [(0, n)]
    starts = list(range(0, n - WINDOW, STEP)) + [n - WINDOW]
    return [(s, s + WINDOW) for s in sorted(set(starts))]


class ESM:
    def __init__(self, fake=False):
        self.fake = fake
        if fake:
            return
        import torch
        import esm
        self.torch = torch
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, seqs):
        """list of strings (each <= 1022) -> list of arrays (len x 1280, float32)."""
        if self.fake:
            return [np.stack([np.full(1280, (ord(c) % 7) / 7.0, np.float32) for c in s]) for s in seqs]
        torch = self.torch
        _, _, toks = self.batch_converter([(str(i), s) for i, s in enumerate(seqs)])
        with torch.no_grad():
            rep = self.model(toks.to(self.device), repr_layers=[33])["representations"][33]
        rep = rep.float().cpu().numpy()
        return [rep[i, 1:len(s) + 1] for i, s in enumerate(seqs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=8000, help="residues per GPU batch")
    ap.add_argument("--fake", action="store_true", help="testing only: no model, deterministic fake vectors")
    args = ap.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)

    rows, seqs = [], {}
    for set_name, root in SETS.items():
        if not root.exists():
            continue
        for sdir in sorted(p for p in root.iterdir() if (p / "stripped.pdb").exists()):
            for ch, res in chains_of(sdir / "stripped.pdb").items():
                s = "".join(r[2] for r in res)
                h = seq_hash(s)
                seqs[h] = s
                rows.append({"set": set_name, "pdb_id": sdir.name, "chain": ch, "seq_hash": h, "length": len(s),
                             "residues": json.dumps([[r[0], r[1]] for r in res])})
    idx = pd.DataFrame(rows)
    idx.to_csv(D / "esm_index.csv", index=False)
    todo = {h: s for h, s in seqs.items() if not (EMB / f"{h}.npy").exists()}
    print(f"{len(idx)} chains in {idx.pdb_id.nunique()} structures; {len(seqs)} unique sequences; "
          f"{len(todo)} still to embed (lengths: median {int(np.median([len(s) for s in seqs.values()]))}, "
          f"max {max(len(s) for s in seqs.values())})")
    if not todo:
        print("nothing to do")
        return

    model = ESM(fake=args.fake)
    # every sequence -> windows; windows batched by length
    pieces = []
    for h, s in todo.items():
        for a, b in windows(len(s)):
            pieces.append((h, a, b, s[a:b]))
    pieces.sort(key=lambda p: len(p[3]))
    acc = {h: (np.zeros((len(s), 1280), np.float32), np.zeros(len(s), np.float32)) for h, s in todo.items()}
    remaining = {h: len(windows(len(s))) for h, s in todo.items()}
    t0, done, i = time.time(), 0, 0
    while i < len(pieces):
        batch = [pieces[i]]
        while i + len(batch) < len(pieces) and (len(batch) + 1) * len(pieces[i + len(batch)][3]) <= args.max_tokens:
            batch.append(pieces[i + len(batch)])
        try:
            embs = model.embed([p[3] for p in batch])
        except RuntimeError as e:  # out of memory: retry one at a time
            if "out of memory" not in str(e).lower() or len(batch) == 1:
                raise
            model.torch.cuda.empty_cache()
            embs = [model.embed([p[3]])[0] for p in batch]
        for (h, a, b, _), e in zip(batch, embs):
            s, c = acc[h]
            s[a:b] += e
            c[a:b] += 1
            remaining[h] -= 1
            if remaining[h] == 0:
                np.save(EMB / f"{h}.npy", (s / c[:, None]).astype(np.float16))
                del acc[h]
                done += 1
        i += len(batch)
        if done and (done % 200 == 0 or i == len(pieces)):
            print(f"  {done}/{len(todo)} sequences  ({time.time() - t0:.0f}s)")
    print(f"done: {done} sequences embedded in {time.time() - t0:.0f}s; saved in {EMB}")


if __name__ == "__main__":
    main()
