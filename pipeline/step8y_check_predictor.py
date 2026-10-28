#!/usr/bin/env python
"""
STEP 8Y: CHECK THAT prism_predict.py REPRODUCES THE TRAINING FEATURES
====================================================================
For a few benchmark structures, runs the predictor's own pipeline (cleaning -> fpocket -> ESM-2 ->
features -> final model) on processed/structures/<PDB>/stripped.pdb and compares with what the
benchmark used (pockets_stripped.csv descriptors, features_stripped.npz ESM means):
  - same number of pockets and same pocket residues?
  - fpocket descriptors identical?
  - ESM pocket vectors nearly identical (cosine similarity ~1)?
  - final-model scores from the predictor's features vs from the benchmark features (should agree)
If all agree, the released tool computes exactly the inputs the model was trained and tested on.

RUN (from ~/allosite, env allosite, after step 8x):
    python scripts/step8y_check_predictor.py --n 5
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prism_predict as pp  # noqa: E402
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--pdb", nargs="*", help="specific PDB IDs (default: random)")
    ap.add_argument("--model-dir", default=str(BASE / "models" / "prism_v1"))
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = s6.load_set("stripped")
    meta = S["meta"]
    bench = pd.read_csv(D / "pockets_stripped.csv", low_memory=False)
    pdbs = args.pdb or random.Random(7).sample(sorted(set(meta.pdb_id)), args.n)
    members = pp.load_members(args.model_dir, device)
    esm = pp.ESM(device)
    fpocket = pp.shutil.which("fpocket")
    ok_all = True
    for pdb in pdbs:
        src = BASE / "processed" / "structures" / pdb / "stripped.pdb"
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.pdb"
            pp.clean_structure(src, clean)
            P = pp.run_fpocket(clean, fpocket)
            emb = {}
            for ch, res in pp.chains_of(clean).items():
                emb[ch] = (esm.embed_chain("".join(r[2] for r in res)), {(r[0], r[1]): i for i, r in enumerate(res)})
        B = bench[bench.pdb_id == pdb].set_index("pocket")
        rows = meta.index[meta.pdb_id == pdb].to_numpy()
        mrow = dict(zip(meta.pocket.iloc[rows], rows))
        n_same_res = n_fp_eq = 0
        cos, s_new, s_old = [], [], []
        rel = {c: 0.0 for c in pp.FP}
        for _, r in P.iterrows():
            p = int(r.pocket)
            if p not in B.index:
                continue
            old_res = {(c, int(n), (ic[0] if ic else "")) for c, n, *ic in json.loads(B.loc[p, "residues"])}
            new_res = set(r.residues)
            n_same_res += old_res == new_res
            a = r[pp.FP].astype(float).to_numpy()
            b = B.loc[p, pp.FP].astype(float).to_numpy()
            d = np.abs(a - b) / (np.abs(b) + 1e-6)
            for c, x in zip(pp.FP, d):
                rel[c] = max(rel[c], float(x))
            # fpocket's volume is a Monte-Carlo estimate (differs slightly run to run): 5 % tolerance
            n_fp_eq += bool(all(x <= (0.05 if c == "volume" else 1e-3) for c, x in zip(pp.FP, d)))
            vecs = [emb[c][0][emb[c][1][(n, ic)]] for c, n, ic in new_res if c in emb and (n, ic) in emb[c][1]]
            if p in mrow and vecs:
                new = np.mean(vecs, 0).astype(np.float16).astype(np.float32)
                old = S["esm"][mrow[p]].astype(np.float32)
                cos.append(float(new @ old / (np.linalg.norm(new) * np.linalg.norm(old) + 1e-9)))
                fp = r[pp.FP].to_numpy(np.float32)[None]
                s_new.append(pp.score(members, new[None], fp, device)[0])
                s_old.append(pp.score(members, old[None], S["fp"][mrow[p]][None], device)[0])
        n = len(P)
        good = n == len(B) and n_same_res == n and n_fp_eq == n and (min(cos) if cos else 0) > 0.995
        ok_all &= good
        print(f"{pdb}: pockets new {n} / benchmark {len(B)}; same residues {n_same_res}/{n}; "
              f"same descriptors {n_fp_eq}/{n}; ESM cosine min {min(cos) if cos else float('nan'):.4f}; "
              f"score max |diff| {np.max(np.abs(np.array(s_new) - np.array(s_old))) if s_new else float('nan'):.4f}"
              f"  -> {'OK' if good else 'CHECK'}")
        diff = {c: round(x, 4) for c, x in rel.items() if x > 1e-3}
        if diff:
            print(f"      descriptors that differ (max relative difference): {diff}")
    print("\nALL OK: the predictor reproduces the benchmark inputs" if ok_all else
          "\nSome differences (see above). Small ESM differences (GPU half precision) and small volume differences "
          "(fpocket Monte-Carlo volume) are harmless; different pockets would mean a different fpocket version.")


if __name__ == "__main__":
    main()
