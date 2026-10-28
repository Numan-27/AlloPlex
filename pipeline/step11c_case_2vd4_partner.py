#!/usr/bin/env python
"""
STEP 11C: 2VD4 failure case on its PRIMARY (partner) structure 2V0H
===================================================================
step11b looked the case up by pdb_id, and the partner table stores the partner PDB (2V0H), so only the
stripped structure was reported. This scores just the pockets of the partner structure with the released
networks that never trained on the family (member_f<k>_s*.pt, k = family fold in repeat 0), then prints
every method's rank and AlloPlex's top-3 pockets.

RUN (from ~/allosite, env allosite; small, fine while other GPU jobs run):
    python scripts/step11c_case_2vd4_partner.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402
import step11_reviewer_analyses as s11  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
SITE = sys.argv[1] if len(sys.argv) > 1 else "2VD4"


def main():
    import torch
    import alloplex_predict as pp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets, tabs, methods = s11.load()
    for n in ("partner", "stripped"):
        S = sets[n]
        t = tabs[n][tabs[n].site_id.astype(str).str.startswith(SITE)]
        if t.empty:
            print(f"{SITE}: no site in the {n} table")
            continue
        pdb = t.pdb_id.iloc[0]
        rows = np.where(S["meta"].pdb_id.to_numpy() == pdb)[0]
        fam = str(S["meta"].family.iloc[rows[0]])
        folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
        k = int(folds[(folds.repeat == 0) & (folds.family == fam)].fold.iloc[0])
        mdir = next(BASE / "models" / m for m in ("alloplex_v1", "prism_v1") if (BASE / "models" / m).exists())
        members = pp.load_members(mdir, device, f"member_f{k}_s*.pt")
        sc = np.full(len(S["y"]), np.nan)
        acc = np.zeros(len(rows))
        for m, mu, sd in members:
            X = np.c_[S["esm"][rows].astype(np.float32), (S["fp"][rows] - mu) / sd].astype(np.float32)
            with torch.no_grad():
                acc += torch.sigmoid(m(torch.tensor(X, device=device))).cpu().numpy()
        sc[rows] = acc / len(members)
        meth = dict(methods)
        meth["AlloPlex (held-out nets)"] = ([{n: sc}], set())

        print(f"\n{SITE} on the {n} structure {pdb} (family {fam}, fold {k}, {len(rows)} pockets)")
        print(t.drop(columns=[c for c in t.columns if c in ("residues",)], errors="ignore").to_string(index=False))
        for m, (runs, _) in meth.items():
            if n not in runs[0]:
                continue
            e = s11.eval_set(t, S, runs[0][n])
            r = e["rank"].iloc[0] if len(e) else np.nan
            print(f"  {m:<26} rank {r}")

        pk = pd.read_csv(s6.D / f"pockets_{n}.csv", usecols=["pdb_id", "pocket", "residues"], low_memory=False)
        order = rows[np.argsort(-np.nan_to_num(sc[rows], nan=-1))][:3]
        for i, j in enumerate(order, 1):
            res = json.loads(pk.residues.iloc[j])
            lab = {1: "SITE", 0: "other", -1: "set aside"}[int(S["y"][j])]
            print(f"  AlloPlex rank {i}: pocket {pk.pocket.iloc[j]} ({lab}), score {sc[j]:.3f}, {len(res)} residues on "
                  f"chain(s) {','.join(sorted({c for c, *_ in res}))}, volume {S['fp'][j, 2]:.0f} A^3, "
                  f"druggability {S['fp'][j, 1]:.2f}")
        for r in t.itertuples():
            if getattr(r, "pos_row", -1) >= 0:
                j = r.pos_row
                rk = 1 + int((sc[rows] > sc[j]).sum())
                print(f"  site pocket {pk.pocket.iloc[j]}: score {sc[j]:.3f} (rank {rk}), volume {S['fp'][j, 2]:.0f} A^3, "
                      f"druggability {S['fp'][j, 1]:.2f}, {int(S['meta'].n_pocket_residues.iloc[j])} residues, "
                      f"best Jaccard {r.best_j:.2f}")


if __name__ == "__main__":
    main()
