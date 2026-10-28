#!/usr/bin/env python
"""
STEP 4Z: IS THE "NEAR THE INTERFACE" SIGNAL JUST "NEAR THE MIDDLE OF THE PROTEIN"?
==================================================================================
Exploratory check of the S3 / random-patch finding. Same pockets as S3a
(primary-set pockets with no interface residue at 5 A^2).

  1. S3 score: minus the distance to the nearest interface residue (as step 4x)
  2. centrality score: minus the distance from the pocket centre to the centre of
     the whole assembly, divided by the assembly's radius of gyration
  3. S3 score again, now ALSO matched on centrality: cells = structure x size
     tercile x centrality tercile. If the interface signal survives this, it is not
     explained by pockets simply lying closer to the middle of the protein.
  + Spearman correlation between the two distances among negative pockets.

Uses the matched-AUC machinery of step4x_primary_test.py (same folder).
RUN (from ~/allosite):  python scripts/step4z_centrality_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4x_primary_test as s4  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
STRUCT = BASE / "processed" / "structures"


def assembly_centre(pdb):
    xyz = []
    for line in open(STRUCT / pdb / "stripped.pdb"):
        if line.startswith("ATOM  "):
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    xyz = np.array(xyz)
    c = xyz.mean(0)
    rg = np.sqrt(((xyz - c) ** 2).sum(1).mean())
    return c, rg


def main():
    P = pd.read_csv(D / "pockets_stripped.csv", low_memory=False,
                    usecols=["pdb_id", "pocket", "n_pocket_residues", "role_primary", "family", "n_iface_5",
                             "dist_iface_5", "center_x", "center_y", "center_z"])
    P = P[P.role_primary.isin(["positive", "negative"]) & (P.n_iface_5 == 0) & P.dist_iface_5.notna()].copy()
    P = P[P[["center_x", "center_y", "center_z"]].notna().all(axis=1)]
    P["y"] = (P.role_primary == "positive").astype(int)
    cen = {}
    for pdb in P.pdb_id.unique():
        cen[pdb] = assembly_centre(pdb)
    xyz = P[["center_x", "center_y", "center_z"]].to_numpy()
    c = np.array([cen[p][0] for p in P.pdb_id])
    rg = np.array([cen[p][1] for p in P.pdb_id])
    P["centrality"] = np.linalg.norm(xyz - c, axis=1) / rg
    P["neg_dist_iface"] = -P.dist_iface_5
    P["neg_centrality"] = -P.centrality

    s4.say("=" * 100)
    s4.say("CENTRALITY CHECK (exploratory): pockets with no interface residue")
    s4.say("=" * 100)
    base = s4.add_cells(P)
    s4.matched_auc(base, "neg_dist_iface", "1 closer to the interface (S3 score)", 2000, kind="exploratory")
    s4.matched_auc(base, "neg_centrality", "2 closer to the assembly centre", 2000, kind="exploratory")
    both = base.copy()
    both["ctercile"] = pd.qcut(both.centrality.rank(method="first"), 3, labels=False)
    both["cell"] = both.cell + "|" + both.ctercile.astype(str)
    s4.matched_auc(both, "neg_dist_iface", "3 closer to the interface, ALSO matched on centrality", 2000,
                   kind="exploratory")
    neg = P[P.y == 0]
    rho = spearmanr(neg.dist_iface_5, neg.centrality).correlation
    s4.say(f"  Spearman correlation, distance to interface vs distance to centre (negatives): {rho:.2f}")
    s4.say("")
    s4.say("How to read: if line 3 stays above 0.5 with a family interval excluding 0.5, the proximity to the")
    s4.say("interface is not explained by pockets lying closer to the middle of the assembly.")
    pd.DataFrame(s4.RESULTS).to_csv(D / "results_step4z.csv", index=False)
    (D / "report_step4z.txt").write_text("\n".join(s4.LOG) + "\n")
    s4.say(f"Saved {D / 'results_step4z.csv'}")


if __name__ == "__main__":
    main()
