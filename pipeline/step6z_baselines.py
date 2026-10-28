#!/usr/bin/env python
"""
STEP 6Z: SIMPLE BASELINES ON THE SAME SITES AS THE MODELS
=========================================================
How much does the trained model add over ranking pockets by a single number?
Baselines (no training, so no folds needed):
    fpocket score | fpocket druggability | pocket size (residues) | pocket volume | random
Scored with exactly the same site-level metrics as step 6x (Top-1, Top-3, site AUROC;
undetected sites = misses; PRIMARY = partner where available, else stripped), then the
MLP (feature set B, mean over repeats) is compared with the best baseline using paired
differences and 95 % family-bootstrap intervals.

RUN (from ~/allosite):  python scripts/step6z_baselines.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

FP_COL = {"fpocket score": 0, "fpocket druggability": 1, "pocket volume": 2}


def site_metrics(sets, tabs, scorer):
    per = {}
    for en in ("stripped", "partner"):
        if en not in sets:
            continue
        sc = scorer(sets[en])
        per[en] = s6.score_sites(tabs[en], sets[en]["meta"], sc)
    prim = per["stripped"].set_index("site_id")
    if "partner" in per:
        part = per["partner"].set_index("site_id")
        prim = pd.concat([part, prim[~prim.index.isin(part.index)]])
    per["primary"] = prim.reset_index()
    return per


def main():
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {"stripped": s6.site_table("stripped", sets["stripped"]["meta"], s6.D / "sites_stripped.csv")}
    if "partner" in sets:
        tabs["partner"] = s6.site_table("partner", sets["partner"]["meta"], s6.D / "sites_partner.csv")
    rng = np.random.default_rng(s6.SEED)
    scorers = {name: (lambda S, c=c: S["fp"][:, c].astype(float)) for name, c in FP_COL.items()}
    scorers["pocket size (residues)"] = lambda S: S["meta"].n_pocket_residues.to_numpy(float)
    scorers["random"] = lambda S: rng.random(len(S["y"]))
    res = {name: site_metrics(sets, tabs, f) for name, f in scorers.items()}

    # the MLP (B), averaged over repeats, from saved predictions
    mlp = {}
    for en in ("stripped", "partner"):
        sc = s6.collect("mlp", "B", en, sets[en]["meta"], [0, 1, 2])
        if sc is None:
            continue
        tabs_r = [s6.score_sites(tabs[en], sets[en]["meta"], s) for s in sc]
        mlp[en] = pd.concat(tabs_r).groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean()
    if "stripped" in mlp:
        prim = mlp["stripped"].set_index("site_id")
        if "partner" in mlp:
            part = mlp["partner"].set_index("site_id")
            prim = pd.concat([part, prim[~prim.index.isin(part.index)]])
        mlp["primary"] = prim.reset_index()
        res["MLP (B, trained)"] = mlp

    s6.say("=" * 96)
    s6.say("BASELINES vs MODEL (same sites, same site-level metrics)")
    s6.say("=" * 96)
    s6.say(f"  {'ranking by':<26}{'set':<10}{'sites':>7}{'Top-1':>9}{'Top-3':>9}{'site AUROC':>12}")
    for name, per in res.items():
        for en in ("primary", "partner", "stripped"):
            if en in per:
                t = per[en]
                s6.say(f"  {name:<26}{en:<10}{len(t):>7}{t.top1.mean():>9.3f}{t.top3.mean():>9.3f}{t.site_auc.mean():>12.3f}")
    if "MLP (B, trained)" in res:
        base = [n for n in res if n not in ("MLP (B, trained)", "random")]
        best = max(base, key=lambda n: res[n]["primary"].top1.mean())
        s6.say("")
        s6.say(f"  MLP (B) minus best baseline ({best}), PRIMARY, 95 % family interval:")
        a = res["MLP (B, trained)"]["primary"].set_index("site_id")
        b = res[best]["primary"].set_index("site_id")
        common = a.index.intersection(b.index)
        for m in ("top1", "top3", "site_auc"):
            d = a.loc[common, m] - b.loc[common, m]
            lo, hi = s6.boot(a.loc[common, "family"].to_numpy(), d.to_numpy())
            s6.say(f"     {m:<9} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    (s6.B / "report_step6z.txt").write_text("\n".join(s6.LOG) + "\n")


if __name__ == "__main__":
    main()
