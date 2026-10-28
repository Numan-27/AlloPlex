#!/usr/bin/env python
"""
STEP 13C: CAN THE SMALL-POCKET BIAS BE CORRECTED BY POCKET-SIZE-WEIGHTED TRAINING? (supervisor point 4)
=====================================================================================================
Cross-validation with the paper's folds (folds.csv) and inner-validation rule (step 6x run()), feature
set B, three variants trained under IDENTICAL settings (same seeds, same folds, same epochs):
    baseline   unweighted focal loss                      (= AlloPlex, but with --seeds networks per fold)
    posbins    allosteric (positive) pockets weighted so that each volume tercile contributes equally
    allbins    the same for the positive AND the negative pockets separately, so that pocket volume
               alone no longer separates the classes in the training loss
Volume terciles are those of the positive training pockets (fpocket volume), recomputed per fold.
Evaluation exactly as in the paper: PRIMARY sites (partner structure where available, else stripped),
Top-1 with ties broken at random; per volume tercile of the site pocket as in the error analysis.

Mean pooling of ESM-2 embeddings is already normalized by pocket size (a mean, not a sum), so
'length-normalized pooling' is what the model already uses; a different pooling (max or attention)
would need the per-residue embeddings, which the benchmark files do not store - not tested here.

RUN (from ~/allosite, env allosite; GPU recommended):
    python scripts/step13c_size_weighted.py --repeats 0 --seeds 3        # 3 variants x 5 folds x 3 seeds
    python scripts/step13c_size_weighted.py --repeats 0,1,2 --seeds 3    # all repeats (3x longer)
    python scripts/step13c_size_weighted.py --report-only               # re-evaluate saved predictions
Output: processed/dataset_v2/bench/preds_step13c/*.npz, figures/data/s13c_size_weighted.csv,
        figures/data/report_step13c.txt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402
import step10a_extra_analyses as s10a  # noqa: E402
import step13_common as c13  # noqa: E402

PRED = s6.B / "preds_step13c"
DATA = s6.BASE / "figures" / "data"
VARIANTS = ("baseline", "posbins", "allbins")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def weights_for(variant, vol, y):
    """per-sample loss weights from the fpocket volume of each training pocket"""
    if variant == "baseline":
        return None
    q = np.quantile(vol[y == 1], [1 / 3, 2 / 3])
    tb = np.digitize(vol, q)                     # 0 small, 1 medium, 2 large
    w = np.ones(len(y), np.float64)
    classes = (1,) if variant == "posbins" else (0, 1)
    for c in classes:
        m = y == c
        for t in range(3):
            share = ((tb == t) & m).sum() / max(1, m.sum())
            if share > 0:
                w[(tb == t) & m] = 1.0 / (3 * share)
    return w.astype(np.float32)


def run_fold(variant, rep, fold, sets, groups, args, device):
    out = PRED / f"{variant}__r{rep}__f{fold}.npz"
    if out.exists():
        return
    S = sets["stripped"]
    g = groups["stripped"]
    lab = S["y"] >= 0
    test_g = set(g["groups_by_fold"][fold])
    is_test = g["group"].isin(test_g).to_numpy()
    train_groups = sorted(set(g["group"][~is_test & lab]))
    rng = np.random.default_rng(s6.SEED + 1000 * rep + fold)                 # same inner split as step 6x
    inner = set(rng.choice(train_groups, max(1, int(s6.INNER_FRAC * len(train_groups))), replace=False))
    is_inner = g["group"].isin(inner).to_numpy()
    tr = np.where(lab & ~is_test & ~is_inner)[0]
    va = np.where(lab & ~is_test & is_inner)[0]
    evals = [e for e in ("stripped", "partner") if e in sets]
    tests = {en: np.where(groups[en]["group"].isin(test_g).to_numpy() & (sets[en]["y"] >= 0))[0] for en in evals}
    Xtr, Xva, *Xtests = s6.standardise(s6.matrix(S, "B", tr), s6.matrix(S, "B", va),
                                       *[s6.matrix(sets[en], "B", tests[en]) for en in evals])
    ytr, yva = S["y"][tr], S["y"][va]
    w = weights_for(variant, S["fp"][tr, 2].astype(float), ytr)
    members = c13.train_members(Xtr, ytr, Xva, yva, range(args.seeds), device, weights=w)
    preds = [np.zeros(len(x)) for x in Xtests]
    for s, state, best in members:
        m = c13.load_state(state, Xtr.shape[1], device)
        for k, X in enumerate(Xtests):
            preds[k] += c13.predict(m, X, device) / len(members)
    np.savez(out, **{f"{en}_rows": tests[en] for en in evals}, **{f"{en}_pred": p for en, p in zip(evals, preds)})


def collect(variant, name, meta, repeats):
    out = []
    for rep in repeats:
        sc = np.full(len(meta), np.nan)
        files = sorted(PRED.glob(f"{variant}__r{rep}__f*.npz"))
        if len(files) < 5:
            return None
        for f in files:
            z = np.load(f)
            if f"{name}_rows" in z:
                sc[z[f"{name}_rows"]] = z[f"{name}_pred"]
        out.append(sc)
    return out


def evaluate(sets, repeats):
    tabs = {n: s6.site_table(n, sets[n]["meta"], s10a.D / f"sites_{n}.csv") for n in sets}
    attrs = s10a.site_attributes(sets, tabs)
    vol_cat = s10a.bins(attrs)["Pocket volume"]
    per = {}
    for v in VARIANTS:
        parts = {}
        for n in sets:
            sc = collect(v, n, sets[n]["meta"], repeats)
            if sc is None:
                break
            t = pd.concat([s6.score_sites(tabs[n], sets[n]["meta"], s) for s in sc])
            parts[n] = t.groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean()
        if len(parts) < len(sets):
            say(f"  {v}: predictions incomplete - skipped")
            continue
        prim = parts["stripped"]
        if "partner" in parts:                                                   # PRIMARY rule
            prim = pd.concat([parts["partner"], prim[~prim.site_id.isin(parts["partner"].site_id)]])
        prim["site_id"] = prim.site_id.astype(str)
        prim["volume"] = prim.site_id.map(vol_cat)
        per[v] = prim.reset_index(drop=True)
    if "baseline" not in per:
        raise SystemExit("baseline predictions missing")
    rows = []
    order = [c for c in s10a.ORDER.get("Pocket volume", []) if c in set(per["baseline"].volume.dropna())]
    say(f"\nPRIMARY Top-1 by variant (repeats {','.join(map(str, repeats))}); 95% family-bootstrap CI; "
        f"difference to baseline paired on sites")
    for v, t in per.items():
        for strat, g in [("all sites", t)] + [(o, t[t.volume == o]) for o in order]:
            m = g.top1.mean()
            lo, hi = s6.boot(g.family.to_numpy(), g.top1.to_numpy())
            row = {"variant": v, "stratum": strat, "sites": len(g), "top1": m, "lo": lo, "hi": hi,
                   "top3": g.top3.mean(), "site_auc": g.site_auc.mean()}
            txt = f"  {v:<9} {strat:<24} n={len(g):>5}  Top-1 {m:.3f} [{lo:.3f}, {hi:.3f}]  Top-3 {g.top3.mean():.3f}"
            if v != "baseline":
                j = g.merge(per["baseline"][["site_id", "top1"]], on="site_id", suffixes=("", "_b"))
                dlo, dhi = s6.boot(j.family.to_numpy(), (j.top1 - j.top1_b).to_numpy())
                row.update(diff=(j.top1 - j.top1_b).mean(), diff_lo=dlo, diff_hi=dhi)
                txt += f"   vs baseline {row['diff']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            say(txt)
            rows.append(row)
    pd.DataFrame(rows).to_csv(DATA / "s13c_size_weighted.csv", index=False)
    say("(the baseline here uses --seeds networks per fold; the paper's main numbers used 10)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", default="0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    repeats = [int(r) for r in a.repeats.split(",")]
    PRED.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    groups = {n: {"group": S["meta"].family.astype(str)} for n, S in sets.items()}
    if not a.report_only:
        import torch
        device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
        t0 = time.time()
        for rep in repeats:
            fr = folds[folds.repeat == rep]
            gbf = {k: list(g.family) for k, g in fr.groupby("fold")}
            for n in sets:
                groups[n]["groups_by_fold"] = gbf
            for v in a.variants.split(","):
                for fold in range(5):
                    run_fold(v, rep, fold, sets, groups, a, device)
                    say(f"  done {v} repeat {rep} fold {fold} ({time.time() - t0:.0f} s)")
    evaluate(sets, repeats)
    (DATA / "report_step13c.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
