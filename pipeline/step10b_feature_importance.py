#!/usr/bin/env python
"""
STEP 10B: WHAT DOES AlloPlex RELY ON? (permutation importance)
==============================================================
Two complementary analyses (paper: Table 3 + Fig S4):

 (a) RETRAINING ablation (cross-validation, in step 6x): ESM-2 only (feature set E) and fpocket
     descriptors only (feature set F) versus both (B). Run once on the GPU (~1-2 h):
        python scripts/step6x_train_models.py --models mlp --features B,E,F --seeds 10 \\
               --report-name report_ablation_features.txt
     (the B runs already exist and are reused; the report's PAIRED DIFFERENCES section has
      "mlp: E minus B" and "mlp: F minus B")

 (b) PERMUTATION IMPORTANCE of the released model (this script): for each family fold k, the three
     released networks that never trained on fold k (member_f<k>_s*.pt) score the stripped pockets of
     fold k. One input at a time is shuffled across pockets (each fpocket descriptor, and the whole
     ESM-2 block); importance = drop in Top-1 and in AUPRC versus the unshuffled inputs
     (mean +/- SD over --n-perm shuffles). Note: fold k was the early-stopping set of those
     members, so absolute values are slightly optimistic; the importances (differences) are what matter.

RUN (from ~/allosite, env allosite, GPU or CPU, ~10 min):
    python scripts/step10b_feature_importance.py
Output: figures/data/s10b_permutation.csv, figures/FigS4_importance.pdf/.png, report_step10b.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step6x_train_models as s6  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
FIG = BASE / "figures"
DATA = FIG / "data"
FP = ["score", "druggability_score", "volume", "polarity_score", "charge_score", "apolar_sasa", "polar_sasa",
      "flexibility", "number_of_alpha_spheres", "mean_local_hydrophobic_density", "n_pocket_residues"]
NICE = {"score": "fpocket score", "druggability_score": "Druggability", "volume": "Volume",
        "polarity_score": "Polarity", "charge_score": "Charge", "apolar_sasa": "Apolar SASA",
        "polar_sasa": "Polar SASA", "flexibility": "Flexibility", "number_of_alpha_spheres": "Alpha spheres",
        "mean_local_hydrophobic_density": "Hydrophobic density", "n_pocket_residues": "Pocket residues",
        "ESM": "ESM-2 embedding (all 1,280)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    import torch
    import alloplex_predict as pp
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    mdir = Path(args.model_dir) if args.model_dir else next(
        (BASE / "models" / m for m in ("alloplex_v1", "prism_v1") if (BASE / "models" / m).exists()), None)
    if mdir is None:
        raise SystemExit("model folder not found (models/alloplex_v1); use --model-dir")
    S = s6.load_set("stripped")
    meta, y = S["meta"], S["y"]
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    f0 = folds[folds.repeat == 0].set_index("family").fold
    fold_of = meta.family.astype(str).map(f0).to_numpy()
    tab = s6.site_table("stripped", meta, D / "sites_stripped.csv")
    lab = y >= 0
    rng = np.random.default_rng(2026)
    members = {k: pp.load_members(mdir, device, f"member_f{k}_s*.pt") for k in range(5)}
    rows_k = {k: np.where(lab & (fold_of == k))[0] for k in range(5)}

    def evaluate(esm, fp):
        sc = np.full(len(y), np.nan)
        for k in range(5):
            r = rows_k[k]
            if len(r):
                sc[r] = pp.score(members[k], esm[r], fp[r], device)
        m = np.isfinite(sc) & lab
        sites = s6.score_sites(tab, meta, sc)
        return sites.top1.mean(), average_precision_score(y[m], sc[m]), sites

    esm = S["esm"].astype(np.float32)
    fp = S["fp"].astype(np.float32)
    base_t1, base_ap, base_sites = evaluate(esm, fp)
    print(f"held-out fold members: Top-1 {base_t1:.3f}, AUPRC {base_ap:.3f} ({len(base_sites)} sites, "
          f"{int(lab.sum())} pockets)")
    labr = np.where(lab)[0]
    rows = []
    for name in ["ESM"] + FP:
        d1, d2 = [], []
        for _ in range(args.n_perm):
            e2, f2 = esm, fp
            perm = labr[rng.permutation(len(labr))]
            if name == "ESM":
                e2 = esm.copy()
                e2[labr] = esm[perm]
            else:
                j = FP.index(name)
                f2 = fp.copy()
                f2[labr, j] = fp[perm, j]
            t1, apr, _ = evaluate(e2, f2)
            d1.append(base_t1 - t1)
            d2.append(base_ap - apr)
        rows.append({"feature": name, "label": NICE[name], "drop_top1": np.mean(d1), "sd_top1": np.std(d1),
                     "drop_auprc": np.mean(d2), "sd_auprc": np.std(d2)})
        print(f"  {NICE[name]:<28} Top-1 drop {np.mean(d1):+.3f} (SD {np.std(d1):.3f})   "
              f"AUPRC drop {np.mean(d2):+.3f} (SD {np.std(d2):.3f})", flush=True)
    out = pd.DataFrame(rows).sort_values("drop_top1", ascending=False)
    DATA.mkdir(parents=True, exist_ok=True)
    out.assign(base_top1=base_t1, base_auprc=base_ap).to_csv(DATA / "s10b_permutation.csv", index=False)
    figure(out)
    lines = [f"baseline (held-out fold members): Top-1 {base_t1:.3f}, AUPRC {base_ap:.3f}"] + [
        f"{r.label:<28} Top-1 drop {r.drop_top1:+.3f} (SD {r.sd_top1:.3f}); AUPRC drop {r.drop_auprc:+.3f} "
        f"(SD {r.sd_auprc:.3f})" for r in out.itertuples()]
    (DATA / "report_step10b.txt").write_text("\n".join(lines) + "\n")


def figure(out):
    import step9x_figures as f9
    plt = f9.style()
    OI = f9.OI
    d = out.sort_values("drop_top1", ascending=True)
    fig, axs = plt.subplots(1, 2, figsize=(f9.W2, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
    cols = [OI["purple"] if f == "ESM" else OI["orange"] for f in d.feature]
    for ax, m, sd, lab, let in ((axs[0], "drop_top1", "sd_top1", "Drop in Top-1", "a"),
                                (axs[1], "drop_auprc", "sd_auprc", "Drop in AUPRC", "b")):
        y = np.arange(len(d))
        ax.barh(y, d[m], xerr=d[sd], color=cols, height=0.65, error_kw=dict(lw=0.6, capsize=1.5))
        ax.axvline(0, color="#333333", lw=0.6)
        ax.set_xlabel(lab)
        f9.letter(ax, let, x=-0.02 if let == "b" else -0.55, y=1.0)
    axs[0].set_yticks(np.arange(len(d)))
    axs[0].set_yticklabels(d.label)
    axs[0].tick_params(axis="y", length=0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"FigS4_importance.{ext}", bbox_inches="tight")
    print("  saved figures/FigS4_importance.pdf/.png")


if __name__ == "__main__":
    main()
