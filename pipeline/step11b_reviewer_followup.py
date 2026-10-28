#!/usr/bin/env python
"""
STEP 11B: FOLLOW-UP ANALYSES FOR THE REVIEWER (no training; a few minutes, GPU optional)
=====================================================================================
  1. FAMILY WEIGHTING (concern 10) - from figures/data/fig5_tools.csv (the paper's matched evaluation,
     each tool scored on the sites it ran on, exactly as Table 5):
       site-weighted mean (paper) and family-weighted mean (every family counts once) for each method,
       and paired AlloPlex-minus-tool differences, family-weighted, with family-bootstrap 95 % CI;
       the same on families absent from ASD 2019; the largest families and their Top-1.
  2. NEGATIVES RETAINED (moderate concern) - the cross-validation runs never scored the pockets that
     were set aside (they overlap an annotated site or a symmetric copy), so AlloPlex is re-scored here
     with the RELEASED networks that never trained on each family (member_f<k>_s*.pt, k = the family's
     fold, as in steps 10b/10d). Top-1 is computed with the standard negatives and with every set-aside
     pocket counted as a negative, for AlloPlex and all tools on the same sites.
  3. CASE STUDIES - ranks of every method for 5OLK (success) and 2VD4 (failure) and the pocket that
     AlloPlex ranked first in 2VD4 (residues), to describe the failure.

RUN (from ~/allosite, env allosite):
    python scripts/step11b_reviewer_followup.py
Output: figures/data/report_step11b.txt, figures/data/s11b_*.csv
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
DATA = BASE / "figures" / "data"
LOG = []
OURS_TOOLS = "MLP (B, ours)"


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def fam_boot(fam, v, B=2000):
    """family-weighted mean (mean of family means) and its family-bootstrap 95 % CI"""
    g = pd.DataFrame({"f": fam, "v": v}).dropna().groupby("f").v.mean().to_numpy()
    rng = np.random.default_rng(s6.SEED)
    bs = g[rng.integers(0, len(g), (B, len(g)))].mean(1)
    return g.mean(), *np.percentile(bs, [2.5, 97.5]), len(g)


# ============================================================================= 1
def family_weighting():
    say("\n1. FAMILY WEIGHTING (matched evaluation as in Table 5; each tool on the sites it ran on)")
    d = pd.read_csv(DATA / "fig5_tools.csv", dtype={"site_id": str, "family": str})
    d = d[d.evaluation == "matched"]
    d["method"] = d.method.replace({OURS_TOOLS: "AlloPlex", "fpocket druggability": "Druggability"})
    ours = d[d.method == "AlloPlex"].set_index("site_id")
    sizes = ours.groupby("family").size().sort_values(ascending=False)
    say(f"  {len(ours)} sites, {len(sizes)} families; largest families (sites, AlloPlex Top-1):")
    for f, n in sizes.head(8).items():
        say(f"    family {f}: {n} sites ({n / len(ours):.1%}), AlloPlex Top-1 {ours[ours.family == f].top1.mean():.3f}")
    rows = []
    for subset, sel in (("all families", None), ("families absent from ASD 2019", "new")):
        say(f"  {subset}")
        for m in ["AlloPlex"] + [x for x in d.method.unique() if x != "AlloPlex"]:
            t = d[d.method == m].set_index("site_id")
            if sel == "new":
                t = t[t.new_family.astype(str).isin(["True", "true", "1", "1.0"])]
            sw = t.top1.mean()
            fw, lo, hi, nf = fam_boot(t.family, t.top1)
            line = f"    {m:<13} sites {len(t):>5}, families {nf:>3}: site-weighted {sw:.3f}; family-weighted {fw:.3f} [{lo:.3f}, {hi:.3f}]"
            row = {"subset": subset, "method": m, "sites": len(t), "families": nf, "site_weighted": sw,
                   "family_weighted": fw, "fw_lo": lo, "fw_hi": hi}
            if m != "AlloPlex":
                j = t[["family", "top1"]].join(ours[["top1"]], rsuffix="_ours", how="inner")
                dd = j.top1_ours - j.top1
                sw_d = s11.ci(j.family, dd)
                fw_d = fam_boot(j.family, dd)
                line += (f"\n{'':18}AlloPlex minus {m}: site-weighted {s11.fmt(*sw_d, sign=True)}; "
                         f"family-weighted {fw_d[0]:+.3f} [{fw_d[1]:+.3f}, {fw_d[2]:+.3f}]")
                row.update(diff_sw=sw_d[0], diff_sw_lo=sw_d[1], diff_sw_hi=sw_d[2], diff_fw=fw_d[0],
                           diff_fw_lo=fw_d[1], diff_fw_hi=fw_d[2])
            say(line)
            rows.append(row)
    pd.DataFrame(rows).to_csv(DATA / "s11b_family_weighting.csv", index=False)


# ============================================================================= 2
def heldout_scores(sets):
    import torch
    import alloplex_predict as pp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = next((BASE / "models" / m for m in ("alloplex_v1", "prism_v1") if (BASE / "models" / m).exists()))
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    fold = folds[folds.repeat == 0].set_index("family").fold
    out = {}
    for n, S in sets.items():
        k_of = S["meta"].family.astype(str).map(fold).fillna(-1).astype(int).to_numpy()
        sc = np.full(len(S["y"]), np.nan)
        for k in range(5):
            rows = np.where(k_of == k)[0]
            if len(rows) == 0:
                continue
            members = pp.load_members(mdir, device, f"member_f{k}_s*.pt")
            acc = np.zeros(len(rows))
            for m, mu, sd in members:
                for a in range(0, len(rows), 20000):
                    r = rows[a:a + 20000]
                    X = np.c_[S["esm"][r].astype(np.float32), (S["fp"][r] - mu) / sd].astype(np.float32)
                    with torch.no_grad():
                        acc[a:a + len(r)] += torch.sigmoid(m(torch.tensor(X, device=device))).cpu().numpy()
            sc[rows] = acc / len(members)
        out[n] = sc
        say(f"  {n}: scored {np.isfinite(sc).sum()} of {len(sc)} pockets with held-out released networks")
    return out


def negatives(sets, tabs, methods):
    say("\n2. NEGATIVES RETAINED - AlloPlex re-scored on ALL pockets with the released networks that never")
    say("   trained on each family; standard negatives vs every set-aside pocket counted as a negative")
    methods = dict(methods)
    methods["AlloPlex"] = ([heldout_scores(sets)], set())
    rows = []
    for neg, lab in (("standard", "standard negatives (paper rule)"), ("all", "set-aside pockets counted as negatives")):
        say(f"  {lab}")
        tb = {m: s11.per_site(sets, tabs, r, set(), neg=neg) for m, (r, f) in methods.items()}
        tb = {m: t[~t.site_id.isin(methods[m][1])] for m, t in tb.items()}  # each tool on the sites it ran on
        s11.compare({"AlloPlex": tb["AlloPlex"], **{m: t for m, t in tb.items() if m != "AlloPlex"}},
                    f"negatives {neg}", rows)
    pd.DataFrame(rows).to_csv(DATA / "s11b_negatives.csv", index=False)
    return methods


# ============================================================================= 3
def cases(sets, tabs, methods):
    say("\n3. CASE STUDIES (ranks among the pockets of the structure used in the paper; AlloPlex = held-out networks)")
    pk = {n: pd.read_csv(s6.D / f"pockets_{'stripped' if n == 'stripped' else 'partner'}.csv",
                         usecols=["pdb_id", "pocket", "residues"], low_memory=False) for n in sets}
    for pdb in ("5OLK", "2VD4"):
        for n in ("partner", "stripped"):
            t = tabs[n][tabs[n].pdb_id == pdb]
            if t.empty:
                continue
            for r in t.itertuples():
                ranks = {}
                for m, (runs, _) in methods.items():
                    e = s11.eval_set(t, sets[n], runs[0][n])
                    ranks[m] = float(e["rank"].iloc[0]) if len(e) else np.nan
                say(f"  {pdb} ({n}) site {r.site_id}, best-pocket Jaccard {r.best_j:.2f}: "
                    + ", ".join(f"{m} {v:.0f}" for m, v in ranks.items() if np.isfinite(v)))
                sc = methods["AlloPlex"][0][0][n]
                rows = np.where(sets[n]["meta"].pdb_id.to_numpy() == pdb)[0]
                order = rows[np.argsort(-np.nan_to_num(sc[rows], nan=-1))][:3]
                for i, j in enumerate(order, 1):
                    res = json.loads(pk[n].residues.iloc[j])
                    chains = sorted({c for c, *_ in res})
                    lab = {1: "SITE", 0: "other", -1: "set aside"}[int(sets[n]["y"][j])]
                    say(f"     AlloPlex rank {i}: pocket {pk[n].pocket.iloc[j]} ({lab}), score {sc[j]:.3f}, "
                        f"{len(res)} residues on chain(s) {','.join(chains)}, volume {sets[n]['fp'][j, 2]:.0f} A^3, "
                        f"druggability {sets[n]['fp'][j, 1]:.2f}")
                if r.pos_row >= 0:
                    j = r.pos_row
                    say(f"     site pocket: score {sc[j]:.3f}, volume {sets[n]['fp'][j, 2]:.0f} A^3, "
                        f"druggability {sets[n]['fp'][j, 1]:.2f}, {int(sets[n]['meta'].n_pocket_residues.iloc[j])} residues")
            break


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    family_weighting()
    sets, tabs, methods = s11.load()
    methods = negatives(sets, tabs, methods)
    cases(sets, tabs, methods)
    (DATA / "report_step11b.txt").write_text("\n".join(LOG) + "\n")
    say("\nsaved figures/data/report_step11b.txt and s11b_*.csv")


if __name__ == "__main__":
    main()
