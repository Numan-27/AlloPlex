#!/usr/bin/env python
"""
STEP 10F: FIGURE S4 - AlloPlex on AlphaFold 3 models versus crystal structures
==============================================================================
Reads the per-site results written by step10d (score):  figures/data/s10d_alphafold.csv
and draws Figure S4 in the style of the other figures (step9x):

  (a) Top-1 and Top-3 on crystal structures and AlphaFold 3 models
      (bars = mean, whiskers = 95% family-bootstrap CI; n sites)
  (b) Rank of each site on the crystal structure versus on the AlphaFold 3 model (log scale),
      filled = site pLDDT >= 70, open = site pLDDT < 70; sites whose pocket was not found are at "n.f."
  (c) Top-3 by model confidence: site pLDDT >= 70 / < 70 and ipTM >= 0.6 / < 0.6

Run from the repository root (same environment as step9x):
    python scripts/step10f_figS4_alphafold.py
Output: figures/FigS4_alphafold.pdf/.png and tables/TableS6_alphafold.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step9x_figures as s9  # noqa: E402  (style, colours, save, letter)

SEED = 20261015
CRYSTAL, MODEL = s9.OI["blue"], s9.OI["orange"]


def boot(fam, vals, B=2000):
    """95% family-bootstrap CI of a site-level mean (families resampled with replacement)."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"f": fam, "v": vals}).dropna()
    g = df.groupby("f").v.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), (B, len(g)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def main():
    f = s9.DATA / "s10d_alphafold.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found - run: python scripts/step10d_alphafold.py score")
    d = pd.read_csv(f, dtype={"family": str, "site_id": str})
    for s in ("exp", "af"):
        d[f"rank_{s}"] = d[f"rank_{s}"].astype(float)
        d[f"top1_{s}"] = (d[f"rank_{s}"] <= 1).astype(float)
        d[f"top3_{s}"] = (d[f"rank_{s}"] <= 3).astype(float)
    fam = d.family.to_numpy()
    n = len(d)
    print(f"{n} sites, {d.pdb_id.nunique()} complexes, {d.family.nunique()} families")

    # ---------------------------------------------------------------- table S6
    rows = []
    for m in ("top1", "top3"):
        e, a = d[f"{m}_exp"], d[f"{m}_af"]
        le, he = boot(fam, e.to_numpy())
        la, ha = boot(fam, a.to_numpy())
        ld, hd = boot(fam, (a - e).to_numpy())
        rows.append({"metric": m.replace("top", "Top-"), "sites": n,
                     "crystal": f"{e.mean():.3f} [{le:.3f}, {he:.3f}]",
                     "AlphaFold 3 model": f"{a.mean():.3f} [{la:.3f}, {ha:.3f}]",
                     "model minus crystal": f"{(a - e).mean():+.3f} [{ld:+.3f}, {hd:+.3f}]"})
    tab = pd.DataFrame(rows)
    (s9.BASE / "tables").mkdir(exist_ok=True)
    tab.to_csv(s9.BASE / "tables" / "TableS6_alphafold.csv", index=False)
    print(tab.to_string(index=False))

    # ---------------------------------------------------------------- figure
    plt = s9.style()
    s9.FIG.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(s9.W2, 2.3),
                            gridspec_kw={"width_ratios": [0.8, 1.05, 1.15], "wspace": 0.45})

    # (a) Top-1 / Top-3
    ax = axs[0]
    x = np.arange(2)
    w = 0.36
    for j, (s, lab, col) in enumerate((("exp", "Crystal", CRYSTAL), ("af", "AlphaFold 3", MODEL))):
        means, err = [], [[], []]
        for m in ("top1", "top3"):
            v = d[f"{m}_{s}"].to_numpy()
            lo, hi = boot(fam, v)
            means.append(v.mean() * 100)
            err[0].append(v.mean() * 100 - lo * 100)
            err[1].append(hi * 100 - v.mean() * 100)
        ax.bar(x + (j - 0.5) * w, means, w, color=col, label=lab, zorder=2)
        ax.errorbar(x + (j - 0.5) * w, means, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.7,
                    capsize=1.8, zorder=3)
        for xi, mv in zip(x + (j - 0.5) * w, means):
            ax.text(xi, 2, f"{mv:.0f}", ha="center", va="bottom", fontsize=5.8, color="white", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Top-1", "Top-3"])
    ax.set_ylabel("Sites (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", handlelength=1.0)
    ax.set_xlabel(f"n = {n} sites")
    s9.letter(ax, "a", x=-0.32)

    # (b) rank on crystal vs model
    ax = axs[1]
    re_, ra = d.rank_exp.to_numpy(), d.rank_af.to_numpy()
    fin = np.r_[re_[np.isfinite(re_)], ra[np.isfinite(ra)]]
    top = max(10.0, np.nanmax(fin)) * 2
    re2, ra2 = np.where(np.isfinite(re_), re_, top), np.where(np.isfinite(ra), ra, top)
    jit = np.random.default_rng(1).uniform(0.9, 1.1, (2, n))
    hi_conf = (d.site_plddt >= 70).to_numpy()
    ax.scatter(re2[hi_conf] * jit[0][hi_conf], ra2[hi_conf] * jit[1][hi_conf], s=10, c=s9.OI["blue"], alpha=0.75,
               lw=0, label="site pLDDT ≥ 70", zorder=3)
    ax.scatter(re2[~hi_conf] * jit[0][~hi_conf], ra2[~hi_conf] * jit[1][~hi_conf], s=10, facecolors="none",
               edgecolors=s9.OI["vermilion"], lw=0.8, label="site pLDDT < 70", zorder=3)
    ax.plot([0.8, top * 1.2], [0.8, top * 1.2], color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, top * 1.25)
    ax.set_ylim(0.8, top * 1.25)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    tk = [t for t in (1, 3, 10, 30, 100) if t < top / 1.5]
    labels = [str(t) for t in tk]
    if (~np.isfinite(re_)).any() or (~np.isfinite(ra)).any():
        tk, labels = tk + [top], labels + ["n.f."]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(tk))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rank on crystal structure")
    ax.set_ylabel("Rank on AlphaFold 3 model")
    ax.legend(loc="upper left", handletextpad=0.2)
    both1 = int(((re_ == 1) & (ra == 1)).sum())
    print(f"  Fig S4b: {both1} sites ranked 1 on both structures (overlapping at the corner)")
    s9.letter(ax, "b", x=-0.3)

    # (c) Top-3 by model confidence
    ax = axs[2]
    groups = [("pLDDT\n≥ 70", d.site_plddt >= 70), ("pLDDT\n< 70", d.site_plddt < 70)]
    if "iptm" in d and d.iptm.notna().any():
        groups += [("ipTM\n≥ 0.6", d.iptm >= 0.6), ("ipTM\n< 0.6", d.iptm < 0.6)]
    groups = [(g, m.to_numpy()) for g, m in groups if m.sum() > 0]
    x = np.arange(len(groups))
    for j, (s, col) in enumerate((("exp", CRYSTAL), ("af", MODEL))):
        vals = [d.loc[m, f"top3_{s}"].mean() * 100 for _, m in groups]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, zorder=2)
    for xi, (_, m) in zip(x, groups):
        ax.text(xi, 101, f"n={int(m.sum())}", ha="center", va="bottom", fontsize=5.6, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Top-3 (%)")
    s9.letter(ax, "c", x=-0.28)

    s9.save(fig, "FigS4_alphafold")
    for lab, g in groups:
        g_ = d[g]
        print(f"  {lab.replace(chr(10), ' '):<12} n={len(g_):>3}: Top-3 crystal {g_.top3_exp.mean():.3f}, "
              f"model {g_.top3_af.mean():.3f}; Top-1 crystal {g_.top1_exp.mean():.3f}, model {g_.top1_af.mean():.3f}")


if __name__ == "__main__":
    main()
