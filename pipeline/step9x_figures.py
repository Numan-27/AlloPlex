#!/usr/bin/env python
"""
STEP 9X: MAIN FIGURES FOR THE PAPER
===================================
Two stages (each figure is independent; one failing does not stop the others):

  collect : reads the result files on the server and writes small tables to figures/data/*.csv
            (per-site results, so confidence intervals are computed in the plotting stage)
  plot    : draws the figures from figures/data/*.csv -> figures/Fig<N>_<name>.pdf + .png (300 dpi)

    python scripts/step9x_figures.py              # collect + plot
    python scripts/step9x_figures.py --plot-only  # re-draw after editing the style

Figures
  Fig1 dataset        sites per complex type, chains per assembly, sites per data source
  Fig2 interface      distance of allosteric vs other pockets to the nearest interface (key set);
                      pre-registered test, random-patch and centrality controls
  Fig3 inflation      Top-1 under the family split vs common shortcuts
  Fig4 models         MLP / GNN / LightGBM variants and pocket baselines (PRIMARY set)
  Fig5 tools          AlloPlex vs DeepAllo, APOP, PASSer, P2Rank, druggability: matched and native
  Fig6 novelty        Top-1 on all sites, proteins new since ASD 2019, families new since ASD 2019
  Fig9 case_5OLK      cross-validation ranking of the pockets of 5OLK (its family held out)

Numbers of the pre-registered interface test and its controls (Fig2b) come from the step-4x,
4y and 4z reports and are entered in REPORTED below.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
D = BASE / "processed" / "dataset_v2"
FIG = BASE / "figures"
DATA = FIG / "data"
SEED = 20261015

# --- style (reference palette, light mode; ours = slot 1 blue, comparison = neutral grey) ---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_D, INK, INK2 = "#b9b8b3", "#7d7c78", "#0b0b0b", "#52514e"
OURS = "MLP (B, ours)"

# pre-registered interface test and controls (reports of steps 4x / 4y / 4z)
REPORTED = [  # label, estimate, low, high, group
    ("Primary: overlap with interface\n(key set, matched AUC)", 0.502, 0.399, 0.611, "primary"),
    ("S2: PPI-annotated sites only\n(circular)", 0.750, 0.686, 0.812, "secondary"),
    ("S3a: pockets not touching the interface, scored by closeness", 0.643, 0.584, 0.693, "secondary"),
    ("S4: hetero / peptide complexes", 0.390, 0.342, 0.487, "secondary"),
    ("Distance to interface,\ncentrality-matched", 0.631, 0.499, 0.693, "control"),
]
REPORTED_EXTRA = [  # label, estimate, low, high (reports of steps 4x / 4y / 4z)
    ("Primary: Mantel-Haenszel odds ratio (pocket overlaps interface)", 1.47, 0.77, 2.79),
    ("Random-patch control: distance AUC, real minus random", 0.096, 0.037, 0.171),
]
RANDOM_PATCH = {"overlap": (0.502, 0.523), "distance": (0.641, 0.545)}  # real vs random-patch AUC


def boot_ci(fam, vals, B=2000):
    """family bootstrap 95 % interval of the mean (same scheme as step 6x)."""
    df = pd.DataFrame({"f": fam, "v": vals}).dropna()
    if df.empty:
        return np.nan, np.nan
    g = df.groupby("f").v.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(g), (B, len(g)))
    return tuple(np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5]))


# ============================================================================= COLLECT
def collect_dataset():
    m = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", low_memory=False)
    if "quality_ok" in m.columns:
        m = m[m.quality_ok.astype(str).isin(["True", "true", "1"])]
    m = m[m.source.isin(["ASD", "AlloBench"])] if "source" in m.columns else m
    keep = [c for c in ("site_id", "pdb_id", "family", "source", "complex_kind", "position_class",
                        "n_modulator_free") if c in m.columns]
    m[keep].to_csv(DATA / "fig1_sites.csv", index=False)
    st = BASE / "processed" / "structures" / "structures.csv"
    if st.exists():
        s = pd.read_csv(st)
        s[s.status == "ok"][["pdb_id", "n_protein_chains"]].to_csv(DATA / "fig1_structures.csv", index=False)


def collect_interface():
    P = pd.read_csv(D / "pockets_stripped.csv", low_memory=False,
                    usecols=lambda c: c in {"pdb_id", "pocket", "label_primary", "label_any", "symmetric_copy",
                                            "n_chains", "dist_iface_5", "frac_iface_5", "sites_primary"})
    m = pd.read_csv(BASE / "processed" / "master" / "allosteric_master.csv", low_memory=False,
                    usecols=["site_id", "position_class"])
    pos = dict(zip(m.site_id.astype(str), m.position_class.astype(str)))
    tf = lambda s: s.astype(str).isin(["True", "true", "1"])
    P = P[P.n_chains.fillna(1) >= 2]
    lp, la, sym = tf(P.label_primary), tf(P.label_any), tf(P.symmetric_copy)
    sites = P.sites_primary.fillna("").astype(str)
    non_ppi = [all(pos.get(s, "") != "PPI interface" for s in x.split(",") if s) for x in sites]
    P["group"] = np.where(lp & np.array(non_ppi), "allosteric (not PPI-annotated)",
                          np.where(lp, "allosteric (PPI-annotated)", np.where(~la & ~sym, "other pockets", "")))
    P = P[P.group != ""]
    P[["pdb_id", "pocket", "group", "dist_iface_5", "frac_iface_5"]].to_csv(DATA / "fig2_pockets.csv", index=False)


def collect_models():
    import step6x_train_models as s6
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import step7y_score_tools as s7
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {n: s6.site_table(n, sets[n]["meta"], D / f"sites_{n}.csv") for n in sets}
    res = s6.evaluate(["mlp", "mlpbce", "lgbm", "gnn"], [0, 1, 2], sets, tabs)
    rows, aprc = [], []
    for (model, feat), per in res.items():
        name = f"{ {'mlp': 'MLP', 'mlpbce': 'MLP-BCE', 'lgbm': 'LightGBM', 'gnn': 'GNN'}.get(model, model.upper())} {feat}"
        for en in ("primary", "partner", "stripped"):
            if en in per:
                t, ap = per[en]
                rows.append(t.assign(method=name, set=en)[["method", "set", "site_id", "family", "top1", "top3", "site_auc"]])
                aprc.append({"method": name, "set": en, "auprc": ap})
    pd.DataFrame(aprc).to_csv(DATA / "tab_auprc.csv", index=False)
    for k, name in ((1, "fpocket druggability"), (0, "fpocket score"), (2, "pocket volume")):
        t = s7.primary_table(sets, tabs, {n: sets[n]["fp"][:, k].astype(float) for n in sets}, [])
        rows.append(t.assign(method=name, set="primary")[["method", "set", "site_id", "family", "top1", "top3", "site_auc"]])
    t = s7.primary_table(sets, tabs, {n: sets[n]["fp"][:, 10].astype(float) for n in sets}, [])
    rows.append(t.assign(method="pocket size", set="primary")[["method", "set", "site_id", "family", "top1", "top3", "site_auc"]])
    pd.concat(rows).to_csv(DATA / "fig4_models.csv", index=False)


def collect_inflation():
    import step6x_train_models as s6
    sets = {n: s6.load_set(n) for n in ("stripped", "holo") if (s6.B / f"features_{n}.npz").exists()}
    st = s6.site_table("stripped", sets["stripped"]["meta"], D / "sites_stripped.csv")
    rows = []
    ref = s6.collect("mlp", "B", "stripped", sets["stripped"]["meta"], [0])
    rows.append(s6.score_sites(st, sets["stripped"]["meta"], ref[0]).assign(split="family split (reference)"))
    for tag, name, setn in (("_gene", "gene-name split", "stripped"), ("_holo", "ligand-bound (holo) inputs", "holo"),
                            ("_random", "random pocket split", "stripped")):
        if setn not in sets:
            continue
        sc = s6.collect("mlp", "B", setn, sets[setn]["meta"], [0], tag)
        if sc is None:
            continue
        tab = st if setn == "stripped" else s6.site_table("holo", sets["holo"]["meta"], D / "sites_holo.csv")
        rows.append(s6.score_sites(tab, sets[setn]["meta"], sc[0]).assign(split=name))
    pd.concat(rows)[["split", "site_id", "family", "top1", "site_auc"]].to_csv(DATA / "fig3_inflation.csv", index=False)


def collect_tools():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import step6x_train_models as s6
    import step7y_score_tools as s7
    T = s7.T
    man = pd.read_csv(T / "manifest.csv")
    man = man[man.roles.str.contains("primary")]
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {n: s6.site_table(n, sets[n]["meta"], D / f"sites_{n}.csv") for n in sets}
    pocket_res = {}
    for setn, csv in s7.SETMAP.values():
        if setn in sets:
            pk = pd.read_csv(D / csv, usecols=["pdb_id", "residues"], low_memory=False)
            pocket_res[setn] = [[(c, int(n)) for c, n, *_ in json.loads(r)] for r in pk.residues]
    site_key = {s: k for k, ss in zip(man.key, man.sites) for s in ss.split(",")}
    results, raw = {}, {}
    for name, fn in s7.TOOLS.items():
        path = T / "results" / fn
        if not path.exists():
            continue
        raw[name] = json.load(open(path))
        sc, failed = s7.scores_for_tool(raw[name], man, sets, pocket_res)
        fs = [s for s, k in site_key.items() if k in failed]
        results[name] = (s7.primary_table(sets, tabs, sc, fs), set(fs))
    sc_mlp = {n: s6.collect("mlp", "B", n, sets[n]["meta"], [0, 1, 2]) for n in sets}
    ours = {OURS: pd.concat([s7.primary_table(sets, tabs, {n: sc_mlp[n][r] for n in sets}, []) for r in range(3)])
            .groupby(["site_id", "family"], as_index=False)[["top1", "top3", "site_auc"]].mean(),
            "fpocket druggability": s7.primary_table(sets, tabs, {n: sets[n]["fp"][:, 1].astype(float) for n in sets}, [])}
    nat, _ = s7.native_eval(man, sets, tabs, pocket_res, raw, {n: sc_mlp[n] for n in sets})
    nov = {}
    asd19 = BASE / "tools" / "DeepAllo" / "source_data" / "ASD_Release_201909_AS.txt"
    if asd19.exists():
        nov, _ = s7.novelty_classes(asd19)
    rows = []
    for name, t in list(ours.items()) + [(k, v[0][~v[0].site_id.isin(v[1])]) for k, v in results.items()]:
        rows.append(t.assign(method=name, evaluation="matched", ran=True)[
            ["method", "evaluation", "site_id", "family", "top1", "top3", "site_auc"]])
    for name, t in nat.items():
        t = t[t.ran]
        rows.append(t.assign(method=name, evaluation="native", site_auc=np.nan)[
            ["method", "evaluation", "site_id", "family", "top1", "top3", "site_auc"]])
    out = pd.concat(rows)
    out["site_id"] = out.site_id.astype(str)
    out["new_protein"] = out.site_id.map(lambda s: nov.get(s, (None, None))[0])
    out["new_family"] = out.site_id.map(lambda s: nov.get(s, (None, None))[1])
    out.to_csv(DATA / "fig5_tools.csv", index=False)


def collect_case(pdb="5OLK"):
    """works for stripped (holo-derived) and partner structures; the partner set is used if the
    PDB ID is a partner structure (as listed by --list-cases)."""
    import step6x_train_models as s6
    setn = "stripped"
    S = s6.load_set("stripped")
    rows = np.where(S["meta"].pdb_id.to_numpy() == pdb)[0]
    if not len(rows) and (s6.B / "features_partner.npz").exists():
        setn = "partner"
        S = s6.load_set("partner")
        rows = np.where(S["meta"].pdb_id.to_numpy() == pdb)[0]
    if not len(rows):
        raise SystemExit(f"{pdb} is in neither the stripped nor the partner set")
    meta = S["meta"]
    sc = s6.collect("mlp", "B", setn, meta, [0, 1, 2])
    cv = np.nanmean(np.vstack([s[rows] for s in sc]), axis=0)
    df = pd.DataFrame({"pocket": meta.pocket.iloc[rows].to_numpy(), "y": S["y"][rows], "cv_score": cv,
                       "druggability": S["fp"][rows, 1], "fpocket_score": S["fp"][rows, 0],
                       "n_res": S["fp"][rows, 10]})
    fam = meta.family.iloc[rows[0]]
    folds = pd.read_csv(s6.B / "folds.csv", dtype={"family": str})
    f = folds[(folds.family == str(fam))]
    df["family"] = fam
    df["test_folds"] = ",".join(f"r{r}f{k}" for r, k in zip(f.repeat, f.fold))
    df["set"] = setn
    df.to_csv(DATA / f"fig7_{pdb}.csv", index=False)
    write_case_pml(pdb)



def _tool_setup():
    """shared loading for the tool / rank collectors (same as step 7y)."""
    import step6x_train_models as s6
    import step7y_score_tools as s7
    man = pd.read_csv(s7.T / "manifest.csv")
    man = man[man.roles.str.contains("primary")]
    sets = {n: s6.load_set(n) for n in ("stripped", "partner") if (s6.B / f"features_{n}.npz").exists()}
    tabs = {n: s6.site_table(n, sets[n]["meta"], D / f"sites_{n}.csv") for n in sets}
    pocket_res = {}
    for setn, csv in s7.SETMAP.values():
        if setn in sets:
            pk = pd.read_csv(D / csv, usecols=["pdb_id", "residues"], low_memory=False)
            pocket_res[setn] = [[(c, int(n)) for c, n, *_ in json.loads(r)] for r in pk.residues]
    return s6, s7, man, sets, tabs, pocket_res


def collect_split():
    """Table 1: the data sets (structures, families, pockets, allosteric pockets, sites)."""
    import step6x_train_models as s6
    rows = []
    for n, lab in (("stripped", "ligand-free holo structures (training + evaluation)"),
                   ("partner", "modulator-free partner structures (evaluation only)"),
                   ("holo", "holo structures with ligands (inflation analysis only)")):
        f = s6.B / f"meta_{n}.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f, keep_default_na=False)
        sites = pd.read_csv(D / f"sites_{n}.csv", low_memory=False) if (D / f"sites_{n}.csv").exists() else None
        rows.append({"data set": lab, "structures": m.pdb_id.nunique(), "families": m.family.nunique(),
                     "pockets": int((m.y >= 0).sum()), "allosteric pockets": int((m.y == 1).sum()),
                     "sites": int((sites.status == "ok").sum()) if sites is not None else ""})
    pd.DataFrame(rows).to_csv(DATA / "tab_split.csv", index=False)


def collect_tsne(n_neg=6000):
    """Fig 3: t-SNE of the pocket features (all allosteric pockets + a random sample of others)."""
    import step6x_train_models as s6
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    S = s6.load_set("stripped")
    rng = np.random.default_rng(SEED)
    pos = np.where(S["y"] == 1)[0]
    neg = rng.choice(np.where(S["y"] == 0)[0], min(n_neg, int((S["y"] == 0).sum())), replace=False)
    idx = np.r_[pos, neg]
    E = S["esm"][idx].astype(np.float32)
    E = PCA(50, random_state=0).fit_transform((E - E.mean(0)) / (E.std(0) + 1e-6))
    F = S["fp"][idx].astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    X = np.c_[E / np.sqrt(50), F / np.sqrt(F.shape[1])]
    Y = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(X)
    meta = S["meta"].iloc[idx]
    pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "allosteric": S["y"][idx] == 1, "kind": meta.kind.to_numpy(),
                  "druggability": S["fp"][idx, 1],
                  "volume": S["fp"][idx, 2]}).to_csv(DATA / "fig3_tsne.csv", index=False)


def collect_ranks():
    """Fig 6: rank of the allosteric pocket among all pockets of its structure, per method (PRIMARY sites)."""
    s6, s7, man, sets, tabs, pocket_res = _tool_setup()
    scores = {}
    sc = {n: s6.collect("mlp", "B", n, sets[n]["meta"], [0, 1, 2]) for n in sets}
    scores[OURS] = ({n: np.nanmean(np.vstack(sc[n]), 0) for n in sets}, set())
    scores["fpocket druggability"] = ({n: sets[n]["fp"][:, 1].astype(float) for n in sets}, set())
    site_key = {s: k for k, ss in zip(man.key, man.sites) for s in ss.split(",")}
    for name, fn in s7.TOOLS.items():
        path = s7.T / "results" / fn
        if path.exists():
            t_sc, failed = s7.scores_for_tool(json.load(open(path)), man, sets, pocket_res)
            scores[name] = (t_sc, {s for s, k in site_key.items() if k in failed})
    psites = set(tabs["partner"].site_id.astype(str)) if "partner" in tabs else set()
    rows = []
    for name, (scs, failed) in scores.items():
        for n in sets:
            meta = sets[n]["meta"]
            y = sets[n]["y"]
            by_pdb = {}
            for i, p in enumerate(meta.pdb_id):
                by_pdb.setdefault(p, []).append(i)
            for sid, pdb, fam, r in zip(tabs[n].site_id.astype(str), tabs[n].pdb_id, tabs[n].family, tabs[n].pos_row):
                if n == "stripped" and sid in psites:
                    continue
                s = scs[n]
                negs = np.array([s[i] for i in by_pdb.get(pdb, []) if y[i] == 0 and not np.isnan(s[i])])
                if r >= 0 and np.isnan(s[r]) and sid not in failed:
                    continue  # not a test pocket for this method
                if sid in failed and name in SUBSET_TOOLS_FIG:
                    continue  # web-only subset tool: not run on this structure
                n_p = len(negs) + 1
                if r < 0 or sid in failed or np.isnan(s[r]):
                    rank = np.inf  # site not found as a pocket / tool failed -> miss
                else:
                    rank = 1 + (negs > s[r]).sum() + 0.5 * (negs == s[r]).sum()
                rows.append({"method": name, "site_id": sid, "pdb_id": pdb, "family": fam, "set": n, "rank": rank,
                             "n_pockets": n_p})
    pd.DataFrame(rows).to_csv(DATA / "fig6_ranks.csv", index=False)


SUBSET_TOOLS_FIG = {"PASSer"}


def _atoms(path, het):
    """{(chain, resnum, resname): xyz array} for ATOM (het=False) or HETATM (het=True, no water) records."""
    out = {}
    for line in open(path):
        rec = line[:6]
        if (rec == "HETATM") if het else (rec == "ATOM  "):
            rn = line[17:20].strip()
            if het and rn in ("HOH", "WAT", "DOD"):
                continue
            try:
                key = (line[21], int(line[22:26]), rn)
                out.setdefault(key, []).append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    return {k: np.array(v) for k, v in out.items()}


def write_case_pml(pdb="5OLK"):
    """Fig 9b: PyMOL script (top-3 CV pockets; the known modulator in green, other ligands thin grey sticks).
    Also prints which bound ligands sit in each of the top-3 pockets (useful for the caption)."""
    d = pd.read_csv(DATA / f"fig7_{pdb}.csv").sort_values("cv_score", ascending=False).reset_index(drop=True)
    setn = d["set"].iloc[0] if "set" in d.columns else "stripped"
    P = pd.read_csv(D / f"pockets_{setn}.csv", low_memory=False, usecols=["pdb_id", "pocket", "residues"])
    P = P[P.pdb_id == pdb].set_index("pocket")
    root = BASE / "processed" / ("structures" if setn == "stripped" else "structures_partner") / pdb
    holo = root / "holo.pdb" if (root / "holo.pdb").exists() else root / "stripped.pdb"
    cdir = FIG / f"case_{pdb}"
    cdir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(holo, cdir / f"{pdb}_holo.pdb")
    # the annotated modulator(s) of this structure, written as their own small file
    mods = {}
    for mj in (root / "modulators.json", BASE / "processed" / "structures" / pdb / "modulators.json"):
        if mj.exists():
            mods = json.load(open(mj))
            break
    lines, codes = [], set()
    for sid, m in mods.items():
        codes.add(m["code"])
        for xyz in m["xyz"]:
            lines.append(f"HETATM{len(lines) + 1:5d}  C   {m['code'][:3]:>3s} M{1:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C")
    if lines:
        (cdir / f"{pdb}_modulator.pdb").write_text("\n".join(lines) + "\nEND\n")
    cols = [("red", "rank 1"), ("orange", "rank 2"), ("purple", "rank 3")]
    pml = [f"load {pdb}_holo.pdb, holo", "bg_color white", "hide everything", "show cartoon, holo and polymer",
           "color grey90, holo and polymer", "set cartoon_transparency, 0.25", "set sphere_scale, 0.5",
           "select other_ligands, holo and organic", "show sticks, other_ligands", "color grey50, other_ligands",
           "set stick_radius, 0.15, other_ligands"]
    if lines:
        pml += [f"load {pdb}_modulator.pdb, modulator", "show spheres, modulator", "color green, modulator",
                "set sphere_scale, 0.7, modulator"]
    lig = _atoms(holo, het=True)
    prot = _atoms(holo, het=False)
    print(f"  annotated modulator(s) of {pdb}: {', '.join(sorted(codes)) or 'not found'}")
    for k, (c, lab) in enumerate(cols):
        if k >= len(d):
            break
        p = int(d.pocket[k])
        res = json.loads(P.loc[p, "residues"]) if p in P.index else []
        sel = " or ".join(f"(chain {r[0]} and resi {r[1]})" for r in res) or "none"
        pml += [f"select pocket_{k + 1}, holo and polymer and ({sel})", f"show spheres, pocket_{k + 1}",
                f"color {c}, pocket_{k + 1}"]
        pa = [xyz for (ch, n, _), xyz in prot.items() if any(ch == r[0] and n == int(r[1]) for r in res)]
        near = []
        if pa:
            pa = np.vstack(pa)
            for (ch, n, rn), xyz in lig.items():
                dm = np.sqrt(((xyz[:, None, :] - pa[None, :, :]) ** 2).sum(-1)).min()
                if dm <= 4.0:
                    near.append(f"{rn} {ch}{n}")
        chains = sorted({r[0] for r in res})
        print(f"  pocket rank {k + 1} ({c}): {'ALLOSTERIC SITE' if d.y[k] == 1 else 'not the annotated site'}; "
              f"chains {','.join(chains)}; {len(res)} residues; ligands within 4 A: {', '.join(near) or 'none'}")
    pml += ["orient holo and polymer", "set ray_opaque_background, 1",
            f"png Fig9b_case_{pdb}_structure.png, width=2400, height=1800, dpi=300, ray=1"]
    (cdir / f"Fig9b_case_{pdb}.pml").write_text("\n".join(pml) + "\n")
    print(f"  saved figures/case_{pdb}/ (structure + Fig9b_case_{pdb}.pml); render inside that folder: "
          f"cd figures/case_{pdb} && pymol -cq Fig9b_case_{pdb}.pml")


# ============================================================================= PLOT
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 7.5,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150})
    return plt


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    print(f"  saved figures/{name}.pdf/.png")


def hbar_ci(ax, labels, means, los, his, colours, xlabel, ref=None, fmt="{:.3f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, height=0.62, color=colours, edgecolor="white", linewidth=1.0, zorder=2)
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=3)
    for yi, m, h in zip(y, means, his):
        ax.text(h + 0.015, yi, fmt.format(m), va="center", ha="left", fontsize=6.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if ref is not None:
        ax.axvline(ref, color=GREY_D, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis="x", color="#e6e5e1", lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)



def plot_workflow(plt):
    """Fig 1: AlloPlex workflow (diagram)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec=INK2, bold=False, size=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=8))
    lb = "#eaf2fc"
    ax.set_ylim(2, 40)
    box(0.5, 17, 15.5, 9, "Protein assembly\n(PDB / mmCIF;\ncomplexes and\nprotein-peptide)", size=6.2)
    box(19.5, 17, 15.5, 9, "Remove ligands,\nwaters, ions,\nnucleic acids", size=6.2)
    box(39, 28, 17, 9, "fpocket:\npockets +\n11 descriptors", size=6.2)
    box(39, 6, 17, 9, "ESM-2 650M:\nper-residue\nembeddings (1280)", size=6.2)
    box(60, 16.5, 17.5, 10, "Pocket vector:\nmean ESM-2 over\npocket residues\n+ descriptors (1291)", fc=lb, size=6.2)
    box(81.5, 16.5, 17.5, 10, "MLP ensemble\n(15 networks)\n-> ranked pockets\n+ PyMOL view", fc=lb, bold=True, size=6.2)
    arrow(16.6, 21.5, 19.0, 21.5)
    arrow(35.6, 23.5, 38.6, 32)
    arrow(35.6, 19.5, 38.6, 11)
    arrow(56.6, 32, 59.6, 24)
    arrow(56.6, 11, 59.6, 19)
    arrow(78.1, 21.5, 81.0, 21.5)
    ax.text(60, 6.5, "Training: 2,187 allosteric and ~113,000 other pockets,\n259 protein families. Evaluation: whole families\n"
                     "held out (5-fold x 3), ligand-free partner structures",
            fontsize=5.9, color=INK2, va="center")
    save(fig, "Fig1_workflow")


def plot_tsne(plt):
    d = pd.read_csv(DATA / "fig3_tsne.csv")
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    n = d[~d.allosteric]
    p = d[d.allosteric]
    ax.scatter(n.x, n.y, s=2, c=GREY, lw=0, label=f"other pockets (random {len(n)})", rasterized=True)
    ax.scatter(p.x, p.y, s=3.5, c=BLUE, lw=0, label=f"allosteric pockets ({len(p)})", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=6, markerscale=3, ncol=2)
    ax.set_title("Pocket features (ESM-2 + fpocket)")
    save(fig, "Fig3_tsne")


RANK_STYLE = {OURS: (BLUE, "-", 2.0), "DeepAllo": (ORANGE, "-", 1.2), "APOP": (AQUA, "-", 1.2),
              "PASSer": ("#eda100", "-", 1.2), "P2Rank": ("#e87ba4", "-", 1.2),
              "fpocket druggability": (GREY_D, (0, (3, 2)), 1.2)}


def plot_ranks(plt):
    d = pd.read_csv(DATA / "fig6_ranks.csv")
    meths = [m for m in RANK_STYLE if m in set(d.method)]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    ks = np.arange(1, 11)
    for m in meths:
        t = d[d.method == m]
        frac = [(t["rank"] <= k).mean() * 100 for k in ks]
        c, ls, lw = RANK_STYLE[m]
        ax.plot(ks, frac, color=c, ls=ls, lw=lw, marker="o", ms=2.5, label=nice(m))
    ax.legend(fontsize=5.8, loc="lower right")
    ax.set_xticks(ks)
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0, 100)
    ax.set_xlabel("top-k pockets considered")
    ax.set_ylabel("allosteric sites found (%)")
    ax.grid(color="#e6e5e1", lw=0.5)
    ax.set_title("a  sites found within the top k pockets")
    ax = axs[1]
    pct = [1, 3, 5, 10]
    w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        t = d[d.method == m]
        rel = t["rank"] / t["n_pockets"]
        vals = [(rel <= q / 100).mean() * 100 for q in pct]
        ax.bar(np.arange(len(pct)) + (j - (len(meths) - 1) / 2) * w, vals, w * 0.92, color=RANK_STYLE[m][0],
               edgecolor="white", label=nice(m))
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([f"top {q} %" for q in pct])
    ax.set_ylabel("allosteric sites (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    ax.set_title("b  rank relative to the number of pockets")
    ax.text(0.02, 0.97, "colours as in a", transform=ax.transAxes, fontsize=6, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "Fig6_ranking")


def plot_fig1(plt):
    s = pd.read_csv(DATA / "fig1_sites.csv")
    s = s[~s.complex_kind.isin(["no protein chain >= 50 aa", "not in PDB"])]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.4, 1, 0.8]})
    order = ["protein-peptide", "hetero-complex", "pseudo-hetero", "homo-oligomer", "homo (assembly only)", "monomer"]
    k = s.complex_kind.value_counts()
    k = k.reindex([o for o in order if o in k.index] + [o for o in k.index if o not in order])
    fam = s.groupby("complex_kind").family.nunique()
    ax = axs[0]
    cols = [GREY if kk == "monomer" else BLUE for kk in k.index]
    y = np.arange(len(k))[::-1]
    ax.barh(y, k.values, color=cols, height=0.62, edgecolor="white")
    for yi, (kk, v) in zip(y, k.items()):
        ax.text(v + k.max() * 0.02, yi, f"{v} ({fam.get(kk, 0)} fam.)", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(k.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, k.max() * 1.45)
    ax.set_xlabel("allosteric sites")
    ax.set_title("a  sites per complex type")
    fig.suptitle(f"{len(s)} allosteric sites, {s.pdb_id.nunique()} structures, {s.family.nunique()} families",
                 x=0.01, ha="left", fontsize=8, color=INK2, y=0.99)
    ax = axs[1]
    if (DATA / "fig1_structures.csv").exists():
        st = pd.read_csv(DATA / "fig1_structures.csv")
        c = st.n_protein_chains.clip(upper=12).value_counts().sort_index()
        ax.bar(c.index, c.values, color=[GREY if i == 1 else BLUE for i in c.index], width=0.75, edgecolor="white")
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
        ax.set_xticklabels(["1", "2", "4", "6", "8", "10", "≥12"])
        ax.set_xlabel("protein chains in the assembly")
        ax.set_ylabel("structures")
        ax.set_title("b  assemblies")
    ax = axs[2]
    src = s.source.value_counts()
    ax.bar(range(len(src)), src.values, color=[BLUE, AQUA, ORANGE][:len(src)], width=0.6, edgecolor="white")
    ax.set_xticks(range(len(src)))
    ax.set_xticklabels(src.index)
    ax.set_ylabel("sites")
    ax.set_title("c  source")
    fig.tight_layout()
    save(fig, "Fig2_dataset")


def plot_fig2(plt):
    P = pd.read_csv(DATA / "fig2_pockets.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1.25, 1.6, 0.95]})
    ax = axs[0]
    for g, col, lw, short in (("other pockets", GREY_D, 1.2, "other pockets"),
                              ("allosteric (not PPI-annotated)", BLUE, 1.8, "allosteric, not PPI-annot."),
                              ("allosteric (PPI-annotated)", ORANGE, 1.2, "allosteric, PPI-annot.")):
        d = np.sort(P[P.group == g].dist_iface_5.fillna(60).clip(upper=60).to_numpy())
        if len(d):
            ax.step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=col, lw=lw,
                    label=f"{short} (n={len(d)})")
    ax.set_xlabel("distance to nearest interface residue (Å)")
    ax.set_ylabel("cumulative fraction of pockets")
    ax.set_xlim(0, 40)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.32, -0.3), fontsize=6, handlelength=1.4, ncol=1)
    ax.set_title("a  multi-chain assemblies")
    ax.grid(color="#e6e5e1", lw=0.5)
    ax = axs[1]
    lab = [r[0] for r in REPORTED]
    cols = [BLUE if r[4] == "primary" else (AQUA if r[4] == "control" else GREY) for r in REPORTED]
    y = np.arange(len(REPORTED))[::-1]
    for yi, r, c in zip(y, REPORTED, cols):
        ax.plot([r[2], r[3]], [yi, yi], color=c, lw=1.4, solid_capstyle="round")
        ax.plot(r[1], yi, "o", ms=4.5, color=c, mec="white", mew=0.8)
        ax.text(r[3] + 0.01, yi, f"{r[1]:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0.5, color=GREY_D, lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.3, 0.88)
    ax.set_xlabel("matched AUC (0.5 = no preference)")
    ax.set_title("b  pre-registered test and controls")
    ax = axs[2]
    x = np.arange(2)
    real = [RANDOM_PATCH["overlap"][0], RANDOM_PATCH["distance"][0]]
    rnd = [RANDOM_PATCH["overlap"][1], RANDOM_PATCH["distance"][1]]
    ax.bar(x - 0.18, real, 0.34, color=BLUE, label="real interface", edgecolor="white")
    ax.bar(x + 0.18, rnd, 0.34, color=GREY, label="random surface patch", edgecolor="white")
    ax.axhline(0.5, color=GREY_D, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(["overlap", "distance"], fontsize=6.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("matched AUC")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("c  random-patch control")
    fig.tight_layout()
    save(fig, "Fig4_interface")


def plot_fig3(plt):
    d = pd.read_csv(DATA / "fig3_inflation.csv")
    order = ["family split (reference)", "gene-name split", "ligand-bound (holo) inputs", "random pocket split"]
    order = [o for o in order if o in set(d.split)]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    m, lo, hi = [], [], []
    for o in order:
        t = d[d.split == o]
        m.append(t.top1.mean())
        a, b = boot_ci(t.family, t.top1)
        lo.append(a)
        hi.append(b)
    hbar_ci(ax, order, m, lo, hi, [BLUE] + [GREY] * (len(order) - 1), "Top-1 (stripped structures, repeat 0)",
            ref=m[0])
    ax.set_xlim(0, 0.95)
    ax.set_title("Shortcut evaluations inflate performance")
    fig.tight_layout()
    save(fig, "FigS1_inflation")


def plot_fig4(plt):
    d = pd.read_csv(DATA / "fig4_models.csv")
    if "set" in d.columns:
        d = d[d.set == "primary"]
    want = ["MLP B", "MLP B+I", "GNN B+I+fp", "GNN B", "GNN B+I", "LightGBM B+I", "LightGBM B",
            "fpocket druggability", "pocket size", "pocket volume", "fpocket score"]
    want = [w for w in want if w in set(d.method)]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    for ax, metric, lab in ((axs[0], "top1", "Top-1"), (axs[1], "site_auc", "site AUROC")):
        m, lo, hi = [], [], []
        for w in want:
            t = d[d.method == w]
            m.append(t[metric].mean())
            a, b = boot_ci(t.family, t[metric])
            lo.append(a)
            hi.append(b)
        cols = [BLUE if w == "MLP B" else (GREY_D if not w.startswith(("fpocket", "pocket")) else GREY) for w in want]
        hbar_ci(ax, ["AlloPlex (MLP B)" if w == "MLP B" else w for w in want], m, lo, hi, cols, lab, ref=m[0])
        ax.set_xlim(0, 1.0 if metric == "site_auc" else 0.55)
    axs[0].set_title("a  Top-1 (PRIMARY set)")
    axs[1].set_title("b  site AUROC")
    fig.tight_layout()
    save(fig, "Fig5_models")


def tool_order(d):
    first = [OURS, "DeepAllo", "APOP", "PASSer", "P2Rank", "fpocket druggability"]
    return [m for m in first if m in set(d.method)] + [m for m in sorted(set(d.method)) if m not in first]


def nice(m):
    return "AlloPlex (ours)" if m == OURS else m


def plot_fig5(plt):
    d = pd.read_csv(DATA / "fig5_tools.csv")
    meths = tool_order(d)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True)
    for ax, ev, title in ((axs[0], "matched", "a  matched to the same pockets"),
                          (axs[1], "native", "b  each method's own pockets")):
        m, lo, hi = [], [], []
        for w in meths:
            t = d[(d.method == w) & (d.evaluation == ev)]
            m.append(t.top1.mean() if len(t) else np.nan)
            a, b = boot_ci(t.family, t.top1) if len(t) else (np.nan, np.nan)
            lo.append(a)
            hi.append(b)
        cols = [BLUE if w == OURS else (GREY if w == "fpocket druggability" else GREY_D) for w in meths]
        hbar_ci(ax, [nice(w) for w in meths], m, lo, hi, cols, "Top-1", ref=m[0])
        ax.set_xlim(0, 0.55)
        ax.set_title(title)
    fig.tight_layout()
    save(fig, "Fig7_tools")


def plot_fig6(plt):
    d = pd.read_csv(DATA / "fig5_tools.csv")
    d = d[d.evaluation == "matched"]
    meths = tool_order(d)
    groups = [("all sites", d), ("new proteins", d[d.new_protein == True]),  # noqa: E712
              ("new families", d[d.new_family == True])]  # noqa: E712
    shades = [BLUE, "#7aaee8", "#c3d9f4"]
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    x = np.arange(len(meths))
    w = 0.26
    for j, (g, t) in enumerate(groups):
        m, lo, hi = [], [], []
        for me in meths:
            tt = t[t.method == me]
            m.append(tt.top1.mean() if len(tt) else np.nan)
            a, b = boot_ci(tt.family, tt.top1) if len(tt) else (np.nan, np.nan)
            lo.append(a)
            hi.append(b)
        n = len(t[t.method == OURS])
        xs = x + (j - 1) * w
        ax.bar(xs, m, w * 0.92, color=shades[j], edgecolor="white", label=f"{g} (n={n} sites)", zorder=2)
        ax.errorbar(xs, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="none",
                    ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([nice(m) for m in meths])
    ax.set_ylabel("Top-1 (matched)")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    ax.legend(ncol=3, fontsize=6.5, loc="upper right")
    ax.set_title("Proteins and families absent from ASD 2019 (the tools' training source)")
    fig.tight_layout()
    save(fig, "Fig8_novelty")


def plot_fig7(plt, pdb="1A3W"):
    f = DATA / f"fig7_{pdb}.csv"
    d = pd.read_csv(f).sort_values("cv_score", ascending=False).reset_index(drop=True)
    d["rank_cv"] = np.arange(1, len(d) + 1)
    d["rank_drug"] = d.druggability.rank(ascending=False, method="min").astype(int)
    top = d.head(15)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    cols = [BLUE if y == 1 else GREY for y in top.y]
    ax.bar(top.rank_cv, top.cv_score, color=cols, width=0.75, edgecolor="white")
    ax.set_xlabel("pocket rank (cross-validation AlloPlex score)")
    ax.set_ylabel("AlloPlex score")
    ax.set_xticks(top.rank_cv)
    kind = "partner structure, " if ("set" in d.columns and d["set"].iloc[0] == "partner") else ""
    ax.set_title(f"a  {pdb}: cross-validation ranking ({kind}family held out)")
    ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    pos = d[d.y == 1]
    ax = axs[1]
    ax.axis("off")
    lines = [f"{len(d)} fpocket pockets in the assembly", f"test folds: {d.test_folds.iloc[0]}", ""]
    for _, r in pos.iterrows():
        lines.append(f"known allosteric site (pocket {int(r.pocket)}):")
        lines.append(f"   AlloPlex rank {int(r.rank_cv)}   (score {r.cv_score:.2f})")
        lines.append(f"   druggability rank {int(r.rank_drug)}   (druggability {r.druggability:.2f})")
    ax.text(0, 0.95, "\n".join(lines), va="top", fontsize=7, color=INK, family="DejaVu Sans")
    ax.set_title("b  AlloPlex vs druggability")
    fig.tight_layout()
    save(fig, f"Fig9_case_{pdb}")



# ============================================================================= PUBLICATION STYLE (v3)
# Okabe-Ito colour-blind-safe palette; a marker per method as second encoding (APOP/PASSer are close
# for deutan vision). Panel letters only; explanations belong in the captions.
OI = {"blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
      "orange": "#E69F00", "sky": "#56B4E9", "grey": "#999999", "lgrey": "#CCCCCC", "dgrey": "#555555"}
METHOD = {OURS: ("AlloPlex", OI["blue"], "o"), "DeepAllo": ("DeepAllo", OI["vermilion"], "s"),
          "APOP": ("APOP", OI["green"], "^"), "PASSer": ("PASSer", OI["purple"], "D"),
          "P2Rank": ("P2Rank", OI["orange"], "v"), "fpocket druggability": ("Druggability", OI["grey"], "x")}
W1, W2 = 3.35, 7.0  # single / double column width (inches; 85 / 178 mm)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = "Arial" if any("Arial" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": fam, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.titlesize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.color": "#333333", "ytick.color": "#333333", "axes.labelcolor": "#111111",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.handlelength": 1.4, "legend.borderaxespad": 0.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "figure.dpi": 150, "lines.linewidth": 1.0})
    return plt


def letter(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def dotci(ax, labels, means, los, his, colours, markers=None, ref=None, xlabel=""):
    """dot-and-whisker, top to bottom."""
    y = np.arange(len(labels))[::-1]
    for yi, m, lo, hi, c, mk in zip(y, means, los, his, colours, markers or ["o"] * len(labels)):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.1, solid_capstyle="round", zorder=2)
        ax.plot(m, yi, marker=mk, ms=4.2, color=c, mec="white" if mk not in ("x",) else c, mew=0.6, zorder=3,
                ls="none")
    if ref is not None:
        ax.axvline(ref, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.6, len(labels) - 0.4)


def ci_of(t, col):
    v = t[col].astype(float)
    lo, hi = boot_ci(t.family, v)
    return v.mean(), lo, hi


# numbers shown in the study workflow (Fig 1a); from Table 1 / Table S1
WF = {"asd": 1418, "allobench": 957, "sites": 2375, "families": 268, "assemblies": 2234, "partners": 375,
      "pockets": 115511, "positives": 2187}


def p_workflow(plt):
    """Fig 1: (a) study workflow: data -> model -> evaluation; (b) how AlloPlex predicts a new structure."""
    from matplotlib.patches import FancyBboxPatch
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W2, 4.1), gridspec_kw={"height_ratios": [2.55, 1], "hspace": 0.08})
    tint, ink, sub_c = "#E8F1F8", "#333333", "#444444"

    def box(a, x, y, w, h, title, sub="", fc="white", ec=ink, lw=0.7, ts=6.8):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9", fc=fc, ec=ec, lw=lw))
        if sub:
            a.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=ts, fontweight="bold")
            a.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=5.8, color=sub_c, linespacing=1.15)
        else:
            a.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=ts, fontweight="bold")

    def arr(a, x1, y1, x2, y2, c=ink):
        a.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7, mutation_scale=7, shrinkA=0, shrinkB=0))

    # ---------------------------------------------------------------- a: study workflow, three lanes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    lanes = [(0.5, "Data"), (34.5, "Model"), (68.5, "Evaluation")]
    for x0, name in lanes:
        ax.add_patch(FancyBboxPatch((x0, 0.5), 31, 62, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="#F6F6F6", ec="none"))
        ax.text(x0 + 15.5, 59.5, name, ha="center", va="center", fontsize=7.5, fontweight="bold", color=OI["blue"])
    # data lane
    bw, bh, x = 27, 10.5, 2.5
    box(ax, x, 44, bw, bh, "Allosteric sites", f"ASD {WF['asd']:,} + AlloBench {WF['allobench']:,}\n"
                                                 f"{WF['sites']:,} sites, {WF['families']} families")
    box(ax, x, 27.5, bw, bh, "Structures", f"{WF['assemblies']:,} ligand-free assemblies\n"
                                           f"{WF['partners']} modulator-free partners")
    box(ax, x, 11, bw, bh, "Pockets and labels", f"fpocket: {WF['pockets']:,} pockets\n"
                                                  f"{WF['positives']:,} allosteric pockets")
    arr(ax, x + bw / 2, 44, x + bw / 2, 38.6)
    arr(ax, x + bw / 2, 27.5, x + bw / 2, 22.1)
    # model lane
    x = 36.5
    box(ax, x, 44, bw, bh, "Pocket features", "ESM-2 (650M) pocket mean\n+ 11 fpocket descriptors")
    box(ax, x, 27.5, bw, bh, "Model selection", "MLP, GNN, LightGBM\n± interface features")
    box(ax, x, 11, bw, bh, "AlloPlex", "MLP, 15-network ensemble", fc=tint, ec=OI["blue"], lw=0.9)
    arr(ax, x + bw / 2, 44, x + bw / 2, 38.6)
    arr(ax, x + bw / 2, 27.5, x + bw / 2, 22.1)
    ax.text(x + bw / 2, 4.6, "family-split cross-validation (5 folds × 3)", ha="center", va="center",
            fontsize=5.8, color=sub_c, style="italic")
    # evaluation lane: five compact boxes
    x, eh = 70.5, 7.6
    items = [("Shortcut checks", "random / gene split, holo inputs"),
             ("Tool benchmark", "DeepAllo, APOP, PASSer, P2Rank"),
             ("Unseen proteins", "absent from ASD 2019"),
             ("Complex types", "monomer to hetero-complex"),
             ("Case study", "5OLK")]
    ys = [48.3 - i * 10.4 for i in range(len(items))]
    for (t, sub), y in zip(items, ys):
        box(ax, x, y, bw, eh, t, sub, ts=6.4)
    # lane-to-lane arrows
    arr(ax, 31.6, 32.75, 36.2, 32.75)          # data lane -> model lane
    arr(ax, 63.6, 16.25, 70.2, 16.25, OI["blue"])  # AlloPlex -> evaluation
    ax.text(-0.5, 63.5, "a", fontsize=9, fontweight="bold", va="top")

    # ---------------------------------------------------------------- b: prediction for a new structure
    bx.set_xlim(0, 100)
    bx.set_ylim(0, 30)
    bx.axis("off")
    box(bx, 1, 11, 14, 8, "Assembly", "PDB / mmCIF")
    box(bx, 19, 11, 14, 8, "Clean", "ligand-free")
    box(bx, 38, 20, 16, 8, "fpocket", "11 descriptors")
    box(bx, 38, 2, 16, 8, "ESM-2", "per residue")
    box(bx, 59, 11, 17, 8, "Pocket vector", "1291 features", fc=tint)
    box(bx, 81, 11, 18, 8, "AlloPlex", "ranked pockets", fc=tint, ec=OI["blue"])
    arr(bx, 15.3, 15, 18.7, 15)
    arr(bx, 33.3, 16.5, 37.7, 23)
    arr(bx, 33.3, 13.5, 37.7, 7)
    arr(bx, 54.3, 24, 58.7, 17)
    arr(bx, 54.3, 6, 58.7, 13)
    arr(bx, 76.3, 15, 80.7, 15)
    bx.text(-0.5, 30, "b", fontsize=9, fontweight="bold", va="top")
    save(fig, "FigS2_study_overview")


def p_architecture(plt):
    """Fig 1 (illustrated): how AlloPlex scores the pockets of a protein complex."""
    from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(W2, 3.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.set_aspect("equal")
    ax.axis("off")
    C = {"esm": ("#EFEAFB", "#7B68C8", "#3E2C8F"), "fp": ("#FDEBDD", "#E08A4F", "#A0461A"),
         "mlp": ("#E8F1FB", "#3C7DC4", "#174A8B"), "out": ("#EAF4E4", "#5E9C45", "#2F6420"),
         "a": ("#E3F2DC", "#5E9C45"), "b": ("#DDEBF8", "#3C7DC4")}
    ink, grey = "#333333", "#555555"

    def card(x, y, w, h, key, lw=0.8):
        fc, ec, _ = C[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ink, lw=0.8, mutation_scale=8, shrinkA=0, shrinkB=0))

    def t(x, y, s, size=6.2, c=ink, w="normal", ha="center", style="normal"):
        ax.text(x, y, s, fontsize=size, color=c, fontweight=w, ha=ha, va="center", style=style)

    # ------------------------------------------------ row 1: complex -> fpocket -> pockets -> features
    ax.add_patch(Ellipse((8, 47.5), 13, 8.5, fc=C["a"][0], ec=C["a"][1], lw=0.8))
    ax.add_patch(Ellipse((18.5, 47.5), 13, 8.5, fc=C["b"][0], ec=C["b"][1], lw=0.8, alpha=0.9))
    ax.add_patch(Ellipse((13.25, 47.5), 3.2, 6.5, fc="#CFE6D8", ec="none", alpha=0.9))
    xs = np.linspace(4, 11.5, 60)
    ax.plot(xs, 48.6 + 1.1 * np.sin((xs - 4) * 2.6), color=C["a"][1], lw=1.1)
    for yy in (49.3, 47.3):
        ax.annotate("", xy=(23.5, yy), xytext=(16.5, yy), arrowprops=dict(arrowstyle="simple,head_width=0.6",
                    fc="#9CC3EA", ec="none", mutation_scale=5))
    t(7.2, 44.6, "Chain A", 5.6, C["a"][1], "bold")
    t(19.5, 44.6, "Chain B", 5.6, C["b"][1], "bold")
    t(13.25, 41.6, "ligand-free assembly", 5.6, grey)
    arr(25.8, 47.5, 29.5, 47.5)
    card(30, 43.5, 15, 8, "fp")
    t(37.5, 48.8, "fpocket", 7, C["fp"][2], "bold")
    t(37.5, 45.8, "pocket detection", 5.8, C["fp"][2])
    arr(45.8, 47.5, 49.5, 47.5)
    for k, x in enumerate((53, 61, 69)):
        ax.add_patch(Circle((x, 47.5), 3.4, fc="white", ec="#7B68C8", lw=0.8, ls=(0, (2.5, 1.5))))
        t(x, 48.4, "pocket", 5.6)
        t(x, 46.4, f"#{k + 1}", 5.6)
    t(75, 47.5, "…", 8)
    t(61, 41.8, f"~{round(WF['pockets'] / WF['assemblies'])} pockets per assembly", 5.8, grey)
    arr(78, 47.5, 82, 47.5)
    t(91, 49.0, "Features per pocket", 6.4)
    t(91, 46.2, "→ 1,291-dim", 6.4)
    arr(91, 43.6, 91, 39.6)

    # ------------------------------------------------ row 2: the two feature groups
    ax.add_patch(FancyBboxPatch((5, 21.5), 90, 17.5, boxstyle="round,pad=0,rounding_size=1.5", fc="none",
                                ec="#AAAAAA", lw=0.7, ls=(0, (3, 2))))
    # ESM-2 card
    card(9, 23.5, 38, 13.5, "esm")
    t(28, 35.0, "ESM-2 (650M)", 7, C["esm"][2], "bold")
    t(28, 32.7, "frozen protein language model", 5.6, C["esm"][1])
    ax.add_patch(FancyBboxPatch((11.5, 29.0), 14, 2.0, boxstyle="round,pad=0,rounding_size=0.5",
                                fc="#DCD3F5", ec="none"))
    t(18.5, 30.0, "M G N C S T …", 5.6, C["esm"][2])
    for k, h in enumerate((4.2, 3.0, 4.6, 3.4, 4.0)):
        ax.add_patch(Rectangle((12 + k * 2.7, 24.3), 2.0, h, fc="#8C7BD6", ec="none"))
    t(37.5, 29.2, "per-residue embeddings", 5.4, C["esm"][2])
    t(37.5, 27.0, "mean over pocket residues", 5.4, C["esm"][2])
    t(37.5, 24.8, "→ 1,280-dim", 6.2, C["esm"][2], "bold")
    # fpocket card
    card(53, 23.5, 38, 13.5, "fp")
    t(72, 35.0, "fpocket descriptors", 7, C["fp"][2], "bold")
    t(72, 32.7, "pocket geometry and chemistry", 5.6, C["fp"][1])
    blob = np.array([[56, 25], [57.2, 29.8], [60.5, 31], [65, 30.2], [66.5, 27], [64.5, 24.4], [59, 24.2]])
    ax.add_patch(Polygon(blob, closed=True, fc="#F6B98F", ec=C["fp"][1], lw=0.7))
    for cx, cy in ((59, 27.8), (62.3, 28.6), (61, 25.9), (63.8, 26.4)):
        ax.add_patch(Circle((cx, cy), 0.9, fc="none", ec=C["fp"][2], lw=0.6, ls=(0, (1.5, 1))))
    t(79, 29.2, "volume, druggability, SASA", 5.8, C["fp"][2])
    t(79, 27.0, "polarity, hydrophobicity, …", 5.8, C["fp"][2])
    t(79, 24.8, "→ 11-dim", 6.2, C["fp"][2], "bold")
    t(28, 20.3, "1,280", 6.2, C["esm"][2], "bold")
    t(72, 20.3, "11", 6.2, C["fp"][2], "bold")
    arr(50, 21.5, 50, 18.7)
    ax.add_patch(FancyBboxPatch((38, 14.6), 24, 4, boxstyle="round,pad=0,rounding_size=1.0", fc="#F2F2F2",
                                ec="#999999", lw=0.7))
    t(50, 16.6, "Concatenate → 1,291-dim", 6.4)
    arr(50, 14.6, 50, 12.6)

    # ------------------------------------------------ row 3: classifier -> ranked pockets
    card(16, 0.8, 38, 11.6, "mlp")
    t(35, 10.2, "MLP classifier", 7, C["mlp"][2], "bold")
    t(35, 7.6, "1,291 → 512 → 256 → 64 → 1", 6.2, C["mlp"][2])
    t(35, 5.2, "focal loss (γ = 2, α = 0.75)  |  AdamW", 5.6, C["mlp"][2])
    t(35, 2.8, "BatchNorm, ReLU, dropout  |  15-network ensemble", 5.6, C["mlp"][2])
    arr(54.5, 6.6, 59.5, 6.6)
    card(60, 0.8, 26, 11.6, "out")
    t(73, 8.8, "Ranked allosteric", 7, C["out"][2], "bold")
    t(73, 6.4, "pockets", 7, C["out"][2], "bold")
    t(73, 3.4, "score per pocket + PyMOL view", 5.4, C["out"][2])
    for k, (h, c) in enumerate(zip((9.5, 7.2, 5.0, 3.0), ("#3E7B2C", "#5E9C45", "#94C27E", "#C6E0B8"))):
        ax.add_patch(Rectangle((88 + k * 2.6, 1.6), 2.0, h, fc=c, ec="none"))
    save(fig, "Fig1_workflow")


def p_dataset(plt):
    s = pd.read_csv(DATA / "fig1_sites.csv")
    s = s[~s.complex_kind.isin(["no protein chain >= 50 aa", "not in PDB"])]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9), gridspec_kw={"width_ratios": [1.5, 1.1, 0.6], "wspace": 0.55})
    order = ["protein-peptide", "hetero-complex", "pseudo-hetero", "homo-oligomer", "homo (assembly only)", "monomer"]
    short = {"protein-peptide": "Protein-peptide", "hetero-complex": "Hetero-complex", "pseudo-hetero": "Pseudo-hetero",
             "homo-oligomer": "Homo-oligomer", "homo (assembly only)": "Homo (assembly)", "monomer": "Monomer"}
    k = s.complex_kind.value_counts().reindex([o for o in order if o in set(s.complex_kind)])
    ax = axs[0]
    y = np.arange(len(k))[::-1]
    ax.barh(y, k.values, height=0.65, color=[OI["grey"] if i == "monomer" else OI["blue"] for i in k.index])
    ax.set_yticks(y)
    ax.set_yticklabels([short[i] for i in k.index])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sites")
    letter(ax, "a", x=-0.55)
    ax = axs[1]
    if (DATA / "fig1_structures.csv").exists():
        st = pd.read_csv(DATA / "fig1_structures.csv")
        c = st.n_protein_chains.clip(upper=12).value_counts().sort_index()
        ax.bar(c.index, c.values, width=0.7, color=[OI["grey"] if i == 1 else OI["blue"] for i in c.index])
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
        ax.set_xticklabels(["1", "2", "4", "6", "8", "10", "≥12"])
        ax.set_xlabel("Chains per assembly")
        ax.set_ylabel("Structures")
        letter(ax, "b", x=-0.3)
    ax = axs[2]
    src = s.source.value_counts()
    ax.bar(range(len(src)), src.values, width=0.6, color=[OI["blue"], OI["sky"]][:len(src)])
    ax.set_xticks(range(len(src)))
    ax.set_xticklabels(src.index, rotation=0)
    ax.set_ylabel("Sites")
    letter(ax, "c", x=-0.55)
    save(fig, "Fig2_dataset")


def p_tsne(plt):
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator
    d = pd.read_csv(DATA / "fig3_tsne.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, W2 * 0.42),
                            gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.3})

    # (a) allosteric versus other pockets
    ax = axs[0]
    n, p = d[~d.allosteric], d[d.allosteric]
    ax.scatter(n.x, n.y, s=1.2, c=OI["lgrey"], lw=0, rasterized=True, label="Other pockets")
    ax.scatter(p.x, p.y, s=2.0, c=OI["blue"], lw=0, rasterized=True, label="Allosteric pockets")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, markerscale=4, handletextpad=0.2)
    letter(ax, "a")

    # (b) the same map coloured by pocket volume
    ax = axs[1]
    v = d.volume.to_numpy(float)
    lo = max(1.0, np.nanpercentile(v[v > 0], 2))
    hi = np.nanpercentile(v, 98)
    sc = ax.scatter(d.x, d.y, s=1.4, c=np.clip(v, lo, hi), cmap="viridis",
                    norm=LogNorm(vmin=lo, vmax=hi), lw=0, rasterized=True)
    ax.scatter(p.x, p.y, s=6.0, facecolors="none", edgecolors="#000000", lw=0.25,
               rasterized=True, label="Allosteric pockets")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.055)
    cb.set_label("Pocket volume (\u00c5$^3$)", labelpad=2)
    cb.ax.tick_params(length=2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), markerscale=2.2, handletextpad=0.2)
    letter(ax, "b")

    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    save(fig, "Fig3_tsne")


def p_interface(plt):
    P = pd.read_csv(DATA / "fig2_pockets.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw={"width_ratios": [1, 1.25, 0.7], "wspace": 0.7})
    ax = axs[0]
    for g, col, lab in (("other pockets", OI["grey"], "Other"), ("allosteric (not PPI-annotated)", OI["blue"], "Allosteric"),
                        ("allosteric (PPI-annotated)", OI["vermilion"], "Allosteric (PPI label)")):
        dd = np.sort(P[P.group == g].dist_iface_5.fillna(60).clip(upper=60).to_numpy())
        if len(dd):
            ax.step(dd, np.arange(1, len(dd) + 1) / len(dd), where="post", color=col, lw=1.1, label=lab)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Distance to interface (Å)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, 0.0), fontsize=6, handlelength=1.0)
    letter(ax, "a", x=-0.3)
    ax = axs[1]
    labs = ["Overlap (primary)", "PPI-annotated sites", "Non-interface pockets", "Hetero/peptide complexes",
            "Distance to interface"]
    cols = [OI["blue"], OI["grey"], OI["grey"], OI["grey"], OI["green"]]
    dotci(ax, labs, [r[1] for r in REPORTED], [r[2] for r in REPORTED], [r[3] for r in REPORTED], cols,
          ref=0.5, xlabel="Matched AUC")
    ax.set_xlim(0.3, 0.85)
    letter(ax, "b", x=-0.62)
    ax = axs[2]
    x = np.arange(2)
    ax.bar(x - 0.18, [RANDOM_PATCH["overlap"][0], RANDOM_PATCH["distance"][0]], 0.34, color=OI["blue"], label="Interface")
    ax.bar(x + 0.18, [RANDOM_PATCH["overlap"][1], RANDOM_PATCH["distance"][1]], 0.34, color=OI["lgrey"],
           label="Random patch")
    ax.axhline(0.5, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(["Overlap", "Distance"])
    ax.set_ylim(0.4, 0.76)
    ax.set_ylabel("Matched AUC")
    ax.legend(loc="upper left")
    letter(ax, "c", x=-0.55)
    save(fig, "Fig4_interface")


def p_models(plt):
    d = pd.read_csv(DATA / "fig4_models.csv")
    if "set" in d.columns:
        d = d[d.set == "primary"]
    rows = [("MLP B", "AlloPlex (MLP)"), ("MLP B+I", "MLP + interface"), ("GNN B+I+fp", "GNN + descriptors"),
            ("GNN B", "GNN"), ("GNN B+I", "GNN + interface"), ("LightGBM B+I", "LightGBM + interface"),
            ("LightGBM B", "LightGBM"), ("fpocket druggability", "Druggability"), ("pocket size", "Pocket size"),
            ("pocket volume", "Pocket volume"), ("fpocket score", "fpocket score")]
    rows = [r for r in rows if r[0] in set(d.method)]
    cols = [OI["blue"] if m == "MLP B" else ("#444444" if not m.startswith(("fpocket", "pocket")) else OI["grey"])
            for m, _ in rows]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, metric, lab, let in ((axs[0], "top1", "Top-1", "a"), (axs[1], "site_auc", "Site AUROC", "b")):
        st = [ci_of(d[d.method == m], metric) for m, _ in rows]
        dotci(ax, [l for _, l in rows], [x[0] for x in st], [x[1] for x in st], [x[2] for x in st], cols,
              ref=st[0][0], xlabel=lab)
        letter(ax, let, x=-0.02 if let == "b" else -0.42)
    axs[0].set_xlim(0.1, 0.55)
    axs[1].set_xlim(0.55, 0.92)
    save(fig, "Fig5_models")


def p_ranking(plt):
    d = pd.read_csv(DATA / "fig6_ranks.csv")
    meths = [m for m in METHOD if m in set(d.method)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})
    ax = axs[0]
    ks = np.arange(1, 11)
    for m in meths:
        t = d[d.method == m]
        lab, c, mk = METHOD[m]
        ax.plot(ks, [(t["rank"] <= k).mean() * 100 for k in ks], color=c, marker=mk, ms=3.2,
                lw=1.6 if m == OURS else 0.9, ls=(0, (3, 2)) if m == "fpocket druggability" else "-",
                mec="white" if mk != "x" else c, mew=0.5, label=lab, zorder=3 if m == OURS else 2)
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(20, 85)
    ax.set_xlabel("Top-k pockets")
    ax.set_ylabel("Sites found (%)")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8)
    letter(ax, "a", x=-0.2)
    ax = axs[1]
    pct = [3, 5, 10]
    w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        t = d[d.method == m]
        rel = t["rank"] / t["n_pockets"]
        lab, c, _ = METHOD[m]
        ax.bar(np.arange(len(pct)) + (j - (len(meths) - 1) / 2) * w, [(rel <= q / 100).mean() * 100 for q in pct],
               w * 0.9, color=c, label=lab)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([f"Top {q}%" for q in pct])
    ax.set_ylabel("Sites (%)")
    letter(ax, "b", x=-0.2)
    save(fig, "Fig6_ranking")


def p_tools(plt):
    d = pd.read_csv(DATA / "fig5_tools.csv")
    meths = [m for m in METHOD if m in set(d.method)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, ev, let in ((axs[0], "matched", "a"), (axs[1], "native", "b")):
        st = [ci_of(d[(d.method == m) & (d.evaluation == ev)], "top1") for m in meths]
        dotci(ax, [METHOD[m][0] for m in meths], [x[0] for x in st], [x[1] for x in st], [x[2] for x in st],
              [METHOD[m][1] for m in meths], [METHOD[m][2] for m in meths], ref=st[0][0],
              xlabel="Top-1 (shared pockets)" if ev == "matched" else "Top-1 (own pockets)")
        ax.set_xlim(0.05, 0.55)
        letter(ax, let, x=-0.02 if let == "b" else -0.3)
    save(fig, "Fig7_tools")


def p_novelty(plt):
    d = pd.read_csv(DATA / "fig5_tools.csv")
    d = d[d.evaluation == "matched"]
    meths = [m for m in METHOD if m in set(d.method)]
    subsets = [("All", None), ("New proteins", "new_protein"), ("New families", "new_family")]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ends = []
    for m in meths:
        lab, c, mk = METHOD[m]
        ys = []
        for _, col in subsets:
            t = d[d.method == m] if col is None else d[(d.method == m) & (d[col].astype(str) == "True")]
            ys.append(t.top1.astype(float).mean())
        ax.plot(range(3), ys, color=c, marker=mk, ms=3.5, lw=1.6 if m == OURS else 0.9,
                ls=(0, (3, 2)) if m == "fpocket druggability" else "-", mec="white" if mk != "x" else c, mew=0.5,
                label=lab, zorder=3 if m == OURS else 2)
        ends.append([ys[-1], lab, c])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):  # keep end labels at least 0.016 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.016)
    for yv, lab, c in ends:
        ax.text(2.08, yv, lab, fontsize=6, va="center", color=c)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s for s, _ in subsets])
    ax.set_xlim(-0.2, 2.75)
    ax.set_ylabel("Top-1")
    ax.set_ylim(0.1, 0.45)
    save(fig, "Fig8_novelty")


def p_case(plt, pdb):
    d = pd.read_csv(DATA / f"fig7_{pdb}.csv").sort_values("cv_score", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    top = d.head(15)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.9), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.45})
    ax = axs[0]
    ax.bar(top["rank"], top.cv_score, width=0.7, color=[OI["blue"] if y == 1 else OI["lgrey"] for y in top.y])
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("Pocket rank")
    ax.set_ylabel("AlloPlex score")
    letter(ax, "a", x=-0.2)
    ax = axs[1]
    r = pd.read_csv(DATA / "fig6_ranks.csv", dtype={"site_id": str})
    r = r[r.pdb_id == pdb]
    meths = [m for m in METHOD if m in set(r.method)]
    if len(r):
        ranks = [r[r.method == m]["rank"].min() for m in meths]
        npk = int(r.n_pockets.max())
        xmax = max(npk, 10) * 1.25
        y = np.arange(len(meths))[::-1]
        for yi, m, rk in zip(y, meths, ranks):
            lab, c, mk = METHOD[m]
            if not np.isfinite(rk):  # tool failed / site not among its pockets
                ax.text(xmax * 0.97, yi, "not found", fontsize=5.5, color="#777777", va="center", ha="right")
                continue
            ax.plot([1, rk], [yi, yi], color=c, lw=0.8, alpha=0.6)
            ax.plot(rk, yi, marker=mk, ms=4.5, color=c, mec="white" if mk != "x" else c, mew=0.6, ls="none")
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD[m][0] for m in meths])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xscale("log")
        ax.set_xlim(0.8, xmax)
        from matplotlib.ticker import FixedLocator, NullFormatter
        tk = [t for t in (1, 3, 10, 30, 100, 300) if t <= xmax]
        ax.xaxis.set_major_locator(FixedLocator(tk))
        ax.set_xticklabels([str(t) for t in tk])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"Rank of the allosteric site (of {npk} pockets)")
        letter(ax, "b", x=-0.42)
    save(fig, f"Fig9_case_{pdb}")


def p_inflation(plt):
    d = pd.read_csv(DATA / "fig3_inflation.csv")
    order = [("family split (reference)", "Family split"), ("gene-name split", "Gene-name split"),
             ("ligand-bound (holo) inputs", "Holo inputs"), ("random pocket split", "Random split")]
    order = [o for o in order if o[0] in set(d.split)]
    st = [ci_of(d[d.split == o], "top1") for o, _ in order]
    fig, ax = plt.subplots(figsize=(W1, 1.4))
    dotci(ax, [l for _, l in order], [x[0] for x in st], [x[1] for x in st], [x[2] for x in st],
          [OI["blue"]] + [OI["grey"]] * (len(order) - 1), ref=st[0][0], xlabel="Top-1")
    ax.set_xlim(0.3, 0.85)
    save(fig, "FigS1_inflation")



def p_newset(plt):
    """Fig 11: independent complexes released from 2024 on. (a) share of sites in the top k; (b) rank of the
    site, AlloPlex vs fpocket druggability (one point per complex)."""
    d = _newset()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.35})
    ax = axs[0]
    ks = np.arange(1, 11)
    sty = {"alloplex": (OI["blue"], "o", 1.6, "-"), "druggability": (OI["grey"], "x", 0.9, (0, (3, 2))),
           "fpocket": (OI["orange"], "v", 0.9, "-")}
    for m, lab in NEW_METHODS:
        c, mk, lw, ls = sty[m]
        r = d[f"rank_{m}"].astype(float)
        ax.plot(ks, [(r <= k).mean() * 100 for k in ks], color=c, marker=mk, ms=3.2, lw=lw, ls=ls,
                mec="white" if mk != "x" else c, mew=0.5, label="Druggability" if m == "druggability" else
                ("fpocket score" if m == "fpocket" else lab))
    ax.set_xticks(ks)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Top-k pockets")
    ax.set_ylabel("Sites found (%)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
    letter(ax, "a", x=-0.2)
    ax = axs[1]
    a = d.rank_alloplex.astype(float).to_numpy()
    b = d.rank_druggability.astype(float).to_numpy()
    top = np.nanmax(np.r_[a[np.isfinite(a)], b[np.isfinite(b)]]) * 2
    a2, b2 = np.where(np.isfinite(a), a, top), np.where(np.isfinite(b), b, top)
    jit = np.random.default_rng(1).uniform(0.9, 1.1, (2, len(a2)))
    new = (d.family_status == "new family").to_numpy()
    ax.scatter(b2[~new] * jit[0][~new], a2[~new] * jit[1][~new], s=9, c=OI["blue"], alpha=0.7, lw=0,
               label="Seen family (held-out)")
    ax.scatter(b2[new] * jit[0][new], a2[new] * jit[1][new], s=9, facecolors="none", edgecolors=OI["vermilion"],
               lw=0.7, label="New family")
    ax.plot([0.8, top * 1.2], [0.8, top * 1.2], color="#BBBBBB", lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, top * 1.2)
    ax.set_ylim(0.8, top * 1.2)
    from matplotlib.ticker import FixedLocator, NullFormatter
    tk = [t for t in (1, 3, 10, 30, 100, 300, 1000) if t <= top * 1.2]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(tk))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(t) for t in tk])
    ax.set_yticklabels([str(t) for t in tk])
    print(f"  Fig 11b: {int(((a == 1) & (b == 1)).sum())} complexes ranked 1 by both (overlapping at the corner)")
    ax.set_xlabel("Rank by druggability")
    ax.set_ylabel("Rank by AlloPlex")
    ax.legend(loc="upper left", handletextpad=0.2)
    ax.text(0.98, 0.04, "AlloPlex better \u2193", transform=ax.transAxes, ha="right", fontsize=5.6, color="#555555")
    letter(ax, "b", x=-0.25)
    save(fig, "Fig11_new_complexes")


KIND_ORDER = [("protein-peptide", "Protein-peptide"), ("hetero-complex", "Hetero-complex"),
              ("pseudo-hetero", "Pseudo-hetero"), ("homo-oligomer", "Homo-oligomer"),
              ("homo (assembly only)", "Homo (assembly)"), ("monomer", "Monomer")]


def tools_with_kind():
    d = pd.read_csv(DATA / "fig5_tools.csv", dtype={"site_id": str})
    k = pd.read_csv(DATA / "fig1_sites.csv", dtype={"site_id": str}, usecols=["site_id", "complex_kind"])
    return d.merge(k.drop_duplicates("site_id"), on="site_id", how="left")


def p_kinds(plt):
    """Fig 10: (a) AlloPlex Top-1 per complex type with 95 % CI; (b) Top-1 of every method per type (heatmap)."""
    from matplotlib.colors import LinearSegmentedColormap
    d = tools_with_kind()
    d = d[d.evaluation == "matched"]
    meths = [m for m in METHOD if m in set(d.method)]
    kinds = [(k, l) for k, l in KIND_ORDER if k in set(d.complex_kind)]
    ours_all = d[d.method == OURS].top1.astype(float).mean()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.08})
    # a: AlloPlex per type
    ax = axs[0]
    st = [ci_of(d[(d.complex_kind == k) & (d.method == OURS)], "top1") for k, _ in kinds]
    ns = [len(d[(d.complex_kind == k) & (d.method == OURS)]) for k, _ in kinds]
    dotci(ax, [f"{l} ({n})" for (_, l), n in zip(kinds, ns)], [x[0] for x in st], [x[1] for x in st],
          [x[2] for x in st], [OI["blue"]] * len(kinds), ref=ours_all, xlabel="AlloPlex Top-1")
    ax.set_xlim(0, 0.9)
    letter(ax, "a", x=-0.62)
    # b: heatmap, rows = complex types (same order), columns = methods
    ax = axs[1]
    M = np.array([[d[(d.complex_kind == k) & (d.method == m)].top1.astype(float).mean() for m in meths]
                  for k, _ in kinds])
    cmap = LinearSegmentedColormap.from_list("w2b", ["#FFFFFF", "#9ECAE1", OI["blue"]])
    ax.imshow(M, cmap=cmap, vmin=0.15, vmax=0.7, aspect="auto")
    for i in range(M.shape[0]):
        best = np.nanargmax(M[i])
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.52 else "#222222",
                        fontweight="bold" if j == best else "normal")
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([METHOD[m][0] for m in meths], rotation=35, ha="right")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    letter(ax, "b", x=-0.04, y=1.02)
    save(fig, "Fig10_complex_types")


def table_kinds():
    d = tools_with_kind()
    meths = [m for m in METHOD if m in set(d.method)]
    rows = []
    for k, lab in KIND_ORDER + [("__multi", "All multi-chain"), ("__all", "All")]:
        if k == "__all":
            sub = d
        elif k == "__multi":
            sub = d[d.complex_kind.isin([x for x, _ in KIND_ORDER if x != "monomer"])]
        else:
            sub = d[d.complex_kind == k]
        if not len(sub):
            continue
        ours = sub[(sub.method == OURS) & (sub.evaluation == "matched")]
        r = {"complex type": lab, "sites": len(ours), "families": ours.family.nunique()}
        for m in meths:
            t = sub[(sub.method == m) & (sub.evaluation == "matched")]
            r[f"{METHOD[m][0]} Top-1"] = mean_ci(t, "top1")
        r["AlloPlex Top-3"] = mean_ci(ours, "top3")
        r["AlloPlex native Top-1"] = mean_ci(sub[(sub.method == OURS) & (sub.evaluation == "native")], "top1")
        best = None
        for m in meths:
            if m == OURS:
                continue
            t = sub[(sub.method == m) & (sub.evaluation == "matched")]
            if len(t) and (best is None or t.top1.astype(float).mean() > best[1]):
                best = (m, t.top1.astype(float).mean(), t)
        if best is not None:
            r["AlloPlex minus best other method (Top-1)"] = f"{METHOD[best[0]][0]}: " + paired_ci(ours, best[2], "top1")
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================= TABLES
TAB = BASE / "tables"


def fmt_ci(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if np.isfinite(lo) else f"{m:.{d}f}"


def mean_ci(t, col):
    if not len(t) or t[col].notna().sum() == 0:
        return ""
    v = t[col].astype(float)
    lo, hi = boot_ci(t.family, v)
    return fmt_ci(v.mean(), lo, hi)


def paired_ci(a, b, col):
    """mean difference a - b on common sites with a family-bootstrap 95 % CI."""
    a = a.drop_duplicates("site_id").set_index("site_id")
    b = b.drop_duplicates("site_id").set_index("site_id")
    c = a.index.intersection(b.index)
    if len(c) < 10:
        return ""
    d = a.loc[c, col].astype(float) - b.loc[c, col].astype(float)
    lo, hi = boot_ci(a.loc[c, "family"], d)
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def table1():
    s = pd.read_csv(DATA / "fig1_sites.csv")
    order = ["protein-peptide", "hetero-complex", "pseudo-hetero", "homo-oligomer", "homo (assembly only)", "monomer"]
    kinds = [k for k in order if k in set(s.complex_kind)] + sorted(set(s.complex_kind) - set(order))
    rows = []
    for k in kinds + ["all"]:
        t = s if k == "all" else s[s.complex_kind == k]
        r = {"complex type": k, "sites": len(t), "structures": t.pdb_id.nunique(), "families": t.family.nunique(),
             "PPI-annotated sites": int((t.position_class == "PPI interface").sum())}
        for src in sorted(s.source.unique()):
            r[f"from {src}"] = int((t.source == src).sum())
        if "n_modulator_free" in t.columns:
            r["with modulator-free partner"] = int((pd.to_numeric(t.n_modulator_free, errors="coerce") > 0).sum())
        rows.append(r)
    multi = s[s.complex_kind.isin(["homo-oligomer", "homo (assembly only)", "hetero-complex", "pseudo-hetero",
                                   "protein-peptide"])]
    key = multi[multi.position_class != "PPI interface"]
    rows.append({"complex type": "KEY SET (multi-chain, not PPI-annotated)", "sites": len(key),
                 "structures": key.pdb_id.nunique(), "families": key.family.nunique()})
    out = pd.DataFrame(rows)
    for c in out.columns[1:]:
        out[c] = [("" if pd.isna(v) else str(int(v))) for v in out[c]]
    return out


def table2():
    rows = [{"analysis": l.replace("\n", " "), "estimate [95 % CI]": fmt_ci(m, lo, hi), "role": g}
            for l, m, lo, hi, g in REPORTED]
    rows += [{"analysis": l, "estimate [95 % CI]": fmt_ci(m, lo, hi), "role": "control" if "patch" in l else "primary"}
             for l, m, lo, hi in REPORTED_EXTRA]
    rows.append({"analysis": "Random-patch control: overlap AUC real / random",
                 "estimate [95 % CI]": f"{RANDOM_PATCH['overlap'][0]:.3f} / {RANDOM_PATCH['overlap'][1]:.3f}",
                 "role": "control"})
    return pd.DataFrame(rows)


def table3(want=None):
    d = pd.read_csv(DATA / "fig4_models.csv")
    if "set" not in d.columns:
        d["set"] = "primary"
    ap = pd.read_csv(DATA / "tab_auprc.csv") if (DATA / "tab_auprc.csv").exists() else pd.DataFrame(columns=["method", "set", "auprc"])
    want = want or ["MLP B", "MLP B+I", "MLP B+I-d", "MLP I", "GNN B+I+fp", "GNN B", "GNN B+I", "LightGBM B",
                    "LightGBM B+I", "LightGBM B+I-d", "LightGBM I", "fpocket druggability", "pocket size",
                    "pocket volume", "fpocket score"]
    rows = []
    for w in [x for x in want if x in set(d.method)]:
        r = {"model / features": "AlloPlex (MLP B)" if w == "MLP B" else w}
        for en in ("primary", "partner", "stripped"):
            t = d[(d.method == w) & (d.set == en)]
            if not len(t):
                continue
            r[f"{en} Top-1"] = mean_ci(t, "top1")
            if en == "primary":
                r["primary Top-3"] = mean_ci(t, "top3")
                r["primary site AUROC"] = mean_ci(t, "site_auc")
            a = ap[(ap.method == w) & (ap.set == en)].auprc
            if en != "primary" and len(a) and pd.notna(a.iloc[0]):
                r[f"{en} AUPRC"] = f"{a.iloc[0]:.3f}"
        ref = d[(d.method == "MLP B") & (d.set == "primary")]
        cur = d[(d.method == w) & (d.set == "primary")]
        r["Top-1 minus AlloPlex (primary)"] = "" if w == "MLP B" else paired_ci(cur, ref, "top1")
        rows.append(r)
    return pd.DataFrame(rows)


def table4():
    d = pd.read_csv(DATA / "fig5_tools.csv")
    meths = tool_order(d)
    rows = []
    for w in meths:
        r = {"method": nice(w)}
        for ev in ("matched", "native"):
            t = d[(d.method == w) & (d.evaluation == ev)]
            r[f"{ev} sites"] = len(t)
            r[f"{ev} Top-1"] = mean_ci(t, "top1")
            r[f"{ev} Top-3"] = mean_ci(t, "top3")
            if ev == "matched":
                r["matched site AUROC"] = mean_ci(t, "site_auc")
            if w != OURS:
                o = d[(d.method == OURS) & (d.evaluation == ev)]
                r[f"AlloPlex minus tool, {ev} Top-1"] = paired_ci(o, t, "top1")
        rows.append(r)
    return pd.DataFrame(rows)


def table5():
    d = pd.read_csv(DATA / "fig5_tools.csv")
    rows = []
    for w in tool_order(d):
        r = {"method": nice(w)}
        for lab, col in (("all sites", None), ("new proteins", "new_protein"), ("new families", "new_family")):
            for ev in ("matched", "native"):
                t = d[(d.method == w) & (d.evaluation == ev)]
                if col:
                    t = t[t[col].astype(str) == "True"]
                r[f"{lab}: {ev} Top-1"] = mean_ci(t, "top1")
                if ev == "matched":
                    r[f"{lab}: n"] = len(t)
            if w != OURS and col:
                o = d[(d.method == OURS) & (d.evaluation == "matched") & (d[col].astype(str) == "True")]
                t = d[(d.method == w) & (d.evaluation == "matched") & (d[col].astype(str) == "True")]
                r[f"{lab}: AlloPlex minus tool (matched)"] = paired_ci(o, t, "top1")
        rows.append(r)
    return pd.DataFrame(rows)


def tableS1():
    d = pd.read_csv(DATA / "fig3_inflation.csv")
    ref = d[d.split == "family split (reference)"]
    rows = []
    for sp in ["family split (reference)", "gene-name split", "ligand-bound (holo) inputs", "random pocket split"]:
        t = d[d.split == sp]
        if len(t):
            rows.append({"evaluation": sp, "sites": len(t), "Top-1": mean_ci(t, "top1"),
                         "site AUROC": mean_ci(t, "site_auc"),
                         "Top-1 minus family split": "" if sp.startswith("family") else paired_ci(t, ref, "top1")})
    return pd.DataFrame(rows)


FEAT_NAMES = {"B": "ESM-2 + fpocket (B)", "B+I": "B + interface (4)", "B+I-d": "B + interface without distance",
              "I": "interface + pocket size only", "E": "ESM-2 pocket mean only (1,280)",
              "F": "11 fpocket descriptors only", "B+I+fp": "B + interface + descriptors at read-out"}


def table_ablation():
    t = table3(["MLP B", "MLP E", "MLP F", "MLP-BCE B", "MLP B+I", "MLP B+I-d", "MLP I", "fpocket druggability",
                "pocket size"])
    t.insert(1, "features", [FEAT_NAMES["B"] if m.startswith("AlloPlex") else
                             "ESM-2 + fpocket (B), plain BCE loss instead of focal" if m == "MLP-BCE B" else
                             FEAT_NAMES.get(m.split()[-1], "") if m.startswith("MLP") else "single fpocket descriptor"
                             for m in t["model / features"]])
    return t


def table_models():
    return table3(["MLP B", "GNN B+I+fp", "GNN B", "GNN B+I", "LightGBM B+I", "LightGBM B", "LightGBM B+I-d"])


def table_split():
    return pd.read_csv(DATA / "tab_split.csv")


def table_hyper():
    return pd.DataFrame([
        ("Pocket detection", "fpocket (default parameters) on the ligand-free assembly"),
        ("Protein language model", "ESM-2 esm2_t33_650M_UR50D, layer 33, windows of 1022 (step 822)"),
        ("Pocket features", "mean ESM-2 vector of pocket residues (1280) + 11 fpocket descriptors = 1291"),
        ("Architecture", "MLP 1291-512-256-64-1, BatchNorm, ReLU, dropout 0.3 / 0.3 / 0.15"),
        ("Loss", "focal loss (gamma 2, alpha 0.75)"),
        ("Optimiser", "AdamW, learning rate 3e-4, weight decay 1e-4, batch 256"),
        ("Schedule / stopping", "ReduceLROnPlateau (patience 10, factor 0.5); early stopping on validation AUPRC (patience 40, max 300 epochs)"),
        ("Validation", "20 % of training families (cross-validation); one family fold per member (final model)"),
        ("Ensemble", "10 seeds per fold (cross-validation); 5 folds x 3 seeds = 15 networks (released model)"),
        ("Cross-validation", "5 folds x 3 repeats, whole families (>= 30 % identity, MMseqs2) per fold"),
    ], columns=["setting", "value"])


def tableS2():
    """how each external tool was run (from the run logs; see the session summary)."""
    return pd.DataFrame([
        {"tool": "DeepAllo", "reference": "Bioinformatics 2025", "input": "one chain (exported primary chain)",
         "pockets": "fpocket (own run)", "score used": "AutoGluon probability", "ran on": "1092 / 1108 chains",
         "notes": "sequence and residue map from the chain's atoms (no SEQRES / RCSB download); 8 chains <= 10 residues and "
                  "8 with < 3 pockets refused by DeepAllo; package versions adjusted (see Methods)"},
        {"tool": "APOP", "reference": "Bioinformatics 2023", "input": "whole ligand-free assembly",
         "pockets": "fpocket (own run)", "score used": "APOP score", "ran on": "997 / 1076 assemblies (<= 2000 residues)",
         "notes": "assemblies > 2000 residues skipped (run time); separate python 3.9 environment"},
        {"tool": "PASSer", "reference": "NAR 2023 (web server)", "input": "whole ligand-free assembly (API upload)",
         "pockets": "fpocket (server)", "score used": "ensemble probability, all pockets", "ran on": "assemblies <= 2000 residues (sites: Table 5)",
         "notes": "https://passer.smu.edu/api; one request at a time"},
        {"tool": "P2Rank 2.5", "reference": "J Cheminform 2018", "input": "whole ligand-free assembly",
         "pockets": "P2Rank (own)", "score used": "pocket probability", "ran on": "1076 / 1076 structures",
         "notes": "generic ligand-binding-site predictor (not allosteric-specific); Java 17"},
        {"tool": "fpocket druggability", "reference": "baseline", "input": "whole ligand-free assembly",
         "pockets": "fpocket", "score used": "druggability score", "ran on": "all", "notes": "same pockets as AlloPlex"},
    ])


# ----------------------------------------------------------------------------- step 10 tables
NEW_METHODS = [("alloplex", "AlloPlex"), ("druggability", "fpocket druggability"), ("fpocket", "fpocket score")]


def _newset():
    d = pd.read_csv(DATA / "s10e_newset.csv")
    d["family"] = d.nearest_training_family.fillna("").astype(str)
    d.loc[d.family == "", "family"] = d.pdb_id  # new families: each entry its own group
    for m, _ in NEW_METHODS:
        r = d[f"rank_{m}"].astype(float)
        d[f"{m}_top1"], d[f"{m}_top3"] = (r <= 1).astype(float), (r <= 3).astype(float)
    return d


def table_newset():
    d = _newset()
    new, seen = d[d.family_status == "new family"], d[d.family_status != "new family"]
    rows = []
    for m, lab in NEW_METHODS:
        r = {"method": lab}
        for name, t in ((f"all ({len(d)})", d), (f"new families ({len(new)})", new), (f"seen families ({len(seen)})", seen)):
            r[f"{name} Top-1"] = mean_ci(t.rename(columns={f"{m}_top1": "v"}), "v")
            r[f"{name} Top-3"] = f"{t[f'{m}_top3'].mean():.3f}"
        if m != "alloplex":
            dd = d["alloplex_top1"] - d[f"{m}_top1"]
            lo, hi = boot_ci(d.family, dd)
            r["AlloPlex minus method, Top-1 (all)"] = f"{dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            r["AlloPlex minus method, Top-1 (all)"] = ""
        rows.append(r)
    return pd.DataFrame(rows)


def table_errors():
    e = pd.read_csv(DATA / "s10a_error_analysis.csv")
    return pd.DataFrame({"site property": e.property, "category": e.category, "sites": e.sites,
                         "families": e.families, "AlloPlex Top-1": [fmt_ci(m, lo, hi) for m, lo, hi in zip(e.top1, e.lo, e.hi)],
                         "AlloPlex Top-3": e.top3.map(lambda x: f"{x:.3f}"),
                         "site found by fpocket": e.detected.map(lambda x: f"{x:.0%}")})


def table_threshold():
    t = pd.read_csv(DATA / "s10a_threshold.csv")
    return pd.DataFrame({"top-pocket score at least": t.threshold.map(lambda x: f"{x:.1f}"),
                         "structures": t.structures, "share of structures": t.share_of_structures.map(lambda x: f"{x:.0%}"),
                         "top pocket is allosteric": [f"{m:.1%} [{lo:.1%}, {hi:.1%}]" for m, lo, hi in zip(t.hit_rate, t.lo, t.hi)]})


def table_runtime():
    t = pd.read_csv(DATA / "s10c_runtime.csv")
    f = lambda x: "" if pd.isna(x) else f"{x:.1f}"
    return pd.DataFrame({"tool": t.tool, "structures": t.structures, "median s / structure": t.median_s.map(f),
                         "mean s / structure": t.mean_s.map(f), "max s": t.max_s.map(f), "note": t.note})


TABLES = [("Table1_datasets", "Data sets used for training and evaluation", table_split),
          ("Table2_hyperparameters", "AlloPlex model and training settings", table_hyper),
          ("Table3_ablation", "Ablation study: input features and loss function (MLP, family-split cross-validation, mean of 3 repeats)", table_ablation),
          ("Table4_models", "Model architectures (family-split cross-validation, mean of 3 repeats)", table_models),
          ("Table5_tools", "Comparison with published tools (PRIMARY sites)", table4),
          ("Table6_novelty", "Proteins and families absent from ASD 2019", table5),
          ("TableS1_dataset_composition", "Dataset composition by complex type", table1),
          ("TableS2_interface_test", "Pre-registered interface test and controls", table2),
          ("TableS3_inflation", "Performance inflation by shortcut evaluations (MLP B, repeat 0)", tableS1),
          ("TableS4_tool_setup", "How the external tools were run", tableS2),
          ("TableS5_complex_types", "Performance by complex type (PRIMARY sites, matched evaluation)", table_kinds),
          ("Table7_new_complexes", "Independent test: complexes released from 2024 on (one allosteric site each)", table_newset),
          ("TableS6_error_analysis", "AlloPlex Top-1 by site property (PRIMARY sites)", table_errors),
          ("TableS7_score_guide", "How often the top-ranked pocket is the allosteric site, by its AlloPlex score", table_threshold),
          ("TableS8_runtime", "Runtime per structure", table_runtime)]


def best_cells(t, rowwise=False):
    """(row, column) of the best value in each metric column (higher is better); rowwise=True: best
    method per row among the '<method> Top-1' columns (tables with methods as columns)."""
    out = set()
    if rowwise:
        cols = [j for j, c in enumerate(t.columns) if c.endswith(" Top-1") and "native" not in c]
        for i in range(len(t)):
            vals = []
            for j in cols:
                try:
                    vals.append(float(str(t.iat[i, j]).split()[0]))
                except (ValueError, IndexError):
                    vals.append(np.nan)
            vals = np.array(vals)
            if np.isfinite(vals).sum() >= 2:
                out |= {(i, cols[k]) for k in np.where(vals == np.nanmax(vals))[0]}
        return out
    for j, c in enumerate(t.columns):
        if not any(k in c for k in ("Top-1", "Top-3", "AUROC", "AUPRC")) or "minus" in c:
            continue
        vals = []
        for v in t[c]:
            try:
                vals.append(float(str(v).split()[0]))
            except (ValueError, IndexError):
                vals.append(np.nan)
        vals = np.array(vals)
        if np.isfinite(vals).sum() >= 2:
            out |= {(i, j) for i in np.where(vals == np.nanmax(vals))[0]}
    return out


def make_tables():
    TAB.mkdir(parents=True, exist_ok=True)
    md, made = [], {}
    for name, title, fn in TABLES:
        try:
            t = fn()
        except FileNotFoundError as e:
            print(f"  {name}: skipped (data not produced yet: {Path(e.filename).name if e.filename else e})")
            continue
        except Exception:
            print(f"  {name} FAILED:\n" + traceback.format_exc(limit=3))
            continue
        t.to_csv(TAB / f"{name}.csv", index=False)
        md += [f"## {name.replace('_', ' ')}: {title}", "", "| " + " | ".join(map(str, t.columns)) + " |",
               "|" + "---|" * len(t.columns)]
        best = best_cells(t, rowwise=name.startswith("TableS5"))
        made[name] = (title, t, best)

        def cell(i, j, v):
            v = "" if (isinstance(v, float) and np.isnan(v)) else str(v)
            return f"**{v}**" if (i, j) in best and v else v
        md += ["| " + " | ".join(cell(i, j, v) for j, v in enumerate(r)) + " |" for i, r in enumerate(t.itertuples(index=False))]
        note = ("Values: mean [95 % family-bootstrap CI]; differences are paired, on the sites both methods were "
                "scored on." if name.startswith(("Table3", "Table4", "Table5", "Table6", "Table7", "TableS3", "TableS5", "TableS6")) else
                "Estimates with 95 % CI as reported by the pre-registered analysis." if name.startswith("TableS2") else "")
        md += ["", (note + (" Best method per row in bold." if name.startswith("TableS5") else
                            " Best value per column in bold.")) if best else note, ""]
        made[name] = made[name] + (note,)
        print(f"  saved tables/{name}.csv")
    (TAB / "all_tables.md").write_text("\n".join(md))
    try:
        with pd.ExcelWriter(TAB / "AlloPlex_tables.xlsx") as xw:
            for name, (title, t, best, note) in made.items():
                t.to_excel(xw, sheet_name=name[:31], index=False, startrow=1)
                ws = xw.sheets[name[:31]]
                if hasattr(ws, "cell"):  # openpyxl
                    from openpyxl.styles import Font
                    ws.cell(row=1, column=1, value=title).font = Font(bold=True)
                    for i, j in best:
                        ws.cell(row=i + 3, column=j + 1).font = Font(bold=True)
                else:  # xlsxwriter
                    ws.write(0, 0, title)
        print("  saved tables/AlloPlex_tables.xlsx and tables/all_tables.md")
    except Exception as e:
        print(f"  (Excel file not written: {e}); tables/all_tables.md saved")
    write_docx(made)


def write_docx(made):
    """all tables in one Word file (three-line style: rules above / below the header and at the end)."""
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Pt
    except ImportError:
        print("  (python-docx not installed: pip install python-docx  -> for tables/AlloPlex_tables.docx)")
        return

    def rule(cell, edge):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            tcPr.append(b)
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "6")
        e.set(qn("w:color"), "000000")
        b.append(e)
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(8)
    for k, (name, (title, t, best, note)) in enumerate(made.items(), 1):
        label = name.split("_")[0].replace("Table", "Table ")
        p = doc.add_paragraph()
        r = p.add_run(f"{label}. ")
        r.bold = True
        p.add_run(title + ".")
        tab = doc.add_table(rows=len(t) + 1, cols=len(t.columns))
        for j, c in enumerate(t.columns):
            cl = tab.cell(0, j)
            cl.text = str(c)
            cl.paragraphs[0].runs[0].bold = True
            rule(cl, "top")
            rule(cl, "bottom")
        for i, row in enumerate(t.itertuples(index=False)):
            for j, v in enumerate(row):
                cl = tab.cell(i + 1, j)
                cl.text = "" if (isinstance(v, float) and np.isnan(v)) else str(v)
                if (i, j) in best and cl.paragraphs[0].runs:
                    cl.paragraphs[0].runs[0].bold = True
                if i == len(t) - 1:
                    rule(cl, "bottom")
        for rw in tab.rows:
            for cl in rw.cells:
                for para in cl.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(7.5)
        if note:
            q = doc.add_paragraph(note + (" Best value per column in bold." if best else ""))
            q.runs[0].italic = True
            q.runs[0].font.size = Pt(7)
        doc.add_paragraph()
    doc.save(TAB / "AlloPlex_tables.docx")
    print("  saved tables/AlloPlex_tables.docx")


# ============================================================================= main
def list_cases(n=15):
    """candidate case studies, chosen by rule (not by eye): sites in families absent from ASD 2019 where
    AlloPlex ranks the allosteric pocket first; shows the other methods' ranks too. Also the clearest failures."""
    r = pd.read_csv(DATA / "fig6_ranks.csv", dtype={"site_id": str})
    nov = pd.read_csv(DATA / "fig5_tools.csv", usecols=["site_id", "new_protein", "new_family"],
                      dtype={"site_id": str}).drop_duplicates("site_id")
    w = r.pivot_table(index=["site_id", "pdb_id", "set"], columns="method", values="rank", aggfunc="first").reset_index()
    w = w.merge(nov, on="site_id", how="left")
    np_ = r.drop_duplicates("site_id").set_index("site_id").n_pockets
    w["n_pockets"] = w.site_id.map(np_)
    cols = [c for c in [OURS, "fpocket druggability", "DeepAllo", "APOP", "PASSer", "P2Rank"] if c in w.columns]
    good = w[(w[OURS] == 1) & (w.new_family.astype(str) == "True")].sort_values("n_pockets", ascending=False)
    nf = w.new_family.astype(str) == "True"
    bad = w[nf & np.isfinite(w[OURS]) & (w[OURS] > 10)].sort_values(OURS, ascending=False)
    n_undet = int((nf & ~np.isfinite(w[OURS])).sum())
    pd.set_option("display.width", 200)
    print(f"SUCCESS candidates (new family, AlloPlex rank 1; most pockets first) - {len(good)} sites:")
    print(good[["site_id", "pdb_id", "set", "n_pockets"] + cols].head(n).to_string(index=False))
    print(f"\nFAILURE examples (new family, pocket found by fpocket but AlloPlex rank > 10) - {len(bad)} sites "
          f"(+ {n_undet} sites with no matching fpocket pocket at all, a miss for every method):")
    print(bad[["site_id", "pdb_id", "set", "n_pockets"] + cols].head(5).to_string(index=False))
    print("\nThen: python scripts/step9x_figures.py --only 9 --case <PDB>   (pdb_id column; stripped or partner)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--only", help="comma list of figure numbers, e.g. 5,6")
    ap.add_argument("--case", default="5OLK")
    ap.add_argument("--tables-only", action="store_true", help="only (re)make the tables from figures/data")
    ap.add_argument("--list-cases", action="store_true", help="suggest case-study structures (after a full run)")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    DATA.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    if args.tables_only:
        make_tables()
        return
    if args.list_cases:
        list_cases()
        return
    jobs = [("1", None, p_architecture), ("S2", None, p_workflow), ("2", collect_dataset, p_dataset), ("3", collect_tsne, p_tsne),
            ("4", collect_interface, p_interface), ("5", collect_models, p_models), ("6", collect_ranks, p_ranking),
            ("7", collect_tools, p_tools), ("8", None, p_novelty),
            ("9", lambda: collect_case(args.case), lambda p: p_case(p, args.case)),
            ("10", None, p_kinds), ("11", None, p_newset), ("S1", collect_inflation, p_inflation), ("T", collect_split, lambda p: None)]
    if not args.plot_only:
        for n, col, _ in jobs:
            if col is None or (only and n not in only):
                continue
            print(f"collect Fig{n} ...", flush=True)
            try:
                col()
            except Exception:
                print(f"  Fig{n} collect FAILED:\n" + traceback.format_exc(limit=3))
    plt = style()
    for n, _, plot in jobs:
        if only and n not in only:
            continue
        try:
            plot(plt)
        except Exception:
            print(f"  Fig{n} plot FAILED:\n" + traceback.format_exc(limit=3))
        plt.close("all")
    print("tables ...")
    make_tables()
    print(f"done; figures in {FIG}, tables in {TAB}")


if __name__ == "__main__":
    main()
