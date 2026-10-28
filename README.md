# AlloPlex

**AlloPlex: an open-source tool for allosteric site prediction across protein–protein interaction complexes,
featuring assembly-level pocket detection and a leakage-controlled benchmark.**

AlloPlex ranks the pockets of a whole protein assembly by how likely each is to be an allosteric site.
It detects pockets with fpocket on the ligand-free assembly, describes each pocket by the mean ESM-2
embedding of its residues and 11 fpocket descriptors, and scores it with an ensemble of 15 multilayer
perceptrons.

| Benchmark (protein-family-held-out cross-validation, 2,349 sites) | Top-1 | Top-3 |
|---|---|---|
| AlloPlex | 0.422 | 0.583 |
| Independent complexes released from 2024 onward (105) | 0.467 | 0.562 |

On protein families absent from the training data of earlier tools, AlloPlex reached a Top-1 of 0.369
(DeepAllo 0.176). AlloPlex is intended for **experimental structures**: on AlphaFold 3 models its Top-1
accuracy roughly halves. Scores rank pockets within one structure and are not calibrated probabilities;
see the score guide in the paper (Table S13).

## Install

```bash
conda env create -f environment.yml
conda activate alloplex
fpocket -h          # fpocket 4.0 must be on PATH (conda-forge or https://github.com/Discngine/fpocket)
```

The ESM-2 weights (esm2_t33_650M_UR50D, MIT licence, Meta) are downloaded automatically by `fair-esm`
on first use (~2.5 GB). A GPU is optional.

## Predict

```bash
python alloplex/alloplex_predict.py examples/1XJE_assembly_ligand_free.pdb --out results
python alloplex/alloplex_predict.py my_complex-assembly1.cif.gz --top 5 --cpu
```

Give the biological assembly you are interested in (for example the RCSB "assembly 1" file); ligands,
waters, ions and nucleic acids are removed automatically. Output per structure in `results/<name>/`:

- `alloplex_pockets.csv` — all pockets ranked by `alloplex_score`, with residues and fpocket descriptors
- `clean.pdb` — the ligand-free structure that was analysed
- `alloplex_view.pml` — PyMOL script colouring the top pockets

Runtime: median 2.4 s per assembly on one GPU (NVIDIA RTX A6000), 11.8 s on eight CPU threads, plus about
13 s once per session to load the models.

## Repository contents

| Folder | Contents |
|---|---|
| `alloplex/` | the predictor |
| `models/alloplex_v1/` | the 15 trained networks (5 family folds × 3 random seeds) |
| `pipeline/` | every script used to build the dataset, train, benchmark and make the figures (`step*.py`) |
| `benchmark/` | family folds, per-site labels and results of every analysis in the paper |
| `examples/` | a ligand-free example assembly (PDB 1XJE) |

Large files (ligand-free and modulator-free structures, pocket feature tables, AlphaFold 3 models) are
archived on Zenodo: DOI_PLACEHOLDER.

## Reproducing the paper

The scripts in `pipeline/` run in numerical order (`step1x` … `step13d`) from a project folder laid out as
in the paper's Supplementary Methods; each script's header gives its command line, inputs and outputs.
Benchmark evaluation uses whole protein families (MMseqs2, ≥ 30 % identity) as cross-validation folds
(`benchmark/folds.csv`); test families are never used for any choice.

## Data sources and licences

- Code: MIT licence (`LICENSE`). Derived data in `benchmark/` and on Zenodo: CC BY 4.0.
- Allosteric site annotations come from the Allosteric Database (ASD, release 2023) and AlloBench; please
  cite them (below) when you use the benchmark. Structures are derived from the RCSB Protein Data Bank.
- ESM-2 code and weights: MIT licence (Meta). fpocket: MIT licence.

## Citation

If you use AlloPlex, please cite the paper (CITATION.cff; reference will be added on publication) and the
data sources:
He J. et al. (2024) ASD2023. *Nucleic Acids Res* 52:D376–D383 · Maity D., Qiao B. (2025) AlloBench.
*ACS Omega* 10:17973–17982.
