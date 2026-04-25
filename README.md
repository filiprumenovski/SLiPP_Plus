# SLiPP++

**Day 1 reformulation of SLiPP (Chou et al. 2024) as a 10-class softmax classifier.**

Reimplements the binary lipid-vs-rest benchmark from the **published SLiPP** work (Chou et al.): read the **preprint PDF** at [doi.org/10.1101/2024.01.26.577452](https://doi.org/10.1101/2024.01.26.577452) (bioRxiv → PDF) and the lab code + `training_pockets.csv` at [github.com/dassamalab/SLiPP_2024](https://github.com/dassamalab/SLiPP_2024). SLiPP++ changes the head to **10-class** softmax over `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE}` while keeping the same curated table and binary-collapse definitions for comparison to **Table 1**.

See `CONTEXT.md` / `RESEARCH_LOG.md` for scope and `reference/README.md` for how to lay out the upstream CSV next to this repo.

## Quickstart

```bash
# One-time
uv sync --extra dev

# End-to-end (ingest -> train -> eval -> figures)
make all
```

Wall-clock budget: under 25 minutes on a 16-core machine. No network required; all training data ships in `reference/SLiPP_2024-main/training_pockets.csv` and the supporting-file xlsx tables in `data/raw/supplementary/`.

## CAVER reliability checks

`slipp_plus.tunnel_features` now enforces fail-fast guards so bad structure roots
do not produce deceptively "successful" but unusable outputs:

- Preflight validates expected structure artifacts (`<id>.pdb`, `<id>_out/pockets`).
- Builds abort if missing-structure fraction exceeds configured limits.
- Output quality gates require strong coverage for
  `tunnel_pocket_context_present` and `tunnel_caver_profile_present`.

Tune with:
- `--max-missing-structure-frac`
- `--min-context-present-frac`
- `--min-profile-present-frac`

Persist reusable CAVER raw analysis outputs for downstream feature extraction with:
- `--analysis-output-root <dir>` (writes one directory per structure containing CAVER `analysis/*.csv`)
- `--analysis-manifest <path.csv>` (writes pocket-to-analysis mapping for joins)

## What runs

| target      | description                                                      |
|-------------|------------------------------------------------------------------|
| `make ingest`  | Parses training CSV + SF2/SF3 xlsx, runs Pandera Rule 1 gate. |
| `make train`   | 25 stratified shuffle splits × {RF, XGB, LGBM} 10-class softmax. |
| `make eval`    | Per-class + macro-F1 + binary-collapse vs paper Table 1 + holdouts. |
| `make figures` | Confusion matrix, per-class ROC, PCA colored by prediction, comparison bars. |
| `make all`     | Chains the four above. |
| `make scratch` | Day 7+: reproduce the data pipeline from raw PDBs via fpocket. |

## Configuration

`configs/day1.yaml` controls seeds, number of iterations, feature set, and models.

- `feature_set: v14` — paper canonical 17 descriptors.
- `feature_set: v14+v22` — adds `surf_pol_vdw22` + `surf_apol_vdw22` (already in CSV, 2.2 Å probe).
- `feature_set: v14+aa` — adds 20 per-pocket amino-acid counts.
- `feature_set: v14+v22+aa` — both.

All four ablations run with the same 25-iter harness and produce comparable metrics.

## Ground truth (published Table 1 — preprint / JCIM)

| dataset                 | F1    | AUROC |
|-------------------------|-------|-------|
| Test (8,380 pockets)    | 0.869 | 0.970 |
| Apo PDB holdout (131)   | 0.726 | 0.828 |
| AlphaFold holdout (177) | 0.643 | 0.851 |

`reports/metrics_table.md` reports ours side-by-side after `make eval`.

## Layout

```
slipp_plus/
├── configs/day1.yaml          # seeds, hparams, feature-set toggle
├── data/raw/supplementary/    # ci5c01076_si_{002..009}.xlsx (ships with repo)
├── reference/SLiPP_2024-main/ # authors' original code + training_pockets.csv
├── processed/                 # parquet outputs (gitignored)
├── models/                    # iter-0 joblib dumps (gitignored)
├── reports/                   # metrics_table.md + figures
├── src/slipp_plus/            # the implementation
└── tests/                     # pandera schema + count parity tests
```
