# Methods

This document summarizes the current SLiPP++ pipeline in code-facing terms. It is meant to be checked against the config files and source modules before submission because this repository contains active experiment branches and historical reports.

## Data Sources

The Day 1 training table is `reference/SLiPP_2024-main/training_pockets.csv`, restored from the public SLiPP 2024 reference repository. It contains 15,219 pocket rows. The validation workbooks are:

- apo-PDB holdout: `data/raw/supplementary/ci5c01076_si_003.xlsx`
- AlphaFold holdout: `data/raw/supplementary/ci5c01076_si_004.xlsx`

The file checksums, class counts, and holdout ID counts are recorded in `DATASHEET.md`.

## Class Definition

SLiPP++ predicts a 10-class label over:

```text
ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE
```

For comparison to Chou et al. Table 1, multiclass predictions are collapsed to a binary lipid-vs-nonlipid label using the canonical lipid codes in `src/slipp_plus/constants.py`. The binary metric implementation is `src/slipp_plus/evaluate.py::binary_collapse`.

## Feature Sets

The paper-aligned Day 1 feature set is `v14`, implemented as the 17-column `SELECTED_17` order in `src/slipp_plus/constants.py`. The order is:

```text
pock_vol, nb_AS, mean_as_ray, mean_as_solv_acc, apol_as_prop,
mean_loc_hyd_dens, hydrophobicity_score, volume_score, polarity_score,
charge_score, flex, prop_polar_atm, as_density, as_max_dst,
surf_pol_vdw14, surf_apol_vdw14, surf_vdw14
```

Additional feature families are configured by `feature_set`:

| Feature set | Definition in code |
|---|---|
| `v14+v22` | `SELECTED_17 + EXTRA_VDW22` |
| `v14+aa` | `SELECTED_17 + AA20` |
| `v49` | `SELECTED_17 + AA20 + AROMATIC_ALIPHATIC_12` |
| `v61` | `v49 + AROMATIC_ALIPHATIC_NORMALIZED_12` |
| `v_sterol` | `v49 + STEROL_CHEMISTRY_SHELL_COLS + POCKET_GEOMETRY_COLS` |
| `v_tunnel` | `v_sterol + TUNNEL_FEATURES_18` |
| `v_lipid_boundary` | `v_sterol + LIPID_BOUNDARY_FEATURES_22` |
| `v_caver_t12` | `v_sterol + CAVER_T12_FEATURES_17` |

The full list of active feature-set aliases is in `src/slipp_plus/constants.py::FEATURE_SETS`.

## Ingestion And Validation

`src/slipp_plus/ingest.py` reads the training CSV and supplementary workbooks, normalizes labels, writes parquet artifacts, and enforces the Rule 1 validation gate through `src/slipp_plus/schemas.py`. The gate requires:

- exactly the configured numeric feature columns
- no missing feature values
- `class_10` in the 10-class vocabulary
- 15,219 training rows
- exact per-class counts listed in `configs/day1.yaml`

If this gate fails, training should not proceed.

## Splits

The baseline split protocol uses 25 stratified shuffle splits, 90/10 train/test, with stratification on `class_10`. Split files are persisted under each config's `processed_dir` as `splits/seed_*.parquet`. Seed initialization is described in `docs/reproducibility.md`.

## Models

Flat multiclass training is implemented in `src/slipp_plus/train.py`. The configured model keys are:

- random forest: `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)`
- XGBoost: `XGBClassifier(objective="multi:softprob", num_class=10, random_state=seed, n_jobs=-1)` with balanced sample weights
- LightGBM: `LGBMClassifier(objective="multiclass", num_class=10, class_weight="balanced", random_state=seed, n_jobs=-1)`

Iteration 0 model bundles are persisted under each config's `models_dir`; all iterations write test probabilities under `processed_dir/predictions/`.

## Ensemble And Tiebreakers

The best-current publication recipe in the handoff is based on `configs/v_sterol.yaml`: `feature_set: v_sterol`, 25 iterations, and RF/XGB/LGBM model keys. Ensemble utilities average model softmax probabilities. PLM/STE tiebreaker logic lives in `src/slipp_plus/plm_ste_tiebreaker.py`, and holdout validation for the ensemble plus that tiebreaker lives in `src/slipp_plus/plm_ste_holdout.py`.

Because this repository has active composite and hierarchy experiments, use `experiments/registry.yaml` and current report artifacts to identify which recipe is being claimed in a manuscript draft.

## Evaluation

`src/slipp_plus/evaluate.py` computes:

- binary-collapsed metrics for paper comparison
- 10-class macro-F1 and per-class precision/recall/F1
- apo-PDB and AlphaFold holdout metrics when compatible model bundles and feature columns are available

The baseline ground-truth comparison values are configured in `configs/day1.yaml::ground_truth` and `configs/v_sterol.yaml::ground_truth`.

## Audit Trail

Experiment logs, report directories, and registry entries are part of the scientific record. Negative, superseded, or abandoned experiments should be preserved with notes rather than deleted.
