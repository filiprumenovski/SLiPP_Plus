# AGENTS.md — Operator Brief

Condensed operator instructions for coding agents (Cursor/Claude/Codex) working on SLiPP++. Authoritative scope lives in `PROMPT.md`; this file is a tl;dr.

## Day 1 scope (the only thing that ships now)

Reimplement Chou et al. 2024 with the classifier head changed from binary to 10-class softmax over `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE}`. No new features, no new pocket detector, no new data.

## Ground truth to hit (paper Table 1)

- Binary test F1 ≈ 0.869, AUROC ≈ 0.970
- Apo-PDB F1 ≈ 0.726, AUROC ≈ 0.828
- AlphaFold F1 ≈ 0.643, AUROC ≈ 0.851

Our binary-collapsed numbers on the test set should be within 0.03 F1 of theirs. If not, the pipeline is wrong, not the reformulation.

## Data — already in the repo

- `reference/SLiPP_2024-main/training_pockets.csv` — 15,219 rows = paper's 5-fold balanced set (1,981 lipid + 3,333 non-lipid + 9,905 PP). All 17 descriptors, per-class labels, 20 AA counts, free `_vdw22` surface variants.
- `data/raw/supplementary/ci5c01076_si_003.xlsx` — Supporting File 2, apo-PDB holdout (119 structures, 17 descriptors baked in).
- `data/raw/supplementary/ci5c01076_si_004.xlsx` — Supporting File 3, AlphaFold holdout (151 structures, 17 descriptors baked in).

**Day 1 does not need to run fpocket.** The from-scratch path (`make scratch`) exists for Day 7+.

## The 17 descriptors (canonical, paper names, exact order)

```python
SELECTED_17 = [
    "pock_vol", "nb_AS", "mean_as_ray", "mean_as_solv_acc", "apol_as_prop",
    "mean_loc_hyd_dens", "hydrophobicity_score", "volume_score", "polarity_score",
    "charge_score", "flex", "prop_polar_atm", "as_density", "as_max_dst",
    "surf_pol_vdw14", "surf_apol_vdw14", "surf_vdw14",
]
```

Source: `reference/SLiPP_2024-main/slipp.py:SELECTED_PARAM`.

## Rule 1 validation gate

`src/slipp_plus/schemas.py` enforces via Pandera:

- Exactly 17 numeric feature columns, no NaN.
- `class_10 ∈ CLASS_10` on every row.
- Training row count == 15,219 (exact).
- Per-class counts match paper exactly: CLR 358, MYR 424, OLA 329, PLM 718, STE 152, ADN 414, B12 373, BGC 526, COA 2020, PP 9905.

If the gate fails, `make ingest` aborts. Do not proceed to training. Debug ingestion.

## Models

- `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)`
- `XGBClassifier(objective="multi:softprob", num_class=10, random_state=seed, n_jobs=-1)` with `sample_weight=compute_sample_weight("balanced", y)`
- `LGBMClassifier(objective="multiclass", num_class=10, class_weight="balanced", random_state=seed, n_jobs=-1)`

sklearn pinned to 1.3.1 (paper version). Do not use 1.5+; RF defaults changed.

## Splits

25 stratified shuffle splits (stratified on `class_10`, not binary), 90/10 train/test, seeded and persisted to `processed/splits/seed_{i}.parquet`. Iteration 0 models dumped to `models/*.joblib`. Iterations 1–24 produce metrics only.

## DO / DO NOT (short list)

**DO:** match sklearn 1.3.1, seed everything, report mean±std, use the paper's exact binary collapse definitions (Methods p.19), keep heme out.

**DO NOT:** add descriptors, tune hyperparameters, collapse PP into nLBP, send any outreach before metrics are in hand.

Full DO/DO-NOT in `PROMPT.md` §8.
