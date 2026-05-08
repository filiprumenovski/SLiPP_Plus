# SLiPP++ Project Context

<!-- This is the LLM entry point. Read this file FIRST at the start of any session. -->
<!-- Last updated: 2026-05-06 -->
<!-- See AGENTS.md for constraints/rules. See RESEARCH_LOG.md for full history. -->

## What This Project Is

SLiPP++ extends the **published SLiPP** binary lipid-binding pocket classifier (Chou et al., *J. Chem. Inf. Model.* **2024**; preprint PDF [doi.org/10.1101/2024.01.26.577452](https://doi.org/10.1101/2024.01.26.577452); official scripts + `training_pockets.csv` at [github.com/dassamalab/SLiPP_2024](https://github.com/dassamalab/SLiPP_2024)) from a binary (lipid vs non-lipid) classifier to a **10-class softmax** over `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE}`. The pipeline trains RF, XGB, and LGBM on **15,219** pocket descriptors from that curated table, evaluates on the same two external holdouts as the paper (apo-PDB, AlphaFold), and reports both 10-class metrics and the **paper’s binary collapse** so test and holdout numbers stay comparable to **Table 1** in the preprint/journal PDF.

The project is in an **active research phase** — past Day 1 reproduction, now optimizing per-lipid-class F1 with chemistry-informed features and binary arbiters.

## Current Best Configuration

```
experiment_id:   exp-017-compact-shape3-shape6-ensemble
feature_set:     probability ensemble of v49+tunnel_shape3 and v49+tunnel_shape
backbone:        two compact family encoders distilled from exp-009 teacher predictions
postprocessing:  mean probability ensemble
config:          scripts/compact_probability_ensemble.py
predictions:     processed/compact_shape3_shape6_ensemble/predictions/test_predictions.parquet
reproduce:       uv run python scripts/compact_probability_ensemble.py
report:          uv run python -m slipp_plus.cli compact-report
```

## Current Best Metrics

| metric                  | value           | paper baseline | Δ       |
|---                      |---              |---             |---      |
| Binary F1 (test)        | 0.904 ± 0.015  | 0.869          | +0.035  |
| Binary AUROC (test)     | 0.989 ± 0.003  | 0.970          | +0.019  |
| 10-class macro-F1       | 0.775 ± 0.017  | —              | new     |
| 5-lipid macro-F1        | 0.676 ± 0.032  | —              | new     |
| CLR F1                  | 0.759           | —              | —       |
| STE F1                  | 0.646           | —              | weakest |
| PLM F1                  | 0.652           | —              | —       |
| MYR F1                  | 0.702           | —              | —       |
| OLA F1                  | 0.620           | —              | —       |
| Apo-PDB F1              | 0.690           | 0.726          | conservative |
| AlphaFold F1            | 0.676           | 0.643          | improved |

## Active Hypotheses

1. **STE confusion is dominated by PLM** (38% of STE errors), not CLR (0.8%). The PLM/STE local neighborhood expert is the current best intervention.
2. **Aromatic residue composition** (TYR, TRP, PHE) is the primary discriminator for sterol-binding pockets — confirmed by XGB gain analysis.
3. **fpocket outperforms P2Rank** on lipid-class pocket detection (validated, exp-008).
4. **Compact family structure beats raw feature accumulation**. The 55-feature tunnel-shape encoder matches the 105-feature tunnel MoE within split noise with a much smaller artifact.

## Known Blockers / Weaknesses

- **STE F1 (0.647)** remains the weakest lipid class. Only 152 training rows, so data scarcity is the floor.
- **External holdouts are conservative** for the compact release candidate: apo-PDB F1 0.696 vs paper 0.726, AlphaFold F1 0.620 vs paper 0.643.
- **Tiebreaker modules** are heavily duplicated (~1,750 LOC of copy-paste). P2 audit item pending.

## What Has Been Tried (and failed / abandoned)

| approach | result | why it failed | experiment |
|---|---|---|---|
| `v_plm_ste` motif features (CRAC/CARC, axial, polar-anchor) | STE F1 0.388 (regressive) | Signal diluted in 10-class softmax | exp-006 |
| CLR/STE binary tiebreaker (margin=0.15) | Fired 2 times in 25 iters | STE's confusion is with PLM, not CLR | exp-003 |
| Pocket elongation via alpha-sphere PCA | Not in top-15 XGB gain | `pock_vol` + `as_max_dst` already capture shape | exp-005 |
| v61 normalized shell fractions | Mixed: helps apo-PDB, hurts AlphaFold | Not a clean replacement for v49 | exp-004 |

## Feature Set Evolution

```
v14 (17 cols)  →  v14+shell (29)                ← shell12 alone is modest
               →  v14+aa (37)                   ← major compact recovery
               →  v49 (+20 AA +12 shell = 49)   ← parsimonious fallback
               →  v49+tunnel_shape (55)         ← current release candidate
               →  v_sterol (+38 chem-refined = 87)
               →  v_tunnel (+18 tunnel = 105)   ← high-complexity reference only
               →  v_plm_ste (+16 motif = 103)   ← abandoned
```

## Suggested Next Experiments

1. **Release hardening for `v49+tunnel_shape_family_encoder`**: freeze config, schema sidecars, and compact-report output.
2. **Holdout story**: decide whether to publish conservative apo/AlphaFold numbers or add a holdout-safe calibration step.
3. **Production path**: make compact inference load only the 55 required columns and avoid heavyweight MoE code paths.
4. **Polars migration**: keep pandas only behind compatibility boundaries such as Pandera and Excel ingest.
5. **Optional science follow-up**: SMOTE or class-balanced encoder training for STE, but only after release candidate cleanup.

## Repository Quick Reference

```
src/slipp_plus/
├── ingest.py              # CSV/XLSX → validated parquets (Rule 1 gate)
├── splits.py              # 25 stratified shuffle splits, seeded
├── train.py               # RF/XGB/LGBM training loop + model persistence
├── evaluate.py            # Metrics + holdout scoring + binary collapse
├── ensemble.py            # Probability-averaging ensemble (polars)
├── figures.py             # Confusion matrix, ROC, PCA plots
├── constants.py           # SOURCE OF TRUTH: features, classes, codes
├── config.py              # Pydantic settings from YAML
├── schemas.py             # Pandera validation gates
├── features.py            # Feature matrix materialization
├── artifact_schema.py     # Schema sidecar for model bundles
│
├── aromatic_aliphatic.py  # Shell residue feature extractor
├── v49.py                 # v49 parquet builder (AA + shell features)
├── v61.py                 # v61 normalized features
├── sterol_features.py     # v_sterol chemistry-refined shells
├── plm_ste_features.py    # v_plm_ste motif features (not recommended)
│
├── sterol_tiebreaker.py   # CLR/STE binary arbiter
├── plm_ste_tiebreaker.py  # PLM/STE binary arbiter (in winning config)
├── hierarchical_experiment.py
├── hierarchical_postprocess.py  # OneVsNeighborsRule abstraction
│
├── cli.py                 # Typer CLI entry point
└── calibration.py         # Calibration analysis

configs/                   # YAML experiment configs
experiments/registry.yaml  # Machine-readable experiment index
reports/                   # Per-experiment markdown + parquet reports
processed/                 # Intermediate parquets + splits
models/                    # Iteration-0 joblib bundles
```

### Key commands

```bash
make all CFG=configs/v_sterol.yaml    # Full pipeline: ingest → train → evaluate → figures
make test                             # 123 pytest tests
uv run python -m slipp_plus.cli --help
```

## Constraints (from AGENTS.md)

- sklearn pinned to 1.3.1 (paper version). Do not use 1.5+.
- Seed everything. Report mean ± std across 25 iterations.
- Binary collapse uses paper Methods p.19 definitions exactly.
- Do NOT add descriptors, tune hyperparameters, or collapse PP into nLBP without explicit approval.
- Training row count must be exactly 15,219 with per-class counts matching the paper.
