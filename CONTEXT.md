# SLiPP++ Project Context

<!-- This is the LLM entry point. Read this file FIRST at the start of any session. -->
<!-- Last updated: 2026-04-24 -->
<!-- See AGENTS.md for constraints/rules. See RESEARCH_LOG.md for full history. -->

## What This Project Is

SLiPP++ extends the **published SLiPP** binary lipid-binding pocket classifier (Chou et al., *J. Chem. Inf. Model.* **2024**; preprint PDF [doi.org/10.1101/2024.01.26.577452](https://doi.org/10.1101/2024.01.26.577452); official scripts + `training_pockets.csv` at [github.com/dassamalab/SLiPP_2024](https://github.com/dassamalab/SLiPP_2024)) from a binary (lipid vs non-lipid) classifier to a **10-class softmax** over `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE}`. The pipeline trains RF, XGB, and LGBM on **15,219** pocket descriptors from that curated table, evaluates on the same two external holdouts as the paper (apo-PDB, AlphaFold), and reports both 10-class metrics and the **paper’s binary collapse** so test and holdout numbers stay comparable to **Table 1** in the preprint/journal PDF.

The project is in an **active research phase** — past Day 1 reproduction, now optimizing per-lipid-class F1 with chemistry-informed features and binary arbiters.

## Current Best Configuration

```
experiment_id:   exp-009-v_sterol-boundary-refactor
feature_set:     v_sterol (87 columns: 17 fpocket + 20 AA + 12 shell + 38 chem-refined)
models:          [rf, xgb, lgbm] → probability-averaged ensemble
postprocessing:  grouped STE-vs-neighbors rescue (threshold=0.50) + OLA-vs-PLM pair head (margin=0.05)
config:          configs/v_sterol.yaml
predictions:     processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet
reproduce:       see reports/v_sterol/boundary_refactor_results.md
```

## Current Best Metrics

| metric                  | value           | paper baseline | Δ       |
|---                      |---              |---             |---      |
| Binary F1 (test)        | 0.899 ± 0.015  | 0.869          | +0.030  |
| Binary AUROC (test)     | 0.986 ± 0.004  | 0.970          | +0.016  |
| 10-class macro-F1       | 0.754 ± 0.016  | —              | new     |
| 5-lipid macro-F1        | 0.641 ± 0.030  | —              | new     |
| CLR F1                  | 0.728 ± 0.047  | —              | —       |
| STE F1                  | 0.576 ± 0.105  | —              | weakest |
| PLM F1                  | 0.649 ± 0.050  | —              | —       |
| MYR F1                  | 0.697           | —              | —       |
| OLA F1                  | 0.553           | —              | —       |
| Apo-PDB F1 (RF)         | 0.716           | 0.726          | −0.010  |
| AlphaFold F1 (RF)        | 0.725           | 0.643          | +0.082  |

## Active Hypotheses

1. **STE confusion is dominated by PLM** (38% of STE errors), not CLR (0.8%). The PLM/STE binary tiebreaker addresses this.
2. **Aromatic residue composition** (TYR, TRP, PHE) is the primary discriminator for sterol-binding pockets — confirmed by XGB gain analysis.
3. **fpocket outperforms P2Rank** on lipid-class pocket detection (validated, exp-008).
4. **Binary arbiters extract pairwise signal** more efficiently than the 10-class softmax can.

## Known Blockers / Weaknesses

- **STE F1 (0.444)** remains the weakest lipid class. Only 152 training rows — data scarcity is the floor.
- **Apo-PDB holdout** slightly underperforms the paper (0.716 vs 0.726) — the aromatic/aliphatic features don't help apo structures as much.
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
v14 (17 cols)  →  v49 (+20 AA +12 shell = 49)  →  v61 (+12 normalized = 61)
                                                 →  v_sterol (+38 chem-refined = 87)  ← current best
                                                 →  v_plm_ste (+16 motif = 103)       ← abandoned
```

## Suggested Next Experiments

1. **Apply binary arbiters to other dominant confusion pairs** — PLM/OLA, STE/COA, CLR/OLA. Each is ~1 hour and should compound.
2. **SMOTE-oversample STE** (152 rows). Quick test, may add 3–5pp to STE F1.
3. **Explicit H-bond donor/acceptor counts per shell** — should sharpen all polar-anchor-dependent classes (CLR, STE).
4. **Re-run P2Rank with conservation** (`-c conservation_hmm`). ~a day of compute. May close the fpocket gap on non-sterol classes.
5. **Split aromatic residues** into PHE/TRP vs TYR/HIS for finer chemical resolution.

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
