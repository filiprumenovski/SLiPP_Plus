# Boundary-Head Refactor Results

Date: 2026-04-24

## Result

The refactored boundary-head stack produces a new best lipid-subclass result on
the 25 stratified v_sterol splits.

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prior best: PLM/STE tiebreaker | 0.738 +/- 0.015 | 0.610 +/- 0.026 | 0.899 +/- 0.015 | 0.986 +/- 0.004 | 0.728 | 0.700 | 0.543 | 0.638 | 0.444 |
| grouped STE rescue | 0.753 +/- 0.016 | 0.640 +/- 0.029 | 0.899 +/- 0.016 | 0.986 +/- 0.004 | 0.728 | 0.701 | 0.546 | 0.647 | 0.576 |
| grouped STE rescue + OLA/PLM pair | 0.754 +/- 0.016 | 0.641 +/- 0.030 | 0.899 +/- 0.016 | 0.986 +/- 0.004 | 0.728 | 0.701 | 0.553 | 0.649 | 0.576 |

Delta versus the prior best:

- 10-class macro-F1: +0.015
- 5-lipid macro-F1: +0.031
- STE F1: +0.132
- OLA F1: +0.009
- PLM F1: +0.011
- Binary F1/AUROC: unchanged within noise

## Promoted Artifact

```text
processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet
```

Supporting reports:

```text
reports/v_sterol/ste_rescue_boundary_refactor.md
reports/v_sterol/ste_rescue_ola_plm_pair_selected.md
reports/v_sterol/ste_rescue_ola_plm_pair_residual_confusions.md
```

## Reproduction

```bash
uv run python -m slipp_plus.cli ste-rescue-sweep \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --predictions processed/v_sterol/predictions/test_predictions.parquet \
  --splits-dir processed/v_sterol/splits \
  --model-bundle models/v_sterol/xgb_multiclass.joblib \
  --output-report reports/v_sterol/ste_rescue_boundary_refactor.md \
  --output-metrics reports/v_sterol/ste_rescue_boundary_refactor.parquet \
  --output-predictions processed/v_sterol/predictions/ste_rescue_boundary_refactor_predictions.parquet \
  --selected-threshold 0.50 \
  --workers 8

uv run python -m slipp_plus.cli pair-tiebreaker-sweep \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --predictions processed/v_sterol/predictions/ste_rescue_boundary_refactor_predictions.parquet \
  --splits-dir processed/v_sterol/splits \
  --model-bundle models/v_sterol/xgb_multiclass.joblib \
  --output-report reports/v_sterol/ste_rescue_ola_plm_pair_selected.md \
  --output-metrics reports/v_sterol/ste_rescue_ola_plm_pair_selected.parquet \
  --output-predictions processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet \
  --selected-margin 0.05 \
  --negative-label PLM \
  --positive-label OLA \
  --workers 8
```

## Decision

Promote the grouped STE-vs-neighbors rescue as the main architectural win from
the boundary-head refactor. Keep the OLA/PLM pair at margin 0.05 as a small
stacked improvement. Do not stack PLM/MYR: the sweep is neutral to regressive
after STE rescue.
