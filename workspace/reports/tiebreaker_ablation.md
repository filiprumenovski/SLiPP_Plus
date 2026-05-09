# Tiebreaker On/Off Ablation

Date: 2026-05-08

## Question

Handoff item 8.5 asks for the lift attributable to the PLM/STE or current
boundary-head tiebreaker stack.

## Method

This report uses persisted `v_sterol` prediction artifacts; no new model
training was run. The off condition is the RF+XGB+LGBM mean-probability
ensemble. The on conditions are the saved PLM/STE and STE-neighbor rescue
postprocessors.

Inputs:

| condition | prediction artifact |
|---|---|
| off: mean-probability ensemble | `processed/v_sterol/predictions/ensemble_predictions.parquet` |
| PLM/STE tiebreaker | `processed/v_sterol/predictions/plm_ste_tiebreaker_predictions.parquet` |
| STE rescue | `processed/v_sterol/predictions/ste_rescue_predictions.parquet` |
| STE rescue boundary refactor | `processed/v_sterol/predictions/ste_rescue_boundary_refactor_predictions.parquet` |
| STE rescue OLA/PLM pair | `processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet` |

## Results

| condition | macro10 F1 | lipid5 macro-F1 | binary F1 | PLM F1 | STE F1 | PLM recall | STE recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| off: ensemble | 0.734 +/- 0.016 | 0.601 +/- 0.028 | 0.899 +/- 0.016 | 0.636 +/- 0.047 | 0.398 +/- 0.097 | 0.623 | 0.349 |
| PLM/STE tiebreaker | 0.738 +/- 0.015 | 0.610 +/- 0.027 | 0.899 +/- 0.016 | 0.638 +/- 0.047 | 0.444 +/- 0.109 | 0.619 | 0.408 |
| STE rescue | 0.753 +/- 0.016 | 0.639 +/- 0.030 | 0.899 +/- 0.016 | 0.648 +/- 0.050 | 0.576 +/- 0.108 | 0.617 | 0.592 |
| STE rescue boundary refactor | 0.753 +/- 0.016 | 0.640 +/- 0.029 | 0.899 +/- 0.016 | 0.647 +/- 0.050 | 0.576 +/- 0.108 | 0.617 | 0.592 |
| STE rescue OLA/PLM pair | 0.754 +/- 0.016 | 0.641 +/- 0.031 | 0.899 +/- 0.016 | 0.649 +/- 0.051 | 0.576 +/- 0.108 | 0.617 | 0.592 |

Paired deltas against the off condition:

| on condition | lipid5 delta | macro10 delta | binary F1 delta | PLM F1 delta | STE F1 delta | STE recall delta |
|---|---:|---:|---:|---:|---:|---:|
| PLM/STE tiebreaker | +0.009 +/- 0.015 | +0.005 +/- 0.008 | +0.000 +/- 0.000 | +0.001 +/- 0.007 | +0.045 +/- 0.072 | +0.059 +/- 0.087 |
| STE rescue | +0.038 +/- 0.017 | +0.019 +/- 0.008 | +0.000 +/- 0.002 | +0.012 +/- 0.008 | +0.178 +/- 0.077 | +0.243 +/- 0.092 |
| STE rescue boundary refactor | +0.038 +/- 0.017 | +0.019 +/- 0.008 | +0.000 +/- 0.002 | +0.011 +/- 0.008 | +0.178 +/- 0.077 | +0.243 +/- 0.092 |
| STE rescue OLA/PLM pair | +0.040 +/- 0.019 | +0.020 +/- 0.009 | +0.000 +/- 0.002 | +0.013 +/- 0.012 | +0.178 +/- 0.077 | +0.243 +/- 0.092 |

## Conclusion

The narrow PLM/STE tiebreaker is directionally useful but small. It improves
STE F1 by about `+0.045` and lipid macro-F1 by about `+0.009`, with no binary
F1 movement.

The broader STE-neighbor rescue is the material intervention. It increases STE
F1 from `0.398` to `0.576`, STE recall from `0.349` to `0.592`, and lipid
macro-F1 by `+0.038` to `+0.040` while leaving binary F1 effectively unchanged.
This validates the tiebreaker/rescue family as a real per-class correction
rather than a binary lipid-vs-rest effect.

This closes handoff item 8.5 from existing artifacts.
