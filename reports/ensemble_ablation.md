# Ensemble vs Best Single Model Ablation

Date: 2026-05-08

## Question

Handoff item 8.4 asks whether the RF+XGB+LGBM mean-probability ensemble beats
the best single model by at least one standard deviation on macro-F1.

## Method

This report uses existing persisted prediction artifacts; no new model training
was run. For each flat RF/XGB/LGBM stack with available per-model predictions,
the script recomputed the mean-probability ensemble per split and compared it
against the best single learner on 5-lipid macro-F1.

Inputs:

| stack | prediction artifact |
|---|---|
| `v49` | `processed/v49/predictions/test_predictions.parquet` |
| `v49+tunnel_shape` | `processed/v49_tunnel_shape/predictions/test_predictions.parquet` |
| `v61` | `processed/v61/predictions/test_predictions.parquet` |
| `v_lipid_boundary` | `processed/v_lipid_boundary/predictions/test_predictions.parquet` |
| `v_tunnel` | `processed/v_tunnel/predictions/test_predictions.parquet` |

The `v_tunnel_aligned` raw metrics exist, but the corresponding
`processed/v_tunnel_aligned/predictions/test_predictions.parquet` artifact is
not present in this checkout, so it is excluded from the paired comparison.

## Results

| stack | best single | best single lipid5 macro-F1 | ensemble lipid5 macro-F1 | paired lipid5 delta | macro10 delta | binary F1 delta | >= 1 std? |
|---|---|---:|---:|---:|---:|---:|---|
| `v49` | XGB | 0.590 +/- 0.033 | 0.596 +/- 0.030 | +0.006 +/- 0.018 | +0.006 | +0.003 | no |
| `v49+tunnel_shape` | LGBM | 0.612 +/- 0.036 | 0.618 +/- 0.035 | +0.006 +/- 0.017 | +0.007 | +0.004 | no |
| `v61` | LGBM | 0.591 +/- 0.031 | 0.600 +/- 0.032 | +0.009 +/- 0.023 | +0.007 | +0.004 | no |
| `v_lipid_boundary` | LGBM | 0.590 +/- 0.027 | 0.599 +/- 0.025 | +0.009 +/- 0.022 | +0.007 | +0.002 | no |
| `v_tunnel` | LGBM | 0.621 +/- 0.039 | 0.633 +/- 0.039 | +0.012 +/- 0.015 | +0.007 | +0.004 | no |

## Conclusion

The mean-probability ensemble is directionally useful: it improves lipid macro,
macro10, and binary F1 over the best single learner on every checked flat stack.
However, the improvement does not meet the handoff criterion of at least one
standard deviation on any checked stack. Treat RF+XGB+LGBM averaging as a
low-risk stabilizer, not as a large standalone scientific effect.

This is a closed weak-positive/negative ablation for item 8.4.
