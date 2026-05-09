# v_sterol Holdout Completion

Queued item: `exp-005-v_sterol-ensemble`.

This closes the missing registry holdout block from existing persisted
prediction artifacts. No retraining was run.

Inputs:

- Test/ensemble artifact: `processed/v_sterol/predictions/test_predictions.parquet`
- apo-PDB predictions: `processed/v_sterol/predictions/holdouts/apo_pdb_ensemble_predictions.parquet`
- AlphaFold predictions: `processed/v_sterol/predictions/holdouts/alphafold_ensemble_predictions.parquet`
- Holdout labels/features: `processed/v_sterol/apo_pdb_holdout.parquet`,
  `processed/v_sterol/alphafold_holdout.parquet`

## Results

| holdout | F1 | AUROC | precision | sensitivity | specificity | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apo-PDB | `0.679` | `0.812` | `0.844` | `0.567` | `0.860` | 38 | 43 | 7 | 29 |
| AlphaFold | `0.708` | `0.864` | `0.962` | `0.560` | `0.966` | 51 | 56 | 2 | 40 |

## Decision

Registry holdouts are now filled for `exp-005-v_sterol-ensemble`. The result is
not competitive with current compact deployable blends, but it is useful audit
evidence for the sterol feature sprint: v_sterol improved internal subclass
metrics but did not solve external recall at the default lipid threshold.
