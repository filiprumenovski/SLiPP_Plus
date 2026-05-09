# Boundary-Refactor Holdout Completion

Queued item: `exp-009-v_sterol-boundary-refactor`.

The first shortcut attempt showed that `make eval
CFG=configs/v_sterol_boundary_refactor.yaml` cannot be used as a complete
test-split report because the generic
`processed/v_sterol/predictions/hierarchical_lipid_predictions.parquet` now
belongs to the later exp-011 composite pair-MoE stack.

The holdout half of that command did use the saved
`models/v_sterol_boundary_refactor/hierarchical_bundle.joblib` bundle and
regenerated:

- `processed/v_sterol/predictions/holdouts/hierarchical_apo_pdb_predictions.parquet`
- `processed/v_sterol/predictions/holdouts/hierarchical_alphafold_predictions.parquet`

## Binary Holdout Results

| holdout | F1 | AUROC | precision | sensitivity |
|---|---:|---:|---:|---:|
| apo-PDB | `0.679` | `0.812` | `0.844` | `0.567` |
| AlphaFold | `0.708` | `0.864` | `0.962` | `0.560` |

## Decision

Closed for binary holdout completion. The boundary-refactor bundle changes
holdout subclass probabilities and predicted 10-class labels (`20` apo-PDB rows
and `25` AlphaFold rows differ from the flat ensemble), but it makes zero binary
lipid/non-lipid decision changes on either holdout. Since the supplementary
holdouts are binary-labeled, the binary F1/AUROC values are identical to the
flat `v_sterol` ensemble holdouts.

Do not use the stale test-split metrics emitted by the generic eval command for
exp-009. Internal exp-009 metrics remain those recorded from
`processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet`.
