# Boundary-Refactor Holdout Attempt

Queued item: `exp-009-v_sterol-boundary-refactor`.

Attempted command:

```bash
make eval CFG=configs/v_sterol_boundary_refactor.yaml
```

## Outcome

Do not use the generated `reports/v_sterol_boundary_refactor/metrics_table.md`
as evidence for `exp-009`.

The config points at `processed/v_sterol` and the generic hierarchical
prediction file:

`processed/v_sterol/predictions/hierarchical_lipid_predictions.parquet`

That artifact is currently tagged as `composite_pair_moe_predictions` and its
internal metrics match `exp-011-family-plus-moe` (`macro-F1 0.762`,
`lipid5 0.660`, `STE 0.635`), not the queued `exp-009` boundary-refactor
artifact (`macro-F1 0.754`, `lipid5 0.641`, `STE 0.576`).

The correct exp-009 internal prediction artifacts are:

- `processed/v_sterol/predictions/ste_rescue_boundary_refactor_predictions.parquet`
- `processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet`

However, matching holdout predictions for the full `ste_rescue_ola_plm_pair`
postprocessing were not present under `processed/v_sterol/predictions/holdouts/`.

## Decision

Leave `exp-009` in the holdout-completion queue. It still needs a holdout path
that applies the grouped STE rescue plus OLA/PLM pair logic to apo-PDB and
AlphaFold predictions, or regenerated holdout predictions with artifact names
that unambiguously match `exp-009`.
