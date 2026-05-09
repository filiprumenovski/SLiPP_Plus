# Family Plus MoE Holdout Completion

Queued item: `exp-011-family-plus-moe`.

This implements holdout inference for saved composite pair/local-MoE bundles by
starting from the family-encoder holdout teacher probabilities and applying the
iteration-0 saved local/pair experts in bundle order.

Command:

```bash
make eval CFG=configs/archive/v_sterol_family_plus_moe.yaml
```

## Results

| holdout | F1 | AUROC | precision | sensitivity |
|---|---:|---:|---:|---:|
| apo-PDB | `0.723` | `0.807` | `0.827` | `0.642` |
| AlphaFold | `0.703` | `0.838` | `0.912` | `0.571` |

## Decision

Registry holdouts are now filled for `exp-011-family-plus-moe`. The result is
useful but not deployable over exp-028: apo-PDB is strong and close to the paper
baseline, but AlphaFold F1 remains below the compact weighted blend
(`0.703` vs `0.724`).
