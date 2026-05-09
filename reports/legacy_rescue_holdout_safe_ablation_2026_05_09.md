# Legacy Rescue Holdout-Safe Ablation, 2026-05-09

## Question

Can the `exp-031` legacy rescue diagnostic be made holdout-safe by selecting
the rescue rule from internal split predictions only?

`exp-031` improved both external F1 scores, but its thresholds were found by a
holdout-scored grid. This ablation tests two holdout-safe replacements.

## Baselines

| condition | binary F1 | lipid5 macro-F1 | apo-PDB F1/AUROC | AlphaFold F1/AUROC |
|---|---:|---:|---:|---:|
| exp-028 deployable anchor | `0.903 +/- 0.016` | `0.670` | `0.717 / 0.801` | `0.724 / 0.855` |
| exp-031 holdout-scored diagnostic | `0.899 +/- 0.017` | `0.666` | `0.748 / 0.765` | `0.733 / 0.733` |

## Internal Threshold Selection

Grid searched:

- `paper17_threshold in {0.25, 0.30, ..., 0.90}`
- `v_sterol_threshold in {0.25, 0.30, ..., 0.90}`
- `margin in {0.00, 0.05, ..., 0.30}`

Objective: mean internal binary F1 across the 25 test splits after applying the
legacy rescue rule to exp-028 non-lipid calls.

Selected rule:

`paper17_threshold=0.50`, `v_sterol_threshold=0.85`, `margin=0.00`

| metric | value |
|---|---:|
| internal binary F1 | `0.903 +/- 0.016` |
| internal lipid5 macro-F1 | `0.669` |
| internal rescue fire rate | `0.03%` |
| apo-PDB F1/AUROC | `0.694 / 0.749` |
| apo-PDB fire rate | `3.4%` |
| AlphaFold F1/AUROC | `0.658 / 0.701` |
| AlphaFold fire rate | `4.0%` |

This selection preserves internal F1 by firing almost never internally, but it
does not transfer externally.

## Logistic Rescue Gate

A small logistic rescue gate was trained only on internal split prediction
features for rows where exp-028 called non-lipid. Features were probability
signals from exp-028, `paper17_family_encoder`, and `v_sterol`; the gate target
was whether the row was truly lipid. The gate threshold was selected by
leave-iteration-out internal predictions.

Selected threshold: `0.95`

| metric | value |
|---|---:|
| internal binary F1 | `0.901 +/- 0.018` |
| internal lipid5 macro-F1 | `0.667` |
| internal rescue fire rate | `2.5%` |
| apo-PDB F1/AUROC | `0.699 / 0.744` |
| apo-PDB fire rate | `13.7%` |
| AlphaFold F1/AUROC | `0.667 / 0.678` |
| AlphaFold fire rate | `14.8%` |

The gate learns a high-precision internal rescue target, but its score is not
calibrated for the external holdouts. It increases external firing without
recovering the holdout-scored exp-031 gains.

## Decision

Record as `exp-032-legacy-rescue-holdout-safe-negative`.

`exp-031` remains a strong diagnostic clue, but neither internal threshold
selection nor the simple logistic gate makes it deployable. The external rescue
signal is therefore a domain-shift problem, not a directly transferable
internal validation rule.
