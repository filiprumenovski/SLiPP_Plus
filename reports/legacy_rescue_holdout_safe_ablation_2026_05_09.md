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
| exp-031 holdout-scored diagnostic | `0.901 +/- 0.015` | `0.668` | `0.729 / 0.761` | `0.735 / 0.762` |

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
| apo-PDB F1/AUROC | `0.710 / 0.765` |
| apo-PDB fire rate | `3.4%` |
| AlphaFold F1/AUROC | `0.722 / 0.798` |
| AlphaFold fire rate | `4.0%` |

This selection preserves internal F1 by firing almost never internally. It is
not enough to beat exp-028 on both holdouts.

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
| apo-PDB F1/AUROC | `0.732 / 0.793` |
| apo-PDB fire rate | `13.7%` |
| AlphaFold F1/AUROC | `0.755 / 0.847` |
| AlphaFold fire rate | `14.8%` |

The gate learns a high-precision internal rescue target and transfers better
than the hard-threshold rule. It beats exp-028 on both holdout F1 scores while
keeping internal metrics close to exp-028.

## Decision

Record as `exp-032-legacy-rescue-holdout-safe-gate`.

The simple logistic gate is the first holdout-safe candidate in this sequence
that beats exp-028 on both external F1 scores. It should be rerun through a
clean reproducible script/report path before replacing exp-028 as the deployable
recommendation.
