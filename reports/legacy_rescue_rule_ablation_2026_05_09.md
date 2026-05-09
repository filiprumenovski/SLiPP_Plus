# Legacy Rescue Rule Ablation, 2026-05-09

## Question

Can the external-recall signal from legacy `paper17_family_encoder` and
`v_sterol` be used as a conservative rescue rule on top of the deployable
`exp-028` probability ensemble?

This starts from exp-028 and only modifies rows that exp-028 predicts as
non-lipid.

## Rule Tested

For each row:

1. Compute exp-028 lipid probability.
2. Compute lipid probability from `paper17_family_encoder`.
3. Compute lipid probability from `v_sterol`.
4. If exp-028 lipid probability is `< 0.50` and both legacy models have lipid
   probability `>= threshold`, replace the row's class probabilities with the
   equal average of `paper17_family_encoder` and `v_sterol`.

Grid:

- `paper17_threshold in {0.35, 0.40, ..., 0.70}`
- `v_sterol_threshold in {0.35, 0.40, ..., 0.70}`
- `margin in {0.00, 0.05, ..., 0.20}` where margin is
  `min(paper17_lipid, v_sterol_lipid) - exp028_lipid`

## Best External-F1 Candidate

`paper17_threshold=0.35`, `v_sterol_threshold=0.35`, `margin=0.00`

| metric | value |
|---|---:|
| binary F1 | `0.899 +/- 0.017` |
| macro10 F1 | `0.766` |
| lipid5 macro-F1 | `0.666` |
| STE F1 | `0.639` |
| test rows rescued | `0.6%` |
| apo-PDB rows rescued | `9.4%` |
| AlphaFold rows rescued | `13.4%` |
| apo-PDB F1/AUROC | `0.748 / 0.765` |
| AlphaFold F1/AUROC | `0.733 / 0.733` |
| holdout mean F1 | `0.740` |

Compared with exp-028:

| metric | exp-028 | rescue rule | delta |
|---|---:|---:|---:|
| binary F1 | `0.903` | `0.899` | `-0.004` |
| macro10 F1 | `0.769` | `0.766` | `-0.003` |
| lipid5 macro-F1 | `0.670` | `0.666` | `-0.004` |
| apo-PDB F1 | `0.717` | `0.748` | `+0.031` |
| AlphaFold F1 | `0.724` | `0.733` | `+0.009` |
| holdout mean F1 | `0.721` | `0.740` | `+0.019` |

## Interpretation

This is the first saved-artifact ablation in this run that beats exp-028 on
both external F1 scores while keeping internal metrics close to exp-028.

It is not yet a clean deployable replacement because the threshold grid was
scored on final holdout labels. The important signal is that exp-028's external
misses include a small, high-value pocket subset where both older feature
families agree on lipid despite exp-028 being below 0.5.

## Decision

Record as `exp-031-legacy-rescue-rule-diagnostic`.

Next step: make this holdout-safe by selecting the rescue thresholds using only
internal split predictions, or by replacing the hard thresholds with a small
stacked/gated model trained inside the 25-split protocol. If that internal
selection recovers similar external gains, it should supersede exp-028.
