# Legacy Rescue Unlabeled Target Selection Audit, 2026-05-09

## Question

Can unlabeled holdout prediction statistics select the exp-037 recall-heavy
`maxlegacy` policy without using holdout labels?

## Method

Using the script-backed gate scores from `scripts/legacy_rescue_gate.py`, sweep
rewrite modes `{mean, paper17, vsterol, maxlegacy}` and thresholds
`0.50..0.99`. For each policy, compute internal metrics plus unlabeled target
statistics on apo-PDB and AlphaFold:

- target predicted lipid-positive rate;
- target mean lipid probability;
- internal fire rate and internal lipid5 macro-F1.

Holdout labels were used only after selection to audit what each unlabeled
criterion would have chosen.

## Result

With internal binary F1 >= `0.895`, unlabeled target statistics do not select
the exp-037 `maxlegacy` policy.

| unlabeled/internal criterion | selected policy | selected holdouts |
|---|---|---:|
| maximize target predicted-positive rate | `mean`, threshold `0.51` | apo `0.738`, AF `0.744` |
| maximize target mean lipid probability | `mean`, threshold `0.52` | apo `0.738`, AF `0.744` |
| maximize internal fire rate | `vsterol`, threshold `0.50` | apo `0.738`, AF `0.741` |
| maximize internal lipid5 macro-F1 | `mean`, threshold `0.99` | apo `0.727`, AF `0.752` |

For comparison, exp-037 (`maxlegacy`, threshold `0.90`) reaches apo-PDB F1
`0.756` and AlphaFold F1 `0.807`, but was found by holdout-scored selection.

## Decision

Record as `exp-040-legacy-rescue-unlabeled-target-selection-negative`.

Naive unlabeled target prevalence and mean-probability objectives are not
sufficient to make exp-037 deployable. They select policies that increase
target lipid calling but do not reproduce the maxlegacy external gain.
