# Legacy Rescue Internal Selection Audit, 2026-05-09

## Question

Can the holdout-positive `exp-037` `maxlegacy`/`0.90` rescue policy be selected
from internal split metrics only?

## Inputs

This audit reads the diagnostic sweep table:

`reports/legacy_rescue_gate_diagnostic_sweep.csv`

The sweep itself was holdout-scored, so this audit is not a promotion path. It
checks whether simple internal-only ranking rules would have independently
chosen the same region.

## Internal-Only Criteria Checked

| criterion | selected policy | internal metrics | holdouts |
|---|---|---:|---:|
| maximize internal binary F1, then lipid5 | `simple`, `C=0.03`, `blend_base50`, `threshold=0.99` | F1 `0.902`, lipid5 `0.669`, fire `0.4%` | apo `0.704`, AF `0.744` |
| maximize internal fire rate with F1 >= `0.895` | `simple`, `C=30`, `blend_base50`, `threshold=0.90` | F1 `0.902`, lipid5 `0.669`, fire `3.2%` | apo `0.704`, AF `0.744` |
| maximize internal fire rate with F1 >= `0.894` | `simple`, `C=30`, `blend_base50`, `threshold=0.90` | F1 `0.902`, lipid5 `0.669`, fire `3.2%` | apo `0.704`, AF `0.744` |

For comparison, the holdout-positive diagnostic candidate is:

| policy | internal metrics | holdouts |
|---|---:|---:|
| `compact`, `C=0.3`, `maxlegacy`, `threshold=0.90` | F1 `0.895`, lipid5 `0.658`, fire `2.9%` | apo `0.756`, AF `0.807` |

## Decision

Record as `exp-038-legacy-rescue-internal-selection-audit`.

Simple internal scalar objectives do not recover the external-positive
`maxlegacy` policy. Internal F1 and lipid5 macro-F1 favor conservative blended
rewrites that preserve split-test quality but lose the external recall gain.
The next useful route is not another scalar threshold on internal F1/fire; it
needs a better internal proxy for external false negatives, such as grouping by
domain-risk features, fitting a selector to source-vs-target probability shift
without labels, or using a leave-family/leave-structure stress split if one can
be constructed from available metadata.
