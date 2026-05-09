# Legacy Rescue Base-FN Selection Audit, 2026-05-09

## Question

Can the exp-037 recall-heavy rescue behavior be selected from internal split
labels by directly targeting base false negatives?

## Method

Using the script-backed gate scores from `scripts/legacy_rescue_gate.py`, sweep
rewrite modes `{mean, paper17, vsterol, maxlegacy}` and thresholds
`0.50..0.99`. For each policy, compute on internal split predictions:

- global binary F1 and lipid5 macro-F1;
- fire rate on rows where exp-028 calls non-lipid;
- rescue precision: fraction of fired rows that are true internal base false
  negatives;
- base-FN recall: fraction of exp-028 internal false negatives that fire.

The intended internal-only selector was: maximize base-FN recall subject to
internal binary F1 >= `0.895` and a reasonable rescue-precision floor.

## Result

No candidate with internal binary F1 >= `0.895` reached rescue precision
`>= 0.50`, `>= 0.55`, `>= 0.60`, `>= 0.65`, `>= 0.70`, or `>= 0.75`.

The unconstrained top base-FN recall policies were low-threshold `mean` or
`vsterol` rewrites:

| policy | internal F1 | lipid5 | fire rate | rescue precision | base-FN recall |
|---|---:|---:|---:|---:|---:|
| `mean`, threshold `0.50` | `0.899` | `0.664` | `10.7%` | `0.107` | `0.793` |
| `vsterol`, threshold `0.50` | `0.899` | `0.665` | `10.7%` | `0.107` | `0.793` |
| `mean`, threshold `0.51` | `0.899` | `0.665` | `10.4%` | `0.108` | `0.786` |

## Decision

Record as `exp-039-legacy-rescue-base-fn-selection-negative`.

Directly maximizing internal base-FN recall is not a useful holdout-safe
selector: it overfires on too many true non-lipid rows, and precision-constrained
versions have no feasible candidate at the current internal F1 floor. The next
selector needs a better proxy for external false negatives than ordinary
internal base-FN membership.
