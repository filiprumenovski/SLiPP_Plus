# v_lipid_boundary bootstrap summary

_Goal: improve lipid subclass classification with current repo data, using `v_sterol` as the base and adding boundary-specific features plus targeted heads._

## Headline

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble + PLM/STE tiebreaker | 0.738 | 0.610 | 0.899 | 0.728 | 0.700 | 0.543 | 0.638 | 0.444 |
| v_sterol + grouped STE rescue, threshold 0.40 | **0.753** | **0.639** | 0.899 | **0.728** | 0.701 | 0.544 | **0.648** | **0.576** |
| v_lipid_boundary ensemble | 0.730 | 0.599 | 0.898 | 0.723 | 0.701 | 0.559 | 0.636 | 0.378 |
| v_lipid_boundary + grouped STE rescue, threshold 0.40 | 0.748 | 0.636 | 0.898 | 0.722 | 0.701 | **0.559** | 0.645 | 0.552 |

## Boundary Sweeps

| sweep | best gate | result | recommendation |
|---|---:|---|---|
| COA vs PLM | margin 0.05 | PLM F1 0.636; no lipid macro gain | Reject for now |
| CLR vs OLA | margin 0.10 | OLA F1 0.557; below plain v_lipid_boundary OLA 0.559 | Reject for now |
| PLM vs OLA | margin 0.05 | OLA F1 0.556; lipid macro drops to 0.598 | Reject for now |
| STE vs {PLM, COA, OLA, MYR} | threshold 0.40 or 0.45 | lipid macro-F1 0.636; STE F1 0.552; PLM F1 0.645 | Keep |

## Interpretation

The new features do not help as a plain 10-way softmax input; they dilute STE signal and regress the ensemble from lipid macro-F1 0.610 to 0.599. The same grouped STE head performs better using only the original `v_sterol` feature columns than with `v_lipid_boundary`, so the real gain is the grouped binary objective, not the new feature extraction.

The pairwise COA/PLM, CLR/OLA, and PLM/OLA arbiters are not worth promoting. Their binary heads find some pair-specific signal, but the deployment gate does not improve the global lipid-class objective.

The `v_lipid_boundary` feature means look chemically plausible, but the confusion matrix shows the plain 10-way model moves true STE away from STE and into PLM/COA/PP. This is consistent with noisy, correlated path/anchor features being diluted by multiclass class weighting rather than a hard extraction failure.

## Grouped STE Rescue Confusion

_Rows are summed across 25 test folds; columns show the main lipid/non-lipid leakage targets._

### v_sterol + grouped STE rescue

| true class | CLR | MYR | OLA | PLM | STE | COA | PP |
|---|---:|---:|---:|---:|---:|---:|---:|
| STE | 5 | 17 | 16 | 71 | 222 | 19 | 25 |

### v_lipid_boundary + grouped STE rescue

| true class | CLR | MYR | OLA | PLM | STE | COA | PP |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLR | 608 | 10 | 77 | 79 | 1 | 54 | 68 |
| MYR | 15 | 754 | 24 | 127 | 5 | 77 | 44 |
| OLA | 67 | 31 | 417 | 136 | 17 | 82 | 75 |
| PLM | 42 | 216 | 66 | 1091 | 139 | 143 | 93 |
| STE | 7 | 22 | 19 | 66 | 209 | 23 | 27 |

The `v_sterol` rescue fired 160 times at threshold 0.40 and raised STE correct calls from 153 in the previous PLM/STE-tiebreaker run to 222. The `v_lipid_boundary` rescue fired 146 times and raised STE correct calls to 209.

## Recommendation

Do not promote `v_lipid_boundary` as the default feature set. Promote the grouped STE rescue pattern using `v_sterol` columns first:

- use `processed/v_sterol/predictions/ste_rescue_predictions.parquet` as the selected threshold-0.40 artifact;
- compare it directly against `v_sterol ensemble + PLM/STE tiebreaker` in downstream writeups;
- if this remains stable on fresh seed panels, replace the existing PLM/STE tiebreaker with the grouped STE head.
