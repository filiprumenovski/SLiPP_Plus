# Legacy Rescue Structure-Stress Audit, 2026-05-09

## Question

Can an internal structure-level stress subset select the exp-037 recall-heavy
`maxlegacy` policy without using holdout labels?

## Method

For each saved stratified split, parse a structure id from `pdb_ligand`, mark
test rows whose structure id is present in that iteration's train split as
`seen_structure`, and mark the rest as `unseen_structure`. Then compare exp-028,
exp-035, and exp-037 binary metrics within each subset.

## Result

The existing random splits are not a useful leave-structure stress test. Almost
every test row shares a structure id with the train split.

| subset | mean rows / iteration | note |
|---|---:|---|
| seen_structure | `1506.0` | enough rows for stable internal metrics |
| unseen_structure | `16.0` | too small; min `8`, max `23` |

Mean binary F1 by subset:

| model | seen_structure F1 | unseen_structure F1 |
|---|---:|---:|
| exp-028 | `0.900` | `0.983` |
| exp-035 | `0.895` | `0.986` |
| exp-037 | `0.887` | `0.984` |

The unseen-structure subset is both tiny and already near-perfect. It cannot
serve as an internal proxy for the external false negatives that exp-037
rescues.

## Decision

Record as `exp-041-legacy-rescue-structure-stress-negative`.

Do not use the current random splits for structure-stress selection. A real
leave-structure or leave-cluster experiment would need retraining with grouped
splits; the existing prediction artifacts cannot answer it reliably.
