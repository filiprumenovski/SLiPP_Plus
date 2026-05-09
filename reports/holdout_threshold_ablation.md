# Holdout Threshold Ablation

Diagnostic threshold sweep over staged holdout probabilities for the compact ensemble candidates. This is not a model-selection protocol; apo-PDB and AlphaFold are final holdouts, so tuned thresholds are evidence about calibration/transfer failure modes, not deployable settings.

## Key Signal

- The holdout regression is mostly a recall/calibration problem, not lipid overcalling. At threshold `0.5`, `exp-019` has high precision but many false negatives.
- Lower thresholds (`0.25`-`0.35`) recover large holdout F1 for every compact ensemble, especially AlphaFold.
- `exp-018` becomes the strongest AlphaFold candidate under a lower threshold, while `exp-021` is strongest on apo-PDB in this diagnostic sweep.
- This suggests the next real fix should be calibration learned inside the 25 split protocol, not hand-tuned holdout thresholds.

## Best Threshold By Condition

| holdout | condition | best_threshold | best_f1 | default_f1 | delta | best_precision | best_recall | default_recall | best_fp | best_fn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alphafold | `exp018_shape3_shape6_shell6` | 0.250 | 0.829 | 0.671 | +0.157 | 0.833 | 0.824 | 0.538 | 15 | 16 |
| alphafold | `exp020_shell6_chem_equal` | 0.250 | 0.820 | 0.698 | +0.122 | 0.839 | 0.802 | 0.571 | 14 | 18 |
| alphafold | `exp021_shell6_chem_20_80` | 0.250 | 0.818 | 0.715 | +0.103 | 0.847 | 0.791 | 0.593 | 13 | 19 |
| alphafold | `exp019_five_way` | 0.250 | 0.807 | 0.623 | +0.184 | 0.863 | 0.758 | 0.473 | 11 | 22 |
| apo_pdb | `exp021_shell6_chem_20_80` | 0.300 | 0.800 | 0.717 | +0.083 | 0.825 | 0.776 | 0.642 | 11 | 15 |
| apo_pdb | `exp019_five_way` | 0.250 | 0.794 | 0.649 | +0.145 | 0.812 | 0.776 | 0.537 | 12 | 15 |
| apo_pdb | `exp020_shell6_chem_equal` | 0.350 | 0.788 | 0.717 | +0.071 | 0.800 | 0.776 | 0.642 | 13 | 15 |
| apo_pdb | `exp018_shape3_shape6_shell6` | 0.300 | 0.776 | 0.712 | +0.064 | 0.776 | 0.776 | 0.627 | 15 | 15 |

## Default 0.5 Threshold

| holdout | condition | f1 | precision | recall | tp | fp | fn | predicted_lipid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| alphafold | `exp021_shell6_chem_20_80` | 0.715 | 0.900 | 0.593 | 54 | 6 | 37 | 60 |
| alphafold | `exp020_shell6_chem_equal` | 0.698 | 0.897 | 0.571 | 52 | 6 | 39 | 58 |
| alphafold | `exp018_shape3_shape6_shell6` | 0.671 | 0.891 | 0.538 | 49 | 6 | 42 | 55 |
| alphafold | `exp019_five_way` | 0.623 | 0.915 | 0.473 | 43 | 4 | 48 | 47 |
| apo_pdb | `exp020_shell6_chem_equal` | 0.717 | 0.811 | 0.642 | 43 | 10 | 24 | 53 |
| apo_pdb | `exp021_shell6_chem_20_80` | 0.717 | 0.811 | 0.642 | 43 | 10 | 24 | 53 |
| apo_pdb | `exp018_shape3_shape6_shell6` | 0.712 | 0.824 | 0.627 | 42 | 9 | 25 | 51 |
| apo_pdb | `exp019_five_way` | 0.649 | 0.818 | 0.537 | 36 | 8 | 31 | 44 |
