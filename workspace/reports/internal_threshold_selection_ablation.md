# Internal Threshold Selection Ablation

This ablation learns one binary lipid-probability threshold per compact ensemble from the 25 internal split predictions, then applies that threshold once to apo-PDB and AlphaFold. This tests whether the low-threshold holdout gains can be recovered without tuning on final holdouts.

## Key Signal

- Internal split optimization chooses thresholds near the default (`0.51`-`0.54`), not the low `0.25`-`0.35` thresholds that maximize holdout F1 diagnostically.
- Applying those internally selected thresholds does not improve external holdouts; it slightly worsens every checked condition.
- Therefore the holdout threshold ablation is evidence of domain shift/calibration mismatch between internal splits and external structures, not a directly transferable threshold-selection recipe.

## Internal Thresholds

| condition | selected_threshold | mean_split_f1 | split_f1_std |
|---|---:|---:|---:|
| `exp018_shape3_shape6_shell6` | 0.530 | 0.902 | 0.018 |
| `exp019_five_way` | 0.540 | 0.905 | 0.017 |
| `exp020_shell6_chem_equal` | 0.510 | 0.904 | 0.017 |
| `exp021_shell6_chem_20_80` | 0.520 | 0.904 | 0.018 |

## Holdout Application

| holdout | condition | selected_threshold | default_f1 | calibrated_f1 | delta | default_recall | calibrated_recall | calibrated_fp | calibrated_fn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alphafold | `exp021_shell6_chem_20_80` | 0.520 | 0.715 | 0.680 | -0.035 | 0.593 | 0.549 | 6 | 41 |
| alphafold | `exp018_shape3_shape6_shell6` | 0.530 | 0.671 | 0.667 | -0.005 | 0.538 | 0.527 | 5 | 43 |
| alphafold | `exp020_shell6_chem_equal` | 0.510 | 0.698 | 0.662 | -0.036 | 0.571 | 0.527 | 6 | 43 |
| alphafold | `exp019_five_way` | 0.540 | 0.623 | 0.607 | -0.016 | 0.473 | 0.451 | 3 | 50 |
| apo_pdb | `exp020_shell6_chem_equal` | 0.510 | 0.717 | 0.706 | -0.011 | 0.642 | 0.627 | 10 | 25 |
| apo_pdb | `exp021_shell6_chem_20_80` | 0.520 | 0.717 | 0.684 | -0.033 | 0.642 | 0.597 | 10 | 27 |
| apo_pdb | `exp018_shape3_shape6_shell6` | 0.530 | 0.712 | 0.678 | -0.034 | 0.627 | 0.582 | 9 | 28 |
| apo_pdb | `exp019_five_way` | 0.540 | 0.649 | 0.636 | -0.012 | 0.537 | 0.522 | 8 | 32 |
