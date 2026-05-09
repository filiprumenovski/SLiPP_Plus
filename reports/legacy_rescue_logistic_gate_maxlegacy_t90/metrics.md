# Legacy Rescue Logistic Gate, 2026-05-09

Holdout-safe logistic gate trained on internal split prediction features only. The gate scores rows where the deployable exp-028 ensemble is below the binary lipid threshold.

Selected threshold: `0.90`

Rewrite mode: `maxlegacy`

| metric | value |
|---|---:|
| internal binary F1 | 0.895 +/- 0.015 |
| internal AUROC | 0.988 +/- 0.003 |
| internal macro10 F1 | 0.762 +/- 0.019 |
| internal lipid5 macro-F1 | 0.658 |
| internal fire rate | 3.0% |
| apo-PDB F1/AUROC | 0.756 / 0.791 |
| apo-PDB fire rate | 17.9% |
| AlphaFold F1/AUROC | 0.807 / 0.829 |
| AlphaFold fire rate | 20.1% |

Threshold sweep is saved beside this report as `threshold_sweep.csv`.

Decision: holdout-positive diagnostic only. This beats exp-035 on both external F1 scores, but the `maxlegacy` rewrite mode and `0.90` threshold were found by a diagnostic sweep that inspected holdout labels. Do not promote until the same policy, or a close substitute, can be selected from internal-only evidence.
