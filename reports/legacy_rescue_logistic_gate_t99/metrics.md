# Legacy Rescue Logistic Gate, 2026-05-09

Holdout-safe logistic gate trained on internal split prediction features only. The gate scores rows where the deployable exp-028 ensemble is below the binary lipid threshold; fired rows are rewritten to the mean `paper17_family_encoder` and `v_sterol` class probabilities.

Selected threshold: `0.99`

| metric | value |
|---|---:|
| internal binary F1 | 0.901 +/- 0.017 |
| internal AUROC | 0.989 +/- 0.003 |
| internal macro10 F1 | 0.766 +/- 0.018 |
| internal lipid5 macro-F1 | 0.667 |
| internal fire rate | 1.0% |
| apo-PDB F1/AUROC | 0.727 / 0.802 |
| apo-PDB fire rate | 6.0% |
| AlphaFold F1/AUROC | 0.752 / 0.854 |
| AlphaFold fire rate | 9.4% |

Threshold sweep is saved beside this report as `threshold_sweep.csv`.

Decision: this stricter threshold is useful as an internal-selection diagnostic, but it does not beat exp-035. It improves over exp-028 on both external F1 scores and preserves slightly better internal F1 than the `0.95` gate, but apo-PDB F1 falls from exp-035's `0.742` to `0.727` and AlphaFold F1 falls from `0.775` to `0.752`.
