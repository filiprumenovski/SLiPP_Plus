# Legacy Rescue Logistic Gate, 2026-05-09

Holdout-safe logistic gate trained on internal split prediction features only. The gate scores rows where the deployable exp-028 ensemble is below the binary lipid threshold; fired rows are rewritten to the mean `paper17_family_encoder` and `v_sterol` class probabilities.

Selected threshold: `0.95`

| metric | value |
|---|---:|
| internal binary F1 | 0.900 +/- 0.018 |
| internal AUROC | 0.988 +/- 0.004 |
| internal macro10 F1 | 0.766 +/- 0.017 |
| internal lipid5 macro-F1 | 0.666 |
| internal fire rate | 2.1% |
| apo-PDB F1/AUROC | 0.742 / 0.793 |
| apo-PDB fire rate | 15.4% |
| AlphaFold F1/AUROC | 0.775 / 0.857 |
| AlphaFold fire rate | 13.4% |

Threshold sweep is saved beside this report as `threshold_sweep.csv`.

Decision: this first-class run improves both external F1 scores over the exp-028 deployable anchor while keeping internal binary F1 within about 0.003. It should supersede exp-028 as the current deployable recommendation unless a later ablation beats it under the same holdout-safe constraints.
