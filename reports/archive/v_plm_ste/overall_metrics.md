# SLiPP++ v_plm_ste ensemble + PLM/STE tiebreaker metrics

_Probability-averaging ensemble of RF + XGB + LGBM with a binary PLM-vs-STE tiebreaker (margin < 0.99), 25 stratified shuffle iterations._

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | PLM F1 | STE F1 | PLM recall | STE recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_plm_ste rf only | 0.653 ± 0.019 | 0.532 ± 0.031 | 0.806 ± 0.022 | 0.977 ± 0.005 | 0.616 ± 0.046 | 0.317 ± 0.080 | 0.579 | 0.248 |
| v_plm_ste xgb only | 0.717 ± 0.018 | 0.578 ± 0.034 | 0.893 ± 0.014 | 0.988 ± 0.003 | 0.617 ± 0.042 | 0.384 ± 0.085 | 0.598 | 0.341 |
| v_plm_ste lgbm only | 0.725 ± 0.015 | 0.591 ± 0.025 | 0.893 ± 0.017 | 0.988 ± 0.003 | 0.630 ± 0.042 | 0.389 ± 0.085 | 0.628 | 0.341 |
| v_plm_ste ensemble (mean prob) | 0.727 ± 0.017 | 0.593 ± 0.030 | 0.895 ± 0.016 | 0.986 ± 0.004 | 0.631 ± 0.042 | 0.388 ± 0.088 | 0.614 | 0.336 |
| v_plm_ste ensemble + plm_ste_tiebreaker | 0.728 ± 0.013 | 0.596 ± 0.022 | 0.895 ± 0.016 | 0.986 ± 0.004 | 0.631 ± 0.038 | 0.402 ± 0.071 | 0.613 | 0.352 |

## PLM vs STE confusion counts (summed over 25 iterations)

| condition | PLM correct | STE correct | PLM→STE | STE→PLM |
|---|---:|---:|---:|---:|
| v_plm_ste rf only | 1042 | 93 | 107 | 133 |
| v_plm_ste xgb only | 1076 | 128 | 128 | 147 |
| v_plm_ste lgbm only | 1130 | 128 | 121 | 153 |
| v_plm_ste ensemble (mean prob) | 1106 | 126 | 121 | 153 |
| v_plm_ste ensemble + plm_ste_tiebreaker | 1103 | 132 | 124 | 147 |

## Tiebreaker diagnostics

- Tiebreaker fired: mean = 12.0 rows/iter (std 3.2), total = 299 over 25 iterations
- Per-iteration fire counts: 8, 13, 12, 10, 14, 12, 11, 11, 6, 10, 12, 19, 16, 10, 14, 9, 11, 6, 10, 18, 12, 16, 12, 11, 16
- Tiebreaker PLM/STE-only binary F1 (STE=positive, on true PLM+STE test rows): 0.455 ± 0.073

### Where do true STE rows get misclassified? (ensemble + tiebreaker, summed over all iterations)

| predicted class | count |
|---|---:|
| PLM | 147 |
| COA | 28 |
| PP | 27 |
| OLA | 18 |
| MYR | 16 |
| CLR | 6 |
| BGC | 1 |

_Note: the tiebreaker only fires when both PLM and STE are in the ensemble's top-2. If STE loses to CLR, OLA, or another class instead of PLM, it is outside this tiebreaker's scope._

## Tiebreaker iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | LEU | 30.9892 |
| 2 | cationic_count_shell3 | 14.7887 |
| 3 | hydrophobicity_score | 14.0080 |
| 4 | ASN | 10.0342 |
| 5 | GLN | 9.6022 |
| 6 | cationic_count_shell1 | 9.1640 |
| 7 | axial_radius_std | 9.1276 |
| 8 | bulky_hydrophobic_count_shell3 | 6.9866 |
| 9 | volume_score | 6.9114 |
| 10 | motif_residue_density | 6.5981 |
| 11 | aromatic_polar_count_shell4 | 6.5385 |
| 12 | LYS | 6.5324 |
| 13 | HIS | 6.2152 |
| 14 | TYR | 6.0975 |
| 15 | polar_neutral_count_shell4 | 6.0234 |
