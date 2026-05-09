# SLiPP++ v_sterol ensemble + PLM/STE tiebreaker metrics

_Probability-averaging ensemble of RF + XGB + LGBM with a binary PLM-vs-STE tiebreaker (margin < 0.99), 25 stratified shuffle iterations._

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | PLM F1 | STE F1 | PLM recall | STE recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol rf only | 0.666 ± 0.016 | 0.547 ± 0.027 | 0.819 ± 0.022 | 0.978 ± 0.005 | 0.619 ± 0.045 | 0.329 ± 0.077 | 0.582 | 0.259 |
| v_sterol xgb only | 0.726 ± 0.015 | 0.594 ± 0.028 | 0.896 ± 0.017 | 0.988 ± 0.003 | 0.627 ± 0.039 | 0.416 ± 0.095 | 0.608 | 0.376 |
| v_sterol lgbm only | 0.730 ± 0.017 | 0.597 ± 0.028 | 0.896 ± 0.016 | 0.989 ± 0.003 | 0.635 ± 0.048 | 0.402 ± 0.094 | 0.627 | 0.355 |
| v_sterol ensemble (mean prob) | 0.734 ± 0.016 | 0.601 ± 0.027 | 0.899 ± 0.015 | 0.986 ± 0.004 | 0.636 ± 0.046 | 0.398 ± 0.095 | 0.623 | 0.349 |
| v_sterol ensemble + plm_ste_tiebreaker | 0.738 ± 0.015 | 0.610 ± 0.026 | 0.899 ± 0.015 | 0.986 ± 0.004 | 0.638 ± 0.046 | 0.444 ± 0.107 | 0.619 | 0.408 |

## PLM vs STE confusion counts (summed over 25 iterations)

| condition | PLM correct | STE correct | PLM→STE | STE→PLM |
|---|---:|---:|---:|---:|
| v_sterol rf only | 1047 | 97 | 109 | 136 |
| v_sterol xgb only | 1094 | 141 | 128 | 144 |
| v_sterol lgbm only | 1129 | 133 | 123 | 150 |
| v_sterol ensemble (mean prob) | 1122 | 131 | 123 | 155 |
| v_sterol ensemble + plm_ste_tiebreaker | 1115 | 153 | 130 | 133 |

## Tiebreaker diagnostics

- Tiebreaker fired: mean = 12.6 rows/iter (std 3.3), total = 315 over 25 iterations
- Per-iteration fire counts: 6, 12, 12, 12, 16, 14, 7, 14, 8, 12, 13, 20, 14, 10, 15, 8, 14, 10, 9, 18, 13, 13, 15, 13, 17
- Tiebreaker PLM/STE-only binary F1 (STE=positive, on true PLM+STE test rows): 0.492 ± 0.084

### Where do true STE rows get misclassified? (ensemble + tiebreaker, summed over all iterations)

| predicted class | count |
|---|---:|
| PLM | 133 |
| PP | 25 |
| COA | 21 |
| OLA | 20 |
| MYR | 18 |
| CLR | 5 |

_Note: the tiebreaker only fires when both PLM and STE are in the ensemble's top-2. If STE loses to CLR, OLA, or another class instead of PLM, it is outside this tiebreaker's scope._

## Tiebreaker iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | LEU | 30.2657 |
| 2 | ASN | 14.3759 |
| 3 | hydrophobicity_score | 13.5733 |
| 4 | cationic_count_shell3 | 10.1366 |
| 5 | cationic_count_shell4 | 10.0549 |
| 6 | cationic_count_shell1 | 8.6711 |
| 7 | GLY | 8.5587 |
| 8 | TYR | 7.5303 |
| 9 | GLU | 6.7688 |
| 10 | as_max_dst | 6.7448 |
| 11 | polar_neutral_count_shell4 | 6.7176 |
| 12 | volume_score | 6.4985 |
| 13 | aromatic_aliphatic_ratio_shell3 | 6.4214 |
| 14 | LYS | 6.3374 |
| 15 | GLN | 6.2927 |
