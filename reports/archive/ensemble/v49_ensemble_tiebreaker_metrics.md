# SLiPP++ v49 ensemble + CLR/STE tiebreaker metrics

_Probability-averaging ensemble of RF + XGB + LGBM with a binary CLR-vs-STE tiebreaker (margin < 0.15), 25 stratified shuffle iterations._

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | STE F1 | CLR recall | STE recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.684 ± 0.020 | 0.562 ± 0.030 | 0.837 ± 0.020 | 0.980 ± 0.005 | 0.700 ± 0.047 | 0.340 ± 0.093 | 0.589 | 0.272 |
| v49 xgb only | 0.725 ± 0.017 | 0.590 ± 0.033 | 0.895 ± 0.015 | 0.988 ± 0.003 | 0.708 ± 0.053 | 0.415 ± 0.089 | 0.669 | 0.389 |
| v49 lgbm only | 0.724 ± 0.015 | 0.588 ± 0.029 | 0.893 ± 0.013 | 0.988 ± 0.004 | 0.716 ± 0.042 | 0.387 ± 0.101 | 0.678 | 0.355 |
| v49 ensemble (mean prob) | 0.731 ± 0.016 | 0.596 ± 0.030 | 0.898 ± 0.016 | 0.986 ± 0.004 | 0.719 ± 0.047 | 0.393 ± 0.087 | 0.671 | 0.357 |
| v49 ensemble + tiebreaker | 0.731 ± 0.016 | 0.596 ± 0.030 | 0.898 ± 0.016 | 0.986 ± 0.004 | 0.719 ± 0.047 | 0.393 ± 0.087 | 0.671 | 0.357 |

## CLR vs STE confusion counts (summed over 25 iterations)

| condition | CLR correct | STE correct | CLR→STE | STE→CLR |
|---|---:|---:|---:|---:|
| v49 rf only | 530 | 102 | 0 | 4 |
| v49 xgb only | 602 | 146 | 3 | 3 |
| v49 lgbm only | 610 | 133 | 4 | 4 |
| v49 ensemble (mean prob) | 604 | 134 | 2 | 4 |
| v49 ensemble + tiebreaker | 604 | 134 | 2 | 4 |

## Tiebreaker diagnostics

- Tiebreaker fired: mean = 0.1 rows/iter (std 0.3), total = 2 over 25 iterations
- Per-iteration fire counts: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
- Tiebreaker sterol-only binary F1 (CLR vs STE, on true sterol test rows): 0.874 ± 0.057

### Where do true STE rows get misclassified? (ensemble, summed over 25 iters)

| predicted class | count |
|---|---:|
| PLM | 155 |
| COA | 23 |
| PP | 22 |
| MYR | 18 |
| OLA | 17 |
| CLR | 4 |
| BGC | 2 |

_Note: the tiebreaker only fires when both CLR and STE are in the ensemble's top-2. If STE loses to PP, COA, or OLA instead of CLR, it is outside the tiebreaker's scope._

## Tiebreaker iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | volume_score | 11.0206 |
| 2 | TYR | 7.3722 |
| 3 | ARG | 6.2221 |
| 4 | mean_as_solv_acc | 5.7738 |
| 5 | polarity_score | 5.7037 |
| 6 | TRP | 5.4057 |
| 7 | prop_polar_atm | 4.2917 |
| 8 | CYS | 3.6637 |
| 9 | surf_apol_vdw14 | 3.4907 |
| 10 | PHE | 3.4089 |
| 11 | LEU | 3.3776 |
| 12 | nb_AS | 3.2737 |
| 13 | ILE | 3.2465 |
| 14 | VAL | 3.2169 |
| 15 | LYS | 3.1816 |
