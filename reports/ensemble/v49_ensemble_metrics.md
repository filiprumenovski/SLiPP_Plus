# SLiPP++ v49 ensemble metrics

_Probability-averaging ensemble of RF + XGB + LGBM, 25 stratified shuffle iterations._

## Headline metrics (mean ± std across 25 iterations)

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.684 ± 0.020 | 0.562 ± 0.030 | 0.837 ± 0.020 | 0.980 ± 0.005 | 0.700 ± 0.047 | 0.340 ± 0.093 |
| v49 xgb only | 0.725 ± 0.017 | 0.590 ± 0.033 | 0.895 ± 0.015 | 0.988 ± 0.003 | 0.708 ± 0.053 | 0.415 ± 0.089 |
| v49 lgbm only | 0.724 ± 0.015 | 0.588 ± 0.029 | 0.893 ± 0.013 | 0.988 ± 0.004 | 0.716 ± 0.042 | 0.387 ± 0.101 |
| v49 ensemble (mean prob) | 0.731 ± 0.016 | 0.596 ± 0.030 | 0.898 ± 0.016 | 0.986 ± 0.004 | 0.719 ± 0.047 | 0.393 ± 0.087 |

## Per-class F1 (mean across iterations)

| condition | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.782 | 0.784 | 0.712 | 0.700 | 0.807 | 0.684 | 0.456 | 0.627 | 0.944 | 0.340 |
| v49 xgb only | 0.829 | 0.857 | 0.793 | 0.708 | 0.854 | 0.682 | 0.522 | 0.623 | 0.973 | 0.415 |
| v49 lgbm only | 0.830 | 0.858 | 0.789 | 0.716 | 0.852 | 0.673 | 0.530 | 0.631 | 0.973 | 0.387 |
| v49 ensemble | 0.837 | 0.866 | 0.800 | 0.719 | 0.859 | 0.684 | 0.547 | 0.636 | 0.974 | 0.393 |

## CLR vs STE confusion counts (summed over 25 iterations)

| condition | CLR→STE | STE→CLR | CLR correct | STE correct |
|---|---:|---:|---:|---:|
| v49 rf only | 0 | 4 | 530 | 102 |
| v49 xgb only | 3 | 3 | 602 | 146 |
| v49 lgbm only | 4 | 4 | 610 | 133 |
| v49 ensemble | 2 | 4 | 604 | 134 |
