# SLiPP++ v49 ensemble metrics

_Probability-averaging ensemble of RF + XGB + LGBM, 25 stratified shuffle iterations._

## Headline metrics (mean ± std across 25 iterations)

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.666 ± 0.016 | 0.547 ± 0.027 | 0.819 ± 0.022 | 0.978 ± 0.005 | 0.696 ± 0.050 | 0.329 ± 0.077 |
| v49 xgb only | 0.726 ± 0.015 | 0.594 ± 0.028 | 0.896 ± 0.017 | 0.988 ± 0.003 | 0.709 ± 0.050 | 0.416 ± 0.095 |
| v49 lgbm only | 0.730 ± 0.017 | 0.597 ± 0.028 | 0.896 ± 0.016 | 0.989 ± 0.003 | 0.714 ± 0.044 | 0.402 ± 0.094 |
| v49 ensemble (mean prob) | 0.734 ± 0.016 | 0.601 ± 0.027 | 0.899 ± 0.015 | 0.986 ± 0.004 | 0.728 ± 0.050 | 0.398 ± 0.095 |

## Per-class F1 (mean across iterations)

| condition | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.766 | 0.749 | 0.683 | 0.696 | 0.794 | 0.685 | 0.406 | 0.619 | 0.935 | 0.329 |
| v49 xgb only | 0.822 | 0.861 | 0.780 | 0.709 | 0.852 | 0.689 | 0.530 | 0.627 | 0.972 | 0.416 |
| v49 lgbm only | 0.834 | 0.862 | 0.790 | 0.714 | 0.853 | 0.697 | 0.535 | 0.635 | 0.972 | 0.402 |
| v49 ensemble | 0.837 | 0.865 | 0.800 | 0.728 | 0.855 | 0.700 | 0.543 | 0.636 | 0.973 | 0.398 |

## CLR vs STE confusion counts (summed over 25 iterations)

| condition | CLR→STE | STE→CLR | CLR correct | STE correct |
|---|---:|---:|---:|---:|
| v49 rf only | 0 | 3 | 517 | 97 |
| v49 xgb only | 0 | 9 | 608 | 141 |
| v49 lgbm only | 1 | 5 | 610 | 133 |
| v49 ensemble | 0 | 5 | 615 | 131 |
