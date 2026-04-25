# SLiPP++ v49 ensemble metrics

_Probability-averaging ensemble of RF + XGB + LGBM, 25 stratified shuffle iterations._

## Headline metrics (mean ± std across 25 iterations)

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.642 ± 0.018 | 0.520 ± 0.030 | 0.801 ± 0.022 | 0.977 ± 0.006 | 0.669 ± 0.050 | 0.303 ± 0.089 |
| v49 xgb only | 0.722 ± 0.017 | 0.586 ± 0.029 | 0.896 ± 0.015 | 0.988 ± 0.003 | 0.705 ± 0.040 | 0.374 ± 0.086 |
| v49 lgbm only | 0.723 ± 0.016 | 0.590 ± 0.027 | 0.896 ± 0.017 | 0.989 ± 0.003 | 0.718 ± 0.038 | 0.369 ± 0.077 |
| v49 ensemble (mean prob) | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.898 ± 0.017 | 0.986 ± 0.004 | 0.723 ± 0.043 | 0.378 ± 0.090 |

## Per-class F1 (mean across iterations)

| condition | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v49 rf only | 0.745 | 0.710 | 0.666 | 0.669 | 0.776 | 0.662 | 0.359 | 0.604 | 0.928 | 0.303 |
| v49 xgb only | 0.820 | 0.848 | 0.794 | 0.705 | 0.852 | 0.688 | 0.538 | 0.625 | 0.972 | 0.374 |
| v49 lgbm only | 0.808 | 0.855 | 0.793 | 0.718 | 0.853 | 0.700 | 0.531 | 0.634 | 0.973 | 0.369 |
| v49 ensemble | 0.819 | 0.860 | 0.797 | 0.723 | 0.854 | 0.701 | 0.559 | 0.636 | 0.972 | 0.378 |

## CLR vs STE confusion counts (summed over 25 iterations)

| condition | CLR→STE | STE→CLR | CLR correct | STE correct |
|---|---:|---:|---:|---:|
| v49 rf only | 0 | 1 | 482 | 87 |
| v49 xgb only | 3 | 12 | 600 | 124 |
| v49 lgbm only | 3 | 6 | 605 | 117 |
| v49 ensemble | 1 | 7 | 608 | 122 |
