# SLiPP++ Day 1 metrics

_Feature set: `v_tunnel`, 25 stratified shuffle iterations._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.892 ± 0.017 | 0.988 ± 0.003 | 0.973 | 0.920 | 0.865 | 0.989 |
| rf | 0.812 ± 0.022 | 0.977 ± 0.005 | 0.958 | 0.954 | 0.707 | 0.995 |
| xgb | 0.892 ± 0.018 | 0.988 ± 0.003 | 0.973 | 0.924 | 0.862 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | 0.679 | 0.833 | 0.923 | 0.537 |
| xgb | 0.617 | 0.763 | 0.825 | 0.493 |
| lgbm | 0.655 | 0.787 | 0.837 | 0.537 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | 0.667 | 0.878 | 0.940 | 0.516 |
| xgb | 0.657 | 0.810 | 0.904 | 0.516 |
| lgbm | 0.662 | 0.824 | 0.922 | 0.516 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| lgbm | 0.728 ± 0.017 | 0.596 ± 0.031 | 0.900 |
| rf | 0.655 ± 0.021 | 0.539 ± 0.034 | 0.859 |
| xgb | 0.726 ± 0.020 | 0.593 ± 0.034 | 0.900 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.830 | 0.861 | 0.781 | 0.722 | 0.853 | 0.676 | 0.543 | 0.619 | 0.973 | 0.419 |
| rf | 0.759 | 0.725 | 0.661 | 0.678 | 0.777 | 0.684 | 0.390 | 0.613 | 0.932 | 0.329 |
| xgb | 0.825 | 0.855 | 0.789 | 0.728 | 0.850 | 0.666 | 0.538 | 0.622 | 0.973 | 0.411 |
