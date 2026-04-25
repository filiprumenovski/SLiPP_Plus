# SLiPP++ Day 1 metrics

_Feature set: `v49`, 25 stratified shuffle iterations._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.893 ± 0.013 | 0.988 ± 0.004 | 0.973 | 0.913 | 0.875 | 0.987 |
| rf | 0.837 ± 0.020 | 0.980 ± 0.005 | 0.963 | 0.961 | 0.742 | 0.995 |
| xgb | 0.895 ± 0.015 | 0.988 ± 0.003 | 0.974 | 0.921 | 0.871 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | 0.709 | 0.848 | 0.907 | 0.582 |
| xgb | 0.685 | 0.788 | 0.864 | 0.567 |
| lgbm | 0.696 | 0.786 | 0.833 | 0.597 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | 0.753 | 0.883 | 0.921 | 0.637 |
| xgb | 0.629 | 0.819 | 0.865 | 0.495 |
| lgbm | 0.707 | 0.855 | 0.898 | 0.582 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| lgbm | 0.724 ± 0.016 | 0.588 ± 0.030 | 0.899 |
| rf | 0.684 ± 0.020 | 0.562 ± 0.031 | 0.875 |
| xgb | 0.725 ± 0.017 | 0.590 ± 0.033 | 0.899 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.830 | 0.858 | 0.789 | 0.716 | 0.852 | 0.673 | 0.530 | 0.631 | 0.973 | 0.387 |
| rf | 0.782 | 0.784 | 0.712 | 0.700 | 0.807 | 0.684 | 0.456 | 0.627 | 0.944 | 0.340 |
| xgb | 0.829 | 0.857 | 0.793 | 0.708 | 0.854 | 0.682 | 0.522 | 0.623 | 0.973 | 0.415 |
