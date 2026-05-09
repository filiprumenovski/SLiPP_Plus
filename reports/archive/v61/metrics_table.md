# SLiPP++ Day 1 metrics

_Feature set: `v61`, 25 stratified shuffle iterations._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.894 ± 0.016 | 0.988 ± 0.004 | 0.973 | 0.914 | 0.875 | 0.988 |
| rf | 0.832 ± 0.021 | 0.979 ± 0.006 | 0.961 | 0.958 | 0.736 | 0.995 |
| xgb | 0.893 ± 0.015 | 0.988 ± 0.003 | 0.973 | 0.917 | 0.870 | 0.988 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | 0.745 | 0.839 | 0.953 | 0.612 |
| xgb | 0.690 | 0.807 | 0.848 | 0.582 |
| lgbm | 0.707 | 0.797 | 0.837 | 0.612 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | 0.690 | 0.879 | 0.926 | 0.549 |
| xgb | 0.685 | 0.830 | 0.879 | 0.560 |
| lgbm | 0.685 | 0.841 | 0.879 | 0.560 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| lgbm | 0.726 ± 0.016 | 0.591 ± 0.031 | 0.900 |
| rf | 0.675 ± 0.021 | 0.552 ± 0.030 | 0.870 |
| xgb | 0.722 ± 0.017 | 0.586 ± 0.034 | 0.898 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.837 | 0.859 | 0.785 | 0.711 | 0.853 | 0.682 | 0.537 | 0.628 | 0.973 | 0.398 |
| rf | 0.778 | 0.769 | 0.700 | 0.694 | 0.802 | 0.687 | 0.417 | 0.629 | 0.941 | 0.330 |
| xgb | 0.827 | 0.847 | 0.789 | 0.713 | 0.852 | 0.668 | 0.519 | 0.624 | 0.973 | 0.405 |
