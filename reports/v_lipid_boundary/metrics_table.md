# SLiPP++ Day 1 metrics

_Feature set: `v_lipid_boundary`, 25 stratified shuffle iterations._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.896 ± 0.018 | 0.989 ± 0.003 | 0.974 | 0.926 | 0.868 | 0.990 |
| rf | 0.801 ± 0.023 | 0.977 ± 0.006 | 0.956 | 0.957 | 0.689 | 0.995 |
| xgb | 0.896 ± 0.016 | 0.988 ± 0.003 | 0.974 | 0.930 | 0.864 | 0.990 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | 0.697 | 0.833 | 0.905 | 0.567 |
| xgb | 0.696 | 0.816 | 0.867 | 0.582 |
| lgbm | 0.712 | 0.795 | 0.824 | 0.627 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | 0.716 | 0.874 | 0.930 | 0.582 |
| xgb | 0.584 | 0.819 | 0.870 | 0.440 |
| lgbm | 0.671 | 0.851 | 0.862 | 0.549 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| lgbm | 0.723 ± 0.016 | 0.590 ± 0.027 | 0.900 |
| rf | 0.642 ± 0.018 | 0.520 ± 0.031 | 0.854 |
| xgb | 0.722 ± 0.017 | 0.586 ± 0.030 | 0.899 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.808 | 0.855 | 0.793 | 0.718 | 0.853 | 0.700 | 0.531 | 0.634 | 0.973 | 0.369 |
| rf | 0.745 | 0.710 | 0.666 | 0.669 | 0.776 | 0.662 | 0.359 | 0.604 | 0.928 | 0.303 |
| xgb | 0.820 | 0.848 | 0.794 | 0.705 | 0.852 | 0.688 | 0.538 | 0.625 | 0.972 | 0.374 |
