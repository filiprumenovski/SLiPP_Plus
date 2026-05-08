# SLiPP++ Day 1 metrics

_Feature set: `v14`, 25 stratified shuffle iterations, pipeline mode: `flat`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.855 ± 0.019 | ± 0.008 | 0.982 ± 0.004 | ± 0.002 | 0.963 | 0.870 | 0.841 | 0.981 |
| rf | 0.825 ± 0.022 | ± 0.009 | 0.976 ± 0.006 | ± 0.002 | 0.960 | 0.943 | 0.733 | 0.993 |
| xgb | 0.860 ± 0.017 | ± 0.007 | 0.982 ± 0.004 | ± 0.002 | 0.964 | 0.883 | 0.838 | 0.983 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | 0.679 | 0.847 | 0.881 | 0.552 |
| xgb | 0.719 | 0.805 | 0.872 | 0.612 |
| lgbm | 0.746 | 0.798 | 0.863 | 0.657 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | 0.732 | 0.855 | 0.903 | 0.615 |
| xgb | 0.692 | 0.796 | 0.809 | 0.604 |
| lgbm | 0.716 | 0.785 | 0.817 | 0.637 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| lgbm | 0.650 ± 0.017 | ± 0.007 | 0.514 ± 0.032 | ± 0.013 | 0.870 |
| rf | 0.620 ± 0.025 | ± 0.010 | 0.495 ± 0.036 | ± 0.015 | 0.861 |
| xgb | 0.650 ± 0.020 | ± 0.008 | 0.514 ± 0.038 | ± 0.016 | 0.873 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.736 | 0.734 | 0.702 | 0.658 | 0.790 | 0.595 | 0.437 | 0.558 | 0.969 | 0.325 |
| rf | 0.708 | 0.668 | 0.638 | 0.659 | 0.764 | 0.590 | 0.413 | 0.564 | 0.954 | 0.248 |
| xgb | 0.733 | 0.723 | 0.707 | 0.669 | 0.795 | 0.588 | 0.436 | 0.564 | 0.971 | 0.313 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.792 | 0.792 | 0.732 | 0.679 | 0.751 | 0.577 | 0.466 | 0.562 | 0.971 | 0.394 |
| rf | 0.916 | 0.954 | 0.928 | 0.778 | 0.715 | 0.613 | 0.621 | 0.604 | 0.922 | 0.397 |
| xgb | 0.788 | 0.789 | 0.751 | 0.702 | 0.759 | 0.572 | 0.482 | 0.571 | 0.969 | 0.365 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.691 | 0.690 | 0.678 | 0.641 | 0.833 | 0.620 | 0.416 | 0.554 | 0.968 | 0.285 |
| rf | 0.580 | 0.521 | 0.490 | 0.573 | 0.821 | 0.574 | 0.314 | 0.529 | 0.988 | 0.189 |
| xgb | 0.690 | 0.672 | 0.673 | 0.642 | 0.836 | 0.610 | 0.401 | 0.559 | 0.973 | 0.283 |
