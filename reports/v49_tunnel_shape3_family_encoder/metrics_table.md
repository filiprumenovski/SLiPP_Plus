# SLiPP++ Day 1 metrics

_Feature set: `v49+tunnel_shape3`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| composite | 0.900 ± 0.015 | ± 0.006 | 0.988 ± 0.004 | ± 0.001 | 0.975 | 0.927 | 0.875 | 0.990 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.667 | 0.801 | 0.809 | 0.567 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.724 | 0.862 | 0.902 | 0.604 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| composite | 0.768 ± 0.018 | ± 0.007 | 0.668 ± 0.031 | ± 0.013 | 0.909 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.841 | 0.850 | 0.813 | 0.747 | 0.861 | 0.700 | 0.610 | 0.642 | 0.975 | 0.638 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.907 | 0.888 | 0.856 | 0.790 | 0.840 | 0.650 | 0.629 | 0.759 | 0.968 | 0.576 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.787 | 0.819 | 0.777 | 0.712 | 0.883 | 0.763 | 0.600 | 0.559 | 0.981 | 0.733 |
