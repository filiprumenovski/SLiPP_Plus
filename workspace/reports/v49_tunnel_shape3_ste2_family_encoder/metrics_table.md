# SLiPP++ Day 1 metrics

_Feature set: `v49+tunnel_shape3`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| composite | 0.896 ± 0.016 | ± 0.007 | 0.988 ± 0.003 | ± 0.001 | 0.974 | 0.925 | 0.870 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.649 | 0.803 | 0.818 | 0.537 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.711 | 0.870 | 0.914 | 0.582 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| composite | 0.760 ± 0.019 | ± 0.008 | 0.657 ± 0.027 | ± 0.011 | 0.906 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.830 | 0.848 | 0.804 | 0.734 | 0.859 | 0.697 | 0.600 | 0.627 | 0.974 | 0.629 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.878 | 0.892 | 0.844 | 0.785 | 0.843 | 0.655 | 0.619 | 0.744 | 0.967 | 0.555 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.788 | 0.812 | 0.770 | 0.696 | 0.875 | 0.749 | 0.592 | 0.546 | 0.982 | 0.741 |
