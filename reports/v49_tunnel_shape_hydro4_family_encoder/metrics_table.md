# SLiPP++ Day 1 metrics

_Feature set: `v49+tunnel_shape_hydro4`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| composite | 0.897 ± 0.017 | ± 0.007 | 0.988 ± 0.003 | ± 0.001 | 0.974 | 0.925 | 0.872 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.649 | 0.805 | 0.818 | 0.537 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.632 | 0.864 | 0.956 | 0.473 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| composite | 0.762 ± 0.021 | ± 0.009 | 0.661 ± 0.032 | ± 0.013 | 0.907 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.833 | 0.840 | 0.806 | 0.743 | 0.858 | 0.696 | 0.587 | 0.636 | 0.975 | 0.640 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.898 | 0.881 | 0.851 | 0.770 | 0.838 | 0.655 | 0.608 | 0.741 | 0.968 | 0.587 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.780 | 0.808 | 0.770 | 0.726 | 0.879 | 0.747 | 0.572 | 0.561 | 0.982 | 0.715 |
