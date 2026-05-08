# SLiPP++ Day 1 metrics

_Feature set: `v14+aa20+shell6+tunnel_shape3`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| composite | 0.898 ± 0.016 | ± 0.006 | 0.988 ± 0.003 | ± 0.001 | 0.974 | 0.925 | 0.872 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.655 | 0.796 | 0.804 | 0.552 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.657 | 0.867 | 0.939 | 0.505 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| composite | 0.764 ± 0.021 | ± 0.008 | 0.666 ± 0.033 | ± 0.013 | 0.908 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.825 | 0.843 | 0.811 | 0.750 | 0.859 | 0.703 | 0.598 | 0.639 | 0.974 | 0.640 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.885 | 0.894 | 0.866 | 0.792 | 0.835 | 0.659 | 0.602 | 0.760 | 0.968 | 0.587 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.778 | 0.800 | 0.765 | 0.719 | 0.884 | 0.761 | 0.599 | 0.556 | 0.981 | 0.720 |
