# SLiPP++ Day 1 metrics

_Feature set: `v_sterol`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| composite | 0.901 ± 0.016 | ± 0.006 | 0.989 ± 0.003 | ± 0.001 | 0.975 | 0.920 | 0.884 | 0.988 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.723 | 0.807 | 0.827 | 0.642 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.703 | 0.838 | 0.912 | 0.571 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| composite | 0.762 ± 0.015 | ± 0.006 | 0.660 ± 0.029 | ± 0.012 | 0.908 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.828 | 0.852 | 0.802 | 0.727 | 0.863 | 0.694 | 0.587 | 0.658 | 0.975 | 0.635 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.896 | 0.906 | 0.859 | 0.746 | 0.836 | 0.658 | 0.600 | 0.740 | 0.971 | 0.590 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.773 | 0.810 | 0.755 | 0.712 | 0.891 | 0.738 | 0.579 | 0.594 | 0.980 | 0.707 |
