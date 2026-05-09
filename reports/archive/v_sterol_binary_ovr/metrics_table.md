# SLiPP++ Day 1 metrics

_Feature set: `v_sterol`, 25 stratified shuffle iterations, pipeline mode: `hierarchical`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| hierarchical | 0.896 ± 0.017 | 0.986 ± 0.004 | 0.974 | 0.942 | 0.855 | 0.992 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.679 | 0.812 | 0.844 | 0.567 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.708 | 0.864 | 0.962 | 0.560 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| hierarchical | 0.726 ± 0.018 | 0.625 ± 0.033 | 0.891 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical | 0.790 | 0.810 | 0.741 | 0.707 | 0.830 | 0.688 | 0.523 | 0.630 | 0.968 | 0.576 |
