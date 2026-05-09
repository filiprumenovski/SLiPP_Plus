# SLiPP++ Day 1 metrics

_Feature set: `v_sterol`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| composite | 0.900 ± 0.015 | 0.989 ± 0.003 | 0.975 | 0.927 | 0.875 | 0.990 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| composite | n/a (artifact is not a hierarchical bundle) | | | |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| composite | n/a (artifact is not a hierarchical bundle) | | | |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| composite | 0.759 ± 0.017 | 0.654 ± 0.031 | 0.907 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.830 | 0.848 | 0.803 | 0.720 | 0.860 | 0.695 | 0.573 | 0.657 | 0.975 | 0.626 |
