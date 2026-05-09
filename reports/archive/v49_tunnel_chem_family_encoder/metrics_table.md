# SLiPP++ Day 1 metrics

_Feature set: `v49+tunnel_chem`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| composite | 0.900 ± 0.018 | 0.988 ± 0.004 | 0.975 | 0.925 | 0.876 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.706 | 0.797 | 0.808 | 0.627 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.698 | 0.853 | 0.897 | 0.571 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| composite | 0.757 ± 0.020 | 0.655 ± 0.032 | 0.906 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.830 | 0.834 | 0.802 | 0.747 | 0.858 | 0.691 | 0.581 | 0.634 | 0.974 | 0.621 |
