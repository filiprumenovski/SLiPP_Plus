# SLiPP++ Day 1 metrics

_Feature set: `v14+shell+tunnel_shape`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| composite | 0.879 ± 0.013 | 0.984 ± 0.003 | 0.970 | 0.916 | 0.846 | 0.988 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.727 | 0.816 | 0.815 | 0.657 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.719 | 0.848 | 0.887 | 0.604 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| composite | 0.698 ± 0.021 | 0.589 ± 0.036 | 0.886 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.756 | 0.753 | 0.743 | 0.684 | 0.812 | 0.644 | 0.509 | 0.578 | 0.972 | 0.533 |
