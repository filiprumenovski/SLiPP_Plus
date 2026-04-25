# SLiPP++ Day 1 metrics

_Feature set: `v_sterol_v2`, 5 stratified shuffle iterations, pipeline mode: `hierarchical`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| hierarchical | 0.897 ± 0.012 | 0.985 ± 0.006 | 0.974 | 0.940 | 0.858 | 0.992 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.697 | 0.821 | 0.905 | 0.567 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.694 | 0.845 | 0.911 | 0.560 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| hierarchical | 0.727 ± 0.007 | 0.609 ± 0.016 | 0.894 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical | 0.788 | 0.856 | 0.761 | 0.684 | 0.847 | 0.643 | 0.494 | 0.632 | 0.970 | 0.592 |
