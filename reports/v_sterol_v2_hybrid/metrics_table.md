# SLiPP++ Day 1 metrics

_Feature set: `v_sterol_v2`, 25 stratified shuffle iterations, pipeline mode: `hierarchical`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| hierarchical | 0.895 ± 0.017 | 0.986 ± 0.004 | 0.974 | 0.946 | 0.851 | 0.993 |

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
| hierarchical | 0.721 ± 0.018 | 0.616 ± 0.032 | 0.889 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical | 0.791 | 0.804 | 0.740 | 0.690 | 0.829 | 0.674 | 0.519 | 0.615 | 0.968 | 0.581 |
