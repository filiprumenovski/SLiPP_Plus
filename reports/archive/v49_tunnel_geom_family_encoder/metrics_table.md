# SLiPP++ Day 1 metrics

_Feature set: `v49+tunnel_geom`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| composite | 0.901 ± 0.017 | 0.988 ± 0.004 | 0.975 | 0.931 | 0.872 | 0.990 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.655 | 0.793 | 0.837 | 0.537 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.602 | 0.868 | 0.952 | 0.440 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| composite | 0.765 ± 0.020 | 0.665 ± 0.036 | 0.908 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.837 | 0.843 | 0.810 | 0.754 | 0.857 | 0.693 | 0.594 | 0.647 | 0.974 | 0.637 |
