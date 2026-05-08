# SLiPP++ Day 1 metrics

_Feature set: `v14+aa20+shell6+tunnel_shape`, 25 stratified shuffle iterations, pipeline mode: `composite`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | - | 0.970 | - | 0.968 | 0.926 | 0.818 | - |
| composite | 0.900 ± 0.017 | ± 0.007 | 0.988 ± 0.003 | ± 0.001 | 0.975 | 0.931 | 0.871 | 0.990 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| hierarchical | 0.711 | 0.798 | 0.796 | 0.642 |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| hierarchical | 0.703 | 0.849 | 0.912 | 0.571 |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |
|---|---|---|---|---|---|
| composite | 0.766 ± 0.019 | ± 0.008 | 0.666 ± 0.031 | ± 0.013 | 0.909 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.833 | 0.852 | 0.811 | 0.751 | 0.860 | 0.704 | 0.584 | 0.648 | 0.975 | 0.642 |

### Per-class precision (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.890 | 0.892 | 0.860 | 0.812 | 0.842 | 0.663 | 0.587 | 0.762 | 0.967 | 0.588 |

### Per-class recall (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| composite | 0.786 | 0.821 | 0.771 | 0.703 | 0.880 | 0.753 | 0.592 | 0.565 | 0.983 | 0.720 |
