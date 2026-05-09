# SLiPP++ Day 1 metrics

_Feature set: `v49+tunnel_shape`, 25 stratified shuffle iterations, pipeline mode: `flat`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.895 ± 0.016 | 0.988 ± 0.004 | 0.973 | 0.916 | 0.875 | 0.988 |
| rf | 0.842 ± 0.022 | 0.979 ± 0.005 | 0.963 | 0.958 | 0.752 | 0.995 |
| xgb | 0.894 ± 0.015 | 0.988 ± 0.003 | 0.973 | 0.920 | 0.870 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | n/a ("holdout missing feature columns: ['tunnel_has_tunnel', 'tunnel_caver_profile_present']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| xgb | n/a ("holdout missing feature columns: ['tunnel_has_tunnel', 'tunnel_caver_profile_present']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| lgbm | n/a ("holdout missing feature columns: ['tunnel_has_tunnel', 'tunnel_caver_profile_present']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | n/a ("holdout missing feature columns: ['tunnel_has_tunnel', 'tunnel_caver_profile_present']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| xgb | n/a ("holdout missing feature columns: ['tunnel_has_tunnel', 'tunnel_caver_profile_present']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| lgbm | n/a ("holdout missing feature columns: ['tunnel_has_tunnel', 'tunnel_caver_profile_present']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| lgbm | 0.736 ± 0.021 | 0.612 ± 0.036 | 0.902 |
| rf | 0.684 ± 0.023 | 0.568 ± 0.033 | 0.873 |
| xgb | 0.732 ± 0.019 | 0.603 ± 0.034 | 0.901 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.828 | 0.860 | 0.784 | 0.758 | 0.854 | 0.689 | 0.562 | 0.637 | 0.974 | 0.414 |
| rf | 0.772 | 0.775 | 0.706 | 0.707 | 0.801 | 0.686 | 0.489 | 0.630 | 0.944 | 0.328 |
| xgb | 0.831 | 0.858 | 0.786 | 0.749 | 0.853 | 0.671 | 0.552 | 0.631 | 0.973 | 0.412 |
