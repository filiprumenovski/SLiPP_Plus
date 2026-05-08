# SLiPP++ Day 1 metrics

_Feature set: `v_tunnel`, 25 stratified shuffle iterations, pipeline mode: `flat`._

## 1. Binary-collapsed on test split (paper Table 1 line 1)

| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |
|---|---|---|---|---|---|---|
| paper (RF) | 0.869 | 0.970 | 0.968 | 0.926 | 0.818 | - |
| lgbm | 0.895 ± 0.015 | 0.988 ± 0.004 | 0.974 | 0.922 | 0.870 | 0.989 |
| rf | 0.817 ± 0.022 | 0.977 ± 0.006 | 0.958 | 0.955 | 0.715 | 0.995 |
| xgb | 0.893 ± 0.016 | 0.988 ± 0.003 | 0.973 | 0.924 | 0.865 | 0.989 |

## 2. Holdouts (paper Table 1 lines 2-3)

### apo-PDB holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.726 | 0.828 | - | - |
| rf | n/a ("holdout missing feature columns: ['tunnel_pocket_context_present', 'tunnel_caver_profile_present', 'tunnel_has_tunnel']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| xgb | n/a ("holdout missing feature columns: ['tunnel_pocket_context_present', 'tunnel_caver_profile_present', 'tunnel_has_tunnel']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| lgbm | n/a ("holdout missing feature columns: ['tunnel_pocket_context_present', 'tunnel_caver_profile_present', 'tunnel_has_tunnel']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |

### AlphaFold holdout

| model | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|
| paper (RF) | 0.643 | 0.851 | - | - |
| rf | n/a ("holdout missing feature columns: ['tunnel_pocket_context_present', 'tunnel_caver_profile_present', 'tunnel_has_tunnel']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| xgb | n/a ("holdout missing feature columns: ['tunnel_pocket_context_present', 'tunnel_caver_profile_present', 'tunnel_has_tunnel']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |
| lgbm | n/a ("holdout missing feature columns: ['tunnel_pocket_context_present', 'tunnel_caver_profile_present', 'tunnel_has_tunnel']. Holdouts only support feature_set=v14. Retrain with feature_set=v14 to evaluate holdouts.") | | | |

## 3. Multi-class (the headline Day 1 result)

| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |
|---|---|---|---|
| lgbm | 0.732 ± 0.017 | 0.606 ± 0.030 | 0.901 |
| rf | 0.657 ± 0.020 | 0.541 ± 0.031 | 0.860 |
| xgb | 0.727 ± 0.022 | 0.600 ± 0.037 | 0.899 |

### Per-class F1 (mean across iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 0.823 | 0.861 | 0.780 | 0.758 | 0.851 | 0.675 | 0.574 | 0.625 | 0.972 | 0.399 |
| rf | 0.755 | 0.732 | 0.661 | 0.681 | 0.780 | 0.675 | 0.399 | 0.620 | 0.932 | 0.332 |
| xgb | 0.813 | 0.857 | 0.777 | 0.745 | 0.850 | 0.674 | 0.566 | 0.622 | 0.972 | 0.393 |
