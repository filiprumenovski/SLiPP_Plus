# Domain-Shift Nearest-Neighbor Ablation

This diagnostic asks whether true-lipid holdout false negatives under `exp-019` are closer, in standardized `v49+tunnel_shape` feature space, to internal lipid pockets or to internal non-lipid/PP pockets. For each holdout pocket, it computes the fraction of lipid labels among the nearest internal neighbors.

## Summary

| holdout | status | n | mean_p_lipid | mean_dist_5 | lipid_frac_5 | lipid_frac_10 | lipid_frac_25 |
|---|---|---:|---:|---:|---:|---:|---:|
| alphafold | FN | 48 | 0.269 | 7.685 | 0.279 | 0.271 | 0.253 |
| alphafold | TP | 43 | 0.753 | 10.119 | 0.712 | 0.702 | 0.641 |
| apo_pdb | FN | 31 | 0.246 | 6.677 | 0.187 | 0.190 | 0.208 |
| apo_pdb | TP | 36 | 0.834 | 9.166 | 0.717 | 0.681 | 0.658 |

## Key Signal

- False negatives are much less lipid-like in local feature space than recalled holdout lipids. Apo-PDB FN nearest-10 lipid fraction is `0.190` vs `0.681` for TP; AlphaFold FN is `0.271` vs `0.702` for TP.
- The hardest false negatives are usually surrounded by internal `PP`/`COA` neighbors, which explains why lowering a global threshold helps recall but does not solve the underlying feature-manifold mismatch.
- This supports a domain-shift interpretation: many external true lipid pockets look like pseudo/non-lipid pockets under the compact descriptors used by the current internal leader.

## apo_pdb: Hardest False Negatives by Neighbor Lipid Fraction

| row | p_lipid | dist5 | lipid_frac_10 | top10_classes |
|---:|---:|---:|---:|---|
| 50 | 0.015 | 4.981 | 0.000 | `PP:9; COA:1` |
| 89 | 0.028 | 6.733 | 0.000 | `PP:10` |
| 47 | 0.048 | 5.955 | 0.000 | `PP:9; COA:1` |
| 49 | 0.088 | 5.879 | 0.000 | `PP:6; COA:4` |
| 63 | 0.089 | 4.256 | 0.000 | `PP:8; COA:2` |
| 36 | 0.129 | 5.780 | 0.000 | `PP:10` |
| 88 | 0.294 | 6.815 | 0.000 | `PP:9; COA:1` |
| 60 | 0.060 | 4.548 | 0.100 | `PP:9; OLA:1` |
| 3 | 0.099 | 3.973 | 0.100 | `PP:9; MYR:1` |
| 92 | 0.168 | 10.593 | 0.100 | `PP:8; COA:1; CLR:1` |
| 84 | 0.299 | 9.459 | 0.100 | `PP:6; COA:3; CLR:1` |
| 52 | 0.299 | 9.698 | 0.100 | `PP:9; CLR:1` |

## alphafold: Hardest False Negatives by Neighbor Lipid Fraction

| row | p_lipid | dist5 | lipid_frac_10 | top10_classes |
|---:|---:|---:|---:|---|
| 82 | 0.029 | 4.271 | 0.000 | `PP:10` |
| 104 | 0.033 | 3.700 | 0.000 | `PP:10` |
| 4 | 0.050 | 5.587 | 0.000 | `PP:10` |
| 31 | 0.073 | 4.032 | 0.000 | `PP:10` |
| 123 | 0.080 | 5.491 | 0.000 | `PP:10` |
| 74 | 0.114 | 9.895 | 0.000 | `COA:5; PP:3; BGC:2` |
| 96 | 0.135 | 6.362 | 0.000 | `PP:9; COA:1` |
| 54 | 0.151 | 5.174 | 0.000 | `PP:9; BGC:1` |
| 89 | 0.212 | 10.301 | 0.000 | `COA:4; ADN:4; PP:2` |
| 69 | 0.224 | 8.306 | 0.000 | `PP:7; COA:3` |
| 110 | 0.227 | 8.392 | 0.000 | `COA:9; PP:1` |
| 78 | 0.240 | 6.227 | 0.000 | `PP:8; COA:2` |
