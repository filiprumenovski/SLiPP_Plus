# Grouped STE rescue sweep

_Binary XGB head trained on STE vs {PLM, COA, OLA, MYR}; applied when top-1 is a neighbor and STE is in the top-4 multiclass probabilities._

Baseline ensemble: 10-class macro-F1 = 0.730 +/- 0.015, 5-lipid macro-F1 = 0.599 +/- 0.025, STE F1 = 0.378 +/- 0.090.

STE-vs-neighbors binary F1 = 0.548 +/- 0.110; mean scale_pos_weight = 22.934.

| threshold | 10-class macro-F1 | 5-lipid macro-F1 | STE F1 | PLM F1 | fired mean | fired total | STE correct | STE->PLM | STE->COA | STE->OLA | STE->MYR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.748 +/- 0.015 | 0.636 +/- 0.025 | 0.552 +/- 0.109 | 0.645 +/- 0.041 | 6.5 | 163 | 209 | 66 | 23 | 19 | 22 |
| 0.40 | 0.748 +/- 0.015 | 0.636 +/- 0.025 | 0.552 +/- 0.109 | 0.645 +/- 0.041 | 5.8 | 146 | 209 | 66 | 23 | 19 | 22 |
| 0.45 | 0.748 +/- 0.015 | 0.636 +/- 0.025 | 0.552 +/- 0.109 | 0.645 +/- 0.041 | 5.1 | 128 | 209 | 66 | 23 | 19 | 22 |
| 0.50 | 0.748 +/- 0.016 | 0.636 +/- 0.025 | 0.552 +/- 0.109 | 0.645 +/- 0.041 | 4.6 | 114 | 209 | 66 | 23 | 19 | 22 |
| 0.55 | 0.748 +/- 0.016 | 0.635 +/- 0.026 | 0.548 +/- 0.111 | 0.645 +/- 0.040 | 4.3 | 107 | 206 | 69 | 23 | 19 | 22 |

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | HIS | 144.6682 |
| 2 | hydrophobicity_score | 127.3226 |
| 3 | LEU | 74.3120 |
| 4 | cationic_count_shell3 | 65.5740 |
| 5 | small_special_count_shell1 | 47.9962 |
| 6 | mean_loc_hyd_dens | 45.7355 |
| 7 | PHE | 43.2379 |
| 8 | lb_p_loop_like_motif_count | 31.8459 |
| 9 | aromatic_count_shell4 | 31.2899 |
| 10 | charge_score | 31.0134 |
| 11 | TRP | 27.6739 |
| 12 | prop_polar_atm | 27.4689 |
| 13 | aromatic_count_shell3 | 26.3254 |
| 14 | lb_tube_beta_branched_fraction | 25.8795 |
| 15 | bulky_hydrophobic_count_shell1 | 25.8565 |
