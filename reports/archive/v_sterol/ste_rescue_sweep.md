# Grouped STE rescue sweep

_Binary XGB head trained on STE vs {PLM, COA, OLA, MYR}; applied when top-1 is a neighbor and STE is in the top-4 multiclass probabilities._

Baseline ensemble: 10-class macro-F1 = 0.734 +/- 0.016, 5-lipid macro-F1 = 0.601 +/- 0.027, STE F1 = 0.398 +/- 0.095.

STE-vs-neighbors binary F1 = 0.569 +/- 0.100; mean scale_pos_weight = 22.934.

| threshold | 10-class macro-F1 | 5-lipid macro-F1 | STE F1 | PLM F1 | fired mean | fired total | STE correct | STE->PLM | STE->COA | STE->OLA | STE->MYR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.753 +/- 0.016 | 0.639 +/- 0.029 | 0.576 +/- 0.105 | 0.648 +/- 0.049 | 7.1 | 177 | 222 | 71 | 19 | 16 | 17 |
| 0.40 | 0.753 +/- 0.016 | 0.639 +/- 0.029 | 0.576 +/- 0.105 | 0.648 +/- 0.049 | 6.4 | 160 | 222 | 71 | 19 | 16 | 17 |
| 0.45 | 0.753 +/- 0.016 | 0.639 +/- 0.029 | 0.576 +/- 0.105 | 0.647 +/- 0.049 | 5.4 | 136 | 222 | 71 | 19 | 16 | 17 |
| 0.50 | 0.753 +/- 0.016 | 0.640 +/- 0.029 | 0.576 +/- 0.105 | 0.647 +/- 0.049 | 4.8 | 119 | 222 | 71 | 19 | 16 | 17 |
| 0.55 | 0.751 +/- 0.016 | 0.635 +/- 0.031 | 0.558 +/- 0.112 | 0.647 +/- 0.050 | 3.9 | 98 | 210 | 79 | 20 | 19 | 17 |

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | HIS | 123.0315 |
| 2 | hydrophobicity_score | 93.5708 |
| 3 | LEU | 82.2344 |
| 4 | aromatic_count_shell4 | 58.1224 |
| 5 | mean_loc_hyd_dens | 40.1182 |
| 6 | cationic_count_shell3 | 31.7621 |
| 7 | SER | 31.4779 |
| 8 | polar_neutral_count_shell3 | 28.7843 |
| 9 | PHE | 28.7051 |
| 10 | charge_score | 26.7715 |
| 11 | aromatic_count_shell3 | 24.8216 |
| 12 | prop_polar_atm | 24.2478 |
| 13 | TYR | 23.4585 |
| 14 | cationic_count_shell2 | 22.4765 |
| 15 | TRP | 22.1746 |
