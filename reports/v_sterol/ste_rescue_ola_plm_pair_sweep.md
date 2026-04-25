# Pair tiebreaker sweep — OLA vs PLM

_Standalone v_sterol ensemble experiment. Positive class = OLA, negative class = PLM. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {PLM, OLA} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.753 ± 0.016, 5-lipid macro-F1 = 0.640 ± 0.029, PLM F1 = 0.647 ± 0.049, OLA F1 = 0.546 ± 0.056.

Pair-only binary head (true PLM+OLA rows, OLA=positive): F1 = 0.706 ± 0.052; mean scale_pos_weight = 2.182.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | PLM F1 | OLA F1 | fired mean | fired total | PLM→OLA | OLA→PLM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.754 ± 0.016 | 0.641 ± 0.030 | 0.649 ± 0.050 | 0.553 ± 0.056 | 1.7 | 42 | 69 | 121 |
| 0.10 | 0.753 ± 0.016 | 0.640 ± 0.031 | 0.650 ± 0.050 | 0.547 ± 0.059 | 3.1 | 77 | 72 | 122 |
| 0.15 | 0.753 ± 0.016 | 0.640 ± 0.031 | 0.651 ± 0.050 | 0.547 ± 0.061 | 4.2 | 106 | 72 | 120 |
| 0.20 | 0.753 ± 0.016 | 0.641 ± 0.030 | 0.652 ± 0.050 | 0.547 ± 0.056 | 5.2 | 130 | 73 | 118 |
| 0.30 | 0.753 ± 0.016 | 0.640 ± 0.030 | 0.651 ± 0.051 | 0.543 ± 0.051 | 7.4 | 184 | 77 | 119 |
| 0.40 | 0.752 ± 0.016 | 0.639 ± 0.030 | 0.651 ± 0.051 | 0.539 ± 0.051 | 9.2 | 229 | 80 | 120 |
| 0.50 | 0.752 ± 0.016 | 0.638 ± 0.029 | 0.651 ± 0.050 | 0.537 ± 0.049 | 10.9 | 273 | 80 | 121 |
| 0.70 | 0.752 ± 0.016 | 0.638 ± 0.029 | 0.649 ± 0.050 | 0.535 ± 0.050 | 13.8 | 345 | 84 | 122 |
| 0.90 | 0.752 ± 0.016 | 0.638 ± 0.029 | 0.649 ± 0.050 | 0.535 ± 0.050 | 16.8 | 421 | 84 | 122 |
| 0.99 | 0.752 ± 0.016 | 0.638 ± 0.029 | 0.649 ± 0.050 | 0.535 ± 0.050 | 19.2 | 479 | 84 | 122 |

## Interpretation

OLA F1 does not improve monotonically across the tested margins.

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | nb_AS | 20.7179 |
| 2 | aromatic_pi_count_shell3 | 9.6012 |
| 3 | aliphatic_count_shell1 | 8.1806 |
| 4 | polar_neutral_count_shell3 | 7.4346 |
| 5 | polar_neutral_count_shell2 | 7.1198 |
| 6 | MET | 6.3878 |
| 7 | HIS | 6.3103 |
| 8 | bulky_hydrophobic_count_shell1 | 6.0830 |
| 9 | aromatic_polar_count_shell2 | 5.9172 |
| 10 | aromatic_pi_count_shell1 | 5.8986 |
| 11 | aromatic_count_shell4 | 5.8891 |
| 12 | aromatic_aliphatic_ratio_shell2 | 5.8854 |
| 13 | aromatic_aliphatic_ratio_shell1 | 5.8155 |
| 14 | LYS | 5.7034 |
| 15 | aromatic_pi_count_shell4 | 5.3930 |
