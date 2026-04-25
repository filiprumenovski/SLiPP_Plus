# Pair tiebreaker sweep — PLM vs MYR

_Standalone v_sterol ensemble experiment. Positive class = PLM, negative class = MYR. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {MYR, PLM} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.753 ± 0.016, 5-lipid macro-F1 = 0.640 ± 0.029, MYR F1 = 0.701 ± 0.060, PLM F1 = 0.647 ± 0.049.

Pair-only binary head (true MYR+PLM rows, PLM=positive): F1 = 0.845 ± 0.026; mean scale_pos_weight = 0.591.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | MYR F1 | PLM F1 | fired mean | fired total | MYR→PLM | PLM→MYR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.753 ± 0.016 | 0.640 ± 0.029 | 0.703 ± 0.058 | 0.648 ± 0.048 | 2.8 | 71 | 143 | 209 |
| 0.10 | 0.753 ± 0.015 | 0.640 ± 0.028 | 0.702 ± 0.058 | 0.648 ± 0.046 | 4.8 | 121 | 141 | 213 |
| 0.15 | 0.753 ± 0.015 | 0.639 ± 0.028 | 0.701 ± 0.058 | 0.645 ± 0.046 | 6.8 | 171 | 136 | 223 |
| 0.20 | 0.753 ± 0.015 | 0.640 ± 0.028 | 0.703 ± 0.057 | 0.645 ± 0.044 | 9.0 | 226 | 132 | 224 |
| 0.30 | 0.752 ± 0.015 | 0.638 ± 0.028 | 0.697 ± 0.057 | 0.643 ± 0.043 | 13.2 | 330 | 137 | 230 |
| 0.40 | 0.751 ± 0.015 | 0.636 ± 0.028 | 0.691 ± 0.058 | 0.640 ± 0.044 | 17.3 | 432 | 142 | 236 |
| 0.50 | 0.750 ± 0.016 | 0.634 ± 0.029 | 0.686 ± 0.059 | 0.636 ± 0.043 | 22.0 | 549 | 145 | 245 |
| 0.70 | 0.750 ± 0.016 | 0.633 ± 0.029 | 0.682 ± 0.061 | 0.635 ± 0.043 | 34.6 | 864 | 150 | 247 |
| 0.90 | 0.750 ± 0.016 | 0.633 ± 0.029 | 0.682 ± 0.061 | 0.635 ± 0.043 | 51.8 | 1295 | 150 | 247 |
| 0.99 | 0.750 ± 0.016 | 0.633 ± 0.029 | 0.682 ± 0.061 | 0.635 ± 0.043 | 55.1 | 1378 | 150 | 247 |

## Interpretation

PLM F1 does not improve monotonically across the tested margins.

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | THR | 14.2247 |
| 2 | LYS | 11.7568 |
| 3 | cationic_count_shell3 | 9.0390 |
| 4 | LEU | 4.8668 |
| 5 | ILE | 4.7411 |
| 6 | charge_score | 4.5714 |
| 7 | GLY | 4.3634 |
| 8 | polar_neutral_count_shell1 | 4.2102 |
| 9 | GLN | 3.7410 |
| 10 | ASP | 3.2950 |
| 11 | HIS | 3.1599 |
| 12 | aromatic_polar_count_shell2 | 3.1076 |
| 13 | mean_loc_hyd_dens | 2.8723 |
| 14 | aromatic_aliphatic_ratio_shell4 | 2.8486 |
| 15 | TYR | 2.8464 |
