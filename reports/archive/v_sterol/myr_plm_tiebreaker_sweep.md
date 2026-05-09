# Pair tiebreaker sweep — MYR vs PLM

_Standalone v_sterol ensemble experiment. Positive class = MYR, negative class = PLM. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {PLM, MYR} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.734 ± 0.016, 5-lipid macro-F1 = 0.601 ± 0.027, PLM F1 = 0.636 ± 0.046, MYR F1 = 0.700 ± 0.061.

Pair-only binary head (true PLM+MYR rows, MYR=positive): F1 = 0.755 ± 0.046; mean scale_pos_weight = 1.691.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | PLM F1 | MYR F1 | fired mean | fired total | PLM→MYR | MYR→PLM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.734 ± 0.015 | 0.601 ± 0.027 | 0.636 ± 0.044 | 0.701 ± 0.060 | 2.8 | 71 | 213 | 144 |
| 0.10 | 0.733 ± 0.015 | 0.601 ± 0.027 | 0.635 ± 0.042 | 0.701 ± 0.060 | 4.9 | 122 | 219 | 141 |
| 0.15 | 0.733 ± 0.015 | 0.601 ± 0.027 | 0.634 ± 0.042 | 0.701 ± 0.062 | 6.9 | 172 | 227 | 134 |
| 0.20 | 0.734 ± 0.015 | 0.602 ± 0.028 | 0.634 ± 0.040 | 0.704 ± 0.062 | 9.1 | 227 | 227 | 130 |
| 0.30 | 0.733 ± 0.015 | 0.600 ± 0.028 | 0.632 ± 0.037 | 0.700 ± 0.063 | 13.3 | 332 | 234 | 132 |
| 0.40 | 0.733 ± 0.015 | 0.599 ± 0.028 | 0.629 ± 0.038 | 0.697 ± 0.062 | 17.4 | 435 | 239 | 134 |
| 0.50 | 0.732 ± 0.015 | 0.598 ± 0.028 | 0.626 ± 0.038 | 0.694 ± 0.065 | 22.1 | 552 | 248 | 135 |
| 0.70 | 0.731 ± 0.015 | 0.597 ± 0.028 | 0.624 ± 0.037 | 0.690 ± 0.065 | 34.7 | 868 | 251 | 139 |
| 0.90 | 0.731 ± 0.015 | 0.596 ± 0.028 | 0.624 ± 0.038 | 0.690 ± 0.065 | 52.0 | 1299 | 252 | 139 |
| 0.99 | 0.731 ± 0.015 | 0.596 ± 0.028 | 0.624 ± 0.038 | 0.690 ± 0.065 | 55.3 | 1382 | 252 | 139 |

## Interpretation

MYR F1 does not improve monotonically across the tested margins.

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | THR | 23.5003 |
| 2 | LYS | 19.7642 |
| 3 | cationic_count_shell3 | 18.7366 |
| 4 | LEU | 9.3913 |
| 5 | GLN | 7.2024 |
| 6 | GLY | 6.8091 |
| 7 | ILE | 6.7808 |
| 8 | aromatic_aliphatic_ratio_shell2 | 6.1450 |
| 9 | charge_score | 6.0062 |
| 10 | HIS | 5.9258 |
| 11 | TRP | 5.6546 |
| 12 | cationic_count_shell2 | 4.9079 |
| 13 | mean_loc_hyd_dens | 4.5503 |
| 14 | MET | 4.5481 |
| 15 | ASP | 4.4589 |
