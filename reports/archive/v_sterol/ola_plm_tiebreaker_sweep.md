# Pair tiebreaker sweep — OLA vs PLM

_Standalone v_sterol ensemble experiment. Positive class = OLA, negative class = PLM. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {PLM, OLA} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.734 ± 0.016, 5-lipid macro-F1 = 0.601 ± 0.027, PLM F1 = 0.636 ± 0.046, OLA F1 = 0.543 ± 0.055.

Pair-only binary head (true PLM+OLA rows, OLA=positive): F1 = 0.706 ± 0.052; mean scale_pos_weight = 2.182.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | PLM F1 | OLA F1 | fired mean | fired total | PLM→OLA | OLA→PLM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.734 ± 0.016 | 0.603 ± 0.027 | 0.638 ± 0.046 | 0.550 ± 0.056 | 1.7 | 42 | 70 | 122 |
| 0.10 | 0.734 ± 0.016 | 0.602 ± 0.028 | 0.638 ± 0.046 | 0.544 ± 0.059 | 3.1 | 77 | 73 | 123 |
| 0.15 | 0.734 ± 0.016 | 0.602 ± 0.028 | 0.639 ± 0.046 | 0.545 ± 0.060 | 4.3 | 107 | 73 | 121 |
| 0.20 | 0.734 ± 0.016 | 0.602 ± 0.028 | 0.640 ± 0.046 | 0.545 ± 0.056 | 5.2 | 131 | 74 | 119 |
| 0.30 | 0.734 ± 0.016 | 0.601 ± 0.027 | 0.639 ± 0.048 | 0.541 ± 0.050 | 7.4 | 185 | 78 | 120 |
| 0.40 | 0.733 ± 0.015 | 0.600 ± 0.026 | 0.639 ± 0.047 | 0.536 ± 0.050 | 9.2 | 230 | 81 | 121 |
| 0.50 | 0.733 ± 0.015 | 0.600 ± 0.026 | 0.639 ± 0.047 | 0.535 ± 0.049 | 11.0 | 274 | 81 | 122 |
| 0.70 | 0.733 ± 0.015 | 0.599 ± 0.026 | 0.637 ± 0.047 | 0.533 ± 0.049 | 13.8 | 346 | 85 | 123 |
| 0.90 | 0.733 ± 0.015 | 0.599 ± 0.026 | 0.637 ± 0.047 | 0.533 ± 0.049 | 16.9 | 422 | 85 | 123 |
| 0.99 | 0.733 ± 0.015 | 0.599 ± 0.026 | 0.637 ± 0.047 | 0.533 ± 0.049 | 19.2 | 480 | 85 | 123 |

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
