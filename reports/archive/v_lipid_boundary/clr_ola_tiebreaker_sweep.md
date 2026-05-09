# Pair tiebreaker sweep — OLA vs CLR

_Standalone v_sterol ensemble experiment. Positive class = OLA, negative class = CLR. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {CLR, OLA} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.730 ± 0.015, 5-lipid macro-F1 = 0.599 ± 0.025, CLR F1 = 0.723 ± 0.043, OLA F1 = 0.559 ± 0.063.

Pair-only binary head (true CLR+OLA rows, OLA=positive): F1 = 0.789 ± 0.042; mean scale_pos_weight = 1.088.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | CLR F1 | OLA F1 | fired mean | fired total | CLR→OLA | OLA→CLR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.730 ± 0.015 | 0.599 ± 0.026 | 0.724 ± 0.045 | 0.555 ± 0.066 | 1.2 | 31 | 76 | 69 |
| 0.10 | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.724 ± 0.042 | 0.557 ± 0.064 | 2.5 | 63 | 77 | 67 |
| 0.15 | 0.730 ± 0.016 | 0.599 ± 0.026 | 0.724 ± 0.044 | 0.555 ± 0.064 | 3.8 | 94 | 76 | 69 |
| 0.20 | 0.729 ± 0.016 | 0.598 ± 0.025 | 0.721 ± 0.046 | 0.553 ± 0.062 | 4.8 | 119 | 78 | 71 |
| 0.30 | 0.729 ± 0.015 | 0.597 ± 0.024 | 0.721 ± 0.046 | 0.550 ± 0.059 | 6.8 | 169 | 78 | 73 |
| 0.40 | 0.728 ± 0.016 | 0.595 ± 0.026 | 0.716 ± 0.047 | 0.545 ± 0.061 | 8.2 | 205 | 82 | 77 |
| 0.50 | 0.728 ± 0.016 | 0.595 ± 0.026 | 0.716 ± 0.049 | 0.544 ± 0.064 | 9.6 | 241 | 83 | 77 |
| 0.70 | 0.728 ± 0.016 | 0.595 ± 0.026 | 0.716 ± 0.049 | 0.544 ± 0.064 | 11.6 | 290 | 83 | 77 |
| 0.90 | 0.728 ± 0.016 | 0.595 ± 0.026 | 0.716 ± 0.049 | 0.544 ± 0.064 | 13.1 | 327 | 83 | 77 |
| 0.99 | 0.728 ± 0.016 | 0.595 ± 0.026 | 0.716 ± 0.049 | 0.544 ± 0.064 | 13.6 | 341 | 83 | 77 |

## Interpretation

OLA F1 does not improve monotonically across the tested margins.

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | aromatic_pi_count_shell3 | 9.6132 |
| 2 | aromatic_count_shell3 | 5.9004 |
| 3 | charge_score | 5.7987 |
| 4 | ALA | 5.6905 |
| 5 | lb_p_loop_like_motif_count | 5.5926 |
| 6 | aromatic_polar_count_shell3 | 5.3193 |
| 7 | aromatic_pi_count_shell4 | 4.5589 |
| 8 | TRP | 4.3494 |
| 9 | ASP | 4.2563 |
| 10 | GLN | 3.9438 |
| 11 | PRO | 3.7171 |
| 12 | aromatic_pi_count_shell2 | 3.1379 |
| 13 | lb_axis_length | 3.0526 |
| 14 | flex | 2.9138 |
| 15 | lb_cationic_anchor_density | 2.9056 |
