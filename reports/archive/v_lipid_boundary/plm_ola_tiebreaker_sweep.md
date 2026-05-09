# Pair tiebreaker sweep — OLA vs PLM

_Standalone v_sterol ensemble experiment. Positive class = OLA, negative class = PLM. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {PLM, OLA} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.730 ± 0.015, 5-lipid macro-F1 = 0.599 ± 0.025, PLM F1 = 0.636 ± 0.039, OLA F1 = 0.559 ± 0.063.

Pair-only binary head (true PLM+OLA rows, OLA=positive): F1 = 0.695 ± 0.053; mean scale_pos_weight = 2.182.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | PLM F1 | OLA F1 | fired mean | fired total | PLM→OLA | OLA→PLM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.729 ± 0.014 | 0.598 ± 0.024 | 0.635 ± 0.038 | 0.556 ± 0.060 | 1.4 | 34 | 71 | 138 |

## Interpretation

OLA F1 moves monotonically upward across the tested margins.

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | nb_AS | 22.0641 |
| 2 | lb_p_loop_like_motif_count | 11.5530 |
| 3 | GLU | 9.6976 |
| 4 | aromatic_pi_count_shell3 | 9.5709 |
| 5 | polar_neutral_count_shell2 | 9.1468 |
| 6 | aromatic_aliphatic_ratio_shell2 | 8.4544 |
| 7 | aromatic_count_shell4 | 8.4043 |
| 8 | polar_neutral_count_shell3 | 7.7406 |
| 9 | aromatic_pi_count_shell1 | 7.5187 |
| 10 | bulky_hydrophobic_count_shell1 | 7.3463 |
| 11 | MET | 7.1392 |
| 12 | TYR | 6.8496 |
| 13 | GLN | 6.5595 |
| 14 | aromatic_count_shell1 | 6.2099 |
| 15 | lb_polar_end_aromatic_count | 6.0476 |
