# Pair tiebreaker sweep — PLM vs COA

_Standalone v_sterol ensemble experiment. Positive class = PLM, negative class = COA. Margin sweep applies the binary head whenever the ensemble top-2 are exactly {COA, PLM} and the top-1/top-2 gap is below the threshold._

Baseline ensemble: 10-class macro-F1 = 0.730 ± 0.015, 5-lipid macro-F1 = 0.599 ± 0.025, COA F1 = 0.854 ± 0.013, PLM F1 = 0.636 ± 0.039.

Pair-only binary head (true COA+PLM rows, PLM=positive): F1 = 0.914 ± 0.025; mean scale_pos_weight = 2.814.

| margin | 10-class macro-F1 | 5-lipid macro-F1 | COA F1 | PLM F1 | fired mean | fired total | COA→PLM | PLM→COA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.854 ± 0.013 | 0.636 ± 0.040 | 1.2 | 31 | 33 | 142 |
| 0.10 | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.854 ± 0.013 | 0.635 ± 0.039 | 2.4 | 60 | 35 | 143 |
| 0.15 | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.854 ± 0.013 | 0.635 ± 0.040 | 3.3 | 82 | 37 | 142 |
| 0.20 | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.853 ± 0.013 | 0.634 ± 0.040 | 4.8 | 119 | 37 | 145 |
| 0.30 | 0.730 ± 0.015 | 0.599 ± 0.025 | 0.853 ± 0.014 | 0.634 ± 0.042 | 7.5 | 187 | 40 | 144 |
| 0.40 | 0.730 ± 0.015 | 0.599 ± 0.024 | 0.853 ± 0.014 | 0.634 ± 0.039 | 9.6 | 241 | 40 | 144 |
| 0.50 | 0.730 ± 0.015 | 0.599 ± 0.024 | 0.853 ± 0.014 | 0.634 ± 0.040 | 11.6 | 290 | 40 | 144 |
| 0.70 | 0.730 ± 0.015 | 0.599 ± 0.024 | 0.853 ± 0.014 | 0.634 ± 0.040 | 18.1 | 452 | 42 | 144 |
| 0.90 | 0.730 ± 0.015 | 0.599 ± 0.024 | 0.853 ± 0.014 | 0.634 ± 0.040 | 25.8 | 645 | 42 | 144 |
| 0.99 | 0.730 ± 0.015 | 0.599 ± 0.024 | 0.853 ± 0.014 | 0.634 ± 0.040 | 31.2 | 781 | 42 | 144 |

## Interpretation

PLM F1 does not improve monotonically across the tested margins.

## Iteration-0 top-15 features (gain)

| rank | feature | gain |
|---:|---|---:|
| 1 | mean_loc_hyd_dens | 41.2547 |
| 2 | apol_as_prop | 36.3151 |
| 3 | charge_score | 33.6754 |
| 4 | GLY | 18.3525 |
| 5 | hydrophobicity_score | 16.3770 |
| 6 | HIS | 15.0249 |
| 7 | LEU | 13.5228 |
| 8 | aromatic_count_shell2 | 12.9427 |
| 9 | small_special_count_shell2 | 12.5869 |
| 10 | THR | 11.6307 |
| 11 | PRO | 9.7692 |
| 12 | LYS | 8.8971 |
| 13 | anionic_count_shell3 | 8.7821 |
| 14 | aromatic_pi_count_shell2 | 8.3525 |
| 15 | volume_score | 7.8961 |
