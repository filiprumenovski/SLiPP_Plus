# Composite Pair-MoE Results

| condition | 10-class macro-F1 | 5-lipid macro-F1 | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher baseline | 0.757 +/- 0.015 | 0.652 +/- 0.029 | 0.715 | 0.692 | 0.572 | 0.653 | 0.626 |
| composite pair-MoE | 0.757 +/- 0.016 | 0.652 +/- 0.029 | 0.715 | 0.697 | 0.572 | 0.653 | 0.623 |

## Delta

- 10-class macro-F1: +0.0001
- 5-lipid macro-F1: +0.0003

## Experts

| expert | labels | margin | fired mean | fired total |
|---|---|---:|---:|---:|
| clr_neighborhood_utility_expert | CLR, OLA, PLM, COA, PP | 0.50 | 1122.5 | 28063 |
| myr_plm_pair_expert | MYR, PLM | 0.05 | 2.2 | 56 |
| plm_ste_pair_expert | PLM, STE | 0.05 | 0.3 | 8 |
