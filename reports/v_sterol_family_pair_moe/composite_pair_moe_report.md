# Composite Pair-MoE Results

| condition | 10-class macro-F1 | 5-lipid macro-F1 | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher baseline | 0.757 +/- 0.015 | 0.652 +/- 0.029 | 0.715 | 0.692 | 0.572 | 0.653 | 0.626 |
| composite pair-MoE | 0.759 +/- 0.017 | 0.654 +/- 0.031 | 0.720 | 0.695 | 0.573 | 0.657 | 0.626 |

## Delta

- 10-class macro-F1: +0.0013
- 5-lipid macro-F1: +0.0026

## Experts

| expert | labels | margin | fired mean | fired total |
|---|---|---:|---:|---:|
| clr_plm_pair_expert | CLR, PLM | 0.15 | 1.6 | 39 |
| clr_ola_pair_expert | CLR, OLA | 0.10 | 3.2 | 79 |
| myr_plm_pair_expert | MYR, PLM | 0.05 | 2.2 | 56 |
| plm_ste_pair_expert | PLM, STE | 0.05 | 0.3 | 8 |
