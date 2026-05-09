# Composite Pair-MoE Results

| condition | 10-class macro-F1 | 5-lipid macro-F1 | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher baseline | 0.757 +/- 0.015 | 0.652 +/- 0.029 | 0.715 | 0.692 | 0.572 | 0.653 | 0.626 |
| composite pair-MoE | 0.762 +/- 0.015 | 0.660 +/- 0.029 | 0.727 | 0.694 | 0.587 | 0.658 | 0.635 |

## Delta

- 10-class macro-F1: +0.0048
- 5-lipid macro-F1: +0.0085

## Experts

| expert | labels | margin | fired mean | fired total |
|---|---|---:|---:|---:|
| clr_neighborhood_expert | CLR, OLA, PLM, COA, PP | 0.75 | 1244.5 | 31112 |
| myr_plm_pair_expert | MYR, PLM | 0.05 | 2.5 | 62 |
| plm_ste_pair_expert | PLM, STE | 0.05 | 0.3 | 8 |
| plm_ste_neighborhood_expert | PLM, STE, COA, MYR, OLA | 0.75 | 592.4 | 14810 |
