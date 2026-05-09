# Composite Pair-MoE Results

| condition | 10-class macro-F1 | 5-lipid macro-F1 | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher baseline | 0.754 +/- 0.016 | 0.641 +/- 0.031 | 0.728 | 0.701 | 0.553 | 0.649 | 0.576 |
| composite pair-MoE | 0.756 +/- 0.017 | 0.644 +/- 0.030 | 0.729 | 0.703 | 0.556 | 0.649 | 0.581 |

## Delta

- 10-class macro-F1: +0.0020
- 5-lipid macro-F1: +0.0026

## Experts

| expert | labels | margin | fired mean | fired total |
|---|---|---:|---:|---:|
| plm_ste_pair_expert | PLM, STE | 0.99 | 12.7 | 317 |
| clr_ola_pair_expert | CLR, OLA | 0.10 | 2.4 | 59 |
| myr_plm_pair_expert | MYR, PLM | 0.05 | 2.8 | 71 |
| coa_adn_pair_expert | COA, ADN | 0.10 | 1.9 | 48 |
| coa_b12_pair_expert | COA, B12 | 0.05 | 1.4 | 35 |
