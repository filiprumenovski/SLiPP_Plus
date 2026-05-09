# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.732 +/- 0.007 | 0.591 +/- 0.015 | 0.895 +/- 0.012 | 0.985 +/- 0.005 | 0.731 +/- 0.032 | 0.697 +/- 0.054 | 0.521 +/- 0.025 | 0.650 +/- 0.030 | 0.358 +/- 0.069 |
| stage1+stage2 hierarchy | 0.722 +/- 0.008 | 0.599 +/- 0.016 | 0.896 +/- 0.012 | 0.985 +/- 0.005 | 0.684 +/- 0.041 | 0.644 +/- 0.069 | 0.488 +/- 0.033 | 0.627 +/- 0.039 | 0.550 +/- 0.054 |
| hierarchy + ste_specialist | 0.726 +/- 0.005 | 0.608 +/- 0.013 | 0.897 +/- 0.011 | 0.985 +/- 0.005 | 0.684 +/- 0.041 | 0.643 +/- 0.070 | 0.491 +/- 0.035 | 0.628 +/- 0.038 | 0.592 +/- 0.063 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.633 +/- 0.023.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 17 total, 3.4 per iteration.
- STE F1 delta vs ensemble: +0.234; vs stage1+stage2 only: +0.042.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 75 |
| STE_correct | 45 |
| STE_as_PLM | 14 |
| PLM_as_STE | 24 |
| STE_as_COA | 2 |
| COA_as_STE | 0 |
| STE_as_OLA | 2 |
| OLA_as_STE | 7 |
| STE_as_MYR | 6 |
| MYR_as_STE | 0 |
| STE_as_PP | 4 |
| PP_as_STE | 0 |
| STE_as_CLR | 1 |
| CLR_as_STE | 0 |

## ste_specialist top features

| rank | feature | gain |
|---:|---|---:|
| 1 | HIS | 123.0315 |
| 2 | hydrophobicity_score | 93.5708 |
| 3 | LEU | 82.2344 |
| 4 | aromatic_count_shell4 | 58.1224 |
| 5 | mean_loc_hyd_dens | 40.1182 |
| 6 | cationic_count_shell3 | 31.7621 |
| 7 | SER | 31.4779 |
| 8 | polar_neutral_count_shell3 | 28.7843 |
| 9 | PHE | 28.7051 |
| 10 | charge_score | 26.7715 |
| 11 | aromatic_count_shell3 | 24.8216 |
| 12 | prop_polar_atm | 24.2478 |

## stage3_nonlipid_family top features

| rank | feature | gain |
|---:|---|---:|
| 1 | TRP | 66.9918 |
| 2 | nb_AS | 64.0700 |
| 3 | charge_density_per_vol | 30.7579 |
| 4 | mean_as_solv_acc | 28.6814 |
| 5 | surf_vdw14 | 26.8931 |
| 6 | branched_aliphatic_aa_frac | 23.3796 |
| 7 | polar_neutral_count_shell3 | 23.1534 |
| 8 | anionic_count_shell2 | 23.0897 |
| 9 | anion_density | 22.7621 |
| 10 | pock_vol | 20.3379 |
| 11 | mean_loc_hyd_dens | 20.1764 |
| 12 | LEU | 19.7214 |

## stage2_lipid_family top features

| rank | feature | gain |
|---:|---|---:|
| 1 | LYS | 9.8300 |
| 2 | aromatic_pi_count_shell1 | 6.0828 |
| 3 | LEU | 4.4919 |
| 4 | polar_neutral_count_shell3 | 4.4776 |
| 5 | VAL | 3.8260 |
| 6 | mean_as_solv_acc | 3.7438 |
| 7 | polar_neutral_count_shell1 | 3.7273 |
| 8 | volume_score | 3.4613 |
| 9 | TYR | 3.3352 |
| 10 | bulky_hydrophobic_count_shell4 | 3.2688 |
| 11 | GLY | 3.2273 |
| 12 | TRP | 3.2136 |

## boundary_ola_vs_plm top features

| rank | feature | gain |
|---:|---|---:|
| 1 | vol_x_polar_surface | 25.5544 |
| 2 | bulky_hydrophobic_count_shell2 | 23.2048 |
| 3 | nb_AS | 14.7791 |
| 4 | polar_neutral_count_shell2 | 12.4478 |
| 5 | aromatic_pi_count_shell3 | 12.1381 |
| 6 | aromatic_aa_frac | 11.3231 |
| 7 | MET | 9.1034 |
| 8 | ARG | 8.9592 |
| 9 | VAL | 7.7251 |
| 10 | aromatic_gradient | 6.8995 |
| 11 | anionic_count_shell3 | 6.4576 |
| 12 | ILE | 6.1093 |
