# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.732 +/- 0.007 | 0.591 +/- 0.015 | 0.895 +/- 0.012 | 0.985 +/- 0.005 | 0.731 +/- 0.032 | 0.697 +/- 0.054 | 0.521 +/- 0.025 | 0.650 +/- 0.030 | 0.358 +/- 0.069 |
| stage1+stage2 hierarchy | 0.723 +/- 0.014 | 0.602 +/- 0.033 | 0.891 +/- 0.017 | 0.985 +/- 0.005 | 0.717 +/- 0.061 | 0.652 +/- 0.044 | 0.517 +/- 0.024 | 0.636 +/- 0.036 | 0.490 +/- 0.094 |
| hierarchy + ste_specialist | 0.732 +/- 0.006 | 0.620 +/- 0.017 | 0.891 +/- 0.017 | 0.985 +/- 0.005 | 0.717 +/- 0.061 | 0.652 +/- 0.044 | 0.515 +/- 0.031 | 0.640 +/- 0.031 | 0.577 +/- 0.041 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.633 +/- 0.036.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 16 total, 3.2 per iteration.
- STE F1 delta vs ensemble: +0.219; vs stage1+stage2 only: +0.087.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 75 |
| STE_correct | 44 |
| STE_as_PLM | 15 |
| PLM_as_STE | 27 |
| STE_as_COA | 4 |
| COA_as_STE | 0 |
| STE_as_OLA | 1 |
| OLA_as_STE | 5 |
| STE_as_MYR | 4 |
| MYR_as_STE | 0 |
| STE_as_PP | 4 |
| PP_as_STE | 0 |
| STE_as_CLR | 2 |
| CLR_as_STE | 1 |

## ste_specialist top features

| rank | feature | gain |
|---:|---|---:|
| 1 | HIS | 172.9975 |
| 2 | hydrophobicity_score | 172.8990 |
| 3 | cationic_count_shell3 | 106.0101 |
| 4 | LEU | 85.9520 |
| 5 | aromatic_count_shell4 | 59.2624 |
| 6 | mean_loc_hyd_dens | 49.6897 |
| 7 | net_charge_per_vol | 38.9767 |
| 8 | polar_neutral_count_shell1 | 36.1617 |
| 9 | LYS | 34.5476 |
| 10 | polar_uncharged_aa_frac | 34.3913 |
| 11 | PHE | 32.0717 |
| 12 | TYR | 31.5708 |

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
| 1 | LYS | 17.6062 |
| 2 | polar_neutral_count_shell3 | 6.7281 |
| 3 | LEU | 6.4641 |
| 4 | polar_neutral_count_shell1 | 6.0339 |
| 5 | aromatic_density | 5.2586 |
| 6 | volume_score | 4.1487 |
| 7 | HIS | 4.1274 |
| 8 | TYR | 3.8157 |
| 9 | ASP | 3.8057 |
| 10 | aromatic_aliphatic_ratio_shell1 | 3.7772 |
| 11 | VAL | 3.7126 |
| 12 | bulky_hydrophobic_count_shell4 | 3.7092 |

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
