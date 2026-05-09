# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.732 +/- 0.007 | 0.591 +/- 0.015 | 0.895 +/- 0.012 | 0.985 +/- 0.005 | 0.731 +/- 0.032 | 0.697 +/- 0.054 | 0.521 +/- 0.025 | 0.650 +/- 0.030 | 0.358 +/- 0.069 |
| stage1+stage2 hierarchy | 0.721 +/- 0.017 | 0.596 +/- 0.038 | 0.892 +/- 0.013 | 0.985 +/- 0.005 | 0.685 +/- 0.046 | 0.635 +/- 0.034 | 0.479 +/- 0.042 | 0.613 +/- 0.052 | 0.569 +/- 0.088 |
| hierarchy + ste_specialist | 0.724 +/- 0.013 | 0.603 +/- 0.031 | 0.892 +/- 0.013 | 0.985 +/- 0.005 | 0.685 +/- 0.046 | 0.636 +/- 0.036 | 0.482 +/- 0.043 | 0.617 +/- 0.050 | 0.596 +/- 0.066 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.616 +/- 0.046.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 10 total, 2.0 per iteration.
- STE F1 delta vs ensemble: +0.238; vs stage1+stage2 only: +0.027.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 75 |
| STE_correct | 48 |
| STE_as_PLM | 12 |
| PLM_as_STE | 28 |
| STE_as_COA | 3 |
| COA_as_STE | 0 |
| STE_as_OLA | 3 |
| OLA_as_STE | 6 |
| STE_as_MYR | 4 |
| MYR_as_STE | 2 |
| STE_as_PP | 4 |
| PP_as_STE | 0 |
| STE_as_CLR | 0 |
| CLR_as_STE | 2 |

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
| 1 | LYS | 8.1778 |
| 2 | LEU | 3.9126 |
| 3 | VAL | 3.4652 |
| 4 | mean_as_solv_acc | 3.3938 |
| 5 | HIS | 3.2725 |
| 6 | TRP | 3.2226 |
| 7 | ASP | 3.0690 |
| 8 | volume_score | 2.9346 |
| 9 | TYR | 2.8546 |
| 10 | aromatic_aliphatic_ratio_shell3 | 2.7454 |
| 11 | THR | 2.7027 |
| 12 | aliphatic_count_shell4 | 2.6708 |

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
