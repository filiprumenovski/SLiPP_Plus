# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.733 +/- 0.013 | 0.602 +/- 0.027 | 0.896 +/- 0.014 | 0.986 +/- 0.004 | 0.732 +/- 0.046 | 0.700 +/- 0.051 | 0.553 +/- 0.062 | 0.633 +/- 0.042 | 0.393 +/- 0.095 |
| stage1+stage2 hierarchy | 0.722 +/- 0.022 | 0.608 +/- 0.037 | 0.893 +/- 0.017 | 0.986 +/- 0.004 | 0.704 +/- 0.040 | 0.677 +/- 0.047 | 0.517 +/- 0.054 | 0.615 +/- 0.042 | 0.526 +/- 0.121 |
| hierarchy + ste_specialist | 0.727 +/- 0.021 | 0.618 +/- 0.034 | 0.893 +/- 0.016 | 0.986 +/- 0.004 | 0.703 +/- 0.040 | 0.678 +/- 0.047 | 0.516 +/- 0.056 | 0.618 +/- 0.042 | 0.573 +/- 0.106 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.641 +/- 0.040.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 77 total, 3.1 per iteration.
- STE F1 delta vs ensemble: +0.180; vs stage1+stage2 only: +0.047.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 375 |
| STE_correct | 228 |
| STE_as_PLM | 60 |
| PLM_as_STE | 147 |
| STE_as_COA | 24 |
| COA_as_STE | 4 |
| STE_as_OLA | 11 |
| OLA_as_STE | 28 |
| STE_as_MYR | 19 |
| MYR_as_STE | 3 |
| STE_as_PP | 19 |
| PP_as_STE | 2 |
| STE_as_CLR | 8 |
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
