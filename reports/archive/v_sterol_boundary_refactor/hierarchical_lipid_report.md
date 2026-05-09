# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.734 +/- 0.016 | 0.601 +/- 0.027 | 0.899 +/- 0.015 | 0.986 +/- 0.004 | 0.728 +/- 0.050 | 0.700 +/- 0.061 | 0.543 +/- 0.055 | 0.636 +/- 0.046 | 0.398 +/- 0.095 |
| stage1+stage2 hierarchy | 0.717 +/- 0.019 | 0.607 +/- 0.035 | 0.895 +/- 0.018 | 0.986 +/- 0.004 | 0.695 +/- 0.048 | 0.668 +/- 0.057 | 0.514 +/- 0.056 | 0.610 +/- 0.046 | 0.546 +/- 0.122 |
| hierarchy + ste_specialist | 0.721 +/- 0.019 | 0.614 +/- 0.034 | 0.895 +/- 0.018 | 0.986 +/- 0.004 | 0.695 +/- 0.048 | 0.669 +/- 0.056 | 0.513 +/- 0.057 | 0.612 +/- 0.047 | 0.581 +/- 0.112 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.641 +/- 0.030.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.5`, top-k `4`.
- Specialist fires: 42 total, 1.7 per iteration.
- STE F1 delta vs ensemble: +0.182; vs stage1+stage2 only: +0.034.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 375 |
| STE_correct | 233 |
| STE_as_PLM | 47 |
| PLM_as_STE | 148 |
| STE_as_COA | 20 |
| COA_as_STE | 7 |
| STE_as_OLA | 14 |
| OLA_as_STE | 24 |
| STE_as_MYR | 28 |
| MYR_as_STE | 6 |
| STE_as_PP | 21 |
| PP_as_STE | 6 |
| STE_as_CLR | 7 |
| CLR_as_STE | 3 |

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
| 1 | TRP | 58.2615 |
| 2 | nb_AS | 39.1322 |
| 3 | mean_as_solv_acc | 25.9576 |
| 4 | surf_vdw14 | 24.3122 |
| 5 | anionic_count_shell2 | 21.5761 |
| 6 | charge_score | 19.2770 |
| 7 | pock_vol | 17.3820 |
| 8 | mean_loc_hyd_dens | 17.2495 |
| 9 | ASP | 17.2337 |
| 10 | LEU | 15.9088 |
| 11 | HIS | 15.6599 |
| 12 | GLY | 15.3567 |

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
| 1 | nb_AS | 20.7179 |
| 2 | aromatic_pi_count_shell3 | 9.6012 |
| 3 | aliphatic_count_shell1 | 8.1806 |
| 4 | polar_neutral_count_shell3 | 7.4346 |
| 5 | polar_neutral_count_shell2 | 7.1198 |
| 6 | MET | 6.3878 |
| 7 | HIS | 6.3103 |
| 8 | bulky_hydrophobic_count_shell1 | 6.0830 |
| 9 | aromatic_polar_count_shell2 | 5.9172 |
| 10 | aromatic_pi_count_shell1 | 5.8986 |
| 11 | aromatic_count_shell4 | 5.8891 |
| 12 | aromatic_aliphatic_ratio_shell2 | 5.8854 |
