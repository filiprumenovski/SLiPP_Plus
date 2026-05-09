# Hierarchical lipid-class experiment

_Stage 1: `trained` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.734 +/- 0.016 | 0.601 +/- 0.027 | 0.899 +/- 0.015 | 0.986 +/- 0.004 | 0.728 +/- 0.050 | 0.700 +/- 0.061 | 0.543 +/- 0.055 | 0.636 +/- 0.046 | 0.398 +/- 0.095 |
| stage1+stage2 hierarchy | 0.718 +/- 0.017 | 0.578 +/- 0.031 | 0.873 +/- 0.015 | 0.983 +/- 0.005 | 0.654 +/- 0.040 | 0.650 +/- 0.052 | 0.466 +/- 0.057 | 0.585 +/- 0.043 | 0.537 +/- 0.113 |
| hierarchy + ste_specialist | 0.721 +/- 0.016 | 0.584 +/- 0.030 | 0.873 +/- 0.015 | 0.983 +/- 0.005 | 0.654 +/- 0.040 | 0.650 +/- 0.052 | 0.466 +/- 0.061 | 0.585 +/- 0.042 | 0.564 +/- 0.109 |

## Stage diagnostics

- Stage-1 lipid gate binary F1: 0.854 +/- 0.014.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.641 +/- 0.030.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 81 total, 3.2 per iteration.
- STE F1 delta vs ensemble: +0.166; vs stage1+stage2 only: +0.027.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 375 |
| STE_correct | 235 |
| STE_as_PLM | 49 |
| PLM_as_STE | 154 |
| STE_as_COA | 13 |
| COA_as_STE | 18 |
| STE_as_OLA | 17 |
| OLA_as_STE | 30 |
| STE_as_MYR | 30 |
| MYR_as_STE | 6 |
| STE_as_PP | 22 |
| PP_as_STE | 7 |
| STE_as_CLR | 9 |
| CLR_as_STE | 3 |

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

## stage1_lipid_gate top features

| rank | feature | gain |
|---:|---|---:|
| 1 | prop_polar_atm | 238.9172 |
| 2 | hydrophobicity_score | 210.3205 |
| 3 | mean_loc_hyd_dens | 147.0713 |
| 4 | apol_as_prop | 101.8641 |
| 5 | pock_vol | 52.5889 |
| 6 | surf_pol_vdw14 | 52.3067 |
| 7 | surf_vdw14 | 49.7492 |
| 8 | HIS | 41.8199 |
| 9 | GLY | 30.4260 |
| 10 | THR | 30.4128 |
| 11 | mean_as_ray | 30.1398 |
| 12 | bulky_hydrophobic_count_shell3 | 29.8863 |
