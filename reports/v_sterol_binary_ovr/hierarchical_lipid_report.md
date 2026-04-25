# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.734 +/- 0.016 | 0.601 +/- 0.027 | 0.899 +/- 0.015 | 0.986 +/- 0.004 | 0.728 +/- 0.050 | 0.700 +/- 0.061 | 0.543 +/- 0.055 | 0.636 +/- 0.046 | 0.398 +/- 0.095 |
| stage1+stage2 hierarchy | 0.713 +/- 0.019 | 0.599 +/- 0.034 | 0.895 +/- 0.017 | 0.986 +/- 0.004 | 0.707 +/- 0.047 | 0.687 +/- 0.059 | 0.523 +/- 0.050 | 0.622 +/- 0.046 | 0.455 +/- 0.113 |
| hierarchy + ste_specialist | 0.726 +/- 0.018 | 0.625 +/- 0.033 | 0.896 +/- 0.017 | 0.986 +/- 0.004 | 0.707 +/- 0.047 | 0.688 +/- 0.058 | 0.523 +/- 0.053 | 0.630 +/- 0.047 | 0.576 +/- 0.108 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.633 +/- 0.029.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 139 total, 5.6 per iteration.
- STE F1 delta vs ensemble: +0.177; vs stage1+stage2 only: +0.121.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 375 |
| STE_correct | 223 |
| STE_as_PLM | 59 |
| PLM_as_STE | 144 |
| STE_as_COA | 21 |
| COA_as_STE | 3 |
| STE_as_OLA | 17 |
| OLA_as_STE | 21 |
| STE_as_MYR | 25 |
| MYR_as_STE | 5 |
| STE_as_PP | 21 |
| PP_as_STE | 3 |
| STE_as_CLR | 4 |
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

## stage2_lipid_CLR top features

| rank | feature | gain |
|---:|---|---:|
| 1 | mean_as_solv_acc | 29.0645 |
| 2 | PRO | 14.2743 |
| 3 | small_special_count_shell4 | 13.5615 |
| 4 | aromatic_aliphatic_ratio_shell3 | 12.5023 |
| 5 | surf_vdw14 | 12.4914 |
| 6 | TRP | 11.9981 |
| 7 | hydrophobicity_score | 11.9219 |
| 8 | SER | 11.6234 |
| 9 | ASP | 11.4175 |
| 10 | polar_neutral_count_shell1 | 11.1957 |
| 11 | VAL | 11.0887 |
| 12 | ILE | 10.6881 |

## stage2_lipid_MYR top features

| rank | feature | gain |
|---:|---|---:|
| 1 | LYS | 68.8664 |
| 2 | LEU | 35.0202 |
| 3 | THR | 22.6169 |
| 4 | TRP | 18.6370 |
| 5 | polar_neutral_count_shell3 | 15.4575 |
| 6 | cationic_count_shell3 | 15.4138 |
| 7 | ASP | 15.0594 |
| 8 | cationic_count_shell2 | 13.5324 |
| 9 | GLY | 11.6836 |
| 10 | charge_score | 10.6043 |
| 11 | mean_loc_hyd_dens | 10.4957 |
| 12 | ILE | 10.4484 |

## stage2_lipid_OLA top features

| rank | feature | gain |
|---:|---|---:|
| 1 | nb_AS | 26.7972 |
| 2 | aromatic_polar_count_shell1 | 24.3645 |
| 3 | aromatic_pi_count_shell3 | 20.2791 |
| 4 | VAL | 19.3603 |
| 5 | GLN | 15.2735 |
| 6 | bulky_hydrophobic_count_shell3 | 14.3648 |
| 7 | aromatic_pi_count_shell1 | 14.1276 |
| 8 | CYS | 13.5757 |
| 9 | aromatic_count_shell4 | 12.7394 |
| 10 | small_special_count_shell2 | 12.3579 |
| 11 | PRO | 11.2203 |
| 12 | THR | 11.1603 |

## stage2_lipid_PLM top features

| rank | feature | gain |
|---:|---|---:|
| 1 | polar_neutral_count_shell3 | 21.5970 |
| 2 | THR | 13.3305 |
| 3 | GLY | 12.6204 |
| 4 | bulky_hydrophobic_count_shell4 | 11.2305 |
| 5 | ILE | 10.9938 |
| 6 | ALA | 9.5112 |
| 7 | cationic_count_shell1 | 7.9697 |
| 8 | PHE | 7.6571 |
| 9 | charge_score | 7.0532 |
| 10 | aliphatic_count_shell4 | 6.9280 |
| 11 | aromatic_polar_count_shell3 | 6.7824 |
| 12 | polar_neutral_count_shell1 | 6.3413 |

## stage2_lipid_STE top features

| rank | feature | gain |
|---:|---|---:|
| 1 | volume_score | 33.8711 |
| 2 | LEU | 33.7301 |
| 3 | TYR | 33.5664 |
| 4 | ASP | 30.2725 |
| 5 | aromatic_polar_count_shell2 | 24.8903 |
| 6 | TRP | 22.3518 |
| 7 | aromatic_polar_count_shell3 | 20.6539 |
| 8 | HIS | 19.5576 |
| 9 | polar_neutral_count_shell3 | 19.3281 |
| 10 | PHE | 19.2024 |
| 11 | polar_neutral_count_shell1 | 18.5276 |
| 12 | anionic_count_shell4 | 18.4901 |
