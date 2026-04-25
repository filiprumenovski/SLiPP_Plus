# Hierarchical lipid-class experiment

_Stage 1: `ensemble` lipid-vs-rest gate. Stage 2: five-way lipid-family XGB. Stage 3: gated one-vs-neighbors specialist head._

## Headline metrics

| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble | 0.706 +/- 0.019 | 0.581 +/- 0.035 | 0.879 +/- 0.019 | 0.984 +/- 0.004 | 0.695 +/- 0.054 | 0.671 +/- 0.061 | 0.516 +/- 0.051 | 0.625 +/- 0.044 | 0.399 +/- 0.104 |
| stage1+stage2 hierarchy | 0.681 +/- 0.021 | 0.570 +/- 0.040 | 0.875 +/- 0.019 | 0.984 +/- 0.004 | 0.666 +/- 0.046 | 0.646 +/- 0.056 | 0.463 +/- 0.055 | 0.595 +/- 0.050 | 0.482 +/- 0.106 |
| hierarchy + ste_specialist | 0.687 +/- 0.019 | 0.582 +/- 0.035 | 0.875 +/- 0.019 | 0.984 +/- 0.004 | 0.666 +/- 0.046 | 0.646 +/- 0.056 | 0.464 +/- 0.055 | 0.599 +/- 0.051 | 0.535 +/- 0.092 |

## Stage diagnostics

- Stage-1 lipid gate uses the existing ensemble lipid probability mass to preserve binary parity.
- Stage-2 lipid-family macro-F1 on true lipid test rows: 0.612 +/- 0.041.
- Specialist `ste_specialist`: positive `STE`, neighbors `PLM, COA, OLA, MYR`, threshold `0.4`, top-k `4`.
- Specialist fires: 92 total, 3.7 per iteration.
- STE F1 delta vs ensemble: +0.136; vs stage1+stage2 only: +0.053.

## STE-focused confusion

| metric | count |
|---|---:|
| STE_support | 375 |
| STE_correct | 205 |
| STE_as_PLM | 69 |
| PLM_as_STE | 139 |
| STE_as_COA | 27 |
| COA_as_STE | 4 |
| STE_as_OLA | 16 |
| OLA_as_STE | 19 |
| STE_as_MYR | 20 |
| MYR_as_STE | 13 |
| STE_as_PP | 21 |
| PP_as_STE | 7 |
| STE_as_CLR | 9 |
| CLR_as_STE | 3 |

## ste_specialist top features

| rank | feature | gain |
|---:|---|---:|
| 1 | anionic_count_shell4 | 113.1585 |
| 2 | prop_polar_atm | 107.6053 |
| 3 | net_charge_per_vol | 63.3169 |
| 4 | polar_neutral_count_shell1 | 59.5312 |
| 5 | cationic_count_shell2 | 44.3499 |
| 6 | branched_aliphatic_aa_frac | 37.8610 |
| 7 | aromatic_pi_count_shell4 | 36.7986 |
| 8 | polar_uncharged_aa_frac | 34.7406 |
| 9 | polar_neutral_count_shell3 | 32.9388 |
| 10 | pocket_lam3 | 32.8793 |
| 11 | aromatic_aliphatic_ratio_shell4 | 30.8627 |
| 12 | pocket_lam2 | 26.7414 |

## stage3_nonlipid_family top features

| rank | feature | gain |
|---:|---|---:|
| 1 | nb_AS | 56.9411 |
| 2 | branched_aliphatic_aa_frac | 34.2442 |
| 3 | charge_density_per_vol | 26.7810 |
| 4 | surf_vdw14 | 25.3881 |
| 5 | polar_neutral_count_shell3 | 24.2645 |
| 6 | mean_as_solv_acc | 23.5050 |
| 7 | anion_density | 23.0983 |
| 8 | anionic_count_shell2 | 19.3009 |
| 9 | charge_balance | 17.1959 |
| 10 | pock_vol | 16.2292 |
| 11 | aromatic_aa_frac | 15.3368 |
| 12 | bulky_hydrophobic_count_shell4 | 14.7697 |

## stage2_lipid_family top features

| rank | feature | gain |
|---:|---|---:|
| 1 | aromatic_density | 5.7857 |
| 2 | polar_neutral_count_shell3 | 4.6204 |
| 3 | vol_per_as | 4.0417 |
| 4 | bulky_hydrophobic_count_shell4 | 3.9984 |
| 5 | polar_hydrophobic_ratio_shell1 | 3.7170 |
| 6 | nb_AS | 3.6959 |
| 7 | charged_aa_frac | 3.6139 |
| 8 | charge_density_per_vol | 3.3967 |
| 9 | planarity_x_aromatic_density | 3.2466 |
| 10 | vol_x_polar_surface | 3.2319 |
| 11 | net_charge_per_vol | 3.0744 |
| 12 | aromatic_gradient | 3.0587 |

## boundary_ola_vs_plm top features

| rank | feature | gain |
|---:|---|---:|
| 1 | vol_x_polar_surface | 26.5061 |
| 2 | nb_AS | 11.9171 |
| 3 | aromatic_pi_count_shell3 | 11.5629 |
| 4 | aromatic_aa_frac | 9.5514 |
| 5 | bulky_hydrophobic_count_shell2 | 8.6169 |
| 6 | aromatic_gradient | 7.8088 |
| 7 | polar_neutral_count_shell2 | 6.7713 |
| 8 | aromatic_aliphatic_ratio_shell3 | 6.4441 |
| 9 | bulky_hydrophobic_count_shell4 | 5.7116 |
| 10 | surf_pol_vdw22_frac | 5.2141 |
| 11 | burial_x_polar_surface | 4.9399 |
| 12 | aromatic_polar_count_shell2 | 4.8093 |
