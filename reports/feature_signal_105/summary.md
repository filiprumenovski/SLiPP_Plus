# 105-feature signal audit

Dataset: `processed/v_tunnel_aligned/full_pockets.parquet`; feature set: `v_tunnel` (105 columns).

## Counts

- Total features: 105
- Constant features: 3
- Univariate MI above global null 95%: 93
- Univariate MI above global null 99%: 91
- Univariate MI above max-null 95%: 87
- LGBM positive gain in any split: 102
- LGBM stable gain (>=20/25 splits and >=0.1% mean gain): 84
- Seed-0 permutation lipid5 drop >=0.001: 48
- Seed-0 permutation lipid5 drop >=0.005: 34
- Any supervised signal by these screens: 94

## Family counts

| family | n | constants | MI 95 | stable gain | permutation >=0.001 | any signal |
|---|---:|---:|---:|---:|---:|---:|
| tunnel | 18 | 3 | 14 | 13 | 4 | 14 |
| v_sterol | 87 | 0 | 79 | 71 | 44 | 80 |

## Top features by mean LGBM gain fraction

| feature | family | gain frac | used splits | MI | perm lipid5 drop |
|---|---|---:|---:|---:|---:|
| `mean_loc_hyd_dens` | v_sterol | 0.0813 | 25 | 0.2764 | 0.0840 |
| `hydrophobicity_score` | v_sterol | 0.0727 | 25 | 0.3153 | 0.0686 |
| `volume_score` | v_sterol | 0.0457 | 25 | 0.1428 | 0.0248 |
| `apol_as_prop` | v_sterol | 0.0426 | 25 | 0.1473 | 0.0452 |
| `surf_vdw14` | v_sterol | 0.0402 | 25 | 0.1294 | 0.0709 |
| `prop_polar_atm` | v_sterol | 0.0395 | 25 | 0.2511 | 0.0399 |
| `mean_as_solv_acc` | v_sterol | 0.0324 | 25 | 0.0792 | 0.0074 |
| `pock_vol` | v_sterol | 0.0313 | 25 | 0.1667 | 0.0643 |
| `LEU` | v_sterol | 0.0270 | 25 | 0.1404 | 0.0555 |
| `TRP` | v_sterol | 0.0269 | 25 | 0.0491 | 0.0045 |
| `tunnel_caver_profile_present` | tunnel | 0.0264 | 25 | 0.0386 | 0.0556 |
| `tunnel_primary_hydrophobicity` | tunnel | 0.0238 | 25 | 0.1163 | 0.0120 |
| `ASP` | v_sterol | 0.0234 | 25 | 0.0608 | 0.0088 |
| `surf_apol_vdw14` | v_sterol | 0.0218 | 25 | 0.1076 | 0.0269 |
| `pocket_burial` | v_sterol | 0.0210 | 25 | 0.0668 | 0.0174 |
| `mean_as_ray` | v_sterol | 0.0200 | 25 | 0.0564 | 0.0274 |
| `GLY` | v_sterol | 0.0181 | 25 | 0.1081 | 0.0121 |
| `flex` | v_sterol | 0.0178 | 25 | 0.0388 | 0.0074 |
| `charge_score` | v_sterol | 0.0150 | 25 | 0.0730 | 0.0034 |
| `surf_pol_vdw14` | v_sterol | 0.0128 | 25 | 0.0850 | 0.0203 |
| `pocket_elongation` | v_sterol | 0.0125 | 25 | 0.0328 | 0.0079 |
| `HIS` | v_sterol | 0.0114 | 25 | 0.0585 | 0.0270 |
| `ILE` | v_sterol | 0.0100 | 25 | 0.0935 | 0.0049 |
| `pocket_lam2` | v_sterol | 0.0094 | 25 | 0.1364 | 0.0051 |
| `VAL` | v_sterol | 0.0093 | 25 | 0.1087 | 0.0095 |

## Tunnel features

| feature | stable gain | gain frac | used splits | MI | perm lipid5 drop | constant |
|---|---:|---:|---:|---:|---:|---:|
| `tunnel_caver_profile_present` | True | 0.0264 | 25 | 0.0386 | 0.0556 | False |
| `tunnel_primary_hydrophobicity` | True | 0.0238 | 25 | 0.1163 | 0.0120 | False |
| `tunnel_primary_bottleneck_radius` | True | 0.0082 | 25 | 0.1307 | 0.0003 | False |
| `tunnel_primary_throughput` | True | 0.0078 | 25 | 0.1315 | -0.0063 | False |
| `tunnel_primary_aromatic_fraction` | True | 0.0077 | 25 | 0.0503 | -0.0029 | False |
| `tunnel_primary_length` | True | 0.0073 | 25 | 0.0564 | -0.0141 | False |
| `tunnel_min_bottleneck` | True | 0.0066 | 25 | 0.0518 | -0.0035 | False |
| `tunnel_max_length` | True | 0.0058 | 25 | 0.0436 | -0.0011 | False |
| `tunnel_total_length` | True | 0.0054 | 25 | 0.0362 | 0.0077 | False |
| `tunnel_length_over_axial` | True | 0.0053 | 25 | 0.0450 | -0.0127 | False |
| `tunnel_primary_curvature` | True | 0.0052 | 25 | 0.0258 | -0.0008 | False |
| `tunnel_primary_charge` | True | 0.0049 | 25 | 0.0337 | -0.0096 | False |
| `tunnel_count` | True | 0.0039 | 25 | 0.0324 | 0.0062 | False |
| `tunnel_extends_beyond_pocket` | False | 0.0000 | 24 | 0.0069 | 0.0000 | False |
| `tunnel_has_tunnel` | False | 0.0000 | 17 | 0.0123 | 0.0000 | False |
| `tunnel_pocket_context_present` | False | 0.0000 | 0 | 0.0029 | 0.0000 | True |
| `tunnel_branching_factor` | False | 0.0000 | 0 | 0.0002 | 0.0000 | True |
| `tunnel_primary_avg_radius` | False | 0.0000 | 0 | 0.0000 | 0.0000 | True |
