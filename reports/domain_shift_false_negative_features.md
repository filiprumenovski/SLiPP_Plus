# Domain-Shift False-Negative Feature Ablation

This diagnostic uses `exp-019` holdout predictions at the default binary threshold (`p_lipid >= 0.5`) and the `v49+tunnel_shape` feature table as a common feature basis. It compares external true-lipid false negatives against internal training lipids and against recalled true-lipid holdout pockets.

## Probability Summary

| holdout | true_lipids | TP | FN | mean_p_lipid_lipids | mean_p_lipid_TP | mean_p_lipid_FN | mean_p_lipid_nonlipids |
|---|---:|---:|---:|---:|---:|---:|---:|
| apo_pdb | 67 | 36 | 31 | 0.562 | 0.834 | 0.246 | 0.225 |
| alphafold | 91 | 43 | 48 | 0.498 | 0.753 | 0.269 | 0.154 |

## Key Signal

- `exp-019` false negatives are true lipid pockets with lipid probability only slightly above non-lipid background, confirming a recall/domain-shift failure rather than lipid overcalling.
- Feature shifts are not identical across apo-PDB and AlphaFold, so a single global threshold or one extra descriptor is unlikely to solve the transfer gap.
- The tables below rank features by standardized separation of false negatives from recalled holdout lipids (`FN - TP`, in internal-lipid standard deviations).

## apo_pdb: FN Lower Than Recalled Lipids

| feature | FN mean | TP mean | internal lipid mean | FN z vs internal | FN-TP sd |
|---|---:|---:|---:|---:|---:|
| `mean_loc_hyd_dens` | 26.717 | 56.976 | 43.889 | -0.751 | -1.323 |
| `TYR` | 0.645 | 2.083 | 1.115 | -0.397 | -1.215 |
| `aliphatic_count_shell4` | 0.548 | 3.611 | 2.204 | -0.615 | -1.137 |
| `bulky_hydrophobic_count_shell4` | 0.452 | 2.750 | 1.638 | -0.569 | -1.104 |
| `as_density` | 4.455 | 7.597 | 6.242 | -0.575 | -1.011 |
| `aromatic_count_shell4` | 0.290 | 1.611 | 0.831 | -0.414 | -1.011 |
| `as_max_dst` | 11.103 | 19.654 | 16.033 | -0.570 | -0.989 |
| `ALA` | 0.935 | 2.306 | 1.538 | -0.422 | -0.960 |
| `PHE` | 1.387 | 3.250 | 2.206 | -0.420 | -0.956 |
| `aliphatic_count_shell3` | 2.613 | 5.278 | 3.977 | -0.489 | -0.954 |
| `LEU` | 2.226 | 5.028 | 3.546 | -0.437 | -0.928 |
| `aromatic_polar_count_shell4` | 0.065 | 0.750 | 0.376 | -0.419 | -0.924 |

## apo_pdb: FN Higher Than Recalled Lipids

| feature | FN mean | TP mean | internal lipid mean | FN z vs internal | FN-TP sd |
|---|---:|---:|---:|---:|---:|
| `pocket_planarity` | 7.324 | 3.366 | 4.330 | 0.793 | 1.049 |
| `pocket_burial` | 0.455 | 0.349 | 0.432 | 0.131 | 0.602 |
| `small_special_count_shell2` | 1.000 | 0.528 | 0.683 | 0.382 | 0.569 |
| `pocket_elongation` | 36.852 | 22.692 | 24.184 | 0.477 | 0.533 |
| `aliphatic_count_shell2` | 3.968 | 3.167 | 3.333 | 0.333 | 0.421 |
| `anionic_count_shell2` | 0.355 | 0.194 | 0.256 | 0.204 | 0.330 |
| `mean_as_solv_acc` | 0.506 | 0.490 | 0.517 | -0.151 | 0.239 |
| `cationic_count_shell1` | 0.032 | 0.000 | 0.022 | 0.071 | 0.216 |
| `bulky_hydrophobic_count_shell2` | 2.968 | 2.639 | 2.650 | 0.187 | 0.193 |
| `flex` | 0.384 | 0.323 | 0.352 | 0.085 | 0.159 |
| `aromatic_pi_count_shell2` | 0.839 | 0.722 | 0.858 | -0.022 | 0.131 |
| `cationic_count_shell2` | 0.323 | 0.250 | 0.369 | -0.070 | 0.111 |

## alphafold: FN Lower Than Recalled Lipids

| feature | FN mean | TP mean | internal lipid mean | FN z vs internal | FN-TP sd |
|---|---:|---:|---:|---:|---:|
| `small_special_count_shell4` | 0.354 | 1.395 | 0.567 | -0.228 | -1.116 |
| `aliphatic_count_shell3` | 3.688 | 6.651 | 3.977 | -0.104 | -1.062 |
| `aliphatic_count_shell4` | 1.500 | 4.279 | 2.204 | -0.262 | -1.032 |
| `bulky_hydrophobic_count_shell3` | 2.521 | 4.884 | 3.069 | -0.239 | -1.028 |
| `ALA` | 1.188 | 2.628 | 1.538 | -0.245 | -1.009 |
| `LEU` | 2.688 | 5.605 | 3.546 | -0.284 | -0.966 |
| `PHE` | 1.521 | 3.349 | 2.206 | -0.352 | -0.938 |
| `as_density` | 5.615 | 8.523 | 6.242 | -0.202 | -0.936 |
| `as_max_dst` | 14.365 | 22.423 | 16.033 | -0.193 | -0.932 |
| `aromatic_count_shell3` | 1.062 | 2.442 | 1.507 | -0.293 | -0.909 |
| `surf_apol_vdw14` | 172.378 | 267.366 | 88.457 | 0.794 | -0.898 |
| `pocket_lam1` | 16.508 | 38.829 | 24.465 | -0.313 | -0.877 |

## alphafold: FN Higher Than Recalled Lipids

| feature | FN mean | TP mean | internal lipid mean | FN z vs internal | FN-TP sd |
|---|---:|---:|---:|---:|---:|
| `cationic_count_shell2` | 0.500 | 0.116 | 0.369 | 0.201 | 0.586 |
| `match_desc_cost` | 8.064 | 4.615 | 4.068 | 0.604 | 0.522 |
| `pocket_burial` | 0.322 | 0.238 | 0.432 | -0.628 | 0.478 |
| `polar_neutral_count_shell2` | 0.875 | 0.442 | 0.845 | 0.030 | 0.423 |
| `aromatic_aliphatic_ratio_shell2` | 0.610 | 0.387 | 0.431 | 0.332 | 0.413 |
| `polar_hydrophobic_ratio_shell2` | 0.509 | 0.310 | 0.402 | 0.210 | 0.390 |
| `mean_as_solv_acc` | 0.505 | 0.482 | 0.517 | -0.167 | 0.343 |
| `prop_polar_atm` | 28.375 | 25.854 | 26.344 | 0.223 | 0.277 |
| `pocket_planarity` | 4.203 | 3.268 | 4.330 | -0.034 | 0.248 |
| `mean_as_ray` | 3.967 | 3.943 | 3.996 | -0.260 | 0.214 |
| `pocket_elongation` | 26.190 | 20.778 | 24.184 | 0.076 | 0.204 |
| `ASN` | 0.583 | 0.442 | 0.535 | 0.054 | 0.159 |
