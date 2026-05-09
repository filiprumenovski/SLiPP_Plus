# v_sterol vs v49 — sterol-targeted feature set

_38 new features layered on top of v49 (17 fpocket descriptors + 20 AA counts + 12 aromatic/aliphatic shells)._

New columns:
- 28 chemistry-shell counts (7 groups x 4 shells)
- 4 polar/hydrophobic ratios per shell
- 5 alpha-sphere PCA features (lam1/2/3 + elongation + planarity)
- 1 pocket burial (distance to protein CA centroid / max CA spread)

## Per-class F1 on test split (mean across 25 iterations)

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE | macro-F1 (10) | macro-F1 (5 lipids) | binary F1 | binary AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rf (v49) | 0.782 | 0.784 | 0.712 | 0.700 | 0.807 | 0.684 | 0.456 | 0.627 | 0.944 | 0.340 | 0.684 | 0.562 | 0.837 | 0.980 |
| rf (v_sterol) | 0.766 | 0.749 | 0.683 | 0.696 | 0.794 | 0.685 | 0.406 | 0.619 | 0.935 | 0.329 | 0.666 | 0.547 | 0.819 | 0.978 |
| **rf Δ** | **-0.017** | **-0.035** | **-0.030** | **-0.004** ⭑ | **-0.013** | **+0.001** | **-0.051** | **-0.008** | **-0.009** | **-0.011** ⭑ | **-0.018** | **-0.015** | **-0.018** | **-0.001** |
| xgb (v49) | 0.829 | 0.857 | 0.793 | 0.708 | 0.854 | 0.682 | 0.522 | 0.623 | 0.973 | 0.415 | 0.725 | 0.590 | 0.895 | 0.988 |
| xgb (v_sterol) | 0.822 | 0.861 | 0.780 | 0.709 | 0.852 | 0.689 | 0.530 | 0.627 | 0.972 | 0.416 | 0.726 | 0.594 | 0.896 | 0.988 |
| **xgb Δ** | **-0.007** | **+0.004** | **-0.012** | **+0.001** ⭑ | **-0.001** | **+0.007** | **+0.008** | **+0.004** | **-0.001** | **+0.001** ⭑ | **+0.000** | **+0.004** | **+0.001** | **+0.001** |
| lgbm (v49) | 0.830 | 0.858 | 0.789 | 0.716 | 0.852 | 0.673 | 0.530 | 0.631 | 0.973 | 0.387 | 0.724 | 0.588 | 0.893 | 0.988 |
| lgbm (v_sterol) | 0.834 | 0.862 | 0.790 | 0.714 | 0.853 | 0.697 | 0.535 | 0.635 | 0.972 | 0.402 | 0.730 | 0.597 | 0.896 | 0.989 |
| **lgbm Δ** | **+0.004** | **+0.005** | **+0.001** | **-0.002** ⭑ | **+0.002** | **+0.024** | **+0.005** | **+0.004** | **-0.000** | **+0.015** ⭑ | **+0.006** | **+0.009** | **+0.003** | **+0.001** |

_⭑ CLR and STE — the sterol classes that motivated this sprint._

## CLR / STE headline (what we are trying to move)

| model | CLR v49 | CLR v_sterol | CLR Δ | STE v49 | STE v_sterol | STE Δ |
|---|---|---|---|---|---|---|
| rf | 0.700 | 0.696 | -0.004 | 0.340 | 0.329 | -0.011 |
| xgb | 0.708 | 0.709 | +0.001 | 0.415 | 0.416 | +0.001 |
| lgbm | 0.716 | 0.714 | -0.002 | 0.387 | 0.402 | +0.015 |

## CLR ↔ STE confusion (summed across all 25 test folds)

| model | variant | CLR rows | CLR → STE | rate | STE rows | STE → CLR | rate |
|---|---|---|---|---|---|---|---|
| rf | v49 | 900 | 0 | 0.000 | 375 | 4 | 0.011 |
| rf | v_sterol | 900 | 0 | 0.000 | 375 | 3 | 0.008 |
| xgb | v49 | 900 | 3 | 0.003 | 375 | 3 | 0.008 |
| xgb | v_sterol | 900 | 0 | 0.000 | 375 | 9 | 0.024 |
| lgbm | v49 | 900 | 4 | 0.004 | 375 | 4 | 0.011 |
| lgbm | v_sterol | 900 | 1 | 0.001 | 375 | 5 | 0.013 |

## Holdouts (binary, iteration-0 model)

### apo-PDB holdout

| model | variant | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|---|
| rf | v49 | 0.709 | 0.848 | 0.907 | 0.582 |
| rf | v_sterol | 0.716 | 0.842 | 0.929 | 0.582 |
| xgb | v49 | 0.685 | 0.788 | 0.864 | 0.567 |
| xgb | v_sterol | 0.667 | 0.771 | 0.809 | 0.567 |
| lgbm | v49 | 0.696 | 0.786 | 0.833 | 0.597 |
| lgbm | v_sterol | 0.678 | 0.795 | 0.812 | 0.582 |

### AlphaFold holdout

| model | variant | F1 | AUROC | precision | sensitivity |
|---|---|---|---|---|---|
| rf | v49 | 0.753 | 0.883 | 0.921 | 0.637 |
| rf | v_sterol | 0.725 | 0.882 | 0.931 | 0.593 |
| xgb | v49 | 0.629 | 0.819 | 0.865 | 0.495 |
| xgb | v_sterol | 0.689 | 0.833 | 0.895 | 0.560 |
| lgbm | v49 | 0.707 | 0.855 | 0.898 | 0.582 |
| lgbm | v_sterol | 0.648 | 0.850 | 0.902 | 0.505 |

## Top XGB feature importance (gain) — iteration-0 v_sterol model

| rank | feature | gain |
|---|---|---|
| 1 | `TRP` | 14.79 |
| 2 | `mean_loc_hyd_dens` | 11.83 |
| 3 | `ASP` | 11.00 |
| 4 | `hydrophobicity_score` | 9.24 |
| 5 | `aromatic_pi_count_shell1` ⬥ | 8.16 |
| 6 | `apol_as_prop` | 7.39 |
| 7 | `LEU` | 7.39 |
| 8 | `polar_neutral_count_shell1` ⬥ | 5.83 |
| 9 | `GLY` | 5.69 |
| 10 | `aromatic_polar_count_shell4` ⬥ | 5.61 |
| 11 | `bulky_hydrophobic_count_shell4` ⬥ | 5.38 |
| 12 | `HIS` | 5.33 |
| 13 | `mean_as_solv_acc` | 5.25 |
| 14 | `charge_score` | 5.12 |
| 15 | `SER` | 5.08 |

_⬥ new sterol feature (not in v49)._
