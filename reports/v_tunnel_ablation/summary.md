# v_tunnel feature ablation summary

Aligned to `processed/v_tunnel_aligned` sorted by `_row_order`; all runs use the
25 canonical splits. Baseline is `v_sterol` features only.

## Result

The CAVER/tunnel signal is real but small. Compact tunnel subsets beat the full
18-column tunnel block in both LGBM and XGB. The gain is mostly CLR/OLA; STE is
flat to slightly negative.

## LGBM screen

| variant | extra | lipid5 | delta | macro10 | binary F1 | CLR | OLA | PLM | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tunnel_shape_avail6 | 6 | 0.608 +/- 0.031 | +0.011 | 0.734 | 0.893 | 0.754 | 0.560 | 0.634 | 0.404 |
| tunnel_geom9 | 9 | 0.608 +/- 0.024 | +0.011 | 0.734 | 0.890 | 0.750 | 0.568 | 0.630 | 0.416 |
| tunnel_chem5 | 5 | 0.607 +/- 0.033 | +0.010 | 0.734 | 0.898 | 0.759 | 0.567 | 0.626 | 0.399 |
| tunnel_all18 | 18 | 0.606 +/- 0.030 | +0.009 | 0.732 | 0.895 | 0.758 | 0.574 | 0.625 | 0.399 |
| tunnel_no_dead15 | 15 | 0.606 +/- 0.030 | +0.009 | 0.732 | 0.895 | 0.758 | 0.574 | 0.625 | 0.399 |
| tunnel_lean13 | 13 | 0.602 +/- 0.028 | +0.006 | 0.730 | 0.895 | 0.763 | 0.558 | 0.623 | 0.397 |
| tunnel_single_best3 | 3 | 0.598 +/- 0.033 | +0.001 | 0.728 | 0.893 | 0.743 | 0.543 | 0.621 | 0.406 |
| v_sterol_only | 0 | 0.597 +/- 0.028 | +0.000 | 0.730 | 0.896 | 0.714 | 0.535 | 0.635 | 0.402 |

## XGB confirmation

| variant | extra | lipid5 | delta | macro10 | binary F1 | CLR | OLA | PLM | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tunnel_shape_avail6 | 6 | 0.609 +/- 0.030 | +0.015 | 0.734 | 0.895 | 0.747 | 0.573 | 0.639 | 0.408 |
| tunnel_chem5 | 5 | 0.608 +/- 0.030 | +0.014 | 0.733 | 0.894 | 0.758 | 0.569 | 0.636 | 0.407 |
| tunnel_geom9 | 9 | 0.605 +/- 0.026 | +0.011 | 0.730 | 0.891 | 0.751 | 0.567 | 0.624 | 0.414 |
| tunnel_all18 | 18 | 0.600 +/- 0.037 | +0.006 | 0.727 | 0.893 | 0.745 | 0.566 | 0.622 | 0.393 |
| v_sterol_only | 0 | 0.594 +/- 0.028 | +0.000 | 0.726 | 0.896 | 0.709 | 0.530 | 0.627 | 0.416 |

## Recommended encoder ablation order

1. `tunnel_shape_avail6`
2. `tunnel_chem5`
3. `tunnel_geom9`
4. `tunnel_all18` only as the negative-control/full-block comparison

## Feature subsets

### tunnel_shape_avail6

- `tunnel_has_tunnel`
- `tunnel_caver_profile_present`
- `tunnel_count`
- `tunnel_primary_bottleneck_radius`
- `tunnel_length_over_axial`
- `tunnel_extends_beyond_pocket`

### tunnel_chem5

- `tunnel_has_tunnel`
- `tunnel_caver_profile_present`
- `tunnel_primary_hydrophobicity`
- `tunnel_primary_charge`
- `tunnel_primary_aromatic_fraction`

### tunnel_geom9

- `tunnel_has_tunnel`
- `tunnel_caver_profile_present`
- `tunnel_count`
- `tunnel_primary_length`
- `tunnel_primary_bottleneck_radius`
- `tunnel_primary_curvature`
- `tunnel_total_length`
- `tunnel_min_bottleneck`
- `tunnel_length_over_axial`

## Interpretation

The full tunnel block carries redundant or dead columns. Dropping the constants
does not change LGBM, and compact subsets perform as well or better than all 18
features. The next encoder retrain should not use the full raw tunnel family as
the primary candidate; start with `tunnel_shape_avail6` and confirm against
`tunnel_chem5`.
