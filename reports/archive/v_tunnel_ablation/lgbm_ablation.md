# v_tunnel LGBM feature ablation

Aligned to `processed/v_tunnel_aligned` sorted by `_row_order`; 25 canonical splits; LGBM only.

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

## Feature subsets

### v_sterol_only

- no tunnel columns

### tunnel_all18

- `tunnel_count`
- `tunnel_primary_length`
- `tunnel_primary_bottleneck_radius`
- `tunnel_primary_avg_radius`
- `tunnel_primary_curvature`
- `tunnel_primary_throughput`
- `tunnel_primary_hydrophobicity`
- `tunnel_primary_charge`
- `tunnel_primary_aromatic_fraction`
- `tunnel_max_length`
- `tunnel_total_length`
- `tunnel_min_bottleneck`
- `tunnel_branching_factor`
- `tunnel_length_over_axial`
- `tunnel_extends_beyond_pocket`
- `tunnel_pocket_context_present`
- `tunnel_caver_profile_present`
- `tunnel_has_tunnel`

### tunnel_no_dead15

- `tunnel_count`
- `tunnel_primary_length`
- `tunnel_primary_bottleneck_radius`
- `tunnel_primary_curvature`
- `tunnel_primary_throughput`
- `tunnel_primary_hydrophobicity`
- `tunnel_primary_charge`
- `tunnel_primary_aromatic_fraction`
- `tunnel_max_length`
- `tunnel_total_length`
- `tunnel_min_bottleneck`
- `tunnel_length_over_axial`
- `tunnel_extends_beyond_pocket`
- `tunnel_caver_profile_present`
- `tunnel_has_tunnel`

### tunnel_lean13

- `tunnel_has_tunnel`
- `tunnel_caver_profile_present`
- `tunnel_count`
- `tunnel_primary_length`
- `tunnel_primary_bottleneck_radius`
- `tunnel_primary_curvature`
- `tunnel_primary_hydrophobicity`
- `tunnel_primary_charge`
- `tunnel_primary_aromatic_fraction`
- `tunnel_total_length`
- `tunnel_min_bottleneck`
- `tunnel_length_over_axial`
- `tunnel_extends_beyond_pocket`

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

### tunnel_chem5

- `tunnel_has_tunnel`
- `tunnel_caver_profile_present`
- `tunnel_primary_hydrophobicity`
- `tunnel_primary_charge`
- `tunnel_primary_aromatic_fraction`

### tunnel_shape_avail6

- `tunnel_has_tunnel`
- `tunnel_caver_profile_present`
- `tunnel_count`
- `tunnel_primary_bottleneck_radius`
- `tunnel_length_over_axial`
- `tunnel_extends_beyond_pocket`

### tunnel_single_best3

- `tunnel_primary_hydrophobicity`
- `tunnel_primary_charge`
- `tunnel_primary_aromatic_fraction`

