# v_tunnel XGB ablation confirmation

Aligned to `processed/v_tunnel_aligned`; 25 canonical splits; selected variants from the LGBM screen.

| variant | extra | lipid5 | delta | macro10 | binary F1 | CLR | OLA | PLM | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tunnel_shape_avail6 | 6 | 0.609 +/- 0.030 | +0.015 | 0.734 | 0.895 | 0.747 | 0.573 | 0.639 | 0.408 |
| tunnel_chem5 | 5 | 0.608 +/- 0.030 | +0.014 | 0.733 | 0.894 | 0.758 | 0.569 | 0.636 | 0.407 |
| tunnel_geom9 | 9 | 0.605 +/- 0.026 | +0.011 | 0.730 | 0.891 | 0.751 | 0.567 | 0.624 | 0.414 |
| tunnel_all18 | 18 | 0.600 +/- 0.037 | +0.006 | 0.727 | 0.893 | 0.745 | 0.566 | 0.622 | 0.393 |
| v_sterol_only | 0 | 0.594 +/- 0.028 | +0.000 | 0.726 | 0.896 | 0.709 | 0.530 | 0.627 | 0.416 |
