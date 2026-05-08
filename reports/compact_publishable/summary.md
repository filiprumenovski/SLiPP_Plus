# Compact publishable-stack checkpoint

Date: 2026-05-06

## Current answer

The release-facing leader is `v49_tunnel_shape_family_encoder`: 55 features, 433K artifact, lipid5 macro-F1 0.666 +/- 0.032.

It matches the 105-feature tunnel MoE within split noise while using 55 instead of 105 features and 433K instead of 1339K.

## Internal 25-split Comparison

| stack | features | artifact | lipid5 macro-F1 | macro10 F1 | binary F1 | CLR | MYR | OLA | PLM | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v49+tunnel_shape` | 55 | 433K | 0.666 +/- 0.032 | 0.766 +/- 0.019 | 0.902 +/- 0.017 | 0.748 | 0.691 | 0.595 | 0.647 | 0.647 |
| `v49+tunnel_geom` | 58 | 434K | 0.665 +/- 0.036 | 0.765 +/- 0.020 | 0.901 +/- 0.017 | 0.754 | 0.693 | 0.594 | 0.647 | 0.637 |
| `v_tunnel+moe` | 105 | 1339K | 0.664 +/- 0.029 | 0.764 +/- 0.016 | 0.902 +/- 0.014 | 0.739 | 0.693 | 0.598 | 0.658 | 0.633 |
| `paper17+aa20+tunnel_shape` | 43 | 364K | 0.660 +/- 0.031 | 0.762 +/- 0.017 | 0.900 +/- 0.019 | 0.750 | 0.703 | 0.604 | 0.632 | 0.613 |
| `v_sterol+moe` | 87 | 1334K | 0.660 +/- 0.029 | 0.762 +/- 0.015 | 0.901 +/- 0.016 | 0.727 | 0.694 | 0.587 | 0.658 | 0.635 |
| `v49+tunnel_chem` | 54 | 433K | 0.655 +/- 0.032 | 0.757 +/- 0.020 | 0.900 +/- 0.018 | 0.747 | 0.691 | 0.581 | 0.634 | 0.621 |
| `v_sterol` | 87 | 513K | 0.652 +/- 0.029 | 0.757 +/- 0.015 | 0.900 +/- 0.015 | 0.715 | 0.692 | 0.572 | 0.653 | 0.626 |
| `v49` | 49 | 365K | 0.649 +/- 0.026 | 0.756 +/- 0.016 | 0.898 +/- 0.016 | 0.703 | 0.699 | 0.572 | 0.639 | 0.631 |
| `paper17+aa20` | 37 | 296K | 0.645 +/- 0.028 | 0.755 +/- 0.020 | 0.901 +/- 0.020 | 0.699 | 0.704 | 0.570 | 0.634 | 0.619 |
| `paper17+shell12+tunnel_shape` | 35 | 362K | 0.589 +/- 0.036 | 0.698 +/- 0.021 | 0.879 +/- 0.013 | 0.684 | 0.644 | 0.509 | 0.578 | 0.533 |
| `paper17+shell12` | 29 | 294K | 0.567 +/- 0.038 | 0.687 +/- 0.024 | 0.876 +/- 0.016 | 0.640 | 0.632 | 0.459 | 0.573 | 0.530 |
| `paper17` | 17 | 225K | 0.520 +/- 0.044 | 0.649 +/- 0.026 | 0.860 +/- 0.017 | 0.586 | 0.609 | 0.403 | 0.522 | 0.479 |

## Ablation Deltas

| comparison | lipid5 delta | interpretation |
|---|---:|---|
| `paper17+shell12` vs `paper17` | +0.047 | shell12 alone is modest |
| `paper17+aa20` vs `paper17` | +0.125 | AA20 carries the major recovery |
| `v49` vs `paper17+aa20` | +0.004 | shell12 adds little once AA20 is present |
| `paper17+shell12+tunnel_shape` vs `paper17+shell12` | +0.023 | tunnel cannot compensate for removing AA20 |
| `paper17+aa20+tunnel_shape` vs `paper17+aa20` | +0.015 | tunnel lift without shell12 |
| `v49+tunnel_shape` vs `paper17+aa20+tunnel_shape` | +0.005 | isolates shell12 value in the tunnel stack |
| `v49+tunnel_chem` vs `v49` | +0.006 | tunnel chemistry is a smaller lift |
| `v49+tunnel_shape` vs `v49` | +0.017 | best compact tunnel lift |
| `v49+tunnel_geom` vs `v49` | +0.016 | ties shape within noise with more columns |
| `v_tunnel+moe` vs `v49+tunnel_shape` | -0.001 | high-complexity reference does not improve the compact leader |

## Interpretation

The main recovery comes from AA20, not shell12 alone: `paper17+shell12` is 0.567, `paper17+aa20` is 0.645, and `v49` is 0.649.

Decision rule: keep the smallest stack whose lipid5 macro-F1 is within 0.01-0.015 of the best observed model and does not degrade STE. Under current results, `v49+tunnel_shape` is the working release candidate; `v49` is the stricter parsimony fallback.
