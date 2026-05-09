# STE Class-Imbalance Ablation

Question: does extra STE weighting inside the family encoder beat the current
unweighted `v49+tunnel_shape3` backbone or recover enough STE signal to justify
promotion?

## Setup

Baseline: `configs/archive/v49_tunnel_shape3_family_encoder.yaml`

Ablation: `configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml`

Change: multiply the existing inverse-frequency cross-entropy weight for `STE`
by `2.0`, then renormalize the class-weight vector to mean `1.0`. All feature
families, teacher predictions, splits, seeds, model dimensions, and evaluation
paths are otherwise unchanged.

Commands:

```bash
make train CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
make eval CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
```

## Internal Split Results

| run | binary F1 | AUROC | macro-F1 10 | lipid5 macro-F1 | CLR | MYR | OLA | PLM | STE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unweighted shape3 baseline | `0.900 +/- 0.015` | `0.988 +/- 0.004` | `0.768 +/- 0.018` | `0.668 +/- 0.031` | `0.747` | `0.700` | `0.610` | `0.642` | `0.638` |
| STE x2 weight | `0.896 +/- 0.016` | `0.988 +/- 0.003` | `0.760 +/- 0.019` | `0.657 +/- 0.027` | `0.734` | `0.697` | `0.600` | `0.627` | `0.629` |
| delta | `-0.004` | `~0.000` | `-0.008` | `-0.011` | `-0.013` | `-0.003` | `-0.010` | `-0.015` | `-0.009` |

STE recall moved in the expected direction, from `0.733 +/- 0.126` to
`0.741 +/- 0.125`, but STE precision fell from `0.576 +/- 0.105` to
`0.555 +/- 0.101`. The net effect is lower STE F1 and lower lipid macro-F1.

## Holdouts

| run | apo-PDB F1 | apo-PDB AUROC | AlphaFold F1 | AlphaFold AUROC |
|---|---:|---:|---:|---:|
| unweighted shape3 baseline | `0.667` | `0.799` | `0.724` | `0.867` |
| STE x2 weight | `0.649` | `0.803` | `0.711` | `0.870` |

The small AUROC lift does not translate into deployable F1. At the fixed
`0.5` lipid threshold, both external F1 scores are lower than the unweighted
shape3 baseline.

## Decision

Closed as a negative ablation. Extra STE class weighting increases STE recall
slightly but pays for it with enough precision and neighboring lipid-class
damage that both STE F1 and lipid5 macro-F1 regress. This does not beat
exp-014, exp-019, or the holdout-weighted exp-021 release candidate.
