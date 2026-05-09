# Compact Weight Grid Sweep

Question: can a simple weighted blend over existing compact components beat the
holdout-weighted exp-021 candidate without new training?

## Setup

Components checked at 0.1 grid resolution:

- `shape3`: `processed/v49_tunnel_shape3`
- `shape6`: `processed/v49_tunnel_shape`
- `shell6`: `processed/v49_shell6_tunnel_shape`
- `shell6shape3`: `processed/v49_shell6_tunnel_shape3`
- `chem`: `processed/v49_tunnel_chem`

The best 0.1-grid candidate was refined at 0.05 resolution over
`shape3 + shell6 + chem`.

Promotion command:

```bash
uv run python scripts/compact_probability_ensemble.py \
  --component-dir processed/v49_tunnel_shape3 \
  --component-dir processed/v49_shell6_tunnel_shape \
  --component-dir processed/v49_tunnel_chem \
  --component-weight 0.1 \
  --component-weight 0.2 \
  --component-weight 0.7 \
  --model-name shape3_shell6_chem_weighted_10_20_70 \
  --report-title "Compact shape3 shell6 chem weighted probability ensemble" \
  --output-predictions-dir processed/compact_shape3_shell6_chem_weighted_10_20_70/predictions \
  --output-report-dir reports/compact_shape3_shell6_chem_weighted_10_20_70
```

## Best Candidates

| blend | weights | lipid5 | binary F1 | STE | apo-PDB F1 | AlphaFold F1 | holdout mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shape3 + shell6 + chem` | `0.10 / 0.20 / 0.70` | `0.670` | `0.903` | `0.644` | `0.717` | `0.724` | `0.720` |
| `shell6 + chem` | `0.20 / 0.80` | `0.664` | `0.902` | `0.629` | `0.717` | `0.715` | `0.716` |
| `shape3 + shell6` | `0.10 / 0.90` | `0.670` | `0.901` | `0.646` | `0.711` | `0.711` | `0.711` |
| `shape3 + shell6 + chem` | `0.10 / 0.30 / 0.60` | `0.673` | `0.903` | `0.644` | `0.706` | `0.715` | `0.711` |

## Promoted Metrics

Report: `reports/compact_shape3_shell6_chem_weighted_10_20_70/metrics.md`

| metric | value |
|---|---:|
| binary F1 | `0.903 +/- 0.016` |
| binary AUROC | `0.989 +/- 0.003` |
| 10-class macro-F1 | `0.769 +/- 0.019` |
| 5-lipid macro-F1 | `0.670 +/- 0.032` |
| apo-PDB F1 / AUROC | `0.717 / 0.801` |
| AlphaFold F1 / AUROC | `0.724 / 0.855` |
| CLR / MYR / OLA / PLM / STE F1 | `0.758 / 0.705 / 0.599 / 0.643 / 0.644` |

## Decision

Promote as the new deployable recommendation. It keeps the holdout-stability
discipline that motivated exp-021, improves AlphaFold F1 by `+0.009`, keeps
apo-PDB F1 tied at `0.717`, and recovers internal lipid5 macro-F1 from `0.664`
to `0.670`.

This does not supersede exp-019 as the internal-validation leader; exp-019 still
has the strongest internal lipid5 macro-F1 but remains holdout-regressive.
