# NEXT_AGENT_HANDOFF.md

This repo preserves both positive and negative experiment results. Do not delete
`reports/`, `logs/`, `experiments/registry.yaml`, `experiments/queued.md`, or
failed/negative ablation notes.

## Current Best Metrics

Deployable recommendation is `exp-028-compact-shape3-shell6-chem-weighted`:

- Binary F1: `0.903 +/- 0.016`
- Binary AUROC: `0.989 +/- 0.003`
- 10-class macro-F1: `0.769 +/- 0.019`
- 5-lipid macro-F1: `0.670 +/- 0.032`
- apo-PDB holdout F1/AUROC: `0.717 / 0.801`
- AlphaFold holdout F1/AUROC: `0.724 / 0.855`
- Component blend: `0.1 shape3 + 0.2 shell6_shape + 0.7 chem`

Internal-validation leader remains `exp-019-compact-five-way-shape-chem-ensemble`:

- Binary F1: `0.906 +/- 0.015`
- 10-class macro-F1: `0.778 +/- 0.017`
- 5-lipid macro-F1: `0.684 +/- 0.030`
- Holdouts regress to apo-PDB `0.649` and AlphaFold `0.623`, so do not deploy
  it without solving the external transfer issue.

## Latest Completed Ablation

`exp-027-ste-class-weight-x2` is closed negative.

- Config: `configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml`
- Report: `reports/ste_imbalance_ablation.md`
- Result: doubling STE's class weight increases STE recall slightly
  (`0.733 -> 0.741`) but lowers STE precision (`0.576 -> 0.555`), STE F1
  (`0.638 -> 0.629`), lipid5 macro-F1 (`0.668 -> 0.657`), and both holdout F1
  scores versus the unweighted `v49+tunnel_shape3` baseline.
- Decision: do not promote simple STE overweighting.

`exp-028-compact-shape3-shell6-chem-weighted` is the newest positive result.

- Report: `reports/compact_weight_grid_sweep.md`
- Metrics: `reports/compact_shape3_shell6_chem_weighted_10_20_70/metrics.md`
- Result: improves exp-021 by keeping apo-PDB F1 tied at `0.717`, raising
  AlphaFold F1 from `0.715` to `0.724`, and raising internal lipid5 macro-F1
  from `0.664` to `0.670`.
- Decision: current deployable recommendation, while exp-019 remains the
  internal-validation leader only.

`exp-029-compact-weight-local-refinement` is closed negative.

- Report: `reports/compact_weight_local_refinement.md`
- Result: a bounded 0.05-resolution local grid around exp-028 did not find a
  better holdout-balanced candidate after allowing small optional mass from
  `shape6`, `shell6_shape3`, `hydro4`, and `geom`.
- Decision: exp-028 is locally stable under this search.

`exp-005-v_sterol-ensemble` holdouts are now complete from existing artifacts.

- Report: `reports/v_sterol_holdout_completion.md`
- Result: apo-PDB F1/AUROC `0.679 / 0.812`; AlphaFold F1/AUROC
  `0.708 / 0.864`.

## Remaining High-Impact Work

1. Prioritize domain-shift fixes that can be learned without tuning on holdout
   labels. The holdout threshold diagnostic showed lower deployable thresholds
   would help externally, but internal threshold selection did not reproduce
   those thresholds.
2. Build a holdout-safe calibration or domain-adaptation experiment using only
   internal split predictions and unlabeled holdout feature distributions.
3. Revisit targeted STE handling only if it is more local than a global class
   weight: e.g. a calibrated PLM/STE/COA/MYR/OLA expert, confidence gating, or
   a data-extension path that adds STE-like pockets.
4. If running more compact ensembles, keep exp-028 as the deployable anchor and
   report both internal lipid5 macro-F1 and apo-PDB/AlphaFold F1. Do not promote
   an internal-only improvement that reproduces exp-019's holdout regression.
5. Keep README current with the deployable recommendation, internal leader, and
   major negative ablations. Stale docs are a known project risk.
6. Remaining queued holdout completions are `exp-009-v_sterol-boundary-refactor`
   and `exp-011-family-plus-moe`; `exp-005` is closed.

## Verification To Re-run

Focused checks used for the latest ablation:

```bash
uv run ruff check src/slipp_plus/composite/backbone_family_encoder.py src/slipp_plus/composite/config.py src/slipp_plus/composite/family_train.py tests/test_pipeline_mode.py tests/test_family_encoder_weights.py
uv run pytest -q tests/test_pipeline_mode.py tests/test_family_encoder_weights.py
make train CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
make eval CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
uv run python scripts/compact_probability_ensemble.py --component-dir processed/v49_tunnel_shape3 --component-dir processed/v49_shell6_tunnel_shape --component-dir processed/v49_tunnel_chem --component-weight 0.1 --component-weight 0.2 --component-weight 0.7 --model-name shape3_shell6_chem_weighted_10_20_70 --report-title "Compact shape3 shell6 chem weighted probability ensemble" --output-predictions-dir processed/compact_shape3_shell6_chem_weighted_10_20_70/predictions --output-report-dir reports/compact_shape3_shell6_chem_weighted_10_20_70
```

Before a release-facing commit, run:

```bash
make test
```
