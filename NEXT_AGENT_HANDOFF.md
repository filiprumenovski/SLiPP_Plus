# NEXT_AGENT_HANDOFF.md

This repo preserves both positive and negative experiment results. Do not delete
`reports/`, `logs/`, `experiments/registry.yaml`, `experiments/queued.md`, or
failed/negative ablation notes.

## Current Best Metrics

Deployable recommendation remains `exp-021-compact-shell6-chem-holdout-weighted`:

- Binary F1: `0.902 +/- 0.016`
- Binary AUROC: `0.989 +/- 0.004`
- 10-class macro-F1: `0.765 +/- 0.021`
- 5-lipid macro-F1: `0.664 +/- 0.034`
- apo-PDB holdout F1/AUROC: `0.717 / 0.802`
- AlphaFold holdout F1/AUROC: `0.715 / 0.855`

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
4. If running more compact ensembles, keep exp-021 as the deployable anchor and
   report both internal lipid5 macro-F1 and apo-PDB/AlphaFold F1. Do not promote
   an internal-only improvement that reproduces exp-019's holdout regression.
5. Keep README current with the deployable recommendation, internal leader, and
   major negative ablations. Stale docs are a known project risk.

## Verification To Re-run

Focused checks used for the latest ablation:

```bash
uv run ruff check src/slipp_plus/composite/backbone_family_encoder.py src/slipp_plus/composite/config.py src/slipp_plus/composite/family_train.py tests/test_pipeline_mode.py tests/test_family_encoder_weights.py
uv run pytest -q tests/test_pipeline_mode.py tests/test_family_encoder_weights.py
make train CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
make eval CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
```

Before a release-facing commit, run:

```bash
make test
```
