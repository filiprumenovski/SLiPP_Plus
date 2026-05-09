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

Internal-validation leader is `exp-030-probability-blend-internal-leader`:

- Binary F1: `0.908 +/- 0.015`
- Binary AUROC: `0.990 +/- 0.003`
- 10-class macro-F1: `0.781 +/- 0.018`
- 5-lipid macro-F1: `0.687 +/- 0.031`
- Component blend: `0.20 compact_shape3_shape6 + 0.20 v_sterol + 0.10 exp-028
  + 0.50 compact five-way`
- Holdouts regress to apo-PDB `0.643` and AlphaFold `0.536`, so do not deploy
  it. It is an internal-only leader and a domain-shift warning.

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

`exp-030-probability-blend-internal-leader` is closed internal-positive but
holdout-negative.

- Report: `reports/probability_blend_sweep_2026_05_09.md`
- Result: improves internal binary F1 to `0.908 +/- 0.015`, macro10 to
  `0.781 +/- 0.018`, lipid5 macro-F1 to `0.687 +/- 0.031`, and STE F1 to
  `0.652`.
- Holdouts: apo-PDB F1/AUROC `0.643 / 0.766`; AlphaFold F1/AUROC
  `0.536 / 0.738`.
- Decision: not deployable. The same sweep found a holdout-mean-positive
  diagnostic (`0.35 paper17_family_encoder + 0.65 v_sterol`: apo-PDB F1
  `0.739`, AlphaFold F1 `0.711`, holdout mean `0.725`), but it loses internal
  subclass quality and AlphaFold F1 versus exp-028.

`exp-031-legacy-rescue-rule-diagnostic` is the strongest current lead.

- Report: `reports/legacy_rescue_rule_ablation_2026_05_09.md`
- Rule: start from exp-028; when exp-028 calls non-lipid but both
  `paper17_family_encoder` has lipid probability `>= 0.35`, `v_sterol` has
  lipid probability `>= 0.55`, and the legacy-minus-exp028 margin is at least
  `0.10`, replace probabilities with the paper17/v_sterol average.
- Result: apo-PDB F1/AUROC `0.729 / 0.761`; AlphaFold F1/AUROC
  `0.735 / 0.762`; internal binary F1 `0.901 +/- 0.015`, lipid5 macro-F1
  `0.668`.
- Decision: do not promote yet because the `0.35` thresholds came from a
  holdout-scored diagnostic grid. This is the next thing to make holdout-safe:
  select thresholds from internal split predictions or train a 25-split rescue
  gate.

`exp-032-legacy-rescue-holdout-safe-gate` is the strongest deployable lead.

- Report: `reports/legacy_rescue_holdout_safe_ablation_2026_05_09.md`
- Internal threshold selection picks a strict rule:
  `paper17 >= 0.50`, `v_sterol >= 0.85`, `margin >= 0.00`.
- Internal hard-threshold selection picks a strict rule:
  `paper17 >= 0.50`, `v_sterol >= 0.85`, `margin >= 0.00`; it reaches
  apo-PDB F1 `0.710` and AlphaFold F1 `0.722`, not enough to beat exp-028.
- A simple logistic rescue gate trained only on internal prediction features is
  positive: internal binary F1 `0.901 +/- 0.018`, lipid5 `0.667`, apo-PDB
  F1/AUROC `0.732 / 0.793`, AlphaFold F1/AUROC `0.755 / 0.847`.
- Decision: first holdout-safe candidate that beats exp-028 on both external
  F1 scores. Make it fully reproducible before marking it deployable.

`exp-033-covariate-shift-threshold-neutral` is closed neutral.

- Report: `reports/covariate_shift_threshold_ablation_2026_05_09.md`
- Method: train a source-vs-holdout domain classifier in exp-028 probability
  space, use unlabeled target density-ratio weights to select a binary lipid
  threshold on internal rows, then apply once to holdouts.
- Result: domain AUROC is high (`0.932` apo-PDB, `0.967` AlphaFold), but
  selected thresholds remain near `0.5`. Holdout F1 stays unchanged from
  exp-028: apo-PDB `0.717`, AlphaFold `0.724`.
- Decision: probability-space covariate reweighting documents domain shift but
  is not a threshold-selection mechanism.

`exp-034-holdout-label-source-audit` closed a row-order trap.

- Report: `reports/holdout_label_source_audit_2026_05_09.md`
- Finding: root holdout files and component-specific holdout feature files have
  the same identities but different row order. Row-position comparison creates
  apparent `class_binary` disagreements, but labels agree exactly after
  aligning by `structure_id` and `ligand`.
- Fix: `scripts/compact_probability_ensemble.py` now prefers canonical root
  labels when present and aligns them to component prediction order by
  `structure_id`/`ligand`.
- Decision: old exp-028 compact metrics remain valid; future ad hoc holdout
  scoring must align labels by identity, not row position.

`exp-005-v_sterol-ensemble` holdouts are now complete from existing artifacts.

- Report: `reports/v_sterol_holdout_completion.md`
- Result: apo-PDB F1/AUROC `0.679 / 0.812`; AlphaFold F1/AUROC
  `0.708 / 0.864`.

`exp-011-family-plus-moe` holdouts are now complete.

- Report: `reports/family_plus_moe_holdout_completion.md`
- Result: apo-PDB F1/AUROC `0.723 / 0.807`; AlphaFold F1/AUROC
  `0.703 / 0.838`.
- Decision: useful but not deployable over exp-028 because AlphaFold is weaker.

`exp-009-v_sterol-boundary-refactor` holdouts are now complete for binary
holdout reporting.

- Reports: `reports/boundary_refactor_holdout_attempt.md`,
  `reports/boundary_refactor_holdout_completion.md`
- Result: apo-PDB F1/AUROC `0.679 / 0.812`; AlphaFold F1/AUROC
  `0.708 / 0.864`.
- Note: the boundary-refactor bundle changes subclass probabilities/classes but
  makes zero binary holdout decision changes versus the flat v_sterol ensemble.

## Remaining High-Impact Work

1. Make exp-032 fully reproducible as a script/report artifact and then decide
   whether it should supersede exp-028 as deployable. It is the first
   holdout-safe candidate to beat exp-028 on both external F1 scores.
2. Prioritize domain-shift fixes that can be learned without tuning on holdout
   labels. The holdout threshold diagnostic showed lower deployable thresholds
   would help externally, but internal threshold selection did not reproduce
   those thresholds.
3. Revisit targeted STE handling only if it is more local than a global class
   weight: e.g. a calibrated PLM/STE/COA/MYR/OLA expert, confidence gating, or
   a data-extension path that adds STE-like pockets.
4. If running more compact ensembles, keep exp-028 as the deployable anchor and
   report both internal lipid5 macro-F1 and apo-PDB/AlphaFold F1. Do not promote
   an internal-only improvement that reproduces exp-019/exp-030 holdout
   regression.
5. Keep README current with the deployable recommendation, internal leader, and
   major negative ablations. Stale docs are a known project risk.
6. Registry holdout-completion gaps from the 2026-05-08 audit are closed:
   `exp-005`, `exp-009`, and `exp-011`.
7. Do not use generic test-split metrics from
   `make eval CFG=configs/v_sterol_boundary_refactor.yaml` for exp-009. That
   command reads `hierarchical_lipid_predictions.parquet`, which is an exp-011
   composite pair-MoE artifact. Use the registry's exp-009 internal metrics.

## Verification To Re-run

Focused checks used for the latest ablation:

```bash
uv run ruff check src/slipp_plus/composite/backbone_family_encoder.py src/slipp_plus/composite/config.py src/slipp_plus/composite/family_train.py tests/test_pipeline_mode.py tests/test_family_encoder_weights.py
uv run pytest -q tests/test_pipeline_mode.py tests/test_family_encoder_weights.py
make train CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
make eval CFG=configs/archive/v49_tunnel_shape3_ste2_family_encoder.yaml
uv run python scripts/compact_probability_ensemble.py --component-dir processed/v49_tunnel_shape3 --component-dir processed/v49_shell6_tunnel_shape --component-dir processed/v49_tunnel_chem --component-weight 0.1 --component-weight 0.2 --component-weight 0.7 --model-name shape3_shell6_chem_weighted_10_20_70 --report-title "Compact shape3 shell6 chem weighted probability ensemble" --output-predictions-dir processed/compact_shape3_shell6_chem_weighted_10_20_70/predictions --output-report-dir reports/compact_shape3_shell6_chem_weighted_10_20_70
uv run python scripts/generate_ablation_matrix.py
```

Before a release-facing commit, run:

```bash
make test
```
