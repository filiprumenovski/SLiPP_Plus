# Queued Experiment Work

This file queues long or non-inline validation work identified from `experiments/registry.yaml`. It preserves incomplete and negative results rather than deleting or overwriting them.

## 2026-05-08 Registry Partial-Holdout Audit

Generated from the registry after `reports/ablation_matrix.md` was created.

### Queue For Holdout Completion

1. `exp-005-v_sterol-ensemble`
   - Config: `configs/v_sterol.yaml`
   - Current gap: registry has no `holdouts` block.
   - Suggested command: rerun/evaluate the saved `v_sterol` iteration-0 RF/XGB/LGBM bundles against current apo-PDB and AlphaFold parquet artifacts, then append `apo_pdb_f1`, `apo_pdb_auroc`, `alphafold_f1`, and `alphafold_auroc` to the registry.

2. `exp-009-v_sterol-boundary-refactor`
   - Config: `configs/v_sterol.yaml`
   - Current gap: registry has no `holdouts` block for the boundary-refactor postprocessing.
   - Suggested command: run the holdout validation path for the saved `ste_rescue_ola_plm_pair` predictions or regenerate the postprocessed holdout predictions, then update the registry with apo-PDB and AlphaFold metrics.

3. `exp-011-family-plus-moe`
   - Config: `configs/v_sterol_family_plus_moe.yaml`
   - Current gap: registry holdout fields are explicitly `null`.
   - Current blocker from registry notes: holdout inference for composite pair/local MoE bundles is not implemented yet.
   - Suggested command: implement or wire composite holdout inference for `models/v_sterol_family_plus_moe/family_plus_moe_bundle.joblib`, then update holdout metrics.

### No Inline Completion Planned

1. `exp-003-v49-ensemble-clr-ste-tb`
   - Reason: negative-targeting experiment; registry notes show the intervention fired only 2 times across 25 iterations. Holdout completion is low priority unless a manuscript specifically discusses CLR/STE tiebreaking.

2. `exp-006-v_plm_ste-features`
   - Reason: explicitly abandoned in registry notes because the feature set was neutral-to-regressive for STE in the 10-class softmax.

3. `exp-008-detector-bakeoff`
   - Reason: detector comparison, not a classifier holdout experiment; apo-PDB/AlphaFold classifier holdout fields are not applicable.

## Missing Publishable Ablations

These handoff items require long model runs or additional implementation and should not be run inline without long-run approval:

1. STE class-imbalance handling: baseline vs class-weighted comparison for STE.

## Closed From Existing Artifacts

1. Ensemble vs best single model (`handoff.md` 8.4)
   - Report: `reports/ensemble_ablation.md`
   - Result: weak positive but below the requested >=1 std bar. RF+XGB+LGBM
     mean-probability ensembles improve lipid macro-F1 on every checked flat
     stack (`+0.006` to `+0.012`), but none by at least one standard deviation.
   - No new model training was run; this used persisted prediction artifacts.

2. Tiebreaker on/off (`handoff.md` 8.5)
   - Report: `reports/tiebreaker_ablation.md`
   - Result: the narrow PLM/STE tiebreaker is small (`+0.009` lipid macro-F1,
     `+0.045` STE F1), while the broader STE-neighbor rescue is material
     (`+0.038` to `+0.040` lipid macro-F1, `+0.178` STE F1).
   - No new model training was run; this used persisted `v_sterol` prediction
     artifacts.

3. Compact subset ensemble sweep
   - Report: `reports/compact_subset_ensemble_sweep/summary.md`
   - Result: the five-way shape/chem blend remains the internal leader
     (`0.684` lipid5 macro-F1), but a chem-heavy `20% shell6_shape / 80% chem`
     weighting is the best holdout-balanced compact blend found so far
     (apo-PDB F1 `0.717`, AlphaFold F1 `0.715`, lipid5 macro-F1 `0.664`).
   - No new model training was run; this swept equal-probability subsets over
     existing compact prediction artifacts plus a targeted shell6/chem weight
     sweep.

4. CAVER/tunnel marginal value (`handoff.md` 8.6)
   - Report: `reports/caver_ablation.md`
   - Result: CAVER/tunnel features are real but small. Compact tunnel shape
     adds about `+0.011` to `+0.017` lipid macro-F1 over matched no-tunnel
     baselines, while the full raw tunnel block is redundant and weaker than
     screened compact subsets.
   - No new model training was run; this consolidates existing v_sterol-aligned
     tunnel screens and compact ladder artifacts.
