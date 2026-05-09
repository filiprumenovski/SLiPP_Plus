# Queued Experiment Work

This file queues long or non-inline validation work identified from `experiments/registry.yaml`. It preserves incomplete and negative results rather than deleting or overwriting them.

## 2026-05-08 Registry Partial-Holdout Audit

Generated from the registry after `reports/ablation_matrix.md` was created.

### Queue For Holdout Completion

All registry holdout-completion gaps from this audit are now closed. Keep this
section for future registry audits.

### No Inline Completion Planned

1. `exp-003-v49-ensemble-clr-ste-tb`
   - Reason: negative-targeting experiment; registry notes show the intervention fired only 2 times across 25 iterations. Holdout completion is low priority unless a manuscript specifically discusses CLR/STE tiebreaking.

2. `exp-006-v_plm_ste-features`
   - Reason: explicitly abandoned in registry notes because the feature set was neutral-to-regressive for STE in the 10-class softmax.

3. `exp-008-detector-bakeoff`
   - Reason: detector comparison, not a classifier holdout experiment; apo-PDB/AlphaFold classifier holdout fields are not applicable.

## Closed From Existing Artifacts

1. Ensemble vs best single model
   - Report: `reports/ensemble_ablation.md`
   - Result: weak positive but below the requested >=1 std bar. RF+XGB+LGBM
     mean-probability ensembles improve lipid macro-F1 on every checked flat
     stack (`+0.006` to `+0.012`), but none by at least one standard deviation.
   - No new model training was run; this used persisted prediction artifacts.

2. Tiebreaker on/off
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

4. CAVER/tunnel marginal value
   - Report: `reports/caver_ablation.md`
   - Result: CAVER/tunnel features are real but small. Compact tunnel shape
     adds about `+0.011` to `+0.017` lipid macro-F1 over matched no-tunnel
     baselines, while the full raw tunnel block is redundant and weaker than
     screened compact subsets.
   - No new model training was run; this consolidates existing v_sterol-aligned
     tunnel screens and compact ladder artifacts.

5. Holdout threshold diagnostic
   - Report: `reports/holdout_threshold_ablation.md`
   - Result: the five-way internal leader's holdout regression is primarily
     recall/calibration-driven. At the default `0.5` lipid-probability
     threshold it has high precision but many false negatives; diagnostic
     thresholds around `0.25`-`0.35` recover large F1 on both holdouts.
   - Do not deploy holdout-tuned thresholds. Use this as evidence for a future
     calibration step learned inside the 25-split protocol.

6. Internal threshold selection
   - Report: `reports/internal_threshold_selection_ablation.md`
   - Result: thresholds selected from the 25 internal splits stay near the
     default (`0.51`-`0.54`) and slightly worsen apo-PDB/AlphaFold F1 for every
     checked compact ensemble.
   - This closes the obvious holdout-safe threshold-calibration idea as a
     negative result; the low-threshold holdout gains are domain-shift evidence,
     not an internally recoverable threshold rule.

7. Domain-shift false-negative feature audit
   - Report: `reports/domain_shift_false_negative_features.md`
   - Result: exp-019 holdout false negatives have lipid probabilities near
     non-lipid background and are depleted in hydrophobic/aromatic shell plus
     size-density signals (`LEU`, `PHE`, shell3/4 aliphatic counts,
     `as_density`, `as_max_dst`). Apo-PDB false negatives also show higher
     planarity/elongation.
   - This suggests external misses are feature-manifold/domain-shift failures,
     not just threshold mistakes.

8. Domain-shift nearest-neighbor audit
   - Report: `reports/domain_shift_nearest_neighbors.md`
   - Result: exp-019 holdout false negatives are locally much less lipid-like
     than recalled holdout lipids. Apo-PDB FN nearest-10 lipid fraction is
     `0.190` vs `0.681` for TP; AlphaFold FN is `0.271` vs `0.702`.
   - The hardest false negatives are often surrounded by internal `PP`/`COA`
     neighbors, explaining why lower thresholds recover recall but do not solve
     the underlying feature-space mismatch.

9. Domain-shift component-rescue audit
   - Report: `reports/domain_shift_component_rescue.md`
   - Result: no single compact component rescues most exp-019 holdout false
     negatives at `p_lipid >= 0.5`. `shell6_shape` and `chem` are best on
     apo-PDB (22.6% rescue each), while AlphaFold rescue is weak and diffuse
     (`shape3` rescues 25.0%).
   - This rules out a simple component override as the next easy fix.

10. STE class-imbalance handling
   - Report: `reports/ste_imbalance_ablation.md`
   - Result: negative. Doubling STE's already inverse-frequency-corrected
     family-encoder class weight increases STE recall slightly (`0.733` to
     `0.741`) but lowers STE precision (`0.576` to `0.555`), STE F1 (`0.638`
     to `0.629`), lipid5 macro-F1 (`0.668` to `0.657`), and both holdout F1
     scores versus the unweighted shape3 baseline.
   - This closes the simple class-weight route; future STE work needs a more
     targeted classifier, calibration, or data-extension intervention.

11. Weighted compact blend promotion
   - Report: `reports/compact_weight_grid_sweep.md`
   - Result: positive. `0.10 shape3 / 0.20 shell6_shape / 0.70 chem` improves
     the prior exp-021 deployable recommendation by tying apo-PDB F1 (`0.717`),
     improving AlphaFold F1 (`0.715` to `0.724`), and improving internal lipid5
     macro-F1 (`0.664` to `0.670`).
   - Registry: `exp-028-compact-shape3-shell6-chem-weighted` is now the
     deployable recommendation.

12. Compact weight local refinement
   - Report: `reports/compact_weight_local_refinement.md`
   - Result: negative. A bounded 0.05-resolution local grid around exp-028 found
     no better holdout-balanced candidate after allowing small optional mass
     from `shape6`, `shell6_shape3`, `hydro4`, and `geom`.

13. `exp-005-v_sterol-ensemble` holdout completion
   - Report: `reports/v_sterol_holdout_completion.md`
   - Result: registry holdouts filled from existing persisted ensemble holdout
     predictions. apo-PDB F1/AUROC `0.679 / 0.812`; AlphaFold F1/AUROC
     `0.708 / 0.864`.
   - No retraining was run.

14. `exp-011-family-plus-moe` holdout completion
   - Report: `reports/family_plus_moe_holdout_completion.md`
   - Result: pair/local-MoE holdout inference now starts from family-encoder
     teacher holdout predictions and applies saved iteration-0 experts. apo-PDB
     F1/AUROC `0.723 / 0.807`; AlphaFold F1/AUROC `0.703 / 0.838`.
   - This closes the composite pair/local-MoE holdout blocker, but does not
     change the deployable recommendation because exp-028 is stronger on
     AlphaFold.

15. `exp-009-v_sterol-boundary-refactor` holdout completion
   - Reports: `reports/boundary_refactor_holdout_attempt.md`,
     `reports/boundary_refactor_holdout_completion.md`
   - Result: the saved boundary-refactor bundle changes holdout subclass
     probabilities/classes but makes zero binary lipid/non-lipid decision
     changes versus the flat `v_sterol` ensemble. Binary holdout metrics
     therefore match exp-005: apo-PDB F1/AUROC `0.679 / 0.812`; AlphaFold
     F1/AUROC `0.708 / 0.864`.
   - Do not use stale generic test-split metrics emitted by
     `make eval CFG=configs/v_sterol_boundary_refactor.yaml` for exp-009.
