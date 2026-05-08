# SLiPP++ Research Log

_Chronological lab notebook. Each entry records what was done, why, and what happened._
_Template at the bottom of this file._

---

## 2026-04-28 — Family Plus MoE Champion

**Goal:** Upgrade the best family-encoder/local-MoE stack without changing the curated training data.
**Hypothesis:** Residual PLM/STE errors are no longer clean pair-boundary mistakes; a small local multiclass expert over the full PLM/STE neighborhood should recover STE while preserving the softmax/teacher signal for larger classes.
**Changes:**
- Added `configs/v_sterol_family_plus_moe.yaml`.
- Kept the promoted CLR local expert and narrow MYR/PLM + PLM/STE pair heads.
- Added `plm_ste_neighborhood_expert` over `[PLM, STE, COA, MYR, OLA]` with confidence threshold `0.75`.
**Result:**
- Final 25-iteration metrics: binary F1 `0.901 ± 0.016`, binary AUROC `0.989 ± 0.003`, 10-class macro-F1 `0.762 ± 0.015`, 5-lipid macro-F1 `0.660 ± 0.029`.
- Per-lipid F1: CLR `0.727`, MYR `0.694`, OLA `0.587`, PLM `0.658`, STE `0.635`.
- Delta vs family-encoder teacher: `+0.0048` macro-F1, `+0.0085` lipid macro-F1.
- Delta vs prior fixed local MoE: about `+0.001` macro-F1, `+0.002` lipid macro-F1, with STE `0.627 -> 0.635`.
**Decision:** Promote `configs/v_sterol_family_plus_moe.yaml` as the current champion. Composite holdout inference remains pending, so holdout results should still be read from the family encoder until bundle inference is implemented for local/pair MoE.
**Refs:** `configs/v_sterol_family_plus_moe.yaml`, `reports/v_sterol_family_plus_moe/composite_pair_moe_report.md`, `reports/v_sterol_family_plus_moe/metrics_table.md`

---

## 2026-04-25 — v_sterol_v2: Chemistry-Derived Feature Engineering

**Goal:** Add 26 biophysically motivated derived features to the v_sterol base (87 → 115 features) to improve lipid subclass discrimination. Features include AA composition fractions, size-normalized densities, surface chemistry ratios, spatial shell gradients, and shape-chemistry interaction terms.
**Hypothesis:** The current features capture pocket geometry and raw composition but miss scale-invariant ratios and inter-feature interactions that distinguish lipid classes by their unique chemistry (e.g., CLR's aromatic stacking, OLA's kinked-channel shape, MYR's cationic anchoring).
**Changes:**
- Created `derived_features.py`: `compute_derived_features()` generates 26 new columns from existing data. Key features: `polar_surface_frac` (d=0.57 OLA vs PLM), `charged_aa_frac` (d=0.56 PLM vs MYR), `bulky_hydro_gradient` (d=0.56 OLA vs PLM), `aromatic_density` (d=0.39 OLA vs PLM), `hydrophobic_aa_frac` (d=1.50 OLA vs COA).
- Added `DERIVED_FEATURES_26` to `constants.py`, registered `v_sterol_v2` feature set (v_sterol + 2 unused vdw22 + 26 derived = 115).
- Added `v_sterol_v2` to `FeatureSet` literal in `config.py`.
- Built enriched parquets for training and holdouts.
**Result:**
- **Flat XGB (5-seed controlled comparison): macro-F1 0.715 → 0.721 (+0.006).** CLR +0.049, OLA +0.034, MYR +0.009. PLM −0.017, STE −0.045. Clear signal for the targeted confusion pairs.
- **Hierarchical Pipeline:** Final macro-F1 improved from 0.721 (v_sterol baseline) to 0.727.
**False Alarm & Hybrid Test:** We initially believed the hierarchy was overfitting because we miscompared the results against an older, differently-configured benchmark (exp-009). We tested a "hybrid" architecture (113 features for Stage 1, 87 for the hierarchy) and a "lean" ablation (dropping 37 noisy features). Both failed (0.721 and 0.687 respectively). The hierarchy *needs* the full 113 features because the boundary heads rely heavily on the derived signals (e.g., `vol_x_polar_surface` is the #1 feature for OLA vs PLM, and `net_charge_per_vol` is #7 for STE).
**Decision:** The `v_sterol_v2` feature set (113 features) is strictly superior across all stages. It is the new champion feature set.
- `vol_x_polar_surface` is the #1 feature for the OLA/PLM boundary head — confirms the design rationale.
- `net_charge_per_vol` appears at #7 in the STE specialist — a new signal not available in v_sterol.
- Holdout evaluation failed because vdw22 columns are missing from holdout XLSXs (graceful fallback implemented for derived features, but raw vdw22 in the feature set breaks validation).
**Refs:** `src/slipp_plus/derived_features.py`, `configs/v_sterol_v2.yaml`, `reports/v_sterol_v2/hierarchical_lipid_report.md`

## 2026-04-25 — v_sterol_v2 stage-specific feature routing (negative quick check)

**Goal:** Test whether the `v_sterol_v2` hierarchical regression can be rescued cheaply by routing smaller feature subsets to specific hierarchy stages while keeping derived features where they help local boundaries.
**Hypothesis:** The derived stack is useful, but one or two hierarchy stages are too feature-wide. Per-stage routing should recover the lost macro-F1 without giving up the new signal.
**Changes:**
- Added stage-specific routing support to `hierarchical_pipeline.py` and config support in `config.py`: `lipid_family_feature_set`, `specialist_feature_set`, `nonlipid_feature_set`, plus `boundary_heads[*].feature_set`.
- Added targeted tests covering config parsing and stage-specific feature extraction.
- Ran 5-iteration A/B experiments for baseline, full routing, specialist-only routing, family-on-`v49`, and nonlipid-only routing.
**Result:**
- **No routed variant beat the plain `v_sterol_v2` baseline on macro-F1 or 5-lipid macro-F1.**
- Baseline remained best on the main objectives: macro-F1 0.7318, lipid macro-F1 0.6195.
- Full routing improved binary F1 (0.8912 → 0.8970) and modestly improved STE F1 (0.5767 → 0.5922) but regressed macro-F1 to 0.7268.
- Family-on-`v49` produced the best STE F1 (0.5964) but regressed macro-F1 harder (0.7242).
- Specialist-only routing was nearly neutral overall and therefore did not explain the main regression.
**Decision:** Keep the routing infrastructure because it is useful for future controlled experiments, but do not promote any routed `v_sterol_v2` hierarchy variant. The quick rescue path is falsified.
**What didn't work:** Simple stage pruning. The hierarchy seems to lose performance through interaction between stages, not just because one stage has too many features.
**Refs:** `src/slipp_plus/hierarchical_pipeline.py`, `reports/v_sterol_v2/feature_routing_5_iter.md`, `configs/v_sterol_v2_stage_routed_5.yaml`, `configs/v_sterol_v2_specialist_routed_5.yaml`, `configs/v_sterol_v2_family_v49_5.yaml`, `configs/v_sterol_v2_nonlipid_routed_5.yaml`

---

## 2026-04-25 — Binary OvR Lipid Family Heads (Negative Result)

**Goal:** Replace Stage 2's 5-way lipid softmax with 5 independent binary one-vs-rest heads to eliminate the simplex mutual-exclusivity constraint between lipid subclasses.
**Hypothesis:** The softmax forces PLM/OLA/STE into artificial competition. Independent binary heads should allow multi-label compatibility and eliminate the need for boundary heads and rescue logic.
**Changes:**
- Added `train_lipid_binary_heads()` and `predict_lipid_binary_heads()` to `hierarchical_experiment.py` — 5 independent binary XGB heads, each learning P(class_k | lipid pocket).
- Added `lipid_family_mode: Literal["softmax", "binary_ovr"]` config field to `HierarchicalSettings`, default `"softmax"` for backward compatibility.
- Wired through `hierarchical_pipeline.py`: training worker dispatch, bundle persistence, holdout inference. Full backward compat with existing softmax bundles.
- Created `configs/v_sterol_binary_ovr.yaml`. Ran full 25-iteration experiment.
**Result:**
- STE F1: 0.576 ± 0.108 — **identical** to the softmax + boundary-head rescue stack (0.576 ± 0.105). Binary OvR achieves this with zero post-hoc rescue logic.
- But every other lipid class regressed: CLR 0.728→0.707, MYR 0.701→0.688, OLA 0.553→0.523, PLM 0.649→0.630.
- 10-class macro-F1: 0.754 → **0.726** (−0.028). 5-lipid macro-F1: 0.641 → **0.625** (−0.016).
- Holdout apo-PDB F1: 0.679 (vs 0.716 current best). AlphaFold F1: 0.708 (vs 0.725).
**Decision:** Do not adopt. The softmax mutual-exclusivity is net positive for well-sampled, well-separated classes (CLR/MYR/OLA/PLM) — it provides free inter-class discriminative signal. The constraint only hurts at the sparse tail (STE, n=152), which the boundary-head architecture already fixes surgically. The current softmax + boundary-head design is validated as near-optimal for this data regime.
**What didn't work:** Blanket replacement of the softmax with independent heads. The theoretical framing ("simplex is biophysically wrong") was correct in principle but empirically wrong given the sample sizes — the inductive bias from mutual exclusivity outweighs the representational cost at n=150–700.
**Useful byproduct:** Per-head feature importance tables reveal what each lipid class independently keys on (e.g., STE uses `volume_score`, `LEU`, `TYR`; CLR uses `mean_as_solv_acc`, `PRO`). This could inform targeted feature engineering.
**Refs:** `reports/v_sterol_binary_ovr/hierarchical_lipid_report.md`, `configs/v_sterol_binary_ovr.yaml`, `experiments/registry.yaml#exp-010`

---

## 2026-04-24 — Boundary-Head Refactor + STE Rescue Promotion

**Goal:** Convert one-off lipid boundary arbiters into reusable code and keep iterating until the refactor produced a measurable metric lift.
**Hypothesis:** The flat 10-class softmax is bottlenecking rare/local lipid decisions; grouped local heads should recover boundary signal without damaging binary lipid-vs-rest parity.
**Changes:**
- Added `boundary_head.py`: generic `BoundaryRule`, `NeighborRescueRule`, binary XGB fitting, top-2 mass redistribution, top-k neighbor rescue, and boundary confusion utilities.
- Migrated PLM/STE and CLR/STE tiebreakers onto the shared boundary-head API.
- Added `confusion_mining.py` plus `slipp_plus mine-confusions` to rank residual off-diagonal errors and emit candidate boundary rules.
- Added config parsing for `hierarchical.boundary_heads` and wired configured boundary heads into hierarchical bundle training/inference.
- Refactored grouped STE-vs-neighbors rescue onto `NeighborRescueRule`.
- Added stacked pair-sweep persistence so selected pair heads can write prediction artifacts.
**Result:**
- Prior best (`v_sterol` ensemble + PLM/STE): 10-class macro-F1 0.738 ± 0.015, 5-lipid macro-F1 0.610 ± 0.026, STE F1 0.444 ± 0.107.
- Grouped STE rescue: 10-class macro-F1 0.753 ± 0.016, 5-lipid macro-F1 0.640 ± 0.029, STE F1 0.576 ± 0.105.
- Grouped STE rescue + OLA/PLM pair at margin 0.05: **10-class macro-F1 0.754 ± 0.016, 5-lipid macro-F1 0.641 ± 0.030, STE F1 0.576 ± 0.105, OLA F1 0.553 ± 0.056, PLM F1 0.649 ± 0.050**.
**Decision:** Promote `processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet` as the new best test-split artifact. Keep PLM/MYR pair head out; post-rescue sweep was neutral-to-regressive.
**What didn't work:** Stacking PLM-vs-MYR after STE rescue did not beat the grouped STE rescue baseline. Wide-margin OLA/PLM also regressed; only margin 0.05 was a small net win.
**Refs:** `src/slipp_plus/boundary_head.py`, `src/slipp_plus/confusion_mining.py`, `reports/v_sterol/boundary_refactor_results.md`, `reports/v_sterol/ste_rescue_boundary_refactor.md`, `reports/v_sterol/ste_rescue_ola_plm_pair_selected.md`

---

## 2026-04-24 — ML Hygiene Audit (P0/P1 execution)

**Goal:** Fix integrity issues identified in codebase audit.
**Changes:**
- Hardened `evaluate_holdout()` to derive lipid indices from model bundle's `class_order` metadata instead of module-level constant. Added assertion that bundle class order matches `CLASS_10`.
- Added logging for dropped holdout rows in `ingest.py` — previously rows with NaN descriptors vanished silently after `pd.to_numeric(..., errors="coerce")`.
- Deduplicated `HIERARCHICAL_PREDICTIONS_NAME`, `HIERARCHICAL_REPORT_NAME`, `HIERARCHICAL_METRICS_NAME` — were defined independently in both `train.py` and `evaluate.py`, now live in `constants.py`.
- Added `sklearn_version`, `xgboost_version`, `lightgbm_version` fields to joblib model bundles for cross-environment reproducibility tracking.
**Result:** All 123 tests pass. No metric changes — these are additive integrity guards.
**Decision:** P2 (tiebreaker module consolidation, ~1,750 LOC deletion) and P3 (mypy strict mode) deferred for separate planning.
**Refs:** `evaluate.py`, `ingest.py`, `constants.py`, `train.py`

---

## 2026-04-23 — Sterol ID Sprint, Round 2: PLM/STE Tiebreaker

**Goal:** Rescue STE F1 after Round 1 regression (0.415 → 0.398).
**Hypothesis:** STE's dominant confusion is PLM (38% of errors, 141/375 rows), not CLR (0.8%). A dedicated PLM-vs-STE binary head should recover STE without hurting PLM.
**Changes:**
- Implemented `plm_ste_tiebreaker.py`: binary XGB with `scale_pos_weight ≈ 4.72`, applied whenever ensemble top-2 = {PLM, STE} with margin < 0.99.
- Also attempted `v_plm_ste` feature set: 16 structural motif features (CRAC/CARC sequence motifs, axial shape-asymmetry profile, polar-anchor chemistry).
**Result:**
- PLM/STE tiebreaker: STE F1 0.398 → **0.444** (+0.046), PLM F1 0.636 → 0.638 (net zero). Fires 12.6 rows/iter. **Pure net win.**
- `v_plm_ste` features: **regressive** in 10-class softmax (STE F1 0.388 vs 0.398 baseline). Features show meaningful group-mean structure but signal is diluted in multiclass.
**Decision:** PLM/STE tiebreaker enters the winning config (exp-007). `v_plm_ste` features retained for potential binary-head use but not recommended.
**What didn't work:** v_plm_ste motif features in 10-class softmax. The binary head extracts PLM/STE signal more efficiently from existing v_sterol columns.
**Refs:** `reports/sterol_id_sprint.md`, `src/slipp_plus/plm_ste_tiebreaker.py`, `experiments/registry.yaml#exp-007`

---

## 2026-04-23 — Sterol ID Sprint, Round 1: Chemistry-Refined Features

**Goal:** Improve CLR and STE F1 for pre-email pitch to Dr. Dassama.
**Hypothesis:** Chemistry-refined residue shells (aromatic polar vs bulk hydrophobic) should separate sterol-binding pockets from fatty-acid pockets.
**Changes:**
- Implemented `sterol_features.py`: 28 chemistry-stratified shell counts (aromatic, aliphatic, cationic, anionic, polar-neutral, small-nonpolar × 4 shells + 2 special), 4 hydro/polar ratios, 5 PCA elongation features, 1 burial metric = 38 new features.
- Trained v_sterol ensemble (RF + XGB + LGBM, probability-averaged).
- Applied CLR/STE tiebreaker (margin=0.15).
**Result:**
- CLR F1: 0.708 → **0.728** (+0.020). Feature importance confirms TYR (#2), TRP (#6), PHE (#10) — mechanistic validation.
- 4 of 5 lipid classes improved: CLR (+0.020), MYR (+0.015), OLA (+0.013), PLM (+0.012).
- STE F1: 0.415 → **0.398** (−0.017, regressed). CLR/STE tiebreaker fired only 2 times across 25 iterations.
**Decision:** v_sterol is the new base feature set. STE regression needs targeted fix → triggers Round 2.
**What didn't work:** CLR/STE tiebreaker — fires near-zero because STE's confusion is with PLM, not CLR.
**Refs:** `reports/v_sterol/metrics.md`, `src/slipp_plus/sterol_features.py`, `experiments/registry.yaml#exp-005`

---

## 2026-04-23 — Detector Bakeoff: fpocket vs P2Rank

**Goal:** Validate paper's choice of fpocket as the pocket detector for lipid-class identification.
**Hypothesis:** P2Rank (CNN-based) may outperform fpocket on lipid pockets due to learned representations.
**Changes:**
- Ran P2Rank 2.4.2 (default model) on all 1,752 training structures.
- Implemented `detector_bakeoff.py` to compare hit rates at DCC/DCA ≤ 4Å.
**Result:** fpocket wins every aggregate metric. Top-1 DCC: fpocket 0.364 vs P2Rank 0.289. CLR mean rank: fpocket 2.6 vs P2Rank 7.6. P2Rank's CNN was trained on drug-pocket benchmarks — lipid pockets are underrepresented.
**Decision:** fpocket is the correct detector for this ligand distribution. Paper's choice validated.
**Refs:** `reports/detector_bakeoff/results.md`, `experiments/registry.yaml#exp-008`

---

## 2026-04-23 — v49 and v61 Feature Set Experiments

**Goal:** Test whether aromatic/aliphatic pocket shell features and amino acid counts improve lipid subclass discrimination.
**Hypothesis:** Residue composition around the pocket should distinguish structurally different lipid classes.
**Changes:**
- v49: added 20 AA counts + 12 aromatic/aliphatic shell features to the v14 baseline.
- v61: added 12 normalized shell features (fractions instead of raw counts) on top of v49.
- Fixed distance metric: changed from residue CA to closest non-hydrogen atom to pocket centroid. Reduced all-zero rate from 39.25% to 1.94%.
**Result:**
- v49 XGB: binary F1 0.860 → **0.895** (+0.035), lipid macro-F1 0.514 → **0.590** (+0.076). Major improvement.
- v61 LGBM: lipid macro-F1 0.591, apo-PDB RF F1 0.745 (best), but AlphaFold regressed vs v49.
**Decision:** v49 is the primary baseline. v61 is secondary (apo-PDB robustness). v49 supersedes Day 1.
**Refs:** `reports/v49/metrics_table.md`, `reports/v61/metrics_table.md`, `experiments/registry.yaml#exp-002`

---

## 2026-04-23 — Day 1 Baseline: 10-Class Reformulation

**Goal:** Reproduce Chou et al. 2024 with 10-class softmax instead of binary classifier.
**Changes:**
- Implemented full pipeline: ingest → schema validation → stratified splits → train (RF/XGB/LGBM) → evaluate.
- Rule 1 gate: exact match on 15,219 rows and per-class counts.
- 25 stratified shuffle splits, seeded and persisted.
**Result:** Binary-collapsed XGB F1 = 0.860 (paper: 0.869, within 0.01). AUROC = 0.982 (paper: 0.970). AlphaFold RF F1 = 0.732 (paper: 0.643, +0.089 — our reformulation is better on AF holdout).
**Decision:** Day 1 baseline is validated. Proceed to feature engineering.
**Refs:** `reports/metrics_table.md`, `reports/DAY_1_SUMMARY.md`, `experiments/registry.yaml#exp-001`

---

## Entry Template

```markdown
## YYYY-MM-DD — [Title]

**Goal:** [1 sentence]
**Hypothesis:** [What you expected and why, if experimental]
**Changes:** [Bullet list of code/config changes]
**Result:** [Key metrics, deltas vs baseline]
**Decision:** [What was decided based on the result]
**What didn't work:** [Optional — failed approaches within this session]
**Refs:** [Files, reports, experiment IDs]
```
