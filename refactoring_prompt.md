# SLiPP++ First-Principles Architectural Audit

## Context

SLiPP++ extends the Stanford SLiPP (2024) binary lipid-pocket classifier into a **flat 10-class softmax** over `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, STE, PP}` trained on three tree ensembles (RF / XGB `multi:softprob` / LGBM `multiclass`). Headline test F1 is 0.869; on AlphaFold OOD holdouts it collapses to 0.643 — a 26% relative drop that is the canonical signature of representational/objective mismatch rather than insufficient capacity.

This audit interrogates three things, ruthlessly, from a biophysical/statistical first-principles standpoint: (1) what the representation can and cannot encode about a lipid-binding pocket, (2) what assumptions the softmax objective bakes in that are false about lipid biology, and (3) which pipeline choices prevent the system from ever discovering a better primitive.

**Ground-truth files consulted:** [constants.py:16-264](src/slipp_plus/constants.py:16), [train.py:28-130](src/slipp_plus/train.py:28), [ingest.py:34-58](src/slipp_plus/ingest.py:34), [splits.py:14-34](src/slipp_plus/splits.py:14), [features.py:11-19](src/slipp_plus/features.py:11), [aromatic_aliphatic.py:19-302](src/slipp_plus/aromatic_aliphatic.py:19), [sterol_features.py:41-238](src/slipp_plus/sterol_features.py:41), [plm_ste_features.py:48-453](src/slipp_plus/plm_ste_features.py:48), [lipid_boundary_features.py:69-354](src/slipp_plus/lipid_boundary_features.py:69), [tunnel_features.py:29-133](src/slipp_plus/tunnel_features.py:29), [caver_analysis.py:51-383](src/slipp_plus/caver_analysis.py:51), [evaluate.py:76-127](src/slipp_plus/evaluate.py:76), [calibration.py:1-142](src/slipp_plus/calibration.py:1).

**Training data reality:** 15,219 pockets, 65% labeled `PP` (pseudo-pocket), STE n=152, ratio PP:STE ≈ 65:1. ~1,752 unique PDB IDs → mean ~8.7 pockets per PDB. The `class_weight="balanced"` flag does not fix this — it only rescales gradients; it cannot manufacture structural diversity STE does not have.

---

## Tier 1 — Fundamental Architecture & Objective Shifts (highest ROI)

### 1.1 The 10-class softmax is the wrong primitive. Kill it.

The objective `multi:softprob` / `multiclass` with `num_class=10` ([train.py:28-55](src/slipp_plus/train.py:28)) encodes four assumptions that are simultaneously false for lipid-binding pockets:

1. **Mutual exclusivity.** A softmax enforces Σp = 1 with unit simplex geometry. Real pockets in ORP/START/SCP2/NPC1 bind multiple sterols; LBP/FABP-family pockets bind interchangeable fatty acids; acyl-CoA pockets accommodate a fatty-acid "envelope." The label generator in [ingest.py:34-58](src/slipp_plus/ingest.py:34) via `lig → LIG_TO_CLASS` assigns one class per `(pdb, pocket)` row, but 25 PDBs in training have *different* lipids in *different* pockets of the *same* protein — the model learns "one pocket, one lipid" as a structural prior that contradicts the proteins it's learning from.
2. **Equidistance in class space.** Cross-entropy treats CLR↔STE (both sterols, shared 27-carbon steroid nucleus, shared π-stack anchors) as equidistant from CLR↔ADN (steroid vs. adenosine). This wastes capacity on impossible confusions and starves the decision surface where it actually matters.
3. **PP is a class.** `PP` (65% of training data) is *not* a lipid class — it is the fpocket false-positive distribution. Forcing it into the simplex drags probability mass off all lipid classes and couples the lipid decision to the pocket-detection noise model. The model's best loss minimizer on ambiguous inputs is "predict PP" because PP is the mode.
4. **Flat taxonomy.** There is a real chemical hierarchy — `{lipid, non-lipid-small-molecule, artifact}` → `{sterol, fatty-acid, acyl-CoA}` → `{CLR, STE}` / `{MYR, PLM, OLA}` — and the current formulation cannot use it.

**Recommended replacement (in order of increasing architectural change):**

- **Minimum viable fix — hierarchical softmax / chained specialists.** Stage A: binary `is_true_pocket` (abstain=PP) — this is the only place fpocket noise belongs. Stage B (on Stage-A positives): coarse chemical family `{sterol, fatty_acid, acyl_CoA, nucleoside_cofactor}` — 4-way softmax over chemically coherent parents. Stage C: within-family specialists (CLR↔STE; MYR↔PLM↔OLA; ADN↔B12↔BGC↔COA). This is exactly the structure the existing [sterol_tiebreaker.py](src/slipp_plus/sterol_tiebreaker.py) and [hierarchical_experiment.py](src/slipp_plus/hierarchical_experiment.py) are already bolting on — promote them from post-hoc patches to the primary objective.
- **Better — chemistry-structured loss.** Replace cross-entropy with a **soft-label CE over a ligand similarity matrix `S ∈ R^{9×9}`** built from (a) Tanimoto over ECFP4 of the canonical ligands and (b) pocket-descriptor Earth-Mover distance between class-conditional fpocket distributions. Target label becomes `y_soft = (1-ε)·one_hot + ε·S[y]`. This penalizes PLM→ADN hard and PLM→OLA lightly, *in proportion to real chemistry*. No architecture change; drops into XGB via custom `objective`.
- **Correct — multi-label + open-set.** The task is fundamentally *ligand-set* prediction with a *novelty* detector: `y ∈ {0,1}^9` for chemical classes (sigmoid BCE per class, no simplex constraint) plus a separate evidential / energy head that maps to "this pocket has no binder in our vocabulary" (current PP). This dissolves the PP-pollutes-simplex problem, permits multi-lipid pockets, and gives calibrated per-class probabilities that downstream users can threshold independently per chemistry.

The [calibration.py:1-50](src/slipp_plus/calibration.py:1) rationale that "multi-class softmax degrades more gracefully on OOD" because mass spreads across 5 lipid classes is the symptom, not a virtue — it's spreading mass because it has no correctly-shaped output space, and `p_lipid = Σ p_k` ([evaluate.py:112-127](src/slipp_plus/evaluate.py:112)) is a hand-rolled marginalization working around the fact that the softmax was never the right shape.

### 1.2 The split is leaking protein identity. Every metric north of AlphaFold is optimistic.

[splits.py:14-34](src/slipp_plus/splits.py:14) uses `StratifiedShuffleSplit(test_size=0.10)` with stratification on `class_10`. With ~8.7 pockets per PDB, this guarantees same-PDB (and therefore same-UniProt, same-fold, near-identical-sequence) pockets on both sides of the split. The 26-point AlphaFold F1 gap is not OOD brittleness — it is the sudden absence of that leakage.

Biology makes this worse than generic data leakage: lipid-binding folds are *deeply* conserved (START, PITP, SCP2, LBP, FABP, ORP, NPC, SR-B, ABC-sterol). A CLR-binding START domain in train vs. test is not two samples — it's the same pocket with different crystal waters.

**Fix:** replace stratified shuffle with **sequence-identity-clustered** splitting (MMseqs2 at 30% identity, or at minimum UniProt-clustered `GroupKFold`), then stratify *within* clusters. Keep the current shuffle as a sanity baseline only. Expect headline test F1 to drop ~10–15 points — **that is the correct number**, and the current AlphaFold holdout number is closer to the real generalization than any in-distribution metric in this repo. The `reports/metrics_table.md` paper-comparability numbers become structurally misleading once you do this; that is desirable, not a bug.

### 1.3 Trees over a flat 103–109-D vector is the wrong computational primitive for 3D pockets.

The end-to-end path ([features.py:11-19](src/slipp_plus/features.py:11)) is `DataFrame[columns].to_numpy(float64)` → `model.fit(X, y)`. Pockets are 3D point clouds of alpha-spheres + a set of contact residues with side chains and sequence context. Collapsing that into 100 floats before any model sees it is the dominant source of representational loss, and no tree ensemble can recover the information that's already gone.

Three concrete bottlenecks the current primitive cannot cross:

- **Permutation and rotation symmetries are thrown away.** The pocket is invariant under SE(3) on the alpha-sphere cloud and under permutation of residue labels, but the feature vector bakes in neither — so the model must *learn* that invariance from 15k examples, and won't.
- **Radial shells are a harsh discretization of a continuous field.** A residue at 5.9 Å goes into shell 2, at 6.1 Å into shell 3 ([aromatic_aliphatic.py:21](src/slipp_plus/aromatic_aliphatic.py:21), [sterol_features.py:184-192](src/slipp_plus/sterol_features.py:184)) — identical biophysics, different features. 4 bins × 7 chemistry groups + 4 bins × aromatic/aliphatic + 5 axial bins × lipid-boundary chemistries is ~80 cells that are each quantizing the same underlying continuous (r, chemistry) density.
- **PCA-axis orientation is not gauge-invariant.** [plm_ste_features.py:441-453](src/slipp_plus/plm_ste_features.py:441) picks the polar end by `density0 ≥ density4`; `axial_radius_gradient` ([plm_ste_features.py:~260](src/slipp_plus/plm_ste_features.py:260)) is the *signed* slope over bins 1–5. For pockets with tied polarity, the sign is arbitrary — the same pocket gets +slope or −slope depending on PCA's eigenvector sign convention. Tree ensembles cannot undo this gauge ambiguity; you are asking them to memorize it.

**Recommended primitive for the pocket representation stage:**
Move pocket embedding to an **SE(3)-equivariant set/graph encoder** (EGNN, SchNet-style continuous-filter, or GearNet on the alpha-sphere ∪ contact-Cα graph with edge features = radial Gaussian basis + chemistry-group one-hot). Pool to a fixed-dim pocket embedding `z ∈ R^d`. Concatenate `z` with the *scalar* fpocket descriptors (pock_vol, hydrophobicity score, flex, as_density — these are legitimate global scalars). Feed *that* to the hierarchical head from §1.1.

You keep tree ensembles at the *outer* classifier if you want — they're fine once the representation is right — but the encoder has to respect the geometry. `fpocket` descriptors are not a representation; they are summary statistics over one.

### 1.4 Per-residue PLM embeddings are missing. The name "PLM_STE" is misleading — audit that first.

Despite [constants.py:230-249](src/slipp_plus/constants.py:230) naming a feature set `v_plm_ste`, **there are no protein-language-model embeddings anywhere in the pipeline**. The `PLM_STE_EXTRA_16` columns ([plm_ste_features.py:48-68](src/slipp_plus/plm_ste_features.py:48)) are *palmitate-vs-steryl* disambiguation features (CRAC/CARC regex motifs, axial radial profile, polar-end chemistry). That is a substantial naming hazard — a future contributor reading "PLM" will assume ESM2 embeddings are feeding the model, and they are not.

**Rename immediately** (`PMS_*` / `palmitate_vs_steryl_*`) and then **actually add PLM embeddings** with a proper pooling strategy:

- ESM2-650M or ESM-C per-residue embeddings, computed once and cached (the PLM cost is a one-time offline pass; it does not recur at training time).
- Pool per-contact-residue, not over the whole protein: aggregate only residues within the pocket's 6 Å shell, **weighted by inverse distance**, or better, by attention conditioned on the pocket geometry embedding from §1.3.
- Mean-pooling over the whole protein or over all contact residues will destroy the signal (1 aromatic anchor at the pocket mouth ≠ 20 aromatics scattered through the fold); do not ship that shortcut.

Per-residue embeddings also fix a silent failure mode the current features already have: for AlphaFold holdout structures, sequence-level features (CRAC motif regex) continue to work, but *sequence-dependent* structural priors (disorder, evolutionary conservation) are invisible to the current pipeline.

---

## Tier 2 — Feature Representation & Fusion Refactoring

### 2.1 Asymmetric Laplace smoothing destroys aromatic/aliphatic symmetry.

[aromatic_aliphatic.py:251-254](src/slipp_plus/aromatic_aliphatic.py:251): `ratio = aromatic / (aliphatic + 1.0)`. For aromatic=10, aliphatic=0 → ratio=10. For aromatic=0, aliphatic=10 → ratio≈0. The feature is scale-free only in one direction. A pocket with 10 aromatics and 0 aliphatics and a pocket with 0 aromatics and 10 aliphatics are equally pure, just on opposite poles — but the feature value differs by a factor of 10⁴.

The same asymmetry is in [sterol_features.py:195-205](src/slipp_plus/sterol_features.py:195) (polar_hydrophobic_ratio), [lipid_boundary_features.py](src/slipp_plus/lipid_boundary_features.py) anchor_charge_balance, and elsewhere.

**Fix:** replace with a symmetric log-odds / centered log-ratio, `feat = log((a + α) / (b + α))` with α = 0.5 (Jeffreys) or α = sqrt(a·b)_prior. Output is signed and symmetric; magnitude scales monotonically with purity in both directions. Trees don't care about the monotone transform but downstream neural heads, SHAP interpretation, and any linear baseline will stop lying.

### 2.2 Discrete radial shells are a hard-coded Nyquist violation.

[aromatic_aliphatic.py:21](src/slipp_plus/aromatic_aliphatic.py:21) `SHELL_EDGES = ((0,3),(3,6),(6,9),(9,12))` — 3 Å bins. The first coordination shell of a C–C van der Waals contact is ~3.8 Å; H-bond donor/acceptor geometry ~2.5–3.5 Å; π-stack ~3.3–4.5 Å. The bin edge at 3 Å *splits the most important biophysical contact shell in half*.

**Fix:** replace binning with **radial basis functions** (K Gaussians with learnable or paper-fixed centers/widths spanning 1.5–12 Å, e.g., `φ_k(r) = exp(-(r-μ_k)²/σ²)`). Each residue contributes `φ_k(r) · chemistry_onehot` to a continuous radial-chemistry tensor. For tree models, pre-compute the K × chemistry matrix and flatten. Per-shell quantization was a 2010-era workaround; RBF expansions are routine in SchNet/DimeNet descriptor pipelines and cost nothing.

### 2.3 Drop primary-tunnel-only summaries. Pockets are not single-tunnel objects.

[tunnel_features.py:29-45](src/slipp_plus/tunnel_features.py:29) and [caver_analysis.py:51-70](src/slipp_plus/caver_analysis.py:51) collapse CAVER output to the primary tunnel (max throughput). For lipid transfer proteins the *second* and *third* tunnels are often the mechanistically important ones — ORPs have a sterol-entry + PI(4)P-exit pair; NPC1 has one sterol tunnel plus a separate cholesterol-handoff route; StarD4/5 tunnels regulate both loading and release. Throwing those away then adding a primary-curvature feature does not recover what was discarded.

**Fix:** represent tunnels as a **set of tuples** `{(length, min_radius, mean_radius, curvature, chemistry_hydrophobicity, chemistry_charge)}_i` and pool via DeepSet (mean + max + sum) or a small attention over tunnels. Keep a `tunnel_count` scalar but let the encoder decide how to weight primary vs secondary tunnels rather than hard-coding `max(throughput)` lexicographic tie-breaking. Minimum: add secondary-tunnel stats (2nd-highest throughput length/radius/curvature) as additional columns — it's a trivial parquet-append and recovers most of the lost signal.

### 2.4 Glycine exclusion is a chemistry bug for lipid binding.

[sterol_features.py:39](src/slipp_plus/sterol_features.py:39) excludes GLY from all seven chemistry groups with the rationale "no side chain → no discriminative sterol-contact signal." For sterols this is defensible. **For the broader lipid label set (MYR/PLM/OLA/COA) it is wrong:** glycine-rich loops (GxxxG transmembrane helix packing, P-loop-analog acyl anchors, glycine hinges at membrane-insertion sites) are a primary mechanical feature of acyl-chain accommodation. The lipid-boundary module partly compensates via `lb_tube_gly_fraction` and `lb_gly_rich_anchor_fraction` ([lipid_boundary_features.py:157-180](src/slipp_plus/lipid_boundary_features.py:157)), but at the sterol-chemistry-shell level GLY is invisible.

**Fix:** add GLY as its own eighth chemistry class (`flexibility_marker` or just `glycine`), with its own 4-shell count columns. 4 columns added; biologically necessary for the acyl-chain classes that now share the objective with sterols.

### 2.5 Missing-tunnel / no-CAVER → 0.0 collapses with real zeros.

[tunnel_features.py:116-133](src/slipp_plus/tunnel_features.py:116), [plm_ste_features.py:289-298](src/slipp_plus/plm_ste_features.py:289), [lipid_boundary_features.py:69-93](src/slipp_plus/lipid_boundary_features.py:69): pockets with no tunnel / <5 alpha-spheres get all-zero defaults. "No tunnel" (an occluded sterol pocket) and "tunnel of length 0" (pathological) and "tunnel of bottleneck 0 Å" (physically impossible) all collapse to the same value. Tree splits cannot separate absence from measurement-of-zero.

**Fix:** add explicit binary `has_tunnel`, `has_caver_profile`, `has_alpha_spheres_ge_5` missingness indicators. Keep the 0.0 imputation (trees handle it fine) but let the model condition on presence. Trivial change; measurable lift on pockets where tunnel features are most discriminative (i.e., closed-vs-open sterol gates).

### 2.6 Per-group feature scaling is missing; magnitudes range across eight orders.

Raw feature matrix mixes eigenvalues in Å² (~1–500), tunnel lengths in Å (5–50), residue counts (0–20), normalized fractions (0–1), and Kyte-Doolittle sums (–40 to +40). Trees are scale-invariant by splitting, yes — but:

- XGB's `hist` method bins per-feature; extreme magnitudes bias the histogram resolution.
- Any future addition of a linear/MLP/neural head on this vector breaks immediately.
- SHAP and permutation importance are distorted, which is how the team will reason about what the model learned — the instrument is miscalibrated.

**Fix:** group-wise standardization fit on training splits only, with scalers persisted alongside models. Applied as a deterministic step in [features.py:11-19](src/slipp_plus/features.py:11). Does not change tree predictions at all; unlocks neural/linear branches and truthful importance.

### 2.7 Canonicalize PCA axis orientation.

[plm_ste_features.py:441-453](src/slipp_plus/plm_ste_features.py:441), [lipid_boundary_features.py:192-218](src/slipp_plus/lipid_boundary_features.py:192). Tied-polarity pockets produce sign-flipped `axial_radius_gradient` and `thick_end_asymmetry` across runs. The fix is to use **signed-quantity-invariant features only**: replace `gradient` with `|gradient|` + `polar_asymmetry_magnitude = |R[0] - R[4]| / mean_R` (already computed as `thick_end_asymmetry`, which is magnitude-only ✓). Drop the raw signed gradient, or emit it with a definitive sign convention based on an external invariant (e.g., sign determined by location of the highest-density cationic residue — which is not gauge-dependent).

---

## Tier 3 — Pipeline & Implementation Optimizations

### 3.1 Feature sets live as parallel universes. Unify under a declarative DAG.

`v49`, `v61`, `v_sterol`, `v_plm_ste`, `v_lipid_boundary`, `v_tunnel`, `v_graph_tunnel`, `v_caver_t12` are each a separate CLI command with hand-wired paths ([cli.py](src/slipp_plus/cli.py) `build-*` subcommands; [constants.py:230-249](src/slipp_plus/constants.py:230) lists them). Dependencies between them (e.g., `v_plm_ste` consumes `v_sterol` parquets) exist only in operator memory.

**Fix:** Snakemake or Prefect DAG with feature groups as nodes and a single `make train FEATURE_SET=…` entry point. Each node emits a schema hash written into the parquet metadata; downstream training refuses to run if the hash doesn't match the config. Stops the silent-column-reorder failure mode (mismatched `FEATURE_SETS[…]` list vs. parquet columns → KeyError if lucky, silently wrong model if not).

### 3.2 Sequence-clustered splits as default; stratified shuffle as sanity.

See §1.2. Implementation: add `GroupKFold` over UniProt ID (or MMseqs2 cluster ID at 30% identity) in [splits.py](src/slipp_plus/splits.py); expose as `split_strategy: grouped_mmseqs30 | stratified_shuffle` in config. **Update the paper-comparison tables with a clear note** that the previous numbers were leakage-inflated — do not try to preserve them for "comparability," because that is comparability to a measurement artifact.

### 3.3 Model persistence needs feature-schema pinning.

[train.py:~130](src/slipp_plus/train.py:130) dumps joblib bundles with no schema hash. If `FEATURE_SETS["v_sterol"]` is edited (reorder, add, drop) after a model is trained, inference silently feeds the wrong columns into the wrong tree splits. **Fix:** serialize alongside the model `{feature_columns: [...], schema_hash: sha256(tuple), training_settings: config_dict}` and assert-check at load. Apply the same discipline to predictions parquet.

### 3.4 Integration test covering ingest→train→eval on a 100-protein fixture.

Current [tests/](tests) directory is unit-level. There is no test that would catch "aromatic_aliphatic refactor changes column order in v49 parquet but not in `FEATURE_SETS[v49]`." One fixture (e.g., 100 PDBs spanning all 10 classes, checked into a test data tarball) running `make all` and asserting macro-F1 within ±2 points of a frozen baseline would catch ~every silent-refactor regression. Fast (<30 s with 100 structures), cheap, and prevents the class of bugs that research code accumulates.

### 3.5 Standardize DataFrame library; kill pandas/polars mixing.

[aromatic_aliphatic.py](src/slipp_plus/aromatic_aliphatic.py) and [ensemble.py](src/slipp_plus/ensemble.py) use polars; most else uses pandas. At 15k rows this does not matter for performance but it is a trap for contributors and causes dtype conversion silent bugs. Pick one (polars if you intend to scale past 100k pockets with PLM embeddings; pandas if not) and enforce in CI.

### 3.6 Separate "paper comparability" reports from "real generalization" reports.

[evaluate.py:76-127](src/slipp_plus/evaluate.py:76) and the `ground_truth` block in configs couple headline metrics to Stanford 2024's numbers. Once §1.2 lands those numbers become non-comparable. Split the reports directory: `reports/paper_replication/` (stratified shuffle, for historical comparison only) and `reports/generalization/` (clustered splits + apo/AlphaFold holdouts, the actual signal). The calibration analysis in [calibration.py:135-142](src/slipp_plus/calibration.py:135) belongs in the latter.

---

## Order of operations (execution sequence, if acted on)

1. **T1.2 clustered split** *first* — every other metric you report after this is now meaningful. Expect and accept the headline F1 drop.
2. **T1.1 hierarchical objective + PP-as-abstain** — biggest single lift, no new data required, the existing `hierarchical_experiment.py` is 80% of the scaffold.
3. **T2.1 / T2.2 / T2.5 / T2.7** (Laplace fix, RBF shells, missingness indicators, PCA canonicalization) — cheap, mechanical, independently useful, and each one stops the model from memorizing an artifact.
4. **T1.4 PLM embeddings** *in parallel with* T1.3 geometry encoder — this is the only change that requires new compute but it's one-time offline.
5. **T1.3 SE(3) encoder + T2.3 tunnel deepsets** — architectural; do after (2) so the new encoder plugs into a correctly-shaped objective.
6. **T3.1–T3.6** pipeline hygiene — alongside each substantive change, not as a separate phase.

## Verification

- After T1.2: clustered-split test F1 should land close to current AlphaFold F1 (~0.64 ± 0.03). If it's much higher, the clustering threshold is too loose; tighten to 25% identity. If it's much lower, the holdout was noisier than the clustered split and current AlphaFold structures are genuinely harder (fine, keep going).
- After T1.1: per-class recall on STE (currently ~0.3–0.5) should exceed 0.6 without harming CLR; PP false-positive rate should drop because PP is now an abstain head rather than a simplex competitor. Inspect the confusion matrix for disappearance of "lipid → PP" confusions on genuine lipid pockets.
- After T1.3/T1.4: ablate {scalar-only, scalar+geom-encoder, scalar+geom+PLM} — expect monotone improvement on clustered + AlphaFold splits. If PLM doesn't help on clustered splits, your pooling is wrong (§1.4 warning).
- Regression guard: the 100-PDB fixture (T3.4) runs on every PR; macro-F1 band ±2 points.

## Critical files to modify

- Objective & split: [train.py](src/slipp_plus/train.py), [splits.py](src/slipp_plus/splits.py), [evaluate.py](src/slipp_plus/evaluate.py), [calibration.py](src/slipp_plus/calibration.py), [hierarchical_experiment.py](src/slipp_plus/hierarchical_experiment.py) (promote from experiment → default path).
- Feature representation: [aromatic_aliphatic.py](src/slipp_plus/aromatic_aliphatic.py) (RBF + symmetric log-odds), [sterol_features.py](src/slipp_plus/sterol_features.py) (GLY re-inclusion), [plm_ste_features.py](src/slipp_plus/plm_ste_features.py) (rename + PCA canonicalization), [lipid_boundary_features.py](src/slipp_plus/lipid_boundary_features.py) (same canonicalization), [tunnel_features.py](src/slipp_plus/tunnel_features.py) / [caver_analysis.py](src/slipp_plus/caver_analysis.py) (secondary tunnels → set features), [features.py](src/slipp_plus/features.py) (group scalers, missingness indicators).
- New: per-residue PLM embedding module + cache; SE(3)-equivariant pocket encoder + fixed-dim `z`; feature-schema hashing.
- Pipeline: [cli.py](src/slipp_plus/cli.py), Snakefile/Prefect DAG, [constants.py](src/slipp_plus/constants.py) (rename `PLM_STE_EXTRA_16` → `PMS_EXTRA_16`), split reports directories.

---

**Closing observation.** Every Tier 1 item is a structural mismatch between the machine-learning primitive and the biology, not a hyperparameter or a feature. The system is currently doing creditable work *despite* the objective; it will do dramatically better work once the objective and the split stop fighting the data. The existing experiments directory (`sterol_tiebreaker`, `plm_ste_tiebreaker`, `hierarchical_experiment`, `ste_rescue_experiment`) is the team's own repeated attempt to patch around the flat-softmax bottleneck from outside — the correct move is to demote those patches and promote their shape to the primary architecture.
