# SLiPP++ Model V2 Spec

Status: approved target architecture for the next major refactor

Audience: downstream coding agents and human operators

Purpose: define the full target model, refactor shape, migration sequence, and acceptance criteria for the architecture expected to produce the next material lift in lipid subclass performance

Read this after [CONTEXT.md](CONTEXT.md) and [AGENTS.md](AGENTS.md).

## 1. Executive Summary

The current system has reached the point where more one-off tiebreakers, more flat feature concatenation, or another small ensemble tweak will not unlock a new level of capability.

The next architecture must do two things together:

1. Generalize the current staged and local-arbitration logic into one topology-driven routed composite model.
2. Replace the current independent RF/XGB/LGBM flat backbone with a shared feature-family-aware representation backbone.

This is not a GNN project. It is not a raw-structure end-to-end project. It is not a detector replacement project. It is a tabular-plus-routed-expert reimplementation that preserves the current data and evaluation contract while materially improving the model's ability to exploit local structure, optional feature families, and high-value confusion neighborhoods.

The final target model is:

1. A shared feature-family-aware backbone encoder.
2. Multiple coarse heads trained from that shared representation.
3. A routed expert layer for high-confusion local boundaries.
4. Closed-world probability combiners that preserve the 10-class simplex.
5. A versioned composite bundle that reconstructs the entire topology for holdout inference.

The backbone replacement is part of the target architecture. It is not optional.

## 2. Why Model V2 Exists

The current repo already shows the governing pattern.

1. The flat backbone is globally strong.
2. The strongest improvements come from local arbitration, not from the global softmax alone.
3. Narrow local objectives extract signal that the 10-class classifier leaves on the table.
4. Existing specialist and boundary logic are effective but duplicated and architecture-specific.
5. Future feature families such as CAVER are likely to be useful locally, not uniformly across all classes.

The system therefore needs an architecture that:

1. shares representation across tasks
2. allows different experts to see different feature families
3. uses learned routing to decide when local experts matter
4. preserves the current evaluation contract
5. can absorb future feature families without spawning bespoke pipelines each time

## 3. What Changes

Model V2 makes two major structural changes.

### 3.1 Composite topology becomes first-class

The current hard-coded distinction between flat mode, hierarchical mode, specialist rules, and pairwise tiebreakers must be replaced by one topology-driven composite system.

Core first-class objects:

1. backbone
2. head
3. expert
4. gate
5. combiner
6. topology
7. composite bundle

### 3.2 The backbone is replaced

The current flat backbone is three independent models trained from scratch on the same raw matrix and then probability-averaged.

That architecture is useful as a teacher, but it is not the end-state backbone.

The new backbone is a shared feature-family-aware tabular encoder that produces one latent embedding per pocket. Coarse heads and local experts consume that representation directly or consume restricted feature views plus that representation.

## 4. Hard Constraints

These constraints are mandatory unless explicitly superseded.

1. Preserve the existing 15,219-row training table and the class-count validation gate.
2. Preserve the current 25 split protocol and seeded reproducibility.
3. Preserve the current binary collapse for paper comparison.
4. Preserve the current final prediction artifact consumed by evaluation and figures.
5. Preserve the existing holdout scoring path conceptually, even if the bundle implementation changes.
6. Do not require GNNs, voxel models, raw atomic graphs, or detector replacement.
7. Do not invent new raw data sources.
8. Keep CAVER and future expensive feature families optional at inference time through availability masking.

## 5. Target Lift

Use these as planning ranges, not guarantees.

Current best reference from [CONTEXT.md](CONTEXT.md):

1. Binary F1: 0.899
2. Binary AUROC: 0.986
3. 10-class macro-F1: 0.754
4. 5-lipid macro-F1: 0.641
5. STE F1: 0.576

Expected end-state target after full Model V2 implementation plus properly integrated CAVER:

1. Binary F1: 0.904 to 0.912
2. Binary AUROC: 0.988 to 0.992
3. 10-class macro-F1: 0.780 to 0.805
4. 5-lipid macro-F1: 0.675 to 0.715
5. STE F1: 0.63 to 0.70

Stretch target:

1. 10-class macro-F1: 0.815 to 0.825
2. 5-lipid macro-F1: 0.725 to 0.740

Interpretation:

1. Most remaining headroom is in subclass discrimination, not binary detection.
2. Most of the lift should come from the architecture and routing change.
3. CAVER should add local signal on top of the new architecture, not substitute for it.

## 6. The Final Model

## 6.1 Overview

The final model has five layers.

1. feature-family input layer
2. shared backbone encoder
3. coarse global heads
4. routed local experts
5. probability combiner and calibration layer

All layers must be representable in one topology and serializable into one composite bundle.

## 6.2 Feature-family input layer

Do not treat all columns as one undifferentiated flat vector.

The backbone must ingest grouped feature families.

Initial feature-family registry:

1. `paper17`
2. `aa20`
3. `shell12`
4. `sterol_chemistry`
5. `pocket_geometry`
6. `boundary22`
7. `graph_tunnel`
8. `caver_t12`
9. `derived`
10. `vdw22`

Each family must define:

1. ordered column list
2. family name
3. schema hash
4. normalization stats
5. availability mask semantics

Rules:

1. Optional families must carry explicit presence indicators.
2. Missing-family absence must not be silently encoded as ordinary zero.
3. Family-level dropout should be supported during training.
4. Downstream experts may request a strict subset of families.

## 6.3 Shared backbone encoder

This is the new backbone. It replaces the current independent flat ensemble as the primary model representation.

Recommended structure:

1. one small encoder per feature family
2. one learned family embedding token per family
3. one availability mask per family
4. a compact fusion stack over family tokens
5. one shared latent pocket embedding

Preferred implementation:

1. family-specific projection MLP or linear block
2. token dimension 32 to 64 per family
3. 2 to 4 residual fusion blocks
4. gated MLP or light attention over family tokens
5. latent embedding width 128 to 192

Acceptable fallback for an initial simpler implementation:

1. family encoders
2. concatenation of family embeddings and masks
3. residual MLP fusion

Design intent:

1. heterogeneous feature families are modeled cleanly
2. optional families can be added without redesigning the whole model
3. experts can consume a shared embedding instead of rebuilding separate backbones
4. routing can depend on learned representation rather than only on top-k margins

This is intentionally not a giant generic transformer and not a giant monolithic MLP over all raw columns.

## 6.4 Backbone outputs

The shared encoder must produce:

1. one latent pocket embedding
2. optional family-specific intermediate embeddings
3. optional uncertainty or confidence summary features

These are then consumed by heads, experts, and gates.

## 6.5 Coarse heads

Train the following coarse heads from the shared embedding:

1. `global_10_class_head`
2. `binary_lipid_head`
3. `lipid_family_head`
4. `nonlipid_family_head` if still justified by error analysis
5. `embedding_projection_head` for prototype or contrastive structure

These heads replace the current fragmentation where the flat ensemble, lipid gate, and family heads all come from separate training surfaces.

## 6.6 Routed local experts

Local experts are mandatory. They are not post-hoc hacks in Model V2.

Expert rules:

1. each expert owns a declared candidate label set
2. each expert has a declared feature-family or feature-set view
3. each expert has a declared gate
4. each expert has a declared combine policy
5. each expert must not rewrite unrelated class probabilities

Preferred expert granularity:

1. confusion-neighborhood experts
2. pair experts only when the boundary is genuinely pair-dominated
3. no blanket one-expert-per-class policy

Initial expert set for v1:

1. `ste_neighbors_expert`
   - focus: STE
   - owned neighborhood: PLM, COA, OLA, MYR, STE
2. `ola_plm_expert`
   - owned neighborhood: OLA, PLM
3. `clr_local_expert`
   - owned neighborhood driven by confusion audit
4. `nonlipid_cleanup_expert`
   - optional, only if it survives validation
5. `caver_access_expert`
   - enabled when CAVER exists
   - focused on access-path-sensitive boundaries

## 6.7 Gates

Every expert must have a gate.

Supported gate kinds:

1. `heuristic`
2. `utility_gate`
3. `candidate_gate`

Default policy:

1. use learned utility gates whenever data support exists
2. use heuristic routing only for parity, low-support experts, or explicit fallback

Gate inputs may include:

1. backbone top-1 and top-2 classes
2. backbone margin and entropy
3. expert confidence
4. shared latent embedding
5. optional raw or family-level features
6. feature-family availability flags

The current specialist utility-gate idea must be generalized into a repo-wide gate abstraction.

## 6.8 Probability combiner

The combiner is mandatory and first-class.

Combiner rules:

1. preserve total probability mass
2. preserve non-candidate class probabilities
3. preserve the final 10-class simplex
4. support multiple local experts in ordered sequence

Required combine modes:

1. `closed_world_redistribute`
2. `pair_swap`
3. `neighbor_rescue`
4. `mixture_override` if later justified

Do not introduce any expert that rewrites the entire class distribution unless separately justified and tested.

## 6.9 Prototype and metric structure

The backbone must support class structure beyond plain cross-entropy.

Required addition:

1. supervised contrastive loss or
2. class prototypes in latent space or
3. prototype-regularized embedding training

Purpose:

1. improve rare-class stability
2. improve routing quality
3. allow nearest-prototype fallback analysis
4. structure local neighborhoods more explicitly

## 7. Training Strategy

## 7.1 Delivery stages

Model V2 must be implemented in two major stages.

### Stage A: composite refactor with current teacher stack

1. build the topology-driven composite runtime
2. use the current RF/XGB/LGBM ensemble as the backbone teacher and parity baseline
3. port current boundary-refactor behavior into the new topology
4. preserve outputs and evaluation unchanged

### Stage B: backbone replacement

1. implement the family-aware shared encoder
2. distill from the Stage A teacher stack
3. swap the flat ensemble backbone for the shared backbone under the same topology
4. re-run parity and then lift experiments

Important:

1. the backbone replacement is mandatory for the final target architecture
2. it is phased after topology parity for risk control

## 7.2 Loss stack

Target end-state loss components:

1. global 10-class cross-entropy
2. binary lipid-vs-nonlipid loss
3. lipid-family loss
4. per-expert local loss
5. prototype or supervised contrastive loss
6. teacher distillation loss during backbone transition

Per-expert local loss may be binary or local multiclass depending on expert type.

## 7.3 Distillation

Backbone replacement must use distillation from the current best teacher stack.

Teacher components:

1. current RF/XGB/LGBM ensemble
2. current staged boundary-refactor outputs where relevant
3. local experts for the corresponding routed neighborhoods

Why distillation is required:

1. reduces regression risk on a modest-size tabular dataset
2. transfers useful decision structure from the current best stack
3. makes the backbone transition realistic instead of brittle

Do not attempt a from-scratch backbone swap without distillation unless a later brief explicitly overrides this.

## 7.4 Split discipline

All expert and gate training must remain split-aligned.

Rules:

1. no expert may be trained on its own evaluation fold
2. utility gates must be trained out-of-fold
3. diagnostics must be based on held-out predictions
4. final reported metrics remain aggregated over the existing 25 iterations

## 8. CAVER Integration

CAVER is a feature family, not a bespoke architecture.

Rules:

1. integrate CAVER through the feature-family input layer
2. allow specific experts to consume CAVER first
3. model CAVER availability explicitly
4. do not require the global head to depend on CAVER
5. only widen CAVER usage after local value is demonstrated

Preferred first uses:

1. access-path-sensitive sterol boundaries
2. OLA versus PLM or MYR
3. compact-site versus tunneled-pocket nonlipid cleanup

## 9. Config Schema

The current single-purpose hierarchical block must evolve into a generalized composite topology schema.

Top-level `Settings` should remain stable where possible:

1. split controls
2. seed controls
3. primary feature-set references
4. paths
5. validation

Add a new conceptual composite schema:

```yaml
pipeline_mode: composite

composite:
  backbone:
    kind: family_encoder
    teacher_init: ensemble_distilled
    feature_families:
      - paper17
      - aa20
      - shell12
      - sterol_chemistry
      - pocket_geometry
      - caver_t12
  heads:
    - name: global_10
      kind: multiclass
      labels: [ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE]
    - name: binary_lipid
      kind: binary
      labels: [nonlipid, lipid]
    - name: lipid_family
      kind: multiclass
      labels: [CLR, MYR, OLA, PLM, STE]
  experts:
    - name: ste_neighbors
      kind: local_multiclass
      labels: [PLM, COA, OLA, MYR, STE]
      gate: utility
      combine: closed_world_redistribute
      feature_set: v_caver_t12
    - name: ola_plm
      kind: binary_boundary
      labels: [OLA, PLM]
      gate: utility
      combine: pair_swap
      feature_set: v_sterol
  execution_order:
    - global_10
    - binary_lipid
    - lipid_family
    - ste_neighbors
    - ola_plm
```

Backward compatibility requirement:

1. current hierarchical YAMLs must normalize into this schema without behavior drift

## 10. Composite Bundle and Artifacts

Replace the current ad hoc hierarchical bundle keys with a versioned composite bundle schema.

Bundle contents must include:

1. bundle version
2. topology definition
3. feature-family registry
4. ordered feature columns per component
5. schema hash per component
6. backbone weights and config
7. head weights and config
8. expert weights and config
9. gate weights and config
10. combiner config
11. calibration config
12. class order
13. settings snapshot
14. library versions

Rules:

1. holdout inference must be reconstructible from the bundle alone
2. the current final prediction parquet contract must remain unchanged
3. route diagnostics may be emitted as separate artifacts

## 11. Repository Refactor Instructions

This is the required downstream implementation sequence.

## 11.1 New core modules

Create a dedicated composite-model layer rather than continuing to overload old staged files.

Recommended module layout:

1. `src/slipp_plus/composite_types.py`
2. `src/slipp_plus/composite_config.py`
3. `src/slipp_plus/composite_topology.py`
4. `src/slipp_plus/composite_bundle.py`
5. `src/slipp_plus/composite_train.py`
6. `src/slipp_plus/composite_infer.py`
7. `src/slipp_plus/composite_combine.py`
8. `src/slipp_plus/composite_gates.py`
9. `src/slipp_plus/composite_diagnostics.py`
10. `src/slipp_plus/backbone_family_encoder.py`
11. `src/slipp_plus/backbone_distill.py`
12. `src/slipp_plus/expert_registry.py`

## 11.2 Existing files to reuse or refactor

High-priority reuse surfaces:

1. `src/slipp_plus/boundary_head.py`
2. `src/slipp_plus/hierarchical_postprocess.py`
3. `src/slipp_plus/specialist_utility_gate.py`
4. `src/slipp_plus/hierarchical_pipeline.py`
5. `src/slipp_plus/artifact_schema.py`
6. `src/slipp_plus/config.py`
7. `src/slipp_plus/train.py`
8. `src/slipp_plus/evaluate.py`

High-priority cleanup:

1. stop production code from depending on `hierarchical_experiment.py` helpers
2. unify neighbor-rescue and boundary-rewrite abstractions
3. stop creating pair-specific pipeline files as the default extension mechanism

## 11.3 Migration order

### Phase 0: no behavior change

1. extract common gate, expert, and combiner abstractions
2. define topology objects
3. define composite bundle schema
4. keep current hierarchical execution alive

### Phase 1: parity topology

1. port the current best boundary-refactor system into the composite topology
2. verify split-level parity
3. verify holdout parity
4. freeze this as the composite baseline

### Phase 2: expert expansion

1. add high-value local experts missing from the parity topology
2. generalize utility gates to all experts
3. add route diagnostics and per-expert reporting

### Phase 3: backbone replacement

1. implement the family-aware backbone encoder
2. distill from the composite teacher stack
3. swap the flat teacher backbone for the shared encoder backbone under the same topology
4. verify parity before claiming gains

### Phase 4: CAVER integration

1. add CAVER as a feature family
2. route CAVER to access-sensitive experts first
3. broaden CAVER use only after local benefit is established

## 12. Diagnostics and Evaluation

The final output artifact consumed by evaluation remains unchanged.

Required new diagnostics:

1. per-expert candidate count
2. per-expert fire rate
3. per-expert keep rate
4. local accuracy delta or F1 delta
5. per-class support inside expert scope
6. top helped and top harmed rows per expert
7. family availability summaries
8. backbone and expert calibration summaries

Do not break the existing evaluation contract while adding these diagnostics.

## 13. Tests

Minimum required tests:

1. config normalization from old hierarchical YAMLs to composite schema
2. deterministic topology execution order
3. probability invariants
4. feature-routing validation per component
5. bundle round-trip integrity
6. holdout inference parity from iteration-0 bundle
7. utility-gate behavior on harmful rewrites
8. missing-family robustness when CAVER or optional families are absent
9. backbone distillation smoke tests

## 14. Acceptance Gates

Each phase must satisfy its gate before the next phase begins.

### Phase 1 parity gate

1. current best boundary-refactor behavior is reproduced within tolerance
2. prediction parquet shape is unchanged
3. holdout inference still works from one bundle

### Phase 2 expert gate

1. each added expert shows positive local value or is disabled
2. no expert silently regresses macro-F1 over the 25-iteration protocol

### Phase 3 backbone gate

1. distilled backbone reaches parity with the composite teacher before lift claims
2. calibration is not materially worse than the teacher
3. rare-class collapse is not accepted as the price of cleaner architecture

### Phase 4 CAVER gate

1. CAVER improves at least one high-value local boundary materially
2. CAVER does not regress the global system beyond tolerated bounds
3. availability handling is correct when CAVER is absent

## 15. Agent DO / DO NOT

DO:

1. preserve the current output artifact contract
2. preserve OOF discipline for gates and experts
3. reuse and unify existing local arbitration logic
4. keep feature routing explicit and schema-validated
5. achieve parity before improvement claims
6. treat the new shared backbone as the final target, not an optional extension
7. use distillation for backbone transition

DO NOT:

1. jump directly to a neural backbone before composite parity exists
2. add another one-off tiebreaker file as the default pattern
3. let experts rewrite unrelated class mass
4. hide missing optional families by collapsing them into ordinary zeros
5. break the current holdout evaluation path
6. claim lift from CAVER or the new backbone without 25-iteration evidence

## 16. Default Initial Topology

Use this as the default target topology for the first full implementation.

1. shared feature-family backbone
2. global 10-class head
3. binary lipid head
4. lipid-family head
5. STE-neighbors expert
6. OLA-vs-PLM expert
7. optional CLR-local expert
8. optional nonlipid cleanup expert
9. composite calibration layer
10. final probability combiner

## 17. Definition of Success

Model V2 is successful if it satisfies all of the following.

1. It subsumes the current best boundary-refactor stack without ad hoc duplication.
2. It replaces the independent flat ensemble backbone with a shared representation backbone.
3. It allows optional feature families such as CAVER to be integrated cleanly.
4. It preserves the current evaluation and paper-comparison contract.
5. It improves subclass performance materially.
6. It reduces engineering cost for future experts and future feature families.
7. It creates a stable seam for future alternate backbones without another architectural rewrite.

## 18. Final Decision Summary

These are the final decisions from this planning thread.

1. The refactor target is a topology-driven routed composite model.
2. The target architecture includes a new shared feature-family backbone.
3. The backbone replacement is phased after composite parity, but it is part of the approved target.
4. Local experts are mandatory first-class objects.
5. Learned utility gates are the default routing policy.
6. Probability rewriting must remain closed-world and simplex-preserving.
7. CAVER is a feature family, not a bespoke model path.
8. Distillation is the preferred path for the backbone swap.
9. Success is measured mainly in subclass lift, not binary lift alone.
