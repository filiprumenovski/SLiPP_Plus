# Next Agent Handoff

Last updated: 2026-05-08

This repo is in publication-polish mode for `handoff.md`. Preserve the full
research audit trail: do not delete experiment logs, generated reports, registry
entries, failed/abandoned experiment notes, or queued work. Negative results are
scientific evidence in this project.

## Current State

- Branch: `main`
- Latest pushed commit at time of writing: `62277da experiments: record shell6 tunnel-shape ablation`
- Worktree was clean when this file was written.
- Current internal leader: `exp-014-v49-tunnel-shape3`
- Current internal best config: `configs/v49_tunnel_shape3_family_encoder.yaml`
- Previous more balanced compact candidate: `exp-012-compact-tunnel-shape`
- Current best report: `reports/v49_tunnel_shape3_family_encoder/metrics_table.md`
- Current best registry entry: `experiments/registry.yaml` entry `exp-014-v49-tunnel-shape3`

Current headline metrics from the registry/README:

| metric | value |
|---|---:|
| Binary F1 | `0.900 +/- 0.015` |
| Binary AUROC | `0.988 +/- 0.004` |
| 10-class macro-F1 | `0.768 +/- 0.018` |
| 5-lipid macro-F1 | `0.668 +/- 0.031` |
| Apo-PDB F1 | `0.667` |
| AlphaFold F1 | `0.724` |

The new internal leader is not a clean all-metric win: apo-PDB regresses vs
exp-012, while AlphaFold improves. Do not hide that.

## Recently Completed

The dirty research snapshot that may contain the best model was staged,
committed, and pushed:

- `e957cb2 wip: preserve composite tunnel research snapshot`

The README was rewritten around the actual release candidate and pushed:

- `aa18641 docs: make README release-candidate focused`

Several `handoff.md` item `6.3` docstring slices were completed and pushed:

- `839f1a9 docs: 6.3 document family encoder stack`
- `f0692c4 docs: 6.3 document caver t12 helpers`
- `2932d4a docs: 6.3 document core helper APIs`
- `6ae8365 docs: 6.3 document split persistence APIs`
- `7163bf7 docs: 6.3 document evaluation APIs`
- `9296bed docs: 6.3 document figure APIs`
- `05ebcdf docs: 6.3 document training entrypoint`
- `ccf85fa docs: 6.3 document composite topology metadata`
- `cf095a1 docs: 6.3 document utility gate serialization`
- `5c235b1 docs: 6.3 document boundary tiebreaker pipeline`

The release-facing README was pushed:

- `aa18641 docs: make README release-candidate focused`

The next-agent root handoff was added:

- `e8be7aa docs: add next agent handoff`

The local quality gate was repaired and pushed:

- `b551bc9 fix: restore local quality gate`
  - Fixed a real Ruff-discovered bug: missing `load_split` import in
    `src/slipp_plus/pair_tiebreaker_experiment.py`.
  - Aligned type checking with the incremental handoff strategy:
    `make typecheck` runs `mypy $(MYPY_TARGETS)`, defaulting to
    `src/slipp_plus/cli.py`, and CI calls `make typecheck`.
- `d8290d7 style: ruff format sweep`
  - Repo-wide `ruff format` sweep after the gate fix.

Reproducibility hardening was pushed:

- `83f4b5f repro: 5.2 write model metadata sidecars`
  - Added `src/slipp_plus/run_metadata.py`.
  - Flat RF/XGB/LGBM bundles, hierarchical bundles, family-encoder bundles,
    and composite pair-MoE bundles now write adjacent
    `.joblib.metadata.json` sidecars with package versions, Python version,
    config path/hash, git commit, UTC timestamp, and seed.
  - Added `tests/test_run_metadata.py`.

Focused verification runs during these ticks:

- `uv run pytest -q tests/test_feature_families.py tests/test_pipeline_mode.py`: passed
- `uv run pytest -q tests/test_caver_analysis.py tests/test_caver_t12_features.py`: passed
- `uv run pytest -q tests/test_splits_determinism.py`: passed
- `uv run pytest -q tests/test_pipeline_mode.py tests/test_regression_day1.py`: passed with slow regression skipped as intended
- `uv run pytest -q tests/test_specialist_utility_gate.py`: passed
- Focused Ruff checks passed for every touched slice
- `make test`: passed after the quality-gate repair and again after the
  metadata-sidecar work (`165 passed, 2 skipped` on the latest run).
- `uv run ruff format --check .`: passed after the format sweep.
- `uv run mypy src/slipp_plus/run_metadata.py`: passed.

## Remaining Work

### 1. Highest-impact non-docstring work

The user explicitly redirected away from docstring cleanup with "skip docstring
shit, start high impact work." Prefer these before returning to item 6.3:

1. Full mypy hardening beyond the current `MYPY_TARGETS` allowlist.
   - Current local gate is intentionally incremental:
     `MYPY_TARGETS ?= src/slipp_plus/cli.py`.
   - `make test` is green, but repo-wide `mypy src` is not complete.
2. Release-candidate reproduction.
   - `make all CFG=configs/v49_tunnel_shape_family_encoder.yaml` has not been
     rerun in this handoff window.
   - If long-run mode is not authorized, do not run it inline; record exact
     command and expected artifacts instead.
3. Long-run ablation closure.
   - See `experiments/queued.md`; these are intentionally queued because they
     need longer model runs or missing holdout implementation.
4. CI state.
   - Local `make test` is green, but GitHub Actions status has not been checked
     or fixed in this window. The user previously said CI can come later.
5. Completion audit.
   - Before marking the active goal complete, map every `handoff.md` item and
     done criterion to actual files, commands, or explicit deferrals. Do not
     treat this handoff file as proof by itself.

### 2. `handoff.md` item 6.3 docstrings remain unfinished

Current audit command:

```bash
uv run python - <<'PY'
from pathlib import Path
import ast
missing = []
for path in sorted(Path("src/slipp_plus").glob("*.py")):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            if ast.get_docstring(node) is None:
                missing.append((str(path), node.lineno, node.name))
print(len(missing))
for row in missing:
    print(f"{row[0]}:{row[1]} {row[2]}")
PY
```

Most recent count after `83f4b5f`: `58` public functions still missing
docstrings. This count did not improve during the high-impact gate/repro work.

Known remaining functions at that point:

```text
src/slipp_plus/aromatic_aliphatic.py:70 write_features
src/slipp_plus/aromatic_aliphatic.py:359 main
src/slipp_plus/calibration.py:357 plot_reliability
src/slipp_plus/calibration.py:438 plot_histograms
src/slipp_plus/calibration.py:503 write_metrics_md
src/slipp_plus/calibration.py:612 write_summary
src/slipp_plus/calibration.py:678 run_calibration
src/slipp_plus/composite_pair_moe.py:279 run_pair_moe_training
src/slipp_plus/detector_bakeoff.py:604 main
src/slipp_plus/ensemble.py:402 main
src/slipp_plus/graph_tunnel_features.py:78 load_alpha_spheres
src/slipp_plus/graph_tunnel_features.py:333 build_training_v_graph_tunnel_parquet
src/slipp_plus/graph_tunnel_features.py:358 build_holdout_v_graph_tunnel_parquet
src/slipp_plus/graph_tunnel_features.py:402 main
src/slipp_plus/hierarchical_experiment.py:50 train_lipid_gate
src/slipp_plus/hierarchical_experiment.py:69 train_lipid_family
src/slipp_plus/hierarchical_experiment.py:153 train_nonlipid_family
src/slipp_plus/hierarchical_experiment.py:218 train_one_vs_neighbors
src/slipp_plus/hierarchical_experiment.py:241 build_specialist_training
src/slipp_plus/hierarchical_experiment.py:530 run_hierarchical_experiment
src/slipp_plus/hierarchical_experiment.py:673 main
src/slipp_plus/hierarchical_pipeline.py:256 load_hierarchical_bundle
src/slipp_plus/hierarchical_pipeline.py:260 save_hierarchical_bundle
src/slipp_plus/lipid_boundary_features.py:348 extract_pocket_lipid_boundary_features
src/slipp_plus/lipid_boundary_features.py:613 main
src/slipp_plus/pair_tiebreaker_experiment.py:58 build_pair_training
src/slipp_plus/pair_tiebreaker_experiment.py:73 train_pair_tiebreaker
src/slipp_plus/pair_tiebreaker_experiment.py:77 apply_pair_tiebreaker
src/slipp_plus/pair_tiebreaker_experiment.py:163 write_pair_experiment_report
src/slipp_plus/pair_tiebreaker_experiment.py:231 run_pair_tiebreaker_experiment
src/slipp_plus/pair_tiebreaker_experiment.py:391 main
src/slipp_plus/plm_ste_features.py:835 main
src/slipp_plus/plm_ste_holdout.py:144 train_iteration0_tiebreaker
src/slipp_plus/plm_ste_holdout.py:172 score_holdout_condition
src/slipp_plus/plm_ste_holdout.py:178 infer_holdout_classes
src/slipp_plus/plm_ste_holdout.py:183 pair_confusion_metrics
src/slipp_plus/plm_ste_holdout.py:222 write_holdout_report
src/slipp_plus/plm_ste_holdout.py:314 run_holdout_validation
src/slipp_plus/plm_ste_holdout.py:419 main
src/slipp_plus/plm_ste_tiebreaker.py:345 write_plm_ste_tiebreaker_report
src/slipp_plus/plm_ste_tiebreaker.py:553 main
src/slipp_plus/ste_rescue_experiment.py:37 build_ste_rescue_training
src/slipp_plus/ste_rescue_experiment.py:56 train_ste_rescue_head
src/slipp_plus/ste_rescue_experiment.py:60 apply_ste_rescue
src/slipp_plus/ste_rescue_experiment.py:194 run_ste_rescue_experiment
src/slipp_plus/ste_rescue_experiment.py:328 main
src/slipp_plus/sterol_features.py:509 main
src/slipp_plus/sterol_tiebreaker.py:278 write_tiebreaker_report
src/slipp_plus/sterol_tiebreaker.py:455 main
src/slipp_plus/tunnel_features.py:129 load_caver_settings
src/slipp_plus/tunnel_features.py:1334 build_training_v_tunnel_parquet
src/slipp_plus/tunnel_features.py:1417 build_holdout_v_tunnel_parquet
src/slipp_plus/tunnel_features.py:1543 main
src/slipp_plus/v49.py:294 main
src/slipp_plus/v49_holdouts.py:219 main
src/slipp_plus/v61.py:77 main
src/slipp_plus/v_sterol_report.py:151 run_report
src/slipp_plus/v_sterol_report.py:331 main
```

Recommended next docstring slices when docstring work is back in scope:

1. `src/slipp_plus/v_sterol_report.py`
2. `src/slipp_plus/hierarchical_pipeline.py`
3. `src/slipp_plus/plm_ste_holdout.py`
4. `src/slipp_plus/pair_tiebreaker_experiment.py`
5. feature-builder CLIs such as `v49.py`, `v61.py`, `sterol_features.py`,
   `aromatic_aliphatic.py`, `tunnel_features.py`

For each slice, run focused Ruff and any related tests. Keep commits small and
push after each coherent tick.

### 3. Long-run ablations remain queued, not done

See `experiments/queued.md`. Do not delete it. It currently queues:

- Holdout completion for `exp-005-v_sterol-ensemble`
- Holdout completion for `exp-009-v_sterol-boundary-refactor`
- Composite holdout inference/metrics for `exp-011-family-plus-moe`
- STE class-imbalance handling ablation
- Ensemble vs best single model ablation
- Tiebreaker on/off ablation
- CAVER/tunnel marginal value ablation

These require long model runs or new implementation. Do not run them inline
unless the user explicitly authorizes long-run mode.

### 4. Full completion criteria are still not met

Do not mark the goal complete yet. `handoff.md` done criteria still need:

- All TODOs completed or explicitly deferred with rationale.
- `make test` passing locally and in CI.
- `mypy src` passing.
- `make all` / release-candidate reproduction checked against README claims.
- `experiments/registry.yaml` and `reports/ablation_matrix.md` synchronized
  after remaining ablations/deferrals.
- CI green on latest push.
- Final release/user-action checklist.

### 5. Important docs and artifacts already exist

These files exist and should be preserved:

- `LICENSE`
- `CITATION.cff`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `DATASHEET.md`
- `RELEASE_NOTES_v0.1.0.md`
- `README.md`
- `docs/api.md`
- `docs/methods.md`
- `docs/reproducibility.md`
- `docs/features/tunnel_features.md`
- `data/holdouts/apo_pdb_ids.csv`
- `data/holdouts/alphafold_ids.csv`
- `reports/ablation_matrix.md`
- `reports/compact_publishable/summary.md`
- `reports/compact_publishable/compact_ladder_metrics.csv`
- `MODEL_V2_SPEC.md`
- `experiments/queued.md`

### 6. Remote note

`git push` succeeds but prints:

```text
remote: This repository moved. Please use the new location:
remote:   https://github.com/filiprumenovski/SLiPP-Plus.git
```

Pushes have still been accepted through the old remote URL.

## Guardrails For The Next Agent

- Do not delete `reports/`, `logs/`, `CONTEXT.md`, `RESEARCH_LOG.md`,
  `experiments/registry.yaml`, or `experiments/queued.md`.
- Do not discard dirty/untracked research files without inspecting them; they may
  contain the best model or a negative result.
- Avoid broad `ruff format .` or `ruff check . --fix` unless the user asks; use
  focused file-level checks.
- Do not spend credits on CI or long ML runs unless the user asks.
- Prefer small commits with messages that reference `handoff.md` item numbers
  where applicable.
- Keep the compact leader story honest: best internal model is strong, external
  holdouts are conservative.
