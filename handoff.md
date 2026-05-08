# SLiPP++ Publication-Polish Autoloop — Handoff Prompt

## Context

**SLiPP++** is a 10-class softmax reformulation of the published binary lipid-pocket classifier from Chou et al. 2024 (J. Chem. Inf. Model.). The repo lives at `/Users/filiprumenovski/Code/slipp_plus` and reproduces Table 1 of the paper while extending it to per-class predictions across `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE}`. Best current config: `v_sterol` ensemble (RF/XGB/LGBM) + STE/PLM tiebreaker; binary F1 0.899±0.015 (paper 0.869), 10-class macro-F1 0.754±0.016.

The codebase is **research-grade and substantively complete**: 44 src modules, 30 tests, 22 YAML configs, machine-readable experiment registry, hardened CAVER pipeline (commit `91bb104`). What it lacks is the **last-mile polish for publication**: documentation, packaging hygiene, CI, reproducibility guarantees, and a closed ablation matrix.

This file is the **handoff prompt for a persistent autoloop agent**. The autoloop should pick the highest-priority unticked item, do it well, verify, commit, and continue on the next iteration. It is not a single one-shot task.

---

## Mission

Bring `slipp_plus` from "good research code" to "publication-quality release" suitable for submission to JCIM / JOSS / Bioinformatics or as a Methods Appendix companion to a paper. Four parallel tracks:

1. **Packaging polish** — installable, citable, redistributable.
2. **Validation depth** — close any open ablations, add the missing scientific controls, document each result against paper Table 1.
3. **Documentation** — public API, scientific methods, data provenance, install/run/cite path.
4. **Quality infrastructure** — CI, lint/type/test green, reproducibility guarantees.

---

## Operating Principles for the Autoloop

- **One coherent unit of work per tick.** Pick the highest-priority unticked TODO, do it, verify, commit with a descriptive message, then return.
- **Never break the build.** Before committing: `make test` (or at least `pytest -q` + `ruff check .`). If the change is large, run `make all` end-to-end on `configs/day1.yaml` and confirm `reports/metrics_table.md` reproduces within tolerance vs the existing one.
- **Read before you write.** This codebase has a strong existing architecture. Reuse `experiments/registry.yaml`, `reports/`, `configs/*.yaml`, and the `src/slipp_plus/` modules — do not invent parallel structures.
- **Cite paper Table 1 when adding metrics tests.** Ground truth is `configs/day1.yaml::ground_truth` and the README table.
- **Keep `RESEARCH_LOG.md` and `experiments/registry.yaml` in sync.** Any new experiment you run must be appended to the registry with `id`, `date`, `config`, `metrics`, `holdouts`, `artifacts`, `notes`. Use today's date in ISO format.
- **Stop the loop early** if you encounter a hard scientific question (e.g. "should STE oversampling be SMOTE or class-weighting?"). Surface it in the commit message body and skip the item rather than guessing.
- **Never run destructive git operations** (force push, reset --hard, branch -D). Never skip hooks (`--no-verify`).
- **Respect the worktree.** Current worktree is `/Users/filiprumenovski/Code/slipp_plus/.claude/worktrees/elated-swartz-87b7c4`. Operate there.
- **Cost discipline.** Long ML runs (`make all` on a non-baseline config = ~25 min) should be queued by writing a config + invocation note to `experiments/queued.md`, not executed inline, unless the user explicitly enables long-run mode in this prompt.

---

## Current-State Triage (verified 2026-05-08)

| Area | State | Action |
|---|---|---|
| Core pipeline (`make all` day1) | Reproduces paper within 0.01 F1 | Keep stable; treat as regression baseline |
| `src/slipp_plus/` (44 modules) | Good factoring | Prune dead `import sys`, unify logging |
| Tests (30 files, 125 functions) | **21/52 fail collection** — `pandera.pandas` import in stale env | Fix import resilience + ensure `uv sync` |
| `pyproject.toml` | MIT declared | `LICENSE` file **missing** — must add |
| `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md` | All **missing** | Add |
| `.github/workflows/` | **Missing** | Add CI: pytest + ruff + mypy on push/PR |
| Logging | 70 `print()` vs 41 `logging.*` declared-but-unused | Unify on logging + typer.echo |
| Type hints | `mypy strict = false`, widespread `Any` | Tighten module by module |
| README | Quickstart good, citation/install thin | Expand: installation block, citation, CAVER setup |
| API docs | None (no Sphinx/mkdocs); 18 docstring'd vs 81 undocumented | Pick one: docstrings + `docs/api.md`, OR Sphinx |
| `docs/features/` | 3 stub markdowns | Expand `tunnel_features.md` per CAVER hardening |
| Examples / notebooks | **None** | Add `examples/quickstart.py` + 1 notebook |
| Data provenance | Implicit; PDB IDs not enumerated | Add `DATASHEET.md` + holdout PDB-ID listings |
| Ablation matrix | 22 configs exist, no consolidated table | Generate `reports/ablation_matrix.md` |
| Experiment registry | `experiments/registry.yaml` exists, well-structured | Append any new runs |
| Reproducibility | Seeds threaded; **no global numpy/random init in CLI entry** | Add explicit master-seed init |
| Versioning | `__version__ = "0.1.0"` hardcoded; no tags | Add CHANGELOG, tag v0.1.0 once polished |
| Zenodo / DOI | None | Reserve DOI when ready to release (user action) |

---

## TODO List (Priority-Ordered)

The autoloop should walk this list top-to-bottom. Each item is sized to fit one tick. **Mark complete in the commit message** by referencing the section number (e.g. `polish: 1.2 add LICENSE file`).

### 1. Packaging Essentials (do first — fast wins, unblocks the rest)

- [ ] **1.1** — Add `LICENSE` file at repo root with MIT license text matching `pyproject.toml::license = { text = "MIT" }`. Author: Filip Rumenovski. Year: 2026.
- [ ] **1.2** — Add `CITATION.cff` at repo root. Cite this software (title, author, version, year, repo URL) and reference Chou et al. 2024 (DOI 10.1101/2024.01.26.577452) as the primary scientific reference. Use [citation-file-format v1.2.0](https://citation-file-format.github.io/) schema.
- [ ] **1.3** — Add `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Seed with `[0.1.0] - 2026-05-08` listing the major capabilities (10-class reformulation, v_sterol ensemble, CAVER hardening, hierarchical lipid pipeline, holdout validation). Use `git log --oneline` for raw material.
- [ ] **1.4** — Add `CONTRIBUTING.md` with: dev install (`uv sync --extra dev`), test command (`make test`), code style (`ruff format`), commit message convention, branch model.
- [ ] **1.5** — Single-source `__version__`. Move version to `src/slipp_plus/__version__.py` with one line `__version__ = "0.1.0"`. Re-export from `__init__.py`. Update `pyproject.toml` to use `[tool.hatch.version] path = "src/slipp_plus/__version__.py"` (dynamic).
- [ ] **1.6** — Add `--version` flag to the Typer CLI in `src/slipp_plus/cli.py` that prints `__version__`.

### 2. Test Suite Repair (must be green before publication)

- [ ] **2.1** — Diagnose `pandera.pandas` import failure. The current `src/slipp_plus/schemas.py:10` uses `import pandera.pandas as pa`, but pandera <0.20 exported the API as `import pandera as pa`. Pin or guard the import: try the new path, fall back to legacy. Verify against `pandera>=0.20,<0.25` (the pinned range). If the fix requires a tighter pin, update `pyproject.toml`.
- [ ] **2.2** — Get `uv run pytest -q` to **collect 0 errors** in this worktree. Run `uv sync --extra dev` if the env is stale; if a test still fails because biopython isn't on the dep tree it expects, mark Bio-dependent tests with `pytest.importorskip("Bio")` rather than moving biopython around (it's already in main deps per `pyproject.toml:36`).
- [ ] **2.3** — Get `uv run pytest -q` to **pass 100%** of collected tests. Investigate any genuine failures; they may indicate real bugs introduced during recent refactors.
- [ ] **2.4** — Add a single **regression test** that runs `make ingest` + a tiny train (1 iteration, 1 model) on `configs/day1.yaml` and asserts binary F1 within ±0.05 of paper 0.869 on a fixed seed. Place in `tests/test_regression_day1.py`. Mark as `@pytest.mark.slow` and gate behind a `--runslow` flag so default `pytest` stays fast.
- [ ] **2.5** — Wire `pytest --runslow` into a separate Makefile target `make test-slow` so day1 reproduction is a one-command check.

### 3. CI / Quality Gates

- [ ] **3.1** — Add `.github/workflows/ci.yml`. Triggers: push to any branch, PR to main. Matrix: Python 3.11 and 3.12, ubuntu-latest. Steps: checkout, install uv, `uv sync --extra dev`, `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src` (non-strict initially). Cache `~/.cache/uv`.
- [ ] **3.2** — Add `.github/workflows/release.yml` (manual dispatch). On tag `v*`, build wheel via `uv build`, upload to GitHub Release. Do **not** auto-publish to PyPI yet — leave that as a user action.
- [ ] **3.3** — Add `.pre-commit-config.yaml` with hooks for `ruff check`, `ruff format`, `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`. Document setup in `CONTRIBUTING.md`.
- [ ] **3.4** — Add CI status badge to `README.md` once the workflow runs green.
- [ ] **3.5** — Tighten mypy incrementally. Pick one module per tick (start with `src/slipp_plus/cli.py`, then `splits.py`, `schemas.py`), add full type hints, add to a per-module strict allowlist in `pyproject.toml::[tool.mypy.overrides]`. Goal: `mypy --strict src/slipp_plus/cli.py` passes by end of polish.

### 4. Logging & CLI Output Hygiene

- [ ] **4.1** — Audit the 12 modules that import `logging` and never use it (e.g. `tunnel_features.py`, `aromatic_aliphatic.py`, `v49.py`, `sterol_features.py`, `detector_bakeoff.py`, `plm_ste_features.py`, `v_sterol_report.py`, `v49_holdouts.py`, `lipid_boundary_features.py`, `v61.py`). For each: either delete the unused logger, or replace nearby `print()` calls with `logger.info()` / `logger.warning()`.
- [ ] **4.2** — Establish a single logging entry point. Add `src/slipp_plus/logging_config.py` with `setup_logging(level: str = "INFO")` that configures root logger (format with timestamp + module + level). Call it from `cli.py::app` callback. Honor `SLIPP_LOG_LEVEL` env var.
- [ ] **4.3** — User-facing CLI messages stay on `typer.echo()` (they're for the operator). Diagnostic messages move to `logging`. Decision rule: if it's part of the "what the command did" narrative, `echo`; if it's "the module did this internally", `logging`.
- [ ] **4.4** — Remove dead `import sys` from modules that don't use it (audit reported in `tunnel_features.py`, `aromatic_aliphatic.py`, `v49.py`, `sterol_features.py`, `detector_bakeoff.py`, `plm_ste_features.py`, `v_sterol_report.py`, `v49_holdouts.py`, `lipid_boundary_features.py`, `v61.py`).

### 5. Reproducibility Hardening

- [ ] **5.1** — Add a master seed-init in the CLI callback (before any subcommand runs): `numpy.random.seed(seed)`, `random.seed(seed)`, set `PYTHONHASHSEED` if unset. Read seed from the active config's `seed_base`. Document in `docs/reproducibility.md`.
- [ ] **5.2** — Embed run metadata in every persisted artifact. When training writes a joblib bundle, also write a sidecar JSON: `{slipp_plus_version, sklearn_version, xgboost_version, lightgbm_version, numpy_version, python_version, config_path, config_sha256, git_commit, timestamp_utc, seed}`. Look at `train.py` for the right hook point.
- [ ] **5.3** — Make `uv.lock` authoritative for reproducible installs. Document in README: "For exact reproduction, use `uv sync --frozen --extra dev`."
- [ ] **5.4** — Verify `processed/{config}/full_pockets.parquet` is byte-identical across two runs with the same seed (parquet metadata may include timestamps; if so, normalize). Add a determinism test under `tests/test_determinism.py`.

### 6. Documentation — Scientific & API

- [ ] **6.1** — Expand `README.md` with: (a) full Installation section (uv install, system fpocket, optional CAVER), (b) "How to cite" block referencing `CITATION.cff`, (c) link to `docs/`, (d) example one-liner for the `v_sterol` ensemble + tiebreaker config (the recommended production setup), (e) mention `make scratch` prerequisites.
- [ ] **6.2** — Create `docs/api.md`. For every public CLI subcommand and every public function in modules called from `cli.py`, write a NumPy-style entry: signature, parameters, returns, raises, example. Auto-generate from docstrings if Sphinx is set up; otherwise hand-curate.
- [ ] **6.3** — Add NumPy-style docstrings to every public function (those not prefixed with `_`) in `src/slipp_plus/`. The audit found 81 undocumented vs 18 documented. Walk modules in this order, one per tick: `cli.py`, `ingest.py`, `train.py`, `eval.py`, `splits.py`, `figures.py`, then feature modules (`sterol_features.py`, `aromatic_aliphatic.py`, `tunnel_features.py`, `lipid_boundary_features.py`, `plm_ste_features.py`), then supporting (`schemas.py`, `constants.py`, `artifact_schema.py`, `binary_collapse.py`, `confusion_mining.py`, `hierarchical_pipeline.py`, `v49.py`, `v61.py`, `v_sterol_report.py`).
- [ ] **6.4** — Expand `docs/features/tunnel_features.md` to cover: CAVER binary version requirement, expected directory layout, parser logic, the `tunnel_pocket_context_present` and `tunnel_caver_profile_present` quality gates, recommended threshold values, failure modes, and the `--analysis-output-root` / `--analysis-manifest` persistence model. Reference `commit 91bb104` for the hardening rationale.
- [ ] **6.5** — Create `docs/methods.md` — a short scientific methods document describing: data source (Chou et al. CSV + supplementary xlsx), training set composition (15,219 rows; per-class breakdown), feature set definitions (v14, v22, +aa, sterol, tunnel, lipid_boundary), model ensemble (RF/XGB/LGBM probability average), tiebreaker logic (PLM/STE motif-based binary head), evaluation protocol (25 stratified shuffle splits, 10-class softmax → binary collapse for paper comparison), holdout sets (apo-PDB, AlphaFold). This becomes citable supplementary material.
- [ ] **6.6** — Create `DATASHEET.md` (datasheet-for-datasets style). Document: training CSV provenance + checksum, supplementary xlsx provenance, holdout PDB ID list (apo + AlphaFold), known data quality issues (e.g., STE has only 152 samples — note this explicitly), license of upstream data.
- [ ] **6.7** — Expand `reference/README.md` with concrete shell commands for first-time setup: `git clone https://github.com/dassamalab/SLiPP_2024 reference/SLiPP_2024-main`, then verify checksum of `training_pockets.csv` (compute and pin in this README).

### 7. Examples / Tutorials

- [ ] **7.1** — Add `examples/quickstart.py` — a single-script end-to-end run on `configs/day1.yaml`. Five sections with header comments: load config → ingest → train (1 iter) → eval → print binary F1. Self-contained so a reviewer can paste-and-run.
- [ ] **7.2** — Add `examples/v_sterol_ensemble.py` — demonstrates the recommended production config (v_sterol + ensemble + tiebreaker), shows how to load predictions and compute confusion mining for STE.
- [ ] **7.3** — Add `examples/README.md` with one-paragraph descriptions of each example.
- [ ] **7.4** — (Optional, later) Add a Jupyter notebook `examples/figure_reproduction.ipynb` that renders the paper-comparison bar plot and confusion matrix from the persisted `reports/raw_metrics.parquet`.

### 8. Ablation Matrix Closure

The repo has 22 configs but no consolidated table mapping config → result → conclusion. Close this gap.

- [ ] **8.1** — Generate `reports/ablation_matrix.md` from `experiments/registry.yaml`. Columns: `experiment_id`, `feature_set`, `binary_f1 ± std`, `macro_f1_10 ± std`, `apo_pdb_f1`, `alphafold_f1`, `key_finding`, `superseded_by`. Sort by date. Mark `is_current_best` row in bold.
- [ ] **8.2** — Identify ablations that are **partial** in the registry (status: in-progress or missing holdout numbers) and either complete them by queuing a run (write `experiments/queued.md` entry) or mark them `abandoned` with reasoning.
- [ ] **8.3** — Add a missing-but-needed ablation: **STE class-imbalance handling**. STE has 152 samples (smallest class). Run two configs: (a) baseline (current), (b) class-weighted. Append both to registry. If results are within 1σ, document that imbalance handling is not the bottleneck (informs paper).
- [ ] **8.4** — Add an ablation: **ensemble vs best single model**. Currently the ensemble is RF+XGB+LGBM averaged. Verify that ensemble > best single on macro-F1 by ≥ 1σ. Document in `reports/ensemble_ablation.md`.
- [ ] **8.5** — Add an ablation: **tiebreaker on/off**. Run `v_sterol` with and without the PLM/STE pair head. Quantify the lift attributable to the tiebreaker per-class.
- [ ] **8.6** — Add an ablation: **CAVER tunnel features marginal value**. Run `v_sterol` vs `v_sterol + caver_t12`. Document delta in `reports/caver_ablation.md`. This is the publishable answer to "do tunnels help?".
- [ ] **8.7** — Wire `src/slipp_plus/v_sterol_ablation.py` (currently script-only, not in CLI per audit) into the Typer CLI as `slipp_plus ablate-v-sterol` and into the Makefile.

### 9. Holdout & Validation Depth

- [ ] **9.1** — List the **apo-PDB holdout** structure IDs in `data/holdouts/apo_pdb_ids.csv` (one column: `pdb_id`). Read them out of the existing supporting xlsx files. 131 expected per paper.
- [ ] **9.2** — List the **AlphaFold holdout** structure IDs / UniProt accessions in `data/holdouts/alphafold_ids.csv`. 177 expected per paper.
- [ ] **9.3** — Add a confidence-interval reporter. Currently `reports/metrics_table.md` shows mean ± std across 25 iterations. Also report 95% CI (bootstrap or t-based). Update the metrics writer in `src/slipp_plus/eval.py`.
- [ ] **9.4** — Add per-class precision/recall (not just F1) to `reports/metrics_table.md`. Useful for the paper's per-class discussion.

### 10. Final Polish

- [ ] **10.1** — Run `ruff check . --fix` and `ruff format .` across the entire repo. Commit as a separate "style: ruff sweep" commit so the diff is auditable.
- [ ] **10.2** — Sweep for and remove TODO/FIXME/XXX comments. Either resolve, file as a registry entry, or move to a dedicated `TODO.md` with rationale.
- [ ] **10.3** — Generate the final `reports/metrics_table.md` from the latest `v_sterol` ensemble run + tiebreaker. Confirm headline numbers match what `README.md` claims. If they drift, update the README.
- [ ] **10.4** — Verify `make all` runs clean from a freshly cloned env (use a sibling worktree or a tmpdir). Time it. Update README's "Wall-clock budget" claim if it has shifted.
- [ ] **10.5** — Write a one-page `RELEASE_NOTES_v0.1.0.md` summarizing what's in the first publication-quality release. This becomes the body of the GitHub Release when v0.1.0 is tagged.
- [ ] **10.6** — Hand off to the user with a final checklist: PyPI publish (user action), Zenodo DOI reservation (user action), git tag `v0.1.0`, draft submission cover letter referencing CITATION.cff.

---

## Verification — How to Confirm a Tick is Done

Before committing, run the relevant subset:

```bash
# Always
uv run ruff check .
uv run ruff format --check .

# After test changes
uv run pytest -q

# After type-relevant changes
uv run mypy src

# After pipeline-touching changes
make ingest && make train && make eval   # ~25min on day1 config

# After ablation runs
# Confirm new entry exists in experiments/registry.yaml with metrics.
```

Reference baselines (must not regress):

| Quantity | Source | Tolerance |
|---|---|---|
| Binary F1 on paper test split | README + Table 1 (0.869) | within 0.02 of historical run |
| AUROC | Table 1 (0.970) | within 0.01 |
| Apo-PDB holdout F1 | Table 1 (0.726) | within 0.02 |
| AlphaFold holdout F1 | Table 1 (0.643) | within 0.05 (looser; ours is +8.2pp on best config) |

---

## Critical File Paths

| Purpose | Path |
|---|---|
| CLI entry | [src/slipp_plus/cli.py](src/slipp_plus/cli.py) |
| Schemas (pandera) | [src/slipp_plus/schemas.py](src/slipp_plus/schemas.py) |
| Training | [src/slipp_plus/train.py](src/slipp_plus/train.py) |
| Eval | [src/slipp_plus/eval.py](src/slipp_plus/eval.py) |
| Splits | [src/slipp_plus/splits.py](src/slipp_plus/splits.py) |
| Constants (per-class counts) | [src/slipp_plus/constants.py](src/slipp_plus/constants.py) |
| Tunnel features (CAVER) | [src/slipp_plus/tunnel_features.py](src/slipp_plus/tunnel_features.py) |
| Sterol features | [src/slipp_plus/sterol_features.py](src/slipp_plus/sterol_features.py) |
| PLM/STE tiebreaker | [src/slipp_plus/plm_ste_tiebreaker.py](src/slipp_plus/plm_ste_tiebreaker.py) |
| Hierarchical pipeline | [src/slipp_plus/hierarchical_pipeline.py](src/slipp_plus/hierarchical_pipeline.py) |
| Day-1 config | [configs/day1.yaml](configs/day1.yaml) |
| Best config | [configs/v_sterol.yaml](configs/v_sterol.yaml) |
| Experiment registry | [experiments/registry.yaml](experiments/registry.yaml) |
| Research log | [RESEARCH_LOG.md](RESEARCH_LOG.md) |
| Project context | [CONTEXT.md](CONTEXT.md) |
| Ground-truth Table 1 | [README.md](README.md) §"Ground truth" |
| Pyproject | [pyproject.toml](pyproject.toml) |
| Makefile | [Makefile](Makefile) |
| Tests | [tests/](tests/) |

---

## Done Criteria (Loop Termination)

The loop has finished when **all** of the following are true:

1. Every TODO above is checked off (or moved to `TODO.md` with explicit rationale for deferral).
2. `make test` passes locally and in CI.
3. `mypy src` passes (non-strict acceptable on a per-module basis with explicit overrides).
4. `make all` on `configs/v_sterol.yaml` reproduces the headline metrics in README within tolerance.
5. `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DATASHEET.md`, `docs/methods.md`, `docs/api.md`, and the expanded `README.md` exist and are non-stub.
6. `experiments/registry.yaml` and `reports/ablation_matrix.md` are in sync; ablations 8.3–8.6 have entries.
7. CI is green on the latest push.
8. A `RELEASE_NOTES_v0.1.0.md` is ready to ship.

When all eight conditions are met, the autoloop should write a final summary to the user (via the next tick's response) noting readiness for `git tag v0.1.0` and stop.

---

## What NOT to Do

- Do not refactor module boundaries or rename public functions without a clear publication-quality justification — downstream notebooks and the registry reference current paths.
- Do not introduce new feature sets or modeling approaches. The science is frozen at the current `v_sterol + tiebreaker` recipe; this loop is polish, not research.
- Do not bump dependency versions without a concrete reason (e.g. CVE, broken import). The `scikit-learn==1.3.1` pin is **paper-pinned** and must not change.
- Do not delete `RESEARCH_LOG.md`, `CONTEXT.md`, or any file in `reports/` — these are the audit trail.
- Do not commit anything in `processed/`, `models/`, or `data/structures/` — `.gitignore` covers them; verify before commit.
- Do not push to `main` directly. Work on the current branch; the user merges.