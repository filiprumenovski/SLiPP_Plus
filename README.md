# SLiPP++

SLiPP++ is a research-grade, reproducible extension of the SLiPP lipid-binding
pocket benchmark from Chou et al. 2024. The original SLiPP task is binary
lipid-vs-rest classification; SLiPP++ keeps the paper's data contract and binary
collapse while reformulating the model as a 10-class softmax over:

`ADN`, `B12`, `BGC`, `CLR`, `COA`, `MYR`, `OLA`, `PLM`, `PP`, and `STE`.

The result is a model stack that can answer the original paper question and the
more useful follow-up: which lipid-like class does this pocket most resemble?

Primary reference: Chou et al., "Prediction of lipid binding sites on protein
structures using machine learning", DOI
[`10.1101/2024.01.26.577452`](https://doi.org/10.1101/2024.01.26.577452).
Upstream code and `training_pockets.csv` come from
[`dassamalab/SLiPP_2024`](https://github.com/dassamalab/SLiPP_2024).

## Current Release Candidate

`exp-018-compact-shape3-shape6-shell6-ensemble` is the current internal
registry leader. It averages probabilities from three complementary compact
family encoders, while preserving each component candidate:

```text
command:     uv run python scripts/compact_probability_ensemble.py
components:  exp-014 v49+tunnel_shape3 + exp-012 v49+tunnel_shape + exp-013 shell6+tunnel_shape
backbone:    three family encoders
report:      reports/compact_shape3_shape6_shell6_ensemble/metrics.md
```

The ensemble confirms that the compact tunnel-shape variants have complementary
errors. It is the internal validation leader, while external holdout performance
remains an explicit caveat.

| Metric | Mean +/- std |
|---|---:|
| Binary F1 | `0.903 +/- 0.017` |
| Binary AUROC | `0.989 +/- 0.003` |
| 10-class macro-F1 | `0.776 +/- 0.015` |
| 5-lipid macro-F1 | `0.678 +/- 0.028` |
| CLR F1 | `0.766` |
| MYR F1 | `0.705` |
| OLA F1 | `0.621` |
| PLM F1 | `0.650` |
| STE F1 | `0.649` |

External holdouts are mixed for this compact ensemble: apo-PDB F1 `0.712`
and AlphaFold F1 `0.671`. The best recorded individual holdout F1s are `0.746`
on apo-PDB from `exp-001-day1-v14` and `0.753` on AlphaFold from
`exp-002-v49-baseline`. Treat holdout performance as an explicit publication
caveat, not as hidden tuning debt.

## Why This Model

The compact ablation ladder shows a clean story:

| Stack | Columns | 5-lipid macro-F1 | Binary F1 |
|---|---:|---:|---:|
| `paper17` | 17 | `0.520 +/- 0.044` | `0.860 +/- 0.017` |
| `paper17+aa20` | 37 | `0.645 +/- 0.028` | `0.901 +/- 0.020` |
| `v49` | 49 | `0.649 +/- 0.026` | `0.898 +/- 0.016` |
| `v49+tunnel_shape3` | 52 | `0.668 +/- 0.031` | `0.900 +/- 0.015` |
| `v49+tunnel_shape` | 55 | `0.666 +/- 0.032` | `0.902 +/- 0.017` |
| `shape3+shape6+shell6 mean ensemble` | 49/52/55 | `0.678 +/- 0.028` | `0.903 +/- 0.017` |
| `shape3+shape6 mean ensemble` | 52/55 | `0.676 +/- 0.032` | `0.904 +/- 0.015` |
| `v_tunnel+moe` | 105 | `0.664 +/- 0.029` | `0.902 +/- 0.014` |

The main recovery comes from amino-acid composition, not raw feature
accumulation. Compact tunnel shape then adds the last useful lift over `v49`,
while the larger tunnel MoE does not improve the release candidate enough to
justify its complexity.

Full audit trail:

- [`experiments/registry.yaml`](experiments/registry.yaml): structured experiment index.
- [`reports/compact_publishable/summary.md`](reports/compact_publishable/summary.md): compact ladder and deltas.
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): decisions, failures, and abandoned ideas.
- [`CONTEXT.md`](CONTEXT.md): current operator state.

## Install

SLiPP++ uses Python `>=3.11,<3.13`, `uv`, and a locked dependency graph.
Scikit-learn is pinned to `1.3.1` to preserve the paper comparison.

```bash
git clone https://github.com/filiprumenovski/SLiPP-Plus.git
cd SLiPP-Plus
uv sync --frozen --extra dev
```

Use `uv sync --extra dev` only when intentionally refreshing the lockfile.

Optional scratch-path dependencies:

- `fpocket` is needed only for raw-PDB pocket extraction.
- Java plus CAVER `3.0.3` at `tools/caver/caver.jar` are needed only for CAVER
  tunnel feature generation.

The default paper reproduction and compact release-candidate paths use data
already present in this checkout.

## Data Contract

The core training table is the paper's curated 5-fold balanced set:

| File | Role |
|---|---|
| `reference/SLiPP_2024-main/training_pockets.csv` | 15,219 pockets with paper descriptors, labels, amino-acid counts, and free surface variants |
| `data/raw/supplementary/ci5c01076_si_003.xlsx` | apo-PDB holdout workbook |
| `data/raw/supplementary/ci5c01076_si_004.xlsx` | AlphaFold holdout workbook |

`make ingest` enforces the Rule 1 gate before training:

- exactly 15,219 training rows;
- exactly the expected per-class counts;
- no missing feature values;
- the paper-aligned 17 descriptor columns in canonical order for `v14`.

Current committed holdout ID lists are workbook-derived and contain 117 apo-PDB
IDs and 149 AlphaFold IDs. Older notes that mention 131/177 are stale for this
checkout. See [`DATASHEET.md`](DATASHEET.md) and
[`data/holdouts/README.md`](data/holdouts/README.md).

## Reproduce

Run the current internal compact leader:

```bash
uv run python scripts/compact_probability_ensemble.py
uv run python -m slipp_plus.cli compact-report
```

Run the paper-aligned Day 1 baseline:

```bash
make all CFG=configs/day1.yaml
```

Run the v-sterol hierarchy line:

```bash
make hierarchical-lipid HIER_CFG=configs/v_sterol_boundary_refactor.yaml
```

Outputs are written to runtime directories:

- `processed/`: validated parquets, splits, and prediction tables.
- `models/`: model bundles plus metadata and schema sidecars.
- `reports/`: metrics tables, raw metrics, figures, and generated summaries.

Some report and log artifacts are intentionally committed when they capture
scientific results. Do not delete negative results or older experiment outputs
just because a newer model is better.

## Make Targets

| Target | Description |
|---|---|
| `make ingest` | Parse CSV/XLSX inputs and run validation gates. |
| `make train` | Train 25 stratified 90/10 splits. |
| `make eval` | Report per-class, macro, binary-collapse, holdout, and CI metrics. |
| `make figures` | Render confusion, ROC, PCA, and comparison figures. |
| `make hierarchical-lipid` | Run the configured staged hierarchy. |
| `make build-caver-batch` | Build one batched CAVER tunnel feature parquet. |
| `make build-v-sterol-ablation` | Materialize a v-sterol ablation feature set. |
| `make test` | Run pytest, Ruff, and mypy. |
| `make test-slow` | Include slow Day 1 regression checks. |
| `make scratch` | Day 7+ raw-PDB path with fpocket. |

`make test` is intentionally broad. During active research, prefer focused
checks for the files being changed, then run the full gate before release.

## Feature Sets

The canonical paper feature order lives in
[`src/slipp_plus/constants.py`](src/slipp_plus/constants.py).

| Feature set | Summary |
|---|---|
| `v14` | Paper-aligned 17 fpocket descriptors. |
| `v14+aa` | Adds 20 amino-acid count features. |
| `v49` | Adds amino-acid counts plus compact residue-shell descriptors. |
| `v49+tunnel_shape` | Current compact release candidate: `v49` plus six tunnel-shape features. |
| `v_sterol` | Sterol-focused chemistry and shape refinements. |
| `v_tunnel` | High-complexity tunnel feature reference stack. |

Feature provenance and interpretation live under [`docs/features/`](docs/features/).

## CAVER Tunnel Features

CAVER is optional. For tunnel runs, install Java and place CAVER at
`tools/caver/caver.jar`, or pass `--caver-jar`.

```bash
uv run python -m slipp_plus.tunnel_features training \
  --base-parquet processed/v_sterol/full_pockets.parquet \
  --source-pdbs-root data/structures/source_pdbs \
  --caver-jar tools/caver/caver.jar \
  --output processed/v_tunnel/full_pockets.parquet \
  --reports-dir reports/v_tunnel
```

The builder fails fast on missing Java/CAVER, validates expected structure
artifacts, enforces coverage gates, and can persist raw CAVER analysis
directories. Details are in
[`docs/features/tunnel_features.md`](docs/features/tunnel_features.md).

## Reproducibility

- Use `uv sync --frozen --extra dev` for lockfile reproduction.
- Seeds are initialized from config `seed_base`.
- Splits are stratified on `class_10`, not binary labels.
- Metrics are reported as mean +/- std over 25 splits.
- Model metadata sidecars record package versions, Python version, config path,
  config SHA-256, git commit, timestamp, and seed.
- The ingest parquet path has a byte-determinism regression test.

See [`docs/reproducibility.md`](docs/reproducibility.md).

## Published Binary Ground Truth

These are the Chou et al. Table 1 binary targets used after SLiPP++ collapses
10-class predictions back to lipid-vs-rest.

| Dataset | F1 | AUROC |
|---|---:|---:|
| Test | `0.869` | `0.970` |
| Apo-PDB holdout | `0.726` | `0.828` |
| AlphaFold holdout | `0.643` | `0.851` |

If the paper-aligned `v14` binary-collapsed test F1 is not within roughly `0.03`
of the published test F1, debug ingestion, splitting, or label collapse before
interpreting multiclass experiments.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
make typecheck
```

`make typecheck` uses the current mypy allowlist (`MYPY_TARGETS`, initially
`src/slipp_plus/cli.py`). Expand that allowlist as modules are hardened.

Useful focused checks for the compact stack:

```bash
uv run pytest -q tests/test_constants.py tests/test_feature_families.py tests/test_pipeline_mode.py
uv run ruff check src/slipp_plus/backbone_family_encoder.py src/slipp_plus/composite_*.py src/slipp_plus/feature_families.py
```

## Documentation Map

- [`CONTEXT.md`](CONTEXT.md): current state, best config, and caveats.
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): decision history and negative results.
- [`experiments/registry.yaml`](experiments/registry.yaml): structured experiment registry.
- [`MODEL_V2_SPEC.md`](MODEL_V2_SPEC.md): composite/family encoder design notes.
- [`docs/api.md`](docs/api.md): CLI and public Python API reference.
- [`docs/methods.md`](docs/methods.md): concise methods appendix.
- [`DATASHEET.md`](DATASHEET.md): data provenance, checksums, counts, and known issues.
- [`reports/compact_publishable/summary.md`](reports/compact_publishable/summary.md): compact release-candidate ladder.
- [`examples/`](examples/): runnable reviewer examples.

Some planning notes are intentionally preserved for auditability. Prefer current
code, configs, registry entries, and generated reports over stale handoff text.

## Cite

If you use SLiPP++, cite both this software and the primary SLiPP scientific
reference. The machine-readable citation file is [`CITATION.cff`](CITATION.cff).

```bibtex
@software{rumenovski_slipp_plus_2026,
  author = {Rumenovski, Filip},
  title = {SLiPP++: 10-class lipid-binding pocket prediction},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/filiprumenovski/SLiPP-Plus}
}
```

## License

SLiPP++ is released under the MIT License. See [`LICENSE`](LICENSE).
