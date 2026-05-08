# SLiPP++

SLiPP++ is a publication-oriented reimplementation of the SLiPP lipid-binding
pocket benchmark from Chou et al. 2024, reformulated from binary
lipid-vs-rest classification into a 10-class softmax task over:

`ADN`, `B12`, `BGC`, `CLR`, `COA`, `MYR`, `OLA`, `PLM`, `PP`, and `STE`.

The project keeps the paper-aligned descriptors, train/test protocol, and
binary-collapse definitions so every multiclass experiment can still be compared
against the original Table 1 binary metrics.

Primary reference: Chou et al., "Prediction of lipid binding sites on protein
structures using machine learning", DOI
[`10.1101/2024.01.26.577452`](https://doi.org/10.1101/2024.01.26.577452).
Upstream code and `training_pockets.csv` are from
[`dassamalab/SLiPP_2024`](https://github.com/dassamalab/SLiPP_2024).

## Current Best

The current registry leader is `exp-012-compact-tunnel-shape`
(`configs/v49_tunnel_shape_family_encoder.yaml`):

| Metric | Mean +/- std |
|---|---:|
| Binary F1 | `0.902 +/- 0.017` |
| Binary AUROC | `0.988 +/- 0.003` |
| 10-class macro-F1 | `0.766 +/- 0.019` |
| Lipid macro-F1 | `0.666 +/- 0.032` |
| STE F1 | `0.647` |

External holdout F1 is not maximized by the same run. Best recorded individual
holdout F1s are `0.746` on apo-PDB from `exp-001-day1-v14` and `0.753` on
AlphaFold from `exp-002-v49-baseline`. See
[`reports/ablation_matrix.md`](reports/ablation_matrix.md) and
[`experiments/registry.yaml`](experiments/registry.yaml) for the audit trail.

## Install

SLiPP++ uses Python `>=3.11,<3.13`, `uv`, and a locked dependency graph. The
paper-aligned scikit-learn version is pinned to `1.3.1`; do not upgrade it when
reproducing paper comparisons.

```bash
git clone https://github.com/filiprumenovski/SLiPP_Plus.git
cd SLiPP_Plus

# Exact local reproduction from uv.lock.
uv sync --frozen --extra dev

# Editable development environment if the lock needs to be refreshed.
uv sync --extra dev
```

Optional scratch-path dependencies:

- `fpocket` is required only for `make scratch` and raw-PDB pocket extraction.
- Java plus CAVER `3.0.3` at `tools/caver/caver.jar` are required only for
  CAVER tunnel feature generation.

The default Day 1 path does not run fpocket or CAVER.

## Data

The main reproducible path uses files already present in this checkout:

| File | Role |
|---|---|
| `reference/SLiPP_2024-main/training_pockets.csv` | 15,219-row training table with paper descriptors and class labels |
| `data/raw/supplementary/ci5c01076_si_003.xlsx` | apo-PDB holdout workbook |
| `data/raw/supplementary/ci5c01076_si_004.xlsx` | AlphaFold holdout workbook |

`make ingest` enforces the Rule 1 validation gate: 15,219 training rows, exactly
17 canonical numeric descriptor columns for the paper-aligned feature set, no
NaN feature values, and exact per-class counts. Current committed holdout ID
lists are file-derived and contain 117 apo-PDB IDs and 149 AlphaFold IDs; older
notes that mention 131/177 are stale for this checkout. See
[`DATASHEET.md`](DATASHEET.md) and [`data/holdouts/README.md`](data/holdouts/README.md).

## Quickstart

Run the paper-aligned Day 1 pipeline:

```bash
make ingest
make train
make eval
make figures
```

Or run the same sequence as one command:

```bash
make all
```

Outputs are written to gitignored runtime directories:

- `processed/`: validated parquet tables and split files.
- `models/`: iteration-0 model bundles plus metadata sidecars.
- `reports/`: metrics tables, raw metrics, plots, and run summaries.

Wall-clock budget for the Day 1 path is under 25 minutes on a 16-core machine.

## Recommended Runs

The stable baseline is `configs/day1.yaml`:

```bash
uv run slipp_plus ingest --config configs/day1.yaml
uv run slipp_plus train --config configs/day1.yaml
uv run slipp_plus eval --config configs/day1.yaml
```

The recommended v-sterol ensemble path is:

```bash
uv run slipp_plus all --config configs/v_sterol.yaml
uv run slipp_plus holdout-plm-ste \
  --config configs/v_sterol.yaml \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --splits-dir processed/v_sterol/splits \
  --output reports/v_sterol/plm_ste_holdout.md \
  --predictions-dir processed/v_sterol/predictions
```

The staged lipid hierarchy helper is:

```bash
make hierarchical-lipid
```

The compact current-best research line is recorded in the registry as
`exp-012-compact-tunnel-shape`.

For a small reviewer-friendly script:

```bash
uv run python examples/quickstart.py
```

## Make Targets

| Target | Description |
|---|---|
| `make ingest` | Parse training CSV and holdout workbooks, then run schema and count gates. |
| `make train` | Train 25 stratified 90/10 splits across RF, XGB, and LGBM. |
| `make eval` | Report per-class, macro, binary-collapse, holdout, and CI metrics. |
| `make figures` | Render confusion, ROC, PCA, and comparison figures. |
| `make hierarchical-lipid` | Run the configured staged hierarchy. |
| `make build-v-sterol-ablation` | Materialize a v-sterol ablation feature set. |
| `make test` | Run pytest, Ruff, Ruff format check, and mypy. |
| `make test-slow` | Include the slow Day 1 regression gate. |
| `make scratch` | Day 7+ raw-PDB path: download structures, run fpocket, then ingest. |

`make test` currently includes repo-wide style/type gates. If local research
changes are in progress, run targeted checks before committing a focused patch.

## Feature Sets

The canonical paper feature set is the 17-column `SELECTED_17` order in
`src/slipp_plus/constants.py`.

| Feature set | Summary |
|---|---|
| `v14` | Paper-aligned 17 descriptors. |
| `v14+v22` | Adds free `_vdw22` surface variants present in the authors' CSV. |
| `v14+aa` | Adds 20 amino-acid count features. |
| `v49` | Adds residue-shell and aromatic/aliphatic descriptors. |
| `v_sterol` | Adds sterol-focused chemistry and shape refinements. |
| `v_tunnel` / tunnel-shape configs | Adds CAVER-derived or compact tunnel-shape features. |

Feature provenance and interpretation live under [`docs/features/`](docs/features/).

## CAVER Tunnel Features

CAVER is optional and is not needed for the baseline reproduction. For tunnel
runs, install Java and place CAVER at `tools/caver/caver.jar`, or pass
`--caver-jar`.

```bash
uv run python -m slipp_plus.tunnel_features training \
  --base-parquet processed/v_sterol/full_pockets.parquet \
  --source-pdbs-root data/structures/source_pdbs \
  --caver-jar tools/caver/caver.jar \
  --output processed/v_tunnel/full_pockets.parquet \
  --reports-dir reports/v_tunnel
```

The builder fails fast on missing Java/CAVER, validates expected structure
artifacts, enforces coverage gates for `tunnel_pocket_context_present` and
`tunnel_caver_profile_present`, and can persist raw CAVER analysis directories
with `--analysis-output-root` plus `--analysis-manifest`. Details are in
[`docs/features/tunnel_features.md`](docs/features/tunnel_features.md).

## Reproducibility

- Use `uv sync --frozen --extra dev` for exact lockfile reproduction.
- Seeds are initialized from config `seed_base`.
- Training writes model metadata sidecars with package versions, Python version,
  config path, config SHA-256, git commit, timestamp, and seed.
- The ingest parquet path has a byte-determinism regression test.

See [`docs/reproducibility.md`](docs/reproducibility.md).

## Published Binary Ground Truth

These are the Chou et al. Table 1 binary targets used for comparison after
SLiPP++ collapses 10-class predictions back to lipid-vs-rest.

| Dataset | F1 | AUROC |
|---|---:|---:|
| Test | `0.869` | `0.970` |
| Apo-PDB holdout | `0.726` | `0.828` |
| AlphaFold holdout | `0.643` | `0.851` |

`reports/metrics_table.md` reports SLiPP++ metrics side-by-side with these
targets after `make eval`.

## Documentation

- [`CONTEXT.md`](CONTEXT.md): current project state and caveats.
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): decision history and negative results.
- [`docs/api.md`](docs/api.md): command-line and public Python API reference.
- [`docs/methods.md`](docs/methods.md): concise methods appendix.
- [`DATASHEET.md`](DATASHEET.md): data provenance, checksums, counts, and known issues.
- [`reports/ablation_matrix.md`](reports/ablation_matrix.md): experiment summary table.
- [`examples/`](examples/): runnable examples for reviewers.

Some older planning notes are intentionally preserved for auditability. Prefer
current code, configs, registry entries, and generated reports over stale handoff
text.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Use `uv run ruff format .` before broad style-only commits. Keep experiment
artifacts, reports, and negative results; they are part of the scientific audit
trail.

## Cite

If you use SLiPP++, cite both this software and the primary SLiPP scientific
reference. The machine-readable citation file is [`CITATION.cff`](CITATION.cff).

```bibtex
@software{rumenovski_slipp_plus_2026,
  author = {Rumenovski, Filip},
  title = {SLiPP++: 10-class lipid-binding pocket prediction},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/filiprumenovski/SLiPP_Plus}
}
```

## License

SLiPP++ is released under the MIT License. See [`LICENSE`](LICENSE).
