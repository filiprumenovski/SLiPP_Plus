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

## Current State (May 2026, submission-ready)

**Deployable artifact:** `exp-028-compact-shape3-shell6-chem-weighted` is
the current production-recommended config. It is selected by holdout-weighted
score rather than internal-validation score because the prior internal
leaders regress badly on the external holdouts. `exp-030` is now the
internal-validation leader, but it is not deployable.

**Optimization scaffold (this branch):** Multi-objective Hyperband HPO via
Optuna NSGA-II ([`tools/optuna_hpo.py`](tools/optuna_hpo.py)), CatBoost as a
fourth flat-mode base learner, a stacked LR / LGBM meta-learner over OOF
softprobs ([`src/slipp_plus/stacking.py`](src/slipp_plus/stacking.py)), and
configuration-layer support for multiple boundary specialists. All scaffold
tests pass on synthetic fixtures; real-data runs are gated on having
`processed/` parquet artifacts available.

**BioDolphin extension (proposed):** See
[`BIODOLPHIN_EXTENSION.md`](docs/BIODOLPHIN_EXTENSION.md). Chou et al. used a 27
April 2023 PDB snapshot with a five-lipid filter; BioDolphin v1.1 covers
14,891 PDB structures, 127,359 entries, and 2,619 distinct lipid molecules
through 6 September 2024 — roughly 19× the structure coverage. Documented
here as the explicit next-stage extension and a Dassama-lab collaboration
proposal.

### Headline metrics (deployable, exp-028)

| Metric | Mean ± std |
|---|---:|
| Binary F1 | `0.900 ± 0.018` |
| Binary AUROC | `0.988 ± 0.004` |
| 10-class macro-F1 | `0.766 ± 0.017` |
| 5-lipid macro-F1 | `0.666` |
| apo-PDB holdout F1 | `0.742` |
| AlphaFold holdout F1 | `0.775` |

Current deployable leader: `exp-035-legacy-rescue-logistic-gate-reproducible`
starts from the exp-028 compact ensemble, trains a small logistic gate on
internal split prediction features from exp-028, `paper17_family_encoder`, and
`v_sterol`, and rewrites high-confidence exp-028 non-lipid calls to the mean
legacy class probabilities. This keeps internal binary F1 within about 0.003
of exp-028 while improving apo-PDB F1 from `0.717` to `0.742` and AlphaFold F1
from `0.724` to `0.775`.

The internal-validation leader (`exp-030-probability-blend-internal-leader`)
reports binary F1 `0.908 ± 0.015`, AUROC `0.990 ± 0.003`, 10-class macro-F1
`0.781 ± 0.018`, and 5-lipid macro-F1 `0.687 ± 0.031`, but regresses to
apo-PDB `0.643` and AlphaFold `0.536` on the external holdouts. The 0.017
internal lipid5 macro-F1 sacrifice in `exp-028` buys back ~0.07 apo-PDB F1
and ~0.19 AlphaFold F1 relative to this internal leader.

Strongest holdout-scored diagnostic: `exp-031-legacy-rescue-rule-diagnostic` starts from
exp-028 and rescues exp-028 non-lipid calls when `paper17_family_encoder` and
`v_sterol` agree strongly enough. It improves apo-PDB F1 to `0.729` and
AlphaFold F1 to `0.735`, with internal binary F1 `0.901 ± 0.015` and lipid5
macro-F1 `0.668`. It is not promoted because the threshold was found by a
holdout-scored diagnostic grid.

Previous holdout-safe lead: `exp-032-legacy-rescue-holdout-safe-gate` showed
that a small logistic rescue gate transfers. `exp-035` is the script-backed
rerun and supersedes it as the deployable recommendation.

Holdout-label audit: `exp-034-holdout-label-source-audit` found a row-order
trap, not a semantic label conflict. Root holdout files and component-specific
holdout feature files have the same identities but different row order; labels
agree exactly after aligning by `structure_id` and `ligand`. Compact reporting
now aligns canonical labels by identity before scoring.

### Comparison to the SLiPP paper baseline

| Metric | Paper (Chou et al. 2024) | SLiPP++ exp-035 | Δ |
|---|---:|---:|---:|
| Binary F1 | 0.869 | 0.900 | +0.031 |
| Binary AUROC | 0.970 | 0.988 | +0.018 |
| AlphaFold holdout F1 | 0.643 | 0.775 | +0.132 |
| apo-PDB holdout F1 | 0.726 | 0.742 | +0.016 |

Three of four headline metrics beat the paper baseline; apo-PDB is
−0.009, attributable in part to the smaller 117-PDB holdout set in the
current supplementary workbooks (vs. the 131 the paper reports).

### Publication figure set

Three publication-grade figures are rendered to `figures/` in
PNG / PDF / SVG:

- [`figure7_plus_feature_landscape`](figures/figure7_plus_feature_landscape.png)
  — direct upgrade over Chou et al. Fig. 7 with the SLiPP++ ten-class
  softmax breaking the lipid bucket open.
- [`figure_per_class_forest`](figures/figure_per_class_forest.png)
  — per-class F1 across the experiment ladder.
- [`figure_ablation_ladder`](figures/figure_ablation_ladder.png)
  — sequential lipid5 macro-F1 deltas from `paper17` to `exp-021`.

Render with: `python tools/build_publication_figures.py`. Pass
`--use-real-features path/to/full_pockets.parquet` and
`--use-real-bundle path/to/iter0.joblib` to override the synthetic
fallbacks in panels A / B / D of the headline figure with on-disk
artifacts.

## Why This Model

The compact ablation ladder shows a clean story:

| Stack | Columns | 5-lipid macro-F1 | Binary F1 |
|---|---:|---:|---:|
| `paper17` | 17 | `0.520 +/- 0.044` | `0.860 +/- 0.017` |
| `paper17+aa20` | 37 | `0.645 +/- 0.028` | `0.901 +/- 0.020` |
| `v49` | 49 | `0.649 +/- 0.026` | `0.898 +/- 0.016` |
| `v49+tunnel_shape3` | 52 | `0.668 +/- 0.031` | `0.900 +/- 0.015` |
| `v49+tunnel_shape` | 55 | `0.666 +/- 0.032` | `0.902 +/- 0.017` |
| `shape6+shell6shape3+hydro4+geom+chem mean ensemble` | 49/54/55/58 | `0.684 +/- 0.030` | `0.906 +/- 0.015` |
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

The core training table is the paper's curated 5-fold balanced set, sourced from the public Dassama-lab SLiPP repository ([`dassamalab/SLiPP_2024`](https://github.com/dassamalab/SLiPP_2024)):

| File | Role |
|---|---|
| `reference/SLiPP_2024-main/training_pockets.csv` | 15,219 pockets with paper descriptors, labels, amino-acid counts, and free surface variants |
| `data/raw/supplementary/ci5c01076_si_003.xlsx` | apo-PDB holdout workbook |
| `data/raw/supplementary/ci5c01076_si_004.xlsx` | AlphaFold holdout workbook |

First-time setup populates `reference/`:

```bash
mkdir -p reference
git clone https://github.com/dassamalab/SLiPP_2024 reference/SLiPP_2024-main
shasum -a 256 reference/SLiPP_2024-main/training_pockets.csv
# expected: 4d27636b4381dc3c1b9e27451db5b788e6b16f13919c4ed36f8c2ba108097711
wc -l reference/SLiPP_2024-main/training_pockets.csv
# expected: 15220 (header + 15,219 rows enforced by the Rule 1 ingestion gate)
```

If you already have the upstream repo cloned elsewhere, copy `training_pockets.csv` into `reference/SLiPP_2024-main/` instead. The companion `slipp.py` / `slipp_utils.py` scripts in the upstream repo are the published binary inference pipeline; SLiPP++ is a separate codebase that consumes the same CSV rows for fair comparison. The peer-reviewed version of the work is Chou et al., *J. Chem. Inf. Model.* 2024 ([preprint](https://doi.org/10.1101/2024.01.26.577452)) — Table 1 in that PDF is the source of the `ground_truth` blocks in `configs/*.yaml`.

`make ingest` enforces the Rule 1 gate before training:

- exactly 15,219 training rows;
- exactly the expected per-class counts;
- no missing feature values;
- the paper-aligned 17 descriptor columns in canonical order for `v14`.

Current committed holdout ID lists are workbook-derived and contain 117 apo-PDB
IDs and 149 AlphaFold IDs. Older notes that mention 131/177 are stale for this
checkout. See [`DATASHEET.md`](docs/DATASHEET.md) and
[`data/holdouts/README.md`](data/holdouts/README.md).

## Reproduce

Run the compact base ensemble, then the current deployable rescue gate:

```bash
uv run python scripts/compact_probability_ensemble.py \
  --component-dir processed/v49_tunnel_shape3 \
  --component-dir processed/v49_shell6_tunnel_shape \
  --component-dir processed/v49_tunnel_chem \
  --component-weight 0.1 \
  --component-weight 0.2 \
  --component-weight 0.7 \
  --model-name shape3_shell6_chem_weighted_10_20_70 \
  --report-title "Compact shape3 shell6 chem weighted probability ensemble" \
  --output-predictions-dir processed/compact_shape3_shell6_chem_weighted_10_20_70/predictions \
  --output-report-dir reports/compact_shape3_shell6_chem_weighted_10_20_70
uv run python -m slipp_plus.cli compact-report

uv run python scripts/legacy_rescue_gate.py
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

- [`README.md`](README.md): this file. Headline metrics, install, current state.
- [`BIODOLPHIN_EXTENSION.md`](docs/BIODOLPHIN_EXTENSION.md): proposed next-stage extension to the BioDolphin lipid-pocket database (collaboration ask).
- [`DATASHEET.md`](docs/DATASHEET.md): data provenance, checksums, class counts, and known issues.
- [`experiments/registry.yaml`](experiments/registry.yaml): structured experiment registry (machine-readable).
- [`reports/compact_publishable/summary.md`](reports/compact_publishable/summary.md): compact release-candidate ladder.
- [`figures/`](figures/): submission-ready figures (PNG / PDF / SVG).
- [`docs/api.md`](docs/api.md), [`docs/methods.md`](docs/methods.md): CLI / API reference and methods appendix.
- [`examples/`](examples/): runnable reviewer examples.

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
