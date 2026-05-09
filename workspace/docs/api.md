# API Reference

This page documents the committed SLiPP++ command-line surface and the public
Python functions those commands call. It is curated from `src/slipp_plus/cli.py`
and the corresponding modules; older planning notes may mention commands or
options that are not part of the committed API.

## Command Line

All commands are available through the console script:

```bash
uv run slipp_plus --help
```

Global options:

| Option | Description |
|---|---|
| `--version` | Print the installed SLiPP++ version and exit. |
| `--help` | Show command help. |

### `slipp_plus ingest`

Signature:

```bash
uv run slipp_plus ingest --config configs/day1.yaml
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/day1.yaml` | YAML settings file. |

Returns:

Prints paths for `full_pockets.parquet`, `apo_pdb_holdout.parquet`,
`alphafold_holdout.parquet`, and `ingest_log.md`.

Raises:

`FileNotFoundError` for missing config or source files. `AssertionError` if the
Rule 1 row-count or per-class-count gate fails. `ValueError` for schema or
column drift.

### `slipp_plus train`

Signature:

```bash
uv run slipp_plus train --config configs/day1.yaml
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/day1.yaml` | YAML settings file. |

Returns:

Prints paths for prediction parquet output, split files, and iteration-0 model
bundles.

Raises:

`FileNotFoundError` for missing processed parquet input. Model-library errors
from scikit-learn, XGBoost, or LightGBM propagate when fitting fails.

### `slipp_plus eval`

Signature:

```bash
uv run slipp_plus eval --config configs/day1.yaml
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/day1.yaml` | YAML settings file. |

Returns:

Prints `metrics_table` and `raw_metrics` output paths. The markdown table
contains per-class metrics, binary-collapse metrics, holdout metrics, and 95%
confidence intervals.

Raises:

`FileNotFoundError` for missing prediction or holdout artifacts. `ValueError`
for malformed prediction frames or unsupported pipeline modes.

### `slipp_plus figures`

Signature:

```bash
uv run slipp_plus figures --config configs/day1.yaml
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/day1.yaml` | YAML settings file. |

Returns:

Prints paths for generated confusion, ROC, PCA, and paper-comparison figures.

Raises:

`FileNotFoundError` for missing raw metrics or predictions. Plotting errors from
Matplotlib or seaborn propagate.

### `slipp_plus all`

Signature:

```bash
uv run slipp_plus all --config configs/day1.yaml
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/day1.yaml` | YAML settings file. |

Returns:

Runs `ingest`, `train`, `eval`, and `figures` in sequence and prints progress.

Raises:

Any exception raised by the four underlying stages.

### `slipp_plus holdout-plm-ste`

Signature:

```bash
uv run slipp_plus holdout-plm-ste \
  --config configs/v_sterol.yaml \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --splits-dir processed/v_sterol/splits \
  --output reports/v_sterol/plm_ste_holdout.md
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/v_sterol.yaml` | YAML settings file. |
| `--full-pockets` | required | Training parquet used to fit the iteration-0 PLM/STE head. |
| `--splits-dir` | `processed/splits` | Directory containing `seed_*.parquet` split files. |
| `--output` | required | Markdown report path for holdout comparison. |
| `--predictions-dir` | none | Optional directory for holdout prediction parquets. |
| `--margin` | `0.99` | Top-2 PLM/STE margin threshold for firing the tiebreaker. |

Returns:

Prints generated holdout report and optional prediction paths.

Raises:

`FileNotFoundError` for missing training, split, model, or holdout artifacts.
`ValueError` for incompatible prediction or feature columns.

### `slipp_plus pair-tiebreaker-sweep`

Signature:

```bash
uv run slipp_plus pair-tiebreaker-sweep \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --predictions processed/v_sterol/predictions/test_predictions.parquet \
  --splits-dir processed/v_sterol/splits \
  --model-bundle models/v_sterol/xgb_multiclass.joblib \
  --output-report reports/pair_sweep.md \
  --output-metrics reports/pair_sweep.parquet \
  --negative-label PLM \
  --positive-label STE
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--full-pockets` | required | Training parquet for pairwise binary heads. |
| `--predictions` | required | Base multiclass prediction parquet. |
| `--splits-dir` | `processed/splits` | Split parquet directory. |
| `--model-bundle` | required | Matching model bundle used to infer feature columns. |
| `--output-report` | required | Markdown sweep report path. |
| `--output-metrics` | required | Parquet sweep metrics path. |
| `--output-predictions` | none | Optional selected augmented prediction parquet. |
| `--selected-margin` | best lipid macro-F1 | Margin to persist with `--output-predictions`. |
| `--negative-label` | required | Baseline class in the pair. |
| `--positive-label` | required | Focal class being rescued. |
| `--margins` | preset grid | Margin thresholds to sweep. |
| `--workers` | `8` | Maximum worker processes. |

Returns:

Prints report and metrics paths, plus selected margin and prediction path when
augmented predictions are requested.

Raises:

`ValueError` for unknown labels, invalid margins, or incompatible prediction
schema. File errors propagate for missing artifacts.

### `slipp_plus mine-confusions`

Signature:

```bash
uv run slipp_plus mine-confusions \
  --predictions processed/v_sterol/predictions/test_predictions.parquet \
  --output-report reports/v_sterol/confusions.md
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--predictions` | required | Prediction parquet to mine. |
| `--output-report` | required | Markdown report path. |
| `--output-table` | none | Optional parquet table path. |
| `--average-models` | `True` | Average RF/XGB/LGBM rows before mining. |
| `--lipid-only` | `True` | Restrict to lipid-vs-lipid confusions. |
| `--min-count` | `1` | Minimum off-diagonal count to report. |
| `--candidate-count` | `5` | Number of candidate boundary rules to emit. |
| `--min-top2-recoverable-fraction` | `0.0` | Minimum top-2 recoverable fraction. |
| `--candidate-margin` | `0.99` | Margin assigned to generated candidate rules. |

Returns:

Prints report path, optional table path, and number of candidate rules.

Raises:

`ValueError` for missing prediction columns.

### `slipp_plus calibration`

Signature:

```bash
uv run slipp_plus calibration --config configs/day1.yaml
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--config`, `-c` | `configs/day1.yaml` | YAML settings file. |

Returns:

Prints paths for calibration metrics and reliability plots.

Raises:

File errors for missing train/eval artifacts. `ValueError` for malformed
probability frames.

### `slipp_plus build-caver-t12`

Signature:

```bash
uv run slipp_plus build-caver-t12 \
  --base-parquet processed/v_sterol/full_pockets.parquet \
  --manifest reports/v_tunnel/analysis_manifest.csv \
  --output processed/v_caver_t12/full_pockets.parquet
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--base-parquet` | required | Base v-sterol parquet to enrich. |
| `--manifest` | required | Pocket-to-analysis manifest, CSV/TSV/parquet. |
| `--output` | required | Output parquet path. |
| `--reports-dir` | `reports/v_caver_t12` | Directory for build reports. |
| `--analysis-root` | none | Root for manifest `analysis_subdir` values. |
| `--holdout` | `False` | Interpret base parquet as a holdout keyed by `structure_id`. |

Returns:

Prints output parquet and report paths.

Raises:

`FileNotFoundError` for missing manifest or analysis directories. `ValueError`
for incompatible base parquet columns.

### `slipp_plus build-lipid-boundary`

Signature:

```bash
uv run slipp_plus build-lipid-boundary \
  --base-parquet processed/v_sterol/full_pockets.parquet \
  --output processed/v_lipid_boundary/full_pockets.parquet
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--base-parquet` | `processed/v_sterol/full_pockets.parquet` | Base v-sterol parquet. |
| `--source-pdbs-root` | `data/structures/source_pdbs` | Root containing protein PDB and fpocket outputs. |
| `--output` | `processed/v_lipid_boundary/full_pockets.parquet` | Output parquet path. |
| `--structural-join-parquet` | `processed/v49/full_pockets.parquet` | Fallback source for `matched_pocket_number`. |
| `--reports-dir` | `reports/v_lipid_boundary` | Directory for build reports. |
| `--workers` | `6` | Maximum worker processes. |
| `--skip-validation` | `False` | Skip full feature-set schema validation. |

Returns:

Prints output and report paths, excluding the verbose warning list.

Raises:

File errors for missing structure artifacts. `ValueError` for row drift,
missing columns, or validation failure.

### `slipp_plus ablate-v-sterol`

Alias: `slipp_plus build-v-sterol-ablation`.

Signature:

```bash
uv run slipp_plus ablate-v-sterol \
  --feature-set v_sterol+derived \
  --v-sterol-dir processed/v_sterol \
  --output-dir processed/v_sterol_derived
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--feature-set` | required | Ablation feature set to materialize. |
| `--v-sterol-dir` | `processed/v_sterol` | Directory containing base v-sterol parquets. |
| `--output-dir` | required | Destination directory for ablation parquets. |
| `--training-csv` | none | Training CSV used to recover raw vdw22 surfaces for training-only ablations. |

Returns:

Prints row counts, holdout support status, output paths, and added columns.

Raises:

`ValueError` for unknown or unsupported feature sets, missing vdw22 source
columns, schema drift, or NaN joins.

### `slipp_plus ste-rescue-sweep`

Signature:

```bash
uv run slipp_plus ste-rescue-sweep \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --predictions processed/v_sterol/predictions/test_predictions.parquet \
  --splits-dir processed/v_sterol/splits \
  --model-bundle models/v_sterol/xgb_multiclass.joblib \
  --output-report reports/ste_rescue.md \
  --output-metrics reports/ste_rescue.parquet
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--full-pockets` | required | Training parquet for STE rescue heads. |
| `--predictions` | required | Base multiclass prediction parquet. |
| `--splits-dir` | `processed/splits` | Split parquet directory. |
| `--model-bundle` | required | Matching model bundle used to infer feature columns. |
| `--output-report` | required | Markdown sweep report path. |
| `--output-metrics` | required | Parquet sweep metrics path. |
| `--output-predictions` | none | Optional selected augmented prediction parquet. |
| `--selected-threshold` | best lipid macro-F1 | Threshold to persist with `--output-predictions`. |
| `--thresholds` | `0.35,0.40,0.45,0.50,0.55` | STE binary probability thresholds. |
| `--workers` | `8` | Maximum worker processes. |

Returns:

Prints report, metrics, selected threshold, and optional predictions path.

Raises:

File errors for missing artifacts. `ValueError` for invalid labels, thresholds,
or prediction schema.

### `slipp_plus hierarchical-lipid`

Signature:

```bash
uv run slipp_plus hierarchical-lipid \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --predictions processed/v_sterol/predictions/test_predictions.parquet
```

Parameters:

| Parameter | Default | Description |
|---|---|---|
| `--full-pockets` | `processed/v_sterol/full_pockets.parquet` | Training parquet for staged hierarchy. |
| `--predictions` | `processed/v_sterol/predictions/test_predictions.parquet` | Base multiclass predictions. |
| `--splits-dir` | `processed/v_sterol/splits` | Split parquet directory. |
| `--model-bundle` | `models/v_sterol/xgb_multiclass.joblib` | Bundle used to infer feature columns. |
| `--output-report` | `reports/v_sterol/hierarchical_lipid_report.md` | Markdown report path. |
| `--output-metrics` | `reports/v_sterol/hierarchical_lipid_metrics.parquet` | Metrics parquet path. |
| `--output-predictions` | `processed/v_sterol/predictions/hierarchical_lipid_predictions.parquet` | Augmented prediction path. |
| `--ste-threshold` | `0.40` | STE specialist probability threshold. |
| `--stage1-source` | `ensemble` | Use ensemble lipid mass or a trained gate. |
| `--workers` | `8` | Maximum worker processes. |

Returns:

Prints report, metrics, fire count, and optional prediction path.

Raises:

File errors for missing artifacts. `ValueError` for incompatible feature or
prediction schema.

### `slipp_plus scratch`

Signature:

```bash
uv run slipp_plus scratch --config configs/day1.yaml
```

Status:

The Day 7+ raw-PDB path is intentionally not implemented for the Day 1 release.
It raises `NotImplementedError` through `download_all`.

## Python Functions

### Configuration

#### `load_settings(path: str | Path) -> Settings`

Load and validate a YAML config.

Parameters:

| Parameter | Description |
|---|---|
| `path` | YAML config path. |

Returns:

`Settings`, including `config_path` and `config_sha256`.

Raises:

`FileNotFoundError` for missing configs. `pydantic.ValidationError` for invalid
config fields.

Example:

```python
from pathlib import Path
from slipp_plus.config import load_settings

settings = load_settings(Path("configs/day1.yaml"))
```

### Ingest

#### `run_ingest(settings: Settings) -> dict[str, Path]`

Produce validated training and holdout parquets plus an ingest log.

Parameters:

| Parameter | Description |
|---|---|
| `settings` | Validated project settings. |

Returns:

Dictionary with `full_pockets`, `apo_pdb_holdout`, `alphafold_holdout`, and
`ingest_log` paths.

Raises:

`AssertionError` for Rule 1 count drift. `ValueError` for schema drift or
missing descriptor columns.

#### `assert_rule_1(full: pandas.DataFrame, settings: Settings) -> dict[str, int]`

Validate exact training row count and per-class counts.

Returns:

Per-class count dictionary ordered by config expectation.

Raises:

`AssertionError` when the total or any class count differs from the config.

### Training

#### `run_training(settings: Settings) -> dict[str, Path]`

Train the configured models over configured train/test splits.

Parameters:

| Parameter | Description |
|---|---|
| `settings` | Validated settings with feature set, models, paths, and seeds. |

Returns:

Paths for split parquet files, prediction parquet files, and iteration-0 model
bundles. Model metadata JSON sidecars are written next to joblib bundles.

Raises:

File errors for missing `full_pockets.parquet`. Model-library errors propagate
from the underlying estimators.

### Evaluation

#### `run_evaluation(settings: Settings) -> dict[str, Path]`

Evaluate test predictions and holdouts, then write markdown and parquet metrics.

Returns:

Dictionary with `metrics_table` and `raw_metrics` paths.

Raises:

File errors for missing predictions or holdout parquets. `ValueError` for
malformed prediction frames.

#### `evaluate_test_predictions(preds: pandas.DataFrame) -> pandas.DataFrame`

Compute per-iteration metrics from prediction rows.

Returns:

A long-form metrics frame containing 10-class and binary-collapse metrics.

#### `evaluate_holdout(bundle: dict, holdout: pandas.DataFrame, dataset: str) -> dict[str, float]`

Evaluate one trained bundle against one holdout frame.

Returns:

Dictionary containing binary F1, AUROC, and related holdout metrics.

### Figures

#### `run_figures(settings: Settings) -> dict[str, Path]`

Generate the standard figure set from predictions and raw metrics.

Returns:

Dictionary of figure names to output paths.

#### `plot_confusion_matrix(preds: pandas.DataFrame, out_path: Path) -> Path`

Write a confusion matrix figure and return its path.

#### `plot_per_class_roc(preds: pandas.DataFrame, out_path: Path) -> Path`

Write per-class ROC curves and return the output path.

#### `plot_pca_colored_by_pred(full: pandas.DataFrame, preds: pandas.DataFrame, features: list[str], out_path: Path) -> Path`

Write a PCA diagnostic figure colored by prediction correctness.

### Calibration

#### `run_calibration(settings: Settings) -> dict[str, Path]`

Train binary baselines, collect probability outputs, compute calibration
metrics, and write reliability figures.

Returns:

Dictionary of calibration metric and plot paths.

### Confusion Mining

#### `mine_confusion_edges(predictions: polars.DataFrame, *, average_models: bool = True, lipid_only: bool = False, min_count: int = 1) -> pandas.DataFrame`

Rank off-diagonal true-label to predicted-label errors.

Returns:

DataFrame with counts, support, probability margins, and top-2 recoverability.

#### `candidate_boundary_rules(confusion_edges: pandas.DataFrame, *, top_n: int = 5, min_count: int = 1, min_top2_recoverable_fraction: float = 0.0, margin: float = 0.99) -> list[BoundaryRule]`

Convert mined confusion rows into candidate single-negative boundary rules.

Returns:

List of `BoundaryRule` objects.

#### `run_confusion_mining(...) -> dict[str, Any]`

Load prediction parquet, mine confusions, optionally write a parquet table, and
write a markdown report.

Returns:

Dictionary containing `report`, optional `table`, mined `edges`, and candidate
`rules`.

### Feature Builders And Experiments

#### `build_v_sterol_ablation_from_v_sterol(v_sterol_dir: Path, output_dir: Path, *, feature_set: str, training_csv: Path | None = None) -> dict[str, object]`

Materialize supported v-sterol ablation parquet sets from existing v-sterol
artifacts.

Returns:

Dictionary containing row counts, output paths, added columns, and whether
holdout validation is supported.

#### `build_training_v_caver_t12_parquet(...) -> dict[str, object]`

Build persisted-output-first CAVER Tier 1-2 features for training parquets.

Returns:

Build summary with output and report paths.

#### `build_training_v_lipid_boundary_parquet(...) -> dict[str, object]`

Build lipid-boundary structural feature parquets for training.

Returns:

Build summary with output and report paths.

#### `run_pair_tiebreaker_experiment(...) -> dict[str, object]`

Sweep a pairwise tiebreaker over configured margins.

Returns:

Report path, metrics path, selected margin, and optional augmented prediction
path.

#### `run_ste_rescue_experiment(...) -> dict[str, object]`

Sweep grouped STE-vs-neighbor rescue thresholds.

Returns:

Report path, metrics path, selected threshold, and optional augmented prediction
path.

#### `run_hierarchical_experiment(...) -> dict[str, object]`

Run the staged lipid hierarchy over persisted flat-model predictions.

Returns:

Report path, metrics path, fire count, and optional augmented predictions.

### Scratch Download Stub

#### `download_all(output_dir: Path) -> None`

Placeholder for the Day 7+ raw-PDB reproduction path.

Raises:

Always raises `NotImplementedError` in the Day 1 release.
