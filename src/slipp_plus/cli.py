"""SLiPP++ CLI: ingest | train | eval | figures | all | scratch."""

from __future__ import annotations

from pathlib import Path

import typer

from .__version__ import __version__
from .config import load_settings
from .logging_config import setup_logging

app = typer.Typer(
    add_completion=False,
    help="SLiPP++ Day 1: 10-class softmax reformulation of Chou et al. 2024.",
)


_CONFIG_OPT = typer.Option("configs/day1.yaml", "--config", "-c",
                           help="Path to YAML configuration.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the SLiPP++ version and exit.",
    ),
) -> None:
    """Run SLiPP++ commands."""
    setup_logging()


def __main__() -> None:
    app()


@app.command()
def ingest(config: Path = _CONFIG_OPT) -> None:
    """CSV + xlsx -> parquet, with Rule 1 gate."""
    from .ingest import run_ingest

    settings = load_settings(config)
    out = run_ingest(settings)
    typer.echo("ingest OK:")
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")


@app.command()
def train(config: Path = _CONFIG_OPT) -> None:
    """25 stratified shuffle iterations x 3 models."""
    from .train import run_training

    settings = load_settings(config)
    out = run_training(settings)
    typer.echo("train OK:")
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")


@app.command("eval")
def evaluate_cmd(config: Path = _CONFIG_OPT) -> None:
    """Metrics table + holdouts."""
    from .evaluate import run_evaluation

    settings = load_settings(config)
    out = run_evaluation(settings)
    typer.echo("eval OK:")
    typer.echo(f"  metrics_table: {out['metrics_table']}")
    typer.echo(f"  raw_metrics: {out['raw_metrics']}")


@app.command("holdout-plm-ste")
def holdout_plm_ste_cmd(
    config: Path = typer.Option("configs/v_sterol.yaml", "--config", "-c", help="Path to YAML configuration."),
    full_pockets: Path = typer.Option(..., help="Training parquet used to fit the iteration-0 PLM/STE head."),
    splits_dir: Path = typer.Option(Path("processed/splits"), help="Directory containing seed_*.parquet split files."),
    output: Path = typer.Option(..., help="Markdown report path for holdout comparison."),
    predictions_dir: Path | None = typer.Option(None, help="Optional directory for holdout prediction parquets."),
    margin: float = typer.Option(0.99, help="Top-2 PLM/STE margin threshold for firing the tiebreaker."),
) -> None:
    """Validate the v_sterol ensemble + PLM/STE tiebreaker on apo/AlphaFold holdouts."""
    from .plm_ste_holdout import run_holdout_validation

    settings = load_settings(config)
    out = run_holdout_validation(
        settings,
        full_pockets_path=full_pockets,
        splits_dir=splits_dir,
        output_path=output,
        predictions_dir=predictions_dir,
        margin=margin,
    )
    typer.echo("holdout-plm-ste OK:")
    for k, v in out["outputs"].items():
        typer.echo(f"  {k}: {v}")


@app.command("pair-tiebreaker-sweep")
def pair_tiebreaker_sweep_cmd(
    full_pockets: Path = typer.Option(..., help="Training parquet used to fit the pairwise binary heads."),
    predictions: Path = typer.Option(..., help="Base multiclass prediction parquet across iterations."),
    splits_dir: Path = typer.Option(Path("processed/splits"), help="Directory containing seed_*.parquet split files."),
    model_bundle: Path = typer.Option(..., help="Any matching multiclass bundle used to infer feature columns."),
    output_report: Path = typer.Option(..., help="Markdown report path for the sweep."),
    output_metrics: Path = typer.Option(..., help="Parquet path for the sweep summary table."),
    output_predictions: Path | None = typer.Option(
        None,
        help="Optional parquet path for selected augmented predictions.",
    ),
    selected_margin: float | None = typer.Option(
        None,
        help="Margin to persist when output_predictions is set; defaults to best lipid macro-F1.",
    ),
    negative_label: str = typer.Option(..., help="The abundant or baseline class in the pair."),
    positive_label: str = typer.Option(..., help="The focal class whose F1 is being tested."),
    margins: list[float] = typer.Option(
        [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 0.99],
        help="Margin thresholds to sweep.",
    ),
    workers: int = typer.Option(8, help="Maximum worker processes."),
) -> None:
    """Run a standalone pairwise tiebreaker margin sweep."""
    from .pair_tiebreaker_experiment import run_pair_tiebreaker_experiment

    result = run_pair_tiebreaker_experiment(
        full_pockets_path=full_pockets,
        predictions_path=predictions,
        splits_dir=splits_dir,
        model_bundle_path=model_bundle,
        output_report=output_report,
        output_metrics=output_metrics,
        output_predictions=output_predictions,
        selected_margin=selected_margin,
        negative_label=negative_label,
        positive_label=positive_label,
        margins=[float(v) for v in margins],
        workers=workers,
    )
    typer.echo("pair-tiebreaker-sweep OK:")
    typer.echo(f"  report: {result['report']}")
    typer.echo(f"  metrics: {result['metrics']}")
    if result["predictions"] is not None:
        typer.echo(f"  selected_margin: {result['selected_margin']}")
        typer.echo(f"  predictions: {result['predictions']}")


@app.command("mine-confusions")
def mine_confusions_cmd(
    predictions: Path = typer.Option(..., help="Prediction parquet to mine."),
    output_report: Path = typer.Option(..., help="Markdown report path."),
    output_table: Path | None = typer.Option(None, help="Optional parquet table path."),
    average_models: bool = typer.Option(
        True,
        help="Average RF/XGB/LGBM rows before mining when a model column is present.",
    ),
    lipid_only: bool = typer.Option(True, help="Only mine lipid-vs-lipid confusions."),
    min_count: int = typer.Option(1, help="Minimum off-diagonal count to report."),
    candidate_count: int = typer.Option(5, help="Number of candidate boundary rules to emit."),
    min_top2_recoverable_fraction: float = typer.Option(
        0.0,
        help="Minimum fraction where true/pred labels both appear in top-2.",
    ),
    candidate_margin: float = typer.Option(
        0.99,
        help="Margin assigned to generated candidate boundary rules.",
    ),
) -> None:
    """Mine recurring confusions into candidate boundary-head rules."""
    from .confusion_mining import run_confusion_mining

    result = run_confusion_mining(
        predictions_path=predictions,
        output_report=output_report,
        output_table=output_table,
        average_models=average_models,
        lipid_only=lipid_only,
        min_count=min_count,
        candidate_count=candidate_count,
        min_top2_recoverable_fraction=min_top2_recoverable_fraction,
        candidate_margin=candidate_margin,
    )
    typer.echo("mine-confusions OK:")
    typer.echo(f"  report: {result['report']}")
    if result["table"] is not None:
        typer.echo(f"  table: {result['table']}")
    typer.echo(f"  candidate_rules: {len(result['rules'])}")


@app.command()
def figures(config: Path = _CONFIG_OPT) -> None:
    """Confusion / ROC / PCA / comparison bars."""
    from .figures import run_figures

    settings = load_settings(config)
    out = run_figures(settings)
    typer.echo("figures OK:")
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")


@app.command()
def all(config: Path = _CONFIG_OPT) -> None:
    """ingest -> train -> eval -> figures."""
    settings = load_settings(config)
    from .evaluate import run_evaluation
    from .figures import run_figures
    from .ingest import run_ingest
    from .train import run_training

    typer.echo("== ingest ==")
    run_ingest(settings)
    typer.echo("== train ==")
    run_training(settings)
    typer.echo("== eval ==")
    run_evaluation(settings)
    typer.echo("== figures ==")
    run_figures(settings)
    typer.echo("== done ==")


@app.command()
def calibration(config: Path = _CONFIG_OPT) -> None:
    """Binary baselines + ECE/Brier/MCE + reliability figures."""
    from .calibration import run_calibration

    settings = load_settings(config)
    out = run_calibration(settings)
    typer.echo("calibration OK:")
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")


@app.command("build-caver-t12")
def build_caver_t12(
    base_parquet: Path = typer.Option(..., help="Base v_sterol parquet to enrich."),
    manifest: Path = typer.Option(..., help="Pocket-to-analysis manifest (csv/tsv/parquet)."),
    output: Path = typer.Option(..., help="Output parquet path for v_caver_t12."),
    reports_dir: Path = typer.Option(Path("reports/v_caver_t12"), help="Directory for build reports."),
    analysis_root: Path | None = typer.Option(None, help="Optional root for manifest analysis_subdir values."),
    holdout: bool = typer.Option(False, help="Interpret the base parquet as a holdout parquet keyed by structure_id."),
) -> None:
    """Build the persisted-output-first CAVER Tier 1-2 feature parquet."""
    from .caver_t12_features import (
        build_holdout_v_caver_t12_parquet,
        build_training_v_caver_t12_parquet,
    )

    if holdout:
        out = build_holdout_v_caver_t12_parquet(
            base_parquet=base_parquet,
            manifest_path=manifest,
            output_path=output,
            reports_dir=reports_dir,
            analysis_root=analysis_root,
        )
    else:
        out = build_training_v_caver_t12_parquet(
            base_parquet=base_parquet,
            manifest_path=manifest,
            output_path=output,
            reports_dir=reports_dir,
            analysis_root=analysis_root,
        )
    typer.echo("build-caver-t12 OK:")
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")


@app.command("build-lipid-boundary")
def build_lipid_boundary(
    base_parquet: Path = typer.Option(
        Path("processed/v_sterol/full_pockets.parquet"),
        help="Base v_sterol parquet to enrich.",
    ),
    source_pdbs_root: Path = typer.Option(
        Path("data/structures/source_pdbs"),
        help="Root containing <CLASS>/<stem>.pdb and <stem>_out/ fpocket outputs.",
    ),
    output: Path = typer.Option(
        Path("processed/v_lipid_boundary/full_pockets.parquet"),
        help="Output parquet path for v_lipid_boundary.",
    ),
    structural_join_parquet: Path = typer.Option(
        Path("processed/v49/full_pockets.parquet"),
        help="Fallback structural join parquet carrying matched_pocket_number.",
    ),
    reports_dir: Path = typer.Option(
        Path("reports/v_lipid_boundary"),
        help="Directory for build reports.",
    ),
    workers: int = typer.Option(6, help="Maximum worker processes."),
    skip_validation: bool = typer.Option(False, help="Skip full feature-set schema validation."),
) -> None:
    """Build the v_lipid_boundary training feature parquet."""
    from .lipid_boundary_features import build_training_v_lipid_boundary_parquet

    out = build_training_v_lipid_boundary_parquet(
        base_parquet=base_parquet,
        source_pdbs_root=source_pdbs_root,
        output_path=output,
        structural_join_parquet=structural_join_parquet,
        reports_dir=reports_dir,
        workers=workers,
        validate_output=not skip_validation,
    )
    typer.echo("build-lipid-boundary OK:")
    for k, v in out.items():
        if k != "warnings_list":
            typer.echo(f"  {k}: {v}")


@app.command("build-v-sterol-ablation")
def build_v_sterol_ablation_cmd(
    feature_set: str = typer.Option(..., help="Ablation feature_set to materialize."),
    v_sterol_dir: Path = typer.Option(
        Path("processed/v_sterol"),
        help="Directory containing the base v_sterol full/holdout parquets.",
    ),
    output_dir: Path = typer.Option(..., help="Destination directory for the ablation parquets."),
    training_csv: Path | None = typer.Option(
        None,
        help="Training CSV used to recover raw vdw22 surfaces for test-only ablations.",
    ),
) -> None:
    """Build an explicit v_sterol feature-family ablation parquet set."""
    from .v_sterol_ablation import build_v_sterol_ablation_from_v_sterol

    out = build_v_sterol_ablation_from_v_sterol(
        v_sterol_dir=v_sterol_dir,
        output_dir=output_dir,
        feature_set=feature_set,
        training_csv=training_csv,
    )
    typer.echo("build-v-sterol-ablation OK:")
    for k, v in out.items():
        typer.echo(f"  {k}: {v}")


@app.command("ste-rescue-sweep")
def ste_rescue_sweep_cmd(
    full_pockets: Path = typer.Option(..., help="Training parquet used to fit STE rescue heads."),
    predictions: Path = typer.Option(..., help="Base multiclass prediction parquet across iterations."),
    splits_dir: Path = typer.Option(Path("processed/splits"), help="Directory containing seed_*.parquet split files."),
    model_bundle: Path = typer.Option(..., help="Any matching multiclass bundle used to infer feature columns."),
    output_report: Path = typer.Option(..., help="Markdown report path for the sweep."),
    output_metrics: Path = typer.Option(..., help="Parquet path for the sweep summary table."),
    output_predictions: Path | None = typer.Option(
        None,
        help="Optional parquet path for the selected augmented predictions.",
    ),
    selected_threshold: float | None = typer.Option(
        None,
        help="Threshold to persist when output_predictions is set; defaults to best lipid macro-F1.",
    ),
    thresholds: list[float] = typer.Option(
        [0.35, 0.40, 0.45, 0.50, 0.55],
        help="STE binary probability thresholds to sweep.",
    ),
    workers: int = typer.Option(8, help="Maximum worker processes."),
) -> None:
    """Run the grouped STE-vs-neighbors rescue threshold sweep."""
    from .ste_rescue_experiment import run_ste_rescue_experiment

    result = run_ste_rescue_experiment(
        full_pockets_path=full_pockets,
        predictions_path=predictions,
        splits_dir=splits_dir,
        model_bundle_path=model_bundle,
        output_report=output_report,
        output_metrics=output_metrics,
        output_predictions=output_predictions,
        selected_threshold=selected_threshold,
        thresholds=[float(v) for v in thresholds],
        workers=workers,
    )
    typer.echo("ste-rescue-sweep OK:")
    typer.echo(f"  report: {result['report']}")
    typer.echo(f"  metrics: {result['metrics']}")
    typer.echo(f"  selected_threshold: {result['selected_threshold']}")
    if result["predictions"] is not None:
        typer.echo(f"  predictions: {result['predictions']}")


@app.command("hierarchical-lipid")
def hierarchical_lipid_cmd(
    full_pockets: Path = typer.Option(
        Path("processed/v_sterol/full_pockets.parquet"),
        help="Training parquet used to fit staged lipid hierarchy.",
    ),
    predictions: Path = typer.Option(
        Path("processed/v_sterol/predictions/test_predictions.parquet"),
        help="Base multiclass prediction parquet across iterations.",
    ),
    splits_dir: Path = typer.Option(
        Path("processed/v_sterol/splits"),
        help="Directory containing seed_*.parquet split files.",
    ),
    model_bundle: Path = typer.Option(
        Path("models/v_sterol/xgb_multiclass.joblib"),
        help="Any matching multiclass bundle used to infer feature columns.",
    ),
    output_report: Path = typer.Option(
        Path("reports/v_sterol/hierarchical_lipid_report.md"),
        help="Markdown report path for the staged hierarchy.",
    ),
    output_metrics: Path = typer.Option(
        Path("reports/v_sterol/hierarchical_lipid_metrics.parquet"),
        help="Parquet path for the staged hierarchy summary table.",
    ),
    output_predictions: Path | None = typer.Option(
        Path("processed/v_sterol/predictions/hierarchical_lipid_predictions.parquet"),
        help="Optional parquet path for augmented staged predictions.",
    ),
    ste_threshold: float = typer.Option(0.40, help="STE specialist probability threshold."),
    stage1_source: str = typer.Option(
        "ensemble",
        help="Use 'ensemble' lipid mass for binary parity, or train a separate 'trained' gate.",
    ),
    workers: int = typer.Option(8, help="Maximum worker processes."),
) -> None:
    """Run Stage 1 lipid gate, Stage 2 lipid-family head, and STE specialist."""
    from .hierarchical_experiment import (
        DEFAULT_STE_NEIGHBORS,
        OneVsNeighborsRule,
        run_hierarchical_experiment,
    )

    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=DEFAULT_STE_NEIGHBORS,
        top_k=4,
        min_positive_proba=ste_threshold,
    )
    result = run_hierarchical_experiment(
        full_pockets_path=full_pockets,
        predictions_path=predictions,
        splits_dir=splits_dir,
        model_bundle_path=model_bundle,
        output_report=output_report,
        output_metrics=output_metrics,
        output_predictions=output_predictions,
        specialist_rule=rule,
        stage1_source=stage1_source,
        workers=workers,
    )
    typer.echo("hierarchical-lipid OK:")
    typer.echo(f"  report: {result['report']}")
    typer.echo(f"  metrics: {result['metrics']}")
    typer.echo(f"  fire_total: {result['fire_total']}")
    if result["predictions"] is not None:
        typer.echo(f"  predictions: {result['predictions']}")


@app.command()
def scratch(config: Path = _CONFIG_OPT) -> None:
    """Day 7+ from-scratch reproduction (not implemented on Day 1)."""
    from .download import download_all

    settings = load_settings(config)
    download_all(settings.paths.processed_dir)


if __name__ == "__main__":
    app()
