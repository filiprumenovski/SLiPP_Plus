"""SLiPP++ CLI: ingest | train | eval | figures | all | scratch."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import load_settings

app = typer.Typer(
    add_completion=False,
    help="SLiPP++ Day 1: 10-class softmax reformulation of Chou et al. 2024.",
)


_CONFIG_OPT = typer.Option("configs/day1.yaml", "--config", "-c",
                           help="Path to YAML configuration.")


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
def all(config: Path = _CONFIG_OPT) -> None:  # noqa: A001 - CLI verb
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
def scratch(config: Path = _CONFIG_OPT) -> None:
    """Day 7+ from-scratch reproduction (not implemented on Day 1)."""
    from .download import download_all

    settings = load_settings(config)
    download_all(settings.paths.processed_dir)


if __name__ == "__main__":
    app()
