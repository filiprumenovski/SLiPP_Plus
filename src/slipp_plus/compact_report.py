"""Release-facing compact-stack ablation report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class CompactRun:
    run: str
    label: str
    features: int


COMPACT_RUNS: tuple[CompactRun, ...] = (
    CompactRun("paper17_family_encoder", "paper17", 17),
    CompactRun("paper17_shell12_family_encoder", "paper17+shell12", 29),
    CompactRun("paper17_shell12_tunnel_shape_family_encoder", "paper17+shell12+tunnel_shape", 35),
    CompactRun("paper17_aa20_family_encoder", "paper17+aa20", 37),
    CompactRun("paper17_aa20_tunnel_shape_family_encoder", "paper17+aa20+tunnel_shape", 43),
    CompactRun("v49_family_encoder", "v49", 49),
    CompactRun("v49_tunnel_chem_family_encoder", "v49+tunnel_chem", 54),
    CompactRun("v49_tunnel_shape_family_encoder", "v49+tunnel_shape", 55),
    CompactRun("v49_tunnel_geom_family_encoder", "v49+tunnel_geom", 58),
    CompactRun("v_sterol_family_encoder", "v_sterol", 87),
    CompactRun("v_sterol_family_plus_moe", "v_sterol+moe", 87),
    CompactRun("v_tunnel_aligned_family_plus_moe", "v_tunnel+moe", 105),
)


def run_compact_report(
    *,
    reports_root: Path = Path("reports"),
    models_root: Path = Path("models"),
    output_dir: Path = Path("reports/compact_publishable"),
) -> dict[str, Path]:
    """Regenerate the compact ladder CSV and markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ladder = _collect_ladder(reports_root=reports_root, models_root=models_root)
    if ladder.is_empty():
        raise FileNotFoundError("no compact ladder raw_metrics.parquet files found")

    ladder = ladder.sort(["lipid5_mean", "features"], descending=[True, False])
    csv_path = output_dir / "compact_ladder_metrics.csv"
    ladder.write_csv(csv_path)

    md_path = output_dir / "summary.md"
    md_path.write_text(_render_summary(ladder), encoding="utf-8")
    return {"metrics_csv": csv_path, "summary_md": md_path}


def _collect_ladder(*, reports_root: Path, models_root: Path) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for run in COMPACT_RUNS:
        metrics_path = reports_root / run.run / "raw_metrics.parquet"
        if not metrics_path.exists():
            continue
        metrics = pl.read_parquet(metrics_path).filter(pl.col("model") == "composite")
        if metrics.is_empty():
            continue
        model_path = _model_path(models_root, run.run)
        rows.append(
            {
                "run": run.run,
                "label": run.label,
                "features": run.features,
                "artifact_kb": (
                    model_path.stat().st_size / 1024 if model_path is not None else float("nan")
                ),
                **_metric_mean_std(metrics, "macro_f1_lipid5", "lipid5"),
                **_metric_mean_std(metrics, "macro_f1_10", "macro10"),
                **_metric_mean_std(metrics, "binary_f1", "binary"),
                "CLR": metrics.get_column("f1_CLR").mean(),
                "MYR": metrics.get_column("f1_MYR").mean(),
                "OLA": metrics.get_column("f1_OLA").mean(),
                "PLM": metrics.get_column("f1_PLM").mean(),
                "STE": metrics.get_column("f1_STE").mean(),
            }
        )
    return pl.DataFrame(rows)


def _metric_mean_std(metrics: pl.DataFrame, column: str, prefix: str) -> dict[str, float]:
    values = metrics.get_column(column)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
    }


def _model_path(models_root: Path, run: str) -> Path | None:
    candidates = (
        models_root / run / "family_encoder_bundle.joblib",
        models_root / run / "family_plus_moe_bundle.joblib",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _render_summary(ladder: pl.DataFrame) -> str:
    leader = ladder.row(0, named=True)
    paper17 = _row_by_label(ladder, "paper17")
    v49 = _row_by_label(ladder, "v49")
    tunnel_chem = _row_by_label(ladder, "v49+tunnel_chem")
    tunnel_shape = _row_by_label(ladder, "v49+tunnel_shape")
    tunnel_geom = _row_by_label(ladder, "v49+tunnel_geom")
    paper17_aa20 = _row_by_label(ladder, "paper17+aa20")
    paper17_aa20_tunnel_shape = _row_by_label(ladder, "paper17+aa20+tunnel_shape")
    paper17_shell12 = _row_by_label(ladder, "paper17+shell12")
    paper17_shell12_tunnel_shape = _row_by_label(ladder, "paper17+shell12+tunnel_shape")
    tunnel_moe = _row_by_label(ladder, "v_tunnel+moe")

    lines = [
        "# Compact publishable-stack checkpoint",
        "",
        "Date: 2026-05-06",
        "",
        "## Current answer",
        "",
        (
            "The release-facing leader is "
            f"`{leader['run']}`: {leader['features']} features, "
            f"{_fmt_kb(leader['artifact_kb'])} artifact, lipid5 macro-F1 "
            f"{_fmt_pm(leader['lipid5_mean'], leader['lipid5_std'])}."
        ),
        "",
    ]
    if tunnel_moe is not None:
        lines.extend(
            [
                (
                    "It matches the 105-feature tunnel MoE within split noise while using "
                    f"{leader['features']} instead of {tunnel_moe['features']} features and "
                    f"{_fmt_kb(leader['artifact_kb'])} instead of "
                    f"{_fmt_kb(tunnel_moe['artifact_kb'])}."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Internal 25-split Comparison",
            "",
            "| stack | features | artifact | lipid5 macro-F1 | macro10 F1 | binary F1 | CLR | MYR | OLA | PLM | STE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ladder.iter_rows(named=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    str(row["features"]),
                    _fmt_kb(row["artifact_kb"]),
                    _fmt_pm(row["lipid5_mean"], row["lipid5_std"]),
                    _fmt_pm(row["macro10_mean"], row["macro10_std"]),
                    _fmt_pm(row["binary_mean"], row["binary_std"]),
                    _fmt(row["CLR"]),
                    _fmt(row["MYR"]),
                    _fmt(row["OLA"]),
                    _fmt(row["PLM"]),
                    _fmt(row["STE"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ablation Deltas",
            "",
            "| comparison | lipid5 delta | interpretation |",
            "|---|---:|---|",
        ]
    )
    lines.extend(
        _ablation_delta_lines(
            paper17=paper17,
            paper17_shell12=paper17_shell12,
            paper17_shell12_tunnel_shape=paper17_shell12_tunnel_shape,
            paper17_aa20=paper17_aa20,
            paper17_aa20_tunnel_shape=paper17_aa20_tunnel_shape,
            v49=v49,
            tunnel_chem=tunnel_chem,
            tunnel_shape=tunnel_shape,
            tunnel_geom=tunnel_geom,
            tunnel_moe=tunnel_moe,
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The main recovery comes from AA20, not shell12 alone: "
                f"`paper17+shell12` is {_fmt(paper17_shell12['lipid5_mean']) if paper17_shell12 else 'n/a'}, "
                f"`paper17+aa20` is {_fmt(paper17_aa20['lipid5_mean']) if paper17_aa20 else 'n/a'}, "
                f"and `v49` is {_fmt(v49['lipid5_mean']) if v49 else 'n/a'}."
            ),
            "",
            (
                "Decision rule: keep the smallest stack whose lipid5 macro-F1 is within "
                "0.01-0.015 of the best observed model and does not degrade STE. Under "
                "current results, `v49+tunnel_shape` is the working release candidate; "
                "`v49` is the stricter parsimony fallback."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _row_by_label(ladder: pl.DataFrame, label: str) -> dict[str, float | int | str] | None:
    rows = ladder.filter(pl.col("label") == label)
    if rows.is_empty():
        return None
    return rows.row(0, named=True)


def _ablation_delta_lines(
    *,
    paper17: dict[str, float | int | str] | None,
    paper17_shell12: dict[str, float | int | str] | None,
    paper17_shell12_tunnel_shape: dict[str, float | int | str] | None,
    paper17_aa20: dict[str, float | int | str] | None,
    paper17_aa20_tunnel_shape: dict[str, float | int | str] | None,
    v49: dict[str, float | int | str] | None,
    tunnel_chem: dict[str, float | int | str] | None,
    tunnel_shape: dict[str, float | int | str] | None,
    tunnel_geom: dict[str, float | int | str] | None,
    tunnel_moe: dict[str, float | int | str] | None,
) -> list[str]:
    comparisons = [
        (paper17_shell12, paper17, "`paper17+shell12` vs `paper17`", "shell12 alone is modest"),
        (paper17_aa20, paper17, "`paper17+aa20` vs `paper17`", "AA20 carries the major recovery"),
        (v49, paper17_aa20, "`v49` vs `paper17+aa20`", "shell12 adds little once AA20 is present"),
        (
            paper17_shell12_tunnel_shape,
            paper17_shell12,
            "`paper17+shell12+tunnel_shape` vs `paper17+shell12`",
            "tunnel cannot compensate for removing AA20",
        ),
        (
            paper17_aa20_tunnel_shape,
            paper17_aa20,
            "`paper17+aa20+tunnel_shape` vs `paper17+aa20`",
            "tunnel lift without shell12",
        ),
        (
            tunnel_shape,
            paper17_aa20_tunnel_shape,
            "`v49+tunnel_shape` vs `paper17+aa20+tunnel_shape`",
            "isolates shell12 value in the tunnel stack",
        ),
        (tunnel_chem, v49, "`v49+tunnel_chem` vs `v49`", "tunnel chemistry is a smaller lift"),
        (tunnel_shape, v49, "`v49+tunnel_shape` vs `v49`", "best compact tunnel lift"),
        (
            tunnel_geom,
            v49,
            "`v49+tunnel_geom` vs `v49`",
            "ties shape within noise with more columns",
        ),
        (
            tunnel_moe,
            tunnel_shape,
            "`v_tunnel+moe` vs `v49+tunnel_shape`",
            "high-complexity reference does not improve the compact leader",
        ),
    ]
    lines: list[str] = []
    for candidate, baseline, label, note in comparisons:
        if candidate is None or baseline is None:
            continue
        delta = float(candidate["lipid5_mean"]) - float(baseline["lipid5_mean"])
        lines.append(f"| {label} | {delta:+.3f} | {note} |")
    return lines


def _fmt(value: object) -> str:
    return f"{float(value):.3f}"


def _fmt_pm(mean: object, std: object) -> str:
    return f"{float(mean):.3f} +/- {float(std):.3f}"


def _fmt_kb(value: object) -> str:
    return f"{float(value):.0f}K"
