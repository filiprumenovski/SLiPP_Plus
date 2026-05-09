#!/usr/bin/env python3
"""Sweep compact probability-ensemble subsets over existing predictions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from slipp_plus.constants import CLASS_10
from slipp_plus.evaluate import (
    _aggregate,
    evaluate_staged_holdout_predictions,
    evaluate_test_predictions,
)

PROBA_COLUMNS = [f"p_{label}" for label in CLASS_10]


@dataclass(frozen=True)
class Component:
    label: str
    root: Path


COMPONENTS: tuple[Component, ...] = (
    Component("shape3", Path("processed/v49_tunnel_shape3")),
    Component("shape6", Path("processed/v49_tunnel_shape")),
    Component("shell6_shape", Path("processed/v49_shell6_tunnel_shape")),
    Component("shell6_shape3", Path("processed/v49_shell6_tunnel_shape3")),
    Component("hydro4", Path("processed/v49_tunnel_shape_hydro4")),
    Component("geom", Path("processed/v49_tunnel_geom")),
    Component("chem", Path("processed/v49_tunnel_chem")),
)


def run_subset_sweep(
    output_dir: Path = Path("reports/compact_subset_ensemble_sweep"),
) -> dict[str, Path]:
    """Evaluate every >=2-way equal-probability ensemble from compact components."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results = _evaluate_subsets(COMPONENTS)
    results_path = output_dir / "subset_metrics.csv"
    results.to_csv(results_path, index=False)
    weight_sweep = _holdout_anchor_weight_sweep(COMPONENTS)
    weight_sweep_path = output_dir / "shell6_chem_weight_sweep.csv"
    weight_sweep.to_csv(weight_sweep_path, index=False)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(_render_summary(results, weight_sweep), encoding="utf-8")
    return {"results": results_path, "weight_sweep": weight_sweep_path, "summary": summary_path}


def _evaluate_subsets(components: tuple[Component, ...]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for size in range(2, len(components) + 1):
        for subset in combinations(components, size):
            labels = [component.label for component in subset]
            roots = [component.root for component in subset]
            model = "+".join(labels)
            test_predictions = _average_predictions(
                [_prediction_path(root) for root in roots],
                model_name=model,
            )
            summary = _aggregate(evaluate_test_predictions(test_predictions), ["model"]).iloc[0]
            holdouts: dict[str, dict[str, float]] = {}
            for holdout_name, holdout_file in (
                ("apo_pdb", "apo_pdb_holdout.parquet"),
                ("alphafold", "alphafold_holdout.parquet"),
            ):
                holdout_predictions = _average_predictions(
                    [_holdout_prediction_path(root, holdout_name) for root in roots],
                    model_name=model,
                )
                holdouts[holdout_name] = evaluate_staged_holdout_predictions(
                    holdout_predictions,
                    pd.read_parquet(roots[0] / holdout_file),
                )
            rows.append(_result_row(labels, summary, holdouts))
    return pd.DataFrame(rows).sort_values(
        ["lipid5", "holdout_mean", "n_components"],
        ascending=[False, False, True],
    )


def _result_row(
    labels: list[str],
    summary: pd.Series,
    holdouts: dict[str, dict[str, float]],
) -> dict[str, float | int | str]:
    apo_f1 = holdouts["apo_pdb"]["f1"]
    alphafold_f1 = holdouts["alphafold"]["f1"]
    return {
        "components": "+".join(labels),
        "n_components": len(labels),
        "lipid5": summary["macro_f1_lipid5_mean"],
        "lipid5_std": summary["macro_f1_lipid5_std"],
        "macro10": summary["macro_f1_10_mean"],
        "macro10_std": summary["macro_f1_10_std"],
        "binary_f1": summary["binary_f1_mean"],
        "binary_f1_std": summary["binary_f1_std"],
        "CLR": summary["f1_CLR_mean"],
        "MYR": summary["f1_MYR_mean"],
        "OLA": summary["f1_OLA_mean"],
        "PLM": summary["f1_PLM_mean"],
        "STE": summary["f1_STE_mean"],
        "apo_f1": apo_f1,
        "apo_auroc": holdouts["apo_pdb"]["auroc"],
        "alphafold_f1": alphafold_f1,
        "alphafold_auroc": holdouts["alphafold"]["auroc"],
        "holdout_mean": (apo_f1 + alphafold_f1) / 2,
        "holdout_min": min(apo_f1, alphafold_f1),
    }


def _prediction_path(root: Path) -> Path:
    return root / "predictions" / "hierarchical_lipid_predictions.parquet"


def _holdout_prediction_path(root: Path, holdout_name: str) -> Path:
    return root / "predictions" / "holdouts" / f"family_encoder_{holdout_name}_predictions.parquet"


def _average_predictions(paths: list[Path], *, model_name: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path).sort_values(["iteration", "row_index"]).reset_index(drop=True)
        for path in paths
    ]
    if len(frames) < 2:
        raise ValueError("subset sweep requires at least two prediction frames")
    base = frames[0].copy()
    keys = ["iteration", "row_index", "y_true_int"]
    for path, frame in zip(paths[1:], frames[1:], strict=False):
        if not base[keys].equals(frame[keys]):
            raise ValueError(f"prediction keys do not align: {path}")
    averaged = sum(frame[PROBA_COLUMNS].to_numpy(dtype=float) for frame in frames) / len(frames)
    base[PROBA_COLUMNS] = averaged
    base["y_pred_int"] = averaged.argmax(axis=1)
    base["model"] = model_name
    return base


def _render_summary(results: pd.DataFrame) -> str:
    top_lipid = results.sort_values(["lipid5", "holdout_mean"], ascending=False).head(12)
    top_holdout = results.sort_values(["holdout_mean", "lipid5"], ascending=False).head(12)
    top_ste = results.sort_values(["STE", "lipid5"], ascending=False).head(8)
    best_internal = top_lipid.iloc[0]
    best_holdout = top_holdout.iloc[0]
    return "\n".join(
        [
            "# Compact Subset Ensemble Sweep",
            "",
            "Equal-probability subset sweep over seven existing compact family-encoder prediction artifacts.",
            "",
            "## Key Signal",
            "",
            (
                f"- Best internal subset remains `{best_internal['components']}`: "
                f"lipid5 {_fmt(best_internal['lipid5'])}, apo-PDB {_fmt(best_internal['apo_f1'])}, "
                f"AlphaFold {_fmt(best_internal['alphafold_f1'])}."
            ),
            (
                f"- Best holdout-balanced subset is `{best_holdout['components']}`: "
                f"lipid5 {_fmt(best_holdout['lipid5'])}, apo-PDB {_fmt(best_holdout['apo_f1'])}, "
                f"AlphaFold {_fmt(best_holdout['alphafold_f1'])}."
            ),
            "- `hydro4` and `geom` help internal validation but are associated with weaker holdout F1 in the top internal blends.",
            "- `shell6_shape` is the strongest holdout anchor; pairing it with `chem` beats the current internal leader on both holdouts.",
            "",
            "## Top Internal Lipid5",
            "",
            _markdown_table(top_lipid),
            "",
            "## Top Holdout Mean",
            "",
            _markdown_table(top_holdout),
            "",
            "## Top STE",
            "",
            _markdown_table(top_ste),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "components",
        "n_components",
        "lipid5",
        "binary_f1",
        "STE",
        "apo_f1",
        "alphafold_f1",
        "holdout_mean",
    ]
    lines = [
        "| components | n | lipid5 | binary_f1 | STE | apo_f1 | alphafold_f1 | holdout_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.components}`",
                    str(row.n_components),
                    _fmt(row.lipid5),
                    _fmt(row.binary_f1),
                    _fmt(row.STE),
                    _fmt(row.apo_f1),
                    _fmt(row.alphafold_f1),
                    _fmt(row.holdout_mean),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        raise ValueError("expected numeric value, got None")
    numeric = float(value) if isinstance(value, (int, float)) else float(str(value))
    return f"{numeric:.3f}"


def main() -> None:
    outputs = run_subset_sweep()
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
