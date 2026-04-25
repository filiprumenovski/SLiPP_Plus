"""Mine recurrent class confusions into candidate boundary heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .boundary_head import BoundaryRule
from .constants import CLASS_10, LIPID_CODES
from .ensemble import PROBA_COLUMNS, average_softprobs, load_predictions

LIPID_SET = frozenset(LIPID_CODES)


def _prediction_frame(predictions: pl.DataFrame, *, average_models: bool = True) -> pl.DataFrame:
    if average_models and "model" in predictions.columns:
        return average_softprobs(predictions)
    needed = {"iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS}
    missing = sorted(needed - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame missing required columns: {missing}")
    return predictions.select(["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS])


def mine_confusion_edges(
    predictions: pl.DataFrame,
    *,
    average_models: bool = True,
    lipid_only: bool = False,
    min_count: int = 1,
) -> pd.DataFrame:
    """Rank off-diagonal class pairs by recurring misclassification burden.

    Rows are oriented as ``true_label -> pred_label``. This orientation maps
    directly to a boundary head where ``true_label`` is the positive class to
    rescue and ``pred_label`` is the local negative class.
    """

    frame = _prediction_frame(predictions, average_models=average_models)
    proba = frame.select(PROBA_COLUMNS).to_numpy().astype(np.float64)
    y_true = frame["y_true_int"].to_numpy().astype(np.int64)
    y_pred = frame["y_pred_int"].to_numpy().astype(np.int64)

    order = np.argsort(-proba, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    margins = proba[np.arange(len(proba)), top1] - proba[np.arange(len(proba)), top2]

    supports = {i: int((y_true == i).sum()) for i in range(len(CLASS_10))}
    rows: list[dict[str, Any]] = []
    for true_idx in range(len(CLASS_10)):
        for pred_idx in range(len(CLASS_10)):
            if true_idx == pred_idx:
                continue
            true_label = CLASS_10[true_idx]
            pred_label = CLASS_10[pred_idx]
            if lipid_only and (
                true_label not in LIPID_SET or pred_label not in LIPID_SET
            ):
                continue
            mask = (y_true == true_idx) & (y_pred == pred_idx)
            count = int(mask.sum())
            if count < min_count:
                continue
            true_scores = proba[mask, true_idx]
            pred_scores = proba[mask, pred_idx]
            recoverable = mask & (
                ((top1 == true_idx) | (top2 == true_idx))
                & ((top1 == pred_idx) | (top2 == pred_idx))
            )
            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                    "true_support": supports[true_idx],
                    "error_fraction_of_true": count / supports[true_idx]
                    if supports[true_idx]
                    else 0.0,
                    "mean_margin": float(np.mean(margins[mask])) if count else 0.0,
                    "median_margin": float(np.median(margins[mask])) if count else 0.0,
                    "mean_true_probability": float(np.mean(true_scores)) if count else 0.0,
                    "mean_pred_probability": float(np.mean(pred_scores)) if count else 0.0,
                    "top2_recoverable_count": int(recoverable.sum()),
                    "top2_recoverable_fraction": int(recoverable.sum()) / count
                    if count
                    else 0.0,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "true_label",
                "pred_label",
                "count",
                "true_support",
                "error_fraction_of_true",
                "mean_margin",
                "median_margin",
                "mean_true_probability",
                "mean_pred_probability",
                "top2_recoverable_count",
                "top2_recoverable_fraction",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["count", "error_fraction_of_true", "top2_recoverable_count"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def candidate_boundary_rules(
    confusion_edges: pd.DataFrame,
    *,
    top_n: int = 5,
    min_count: int = 1,
    min_top2_recoverable_fraction: float = 0.0,
    margin: float = 0.99,
) -> list[BoundaryRule]:
    """Convert mined confusion rows into single-negative boundary rules."""

    rules: list[BoundaryRule] = []
    for row in confusion_edges.itertuples(index=False):
        if int(row.count) < min_count:
            continue
        if float(row.top2_recoverable_fraction) < min_top2_recoverable_fraction:
            continue
        positive = str(row.true_label)
        negative = str(row.pred_label)
        rules.append(
            BoundaryRule(
                name=f"{positive.lower()}_vs_{negative.lower()}",
                positive_label=positive,
                negative_labels=(negative,),
                margin=margin,
                max_rank=2,
            )
        )
        if len(rules) >= top_n:
            break
    return rules


def write_confusion_mining_report(
    output_path: Path,
    confusion_edges: pd.DataFrame,
    *,
    rules: list[BoundaryRule] | None = None,
    title: str = "Confusion mining report",
    top_n: int = 20,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shown = confusion_edges.head(top_n)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(
            "| rank | true | predicted | count | true support | error share | "
            "top-2 recoverable | mean margin | mean p(true) | mean p(pred) |\n"
        )
        handle.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(shown.itertuples(index=False), start=1):
            handle.write(
                f"| {rank} | {row.true_label} | {row.pred_label} "
                f"| {int(row.count)} | {int(row.true_support)} "
                f"| {float(row.error_fraction_of_true):.3f} "
                f"| {float(row.top2_recoverable_fraction):.3f} "
                f"| {float(row.mean_margin):.3f} "
                f"| {float(row.mean_true_probability):.3f} "
                f"| {float(row.mean_pred_probability):.3f} |\n"
            )
        if rules:
            handle.write("\n## Candidate Boundary Rules\n\n")
            handle.write("| rank | name | positive | negative | margin | max rank |\n")
            handle.write("|---:|---|---|---|---:|---:|\n")
            for rank, rule in enumerate(rules, start=1):
                handle.write(
                    f"| {rank} | {rule.name} | {rule.positive_label} "
                    f"| {', '.join(rule.negative_labels)} | {rule.margin:.2f} "
                    f"| {rule.max_rank or len(rule.candidate_labels)} |\n"
                )
    return output_path


def run_confusion_mining(
    *,
    predictions_path: Path,
    output_report: Path,
    output_table: Path | None = None,
    average_models: bool = True,
    lipid_only: bool = True,
    min_count: int = 1,
    candidate_count: int = 5,
    min_top2_recoverable_fraction: float = 0.0,
    candidate_margin: float = 0.99,
) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    edges = mine_confusion_edges(
        predictions,
        average_models=average_models,
        lipid_only=lipid_only,
        min_count=min_count,
    )
    rules = candidate_boundary_rules(
        edges,
        top_n=candidate_count,
        min_count=min_count,
        min_top2_recoverable_fraction=min_top2_recoverable_fraction,
        margin=candidate_margin,
    )
    if output_table is not None:
        output_table.parent.mkdir(parents=True, exist_ok=True)
        edges.to_parquet(output_table, index=False)
    report = write_confusion_mining_report(output_report, edges, rules=rules)
    return {"report": report, "table": output_table, "edges": edges, "rules": rules}
