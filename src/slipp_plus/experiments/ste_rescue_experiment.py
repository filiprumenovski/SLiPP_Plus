"""Grouped STE-vs-neighbor rescue head.

This experiment trains a binary XGB head for ``STE`` against the main residual
neighbor set ``{PLM, COA, OLA, MYR}``, then sweeps STE probability thresholds
on top of an existing multiclass prediction parquet.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier

from ..boundary_head import (
    NeighborRescueRule,
    apply_neighbor_rescue_head,
    build_boundary_training,
    gain_importance,
    train_boundary_head,
)
from ..constants import CLASS_10
from ..ensemble import PROBA_COLUMNS, average_softprobs, load_predictions, score_summary
from ..splits import load_split

STE_LABEL = "STE"
NEIGHBOR_LABELS: tuple[str, ...] = ("PLM", "COA", "OLA", "MYR")
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55)


def build_ste_rescue_training(
    full_pockets: pd.DataFrame,
    feature_columns: list[str],
    split_parquet: Path,
    neighbor_labels: tuple[str, ...] = NEIGHBOR_LABELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rule = NeighborRescueRule(
        name="ste_rescue",
        positive_label=STE_LABEL,
        neighbor_labels=neighbor_labels,
    )
    return build_boundary_training(
        full_pockets,
        feature_columns,
        split_parquet,
        rule.boundary_rule,
    )


def train_ste_rescue_head(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> XGBClassifier:
    return train_boundary_head(X_tr, y_tr, seed=seed)


def apply_ste_rescue(
    ensemble_df: pl.DataFrame,
    ste_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    *,
    threshold: float,
    neighbor_labels: tuple[str, ...] = NEIGHBOR_LABELS,
) -> pl.DataFrame:
    rule = NeighborRescueRule(
        name="ste_rescue",
        positive_label=STE_LABEL,
        neighbor_labels=neighbor_labels,
        threshold=threshold,
        top_k=4,
        fired_column="ste_rescue_fired",
        score_column="p_STE_rescue",
    )
    return apply_neighbor_rescue_head(ensemble_df, ste_proba, row_index_lookup, rule)


def _binary_f1(y_true: np.ndarray, p_positive: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p_positive >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _worker(
    *,
    iteration: int,
    split_path: Path,
    full_pockets_path: Path,
    feature_columns: list[str],
    seed: int,
    persist_importance: bool,
) -> dict[str, Any]:
    full = pd.read_parquet(full_pockets_path)
    X_tr, y_tr, X_te_neighbor, y_te_neighbor, _ = build_ste_rescue_training(
        full, feature_columns, split_path
    )
    model = train_ste_rescue_head(X_tr, y_tr, seed=seed)

    _, test_idx = load_split(split_path)
    X_all = full[feature_columns].to_numpy(dtype=np.float64)
    p_ste = model.predict_proba(X_all[test_idx])[:, 1]
    p_neighbor = model.predict_proba(X_te_neighbor)[:, 1]

    importance: dict[str, float] | None = None
    if persist_importance:
        importance = gain_importance(model, feature_columns)

    return {
        "iteration": iteration,
        "row_index_lookup": test_idx.astype(np.int64),
        "ste_proba": p_ste,
        "neighbor_binary_f1": _binary_f1(y_te_neighbor, p_neighbor),
        "scale_pos_weight": float(
            ((y_tr == 0).sum() / (y_tr == 1).sum()) if (y_tr == 1).sum() else 1.0
        ),
        "feature_importance": importance,
    }


def _ste_confusion(df: pl.DataFrame) -> dict[str, int]:
    ste = CLASS_10.index(STE_LABEL)
    y_true = df["y_true_int"].to_numpy()
    y_pred = df["y_pred_int"].to_numpy()
    out = {
        "STE_correct": int(((y_true == ste) & (y_pred == ste)).sum()),
        "STE_support": int((y_true == ste).sum()),
    }
    for label in NEIGHBOR_LABELS:
        idx = CLASS_10.index(label)
        out[f"STE_as_{label}"] = int(((y_true == ste) & (y_pred == idx)).sum())
        out[f"{label}_as_STE"] = int(((y_true == idx) & (y_pred == ste)).sum())
    return out


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def _write_report(
    output_report: Path,
    *,
    base_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    neighbor_binary_f1_mean: float,
    neighbor_binary_f1_std: float,
    scale_pos_weight_mean: float,
    feature_importance: dict[str, float],
) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        handle.write("# Grouped STE rescue sweep\n\n")
        handle.write(
            "_Binary XGB head trained on STE vs {PLM, COA, OLA, MYR}; applied when "
            "top-1 is a neighbor and STE is in the top-4 multiclass probabilities._\n\n"
        )
        handle.write(
            "Baseline ensemble: "
            f"10-class macro-F1 = {_fmt(base_summary['macro_f1_mean'], base_summary['macro_f1_std'])}, "
            f"5-lipid macro-F1 = {_fmt(base_summary['lipid_macro_f1_mean'], base_summary['lipid_macro_f1_std'])}, "
            f"STE F1 = {_fmt(*base_summary['per_class_f1']['STE'])}.\n\n"
        )
        handle.write(
            "STE-vs-neighbors binary F1 = "
            f"{_fmt(neighbor_binary_f1_mean, neighbor_binary_f1_std)}; "
            f"mean scale_pos_weight = {scale_pos_weight_mean:.3f}.\n\n"
        )
        handle.write(
            "| threshold | 10-class macro-F1 | 5-lipid macro-F1 | STE F1 | PLM F1 | fired mean | fired total | STE correct | STE->PLM | STE->COA | STE->OLA | STE->MYR |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['threshold']:.2f} "
                f"| {_fmt(row['macro_f1_mean'], row['macro_f1_std'])} "
                f"| {_fmt(row['lipid_macro_f1_mean'], row['lipid_macro_f1_std'])} "
                f"| {_fmt(row['STE_f1_mean'], row['STE_f1_std'])} "
                f"| {_fmt(row['PLM_f1_mean'], row['PLM_f1_std'])} "
                f"| {row['fire_mean']:.1f} | {row['fire_total']} "
                f"| {row['STE_correct']} | {row['STE_as_PLM']} | {row['STE_as_COA']} "
                f"| {row['STE_as_OLA']} | {row['STE_as_MYR']} |\n"
            )
        if feature_importance:
            top = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:15]
            handle.write("\n## Iteration-0 top-15 features (gain)\n\n")
            handle.write("| rank | feature | gain |\n|---:|---|---:|\n")
            for rank, (name, gain) in enumerate(top, start=1):
                handle.write(f"| {rank} | {name} | {gain:.4f} |\n")


def run_ste_rescue_experiment(
    *,
    full_pockets_path: Path,
    predictions_path: Path,
    splits_dir: Path,
    model_bundle_path: Path,
    output_report: Path,
    output_metrics: Path,
    thresholds: list[float],
    output_predictions: Path | None = None,
    selected_threshold: float | None = None,
    workers: int = 8,
    seed_base: int = 42,
) -> dict[str, Any]:
    bundle = joblib.load(model_bundle_path)
    feature_columns = list(bundle["feature_columns"])
    ensemble_df = average_softprobs(load_predictions(predictions_path))
    base_summary = score_summary(ensemble_df)

    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if not split_files:
        raise FileNotFoundError(f"no split files found in {splits_dir}")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _worker,
                iteration=i,
                split_path=split_path,
                full_pockets_path=full_pockets_path,
                feature_columns=feature_columns,
                seed=seed_base + i,
                persist_importance=(i == 0),
            )
            for i, split_path in enumerate(split_files)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["iteration"])

    rows: list[dict[str, Any]] = []
    augmented_by_threshold: dict[float, pl.DataFrame] = {}
    for threshold in thresholds:
        frames: list[pl.DataFrame] = []
        fires: list[int] = []
        for result in results:
            sub = ensemble_df.filter(pl.col("iteration") == result["iteration"])
            augmented = apply_ste_rescue(
                sub,
                result["ste_proba"],
                result["row_index_lookup"],
                threshold=threshold,
            )
            frames.append(augmented)
            fires.append(int(augmented["ste_rescue_fired"].sum()))
        augmented_all = pl.concat(frames).sort(["iteration", "row_index"])
        augmented_by_threshold[float(threshold)] = augmented_all
        scoring_frame = augmented_all.select(
            ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
        )
        summary = score_summary(scoring_frame)
        confusion = _ste_confusion(scoring_frame)
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1_mean": summary["macro_f1_mean"],
                "macro_f1_std": summary["macro_f1_std"],
                "lipid_macro_f1_mean": summary["lipid_macro_f1_mean"],
                "lipid_macro_f1_std": summary["lipid_macro_f1_std"],
                "STE_f1_mean": summary["per_class_f1"]["STE"][0],
                "STE_f1_std": summary["per_class_f1"]["STE"][1],
                "PLM_f1_mean": summary["per_class_f1"]["PLM"][0],
                "PLM_f1_std": summary["per_class_f1"]["PLM"][1],
                "fire_mean": float(np.mean(fires)),
                "fire_total": int(np.sum(fires)),
                **confusion,
            }
        )

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_metrics, index=False)
    selected = selected_threshold
    if selected is None and rows:
        selected = max(
            rows,
            key=lambda r: (
                float(r["lipid_macro_f1_mean"]),
                float(r["STE_f1_mean"]),
                -abs(float(r["threshold"]) - 0.40),
            ),
        )["threshold"]
    if output_predictions is not None and selected is not None:
        chosen = augmented_by_threshold[float(selected)]
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        chosen.write_parquet(output_predictions)
    neighbor_f1s = [r["neighbor_binary_f1"] for r in results]
    weights = [r["scale_pos_weight"] for r in results]
    feature_importance = next(
        (r["feature_importance"] for r in results if r["feature_importance"]), {}
    )
    _write_report(
        output_report,
        base_summary=base_summary,
        rows=rows,
        neighbor_binary_f1_mean=float(np.mean(neighbor_f1s)),
        neighbor_binary_f1_std=float(np.std(neighbor_f1s)),
        scale_pos_weight_mean=float(np.mean(weights)),
        feature_importance=feature_importance,
    )
    return {
        "report": output_report,
        "metrics": output_metrics,
        "predictions": output_predictions,
        "selected_threshold": selected,
        "rows": rows,
        "base_summary": base_summary,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped STE rescue threshold sweep.")
    parser.add_argument("--full-pockets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--selected-threshold", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_ste_rescue_experiment(
        full_pockets_path=args.full_pockets,
        predictions_path=args.predictions,
        splits_dir=args.splits_dir,
        model_bundle_path=args.model_bundle,
        output_report=args.output_report,
        output_metrics=args.output_metrics,
        output_predictions=args.output_predictions,
        selected_threshold=args.selected_threshold,
        thresholds=list(args.thresholds),
        workers=args.workers,
        seed_base=args.seed_base,
    )
    print(f"wrote {result['report']}")
    print(f"wrote {result['metrics']}")
    if result["predictions"] is not None:
        print(f"wrote {result['predictions']}")


if __name__ == "__main__":
    main()
