"""Standalone pairwise tiebreaker margin-sweep experiments.

This module generalizes the PLM/STE tiebreaker pattern into an independent
experiment surface. It trains a binary XGB head on one confusion pair per
iteration, scores the full test fold, and sweeps a set of margin thresholds
to measure whether routing the pairwise arbiter improves the focal class F1.

The output is deliberately exploratory: markdown + parquet summaries only.
It does not modify the production pipeline or any persisted ensemble outputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier

from .boundary_head import (
    BoundaryRule,
    apply_boundary_head,
    build_boundary_training,
    gain_importance,
    train_boundary_head,
)
from .constants import CLASS_10
from .ensemble import PROBA_COLUMNS, average_softprobs, load_predictions, score_summary
from .splits import load_split

DEFAULT_MARGINS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 0.99)


def _pair_rule(negative_label: str, positive_label: str, margin: float) -> BoundaryRule:
    return BoundaryRule(
        name=f"{positive_label.lower()}_vs_{negative_label.lower()}",
        positive_label=positive_label,
        negative_labels=(negative_label,),
        margin=margin,
        max_rank=2,
        fired_column="tiebreaker_fired",
        score_column=f"p_{positive_label}_binary",
    )


def _load_prediction_substrate(predictions_path: Path) -> pl.DataFrame:
    """Load raw long-format predictions or an already-averaged/staged parquet."""

    df = load_predictions(predictions_path)
    if "model" in df.columns:
        return average_softprobs(df)
    needed = {"iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"prediction parquet missing required columns: {missing}")
    return df.select(["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"])


def build_pair_training(
    full_pockets: pd.DataFrame,
    feature_columns: list[str],
    split_parquet: Path,
    negative_label: str,
    positive_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return build_boundary_training(
        full_pockets,
        feature_columns,
        split_parquet,
        _pair_rule(negative_label, positive_label, margin=DEFAULT_MARGINS[0]),
    )


def train_pair_tiebreaker(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> XGBClassifier:
    return train_boundary_head(X_tr, y_tr, seed=seed)


def apply_pair_tiebreaker(
    ensemble_df: pl.DataFrame,
    positive_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    *,
    negative_label: str,
    positive_label: str,
    margin: float,
) -> pl.DataFrame:
    return apply_boundary_head(
        ensemble_df,
        positive_proba,
        row_index_lookup,
        _pair_rule(negative_label, positive_label, margin=margin),
    )


def _pair_only_binary_f1(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    pred = (p_positive >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _pair_confusion(df: pl.DataFrame, negative_label: str, positive_label: str) -> dict[str, int]:
    neg_idx = CLASS_10.index(negative_label)
    pos_idx = CLASS_10.index(positive_label)
    y_true = df["y_true_int"].to_numpy()
    y_pred = df["y_pred_int"].to_numpy()
    return {
        f"{negative_label}_correct": int(((y_true == neg_idx) & (y_pred == neg_idx)).sum()),
        f"{positive_label}_correct": int(((y_true == pos_idx) & (y_pred == pos_idx)).sum()),
        f"{negative_label}_as_{positive_label}": int(((y_true == neg_idx) & (y_pred == pos_idx)).sum()),
        f"{positive_label}_as_{negative_label}": int(((y_true == pos_idx) & (y_pred == neg_idx)).sum()),
        f"{negative_label}_support": int((y_true == neg_idx).sum()),
        f"{positive_label}_support": int((y_true == pos_idx).sum()),
    }


def _worker(
    *,
    iteration: int,
    split_path: Path,
    full_pockets_path: Path,
    feature_columns: list[str],
    negative_label: str,
    positive_label: str,
    seed: int,
    persist_importance: bool,
) -> dict[str, Any]:
    full = pd.read_parquet(full_pockets_path)
    X_tr, y_tr, X_te_pair, y_te_pair, _ = build_pair_training(
        full,
        feature_columns,
        split_path,
        negative_label,
        positive_label,
    )
    model = train_pair_tiebreaker(X_tr, y_tr, seed=seed)

    _, test_idx = load_split(split_path)
    X_all = full[feature_columns].to_numpy(dtype=np.float64)
    X_te_full = X_all[test_idx]
    positive_proba = model.predict_proba(X_te_full)[:, 1]

    importance: dict[str, float] | None = None
    if persist_importance:
        importance = gain_importance(model, feature_columns)

    return {
        "iteration": iteration,
        "row_index_lookup": test_idx.astype(np.int64),
        "positive_proba": positive_proba,
        "pair_binary_f1": float(_pair_only_binary_f1(y_te_pair, model.predict_proba(X_te_pair)[:, 1])),
        "scale_pos_weight": float(((y_tr == 0).sum() / (y_tr == 1).sum()) if (y_tr == 1).sum() else 1.0),
        "feature_importance": importance,
    }


def _monotonic_non_decreasing(values: list[float], atol: float = 1e-9) -> bool:
    return all(b + atol >= a for a, b in pairwise(values))


def write_pair_experiment_report(
    output_path: Path,
    *,
    negative_label: str,
    positive_label: str,
    rows: list[dict[str, Any]],
    base_summary: dict[str, Any],
    pair_binary_f1_mean: float,
    pair_binary_f1_std: float,
    scale_pos_weight_mean: float,
    feature_importance: dict[str, float],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positive_f1_values = [float(r[f"{positive_label}_f1_mean"]) for r in rows]
    monotonic = _monotonic_non_decreasing(positive_f1_values)

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f} ± {std:.3f}"

    with output_path.open("w") as f:
        f.write(f"# Pair tiebreaker sweep — {positive_label} vs {negative_label}\n\n")
        f.write(
            f"_Standalone v_sterol ensemble experiment. Positive class = {positive_label}, "
            f"negative class = {negative_label}. Margin sweep applies the binary head whenever "
            f"the ensemble top-2 are exactly {{{negative_label}, {positive_label}}} and the top-1/top-2 gap is below the threshold._\n\n"
        )
        f.write(
            f"Baseline ensemble: 10-class macro-F1 = {_fmt(base_summary['macro_f1_mean'], base_summary['macro_f1_std'])}, "
            f"5-lipid macro-F1 = {_fmt(base_summary['lipid_macro_f1_mean'], base_summary['lipid_macro_f1_std'])}, "
            f"{negative_label} F1 = {_fmt(*base_summary['per_class_f1'][negative_label])}, "
            f"{positive_label} F1 = {_fmt(*base_summary['per_class_f1'][positive_label])}.\n\n"
        )
        f.write(
            f"Pair-only binary head (true {negative_label}+{positive_label} rows, {positive_label}=positive): "
            f"F1 = {_fmt(pair_binary_f1_mean, pair_binary_f1_std)}; mean scale_pos_weight = {scale_pos_weight_mean:.3f}.\n\n"
        )
        f.write(
            "| margin | 10-class macro-F1 | 5-lipid macro-F1 | "
            f"{negative_label} F1 | {positive_label} F1 | fired mean | fired total | "
            f"{negative_label}→{positive_label} | {positive_label}→{negative_label} |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['margin']:.2f} | {_fmt(row['macro_f1_mean'], row['macro_f1_std'])} "
                f"| {_fmt(row['lipid_macro_f1_mean'], row['lipid_macro_f1_std'])} "
                f"| {_fmt(row[f'{negative_label}_f1_mean'], row[f'{negative_label}_f1_std'])} "
                f"| {_fmt(row[f'{positive_label}_f1_mean'], row[f'{positive_label}_f1_std'])} "
                f"| {row['fire_mean']:.1f} | {row['fire_total']} "
                f"| {row[f'{negative_label}_as_{positive_label}']} | {row[f'{positive_label}_as_{negative_label}']} |\n"
            )

        f.write("\n## Interpretation\n\n")
        if monotonic:
            verdict = f"{positive_label} F1 moves monotonically upward across the tested margins."
        else:
            verdict = f"{positive_label} F1 does not improve monotonically across the tested margins."
        f.write(verdict + "\n")

        if feature_importance:
            top = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:15]
            f.write("\n## Iteration-0 top-15 features (gain)\n\n")
            f.write("| rank | feature | gain |\n|---:|---|---:|\n")
            for rank, (name, gain) in enumerate(top, start=1):
                f.write(f"| {rank} | {name} | {gain:.4f} |\n")
    return output_path


def run_pair_tiebreaker_experiment(
    *,
    full_pockets_path: Path,
    predictions_path: Path,
    splits_dir: Path,
    model_bundle_path: Path,
    output_report: Path,
    output_metrics: Path,
    negative_label: str,
    positive_label: str,
    margins: list[float],
    output_predictions: Path | None = None,
    selected_margin: float | None = None,
    workers: int = 8,
    seed_base: int = 42,
) -> dict[str, Any]:
    if negative_label == positive_label:
        raise ValueError("negative_label and positive_label must differ")
    if negative_label not in CLASS_10 or positive_label not in CLASS_10:
        raise ValueError("labels must be members of CLASS_10")

    bundle = joblib.load(model_bundle_path)
    feature_columns = list(bundle["feature_columns"])
    ensemble_df = _load_prediction_substrate(predictions_path)
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
                negative_label=negative_label,
                positive_label=positive_label,
                seed=seed_base + i,
                persist_importance=(i == 0),
            )
            for i, split_path in enumerate(split_files)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["iteration"])

    pair_binary_f1s = [r["pair_binary_f1"] for r in results]
    scale_pos_weights = [r["scale_pos_weight"] for r in results]
    feature_importance = next((r["feature_importance"] for r in results if r["feature_importance"]), {})

    rows: list[dict[str, Any]] = []
    augmented_by_margin: dict[float, pl.DataFrame] = {}
    neg_idx = CLASS_10.index(negative_label)
    pos_idx = CLASS_10.index(positive_label)
    for margin in margins:
        augmented_per_iter: list[pl.DataFrame] = []
        fires: list[int] = []
        for result in results:
            sub = ensemble_df.filter(pl.col("iteration") == result["iteration"])
            augmented = apply_pair_tiebreaker(
                sub,
                result["positive_proba"],
                result["row_index_lookup"],
                negative_label=negative_label,
                positive_label=positive_label,
                margin=margin,
            )
            augmented_per_iter.append(augmented)
            fires.append(int(augmented["tiebreaker_fired"].sum()))

        augmented_all = pl.concat(augmented_per_iter).sort(["iteration", "row_index"])
        augmented_by_margin[float(margin)] = augmented_all
        summary = score_summary(augmented_all.select(["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]))
        confusion = _pair_confusion(augmented_all, negative_label, positive_label)
        rows.append(
            {
                "margin": float(margin),
                "macro_f1_mean": summary["macro_f1_mean"],
                "macro_f1_std": summary["macro_f1_std"],
                "lipid_macro_f1_mean": summary["lipid_macro_f1_mean"],
                "lipid_macro_f1_std": summary["lipid_macro_f1_std"],
                f"{negative_label}_f1_mean": summary["per_class_f1"][negative_label][0],
                f"{negative_label}_f1_std": summary["per_class_f1"][negative_label][1],
                f"{positive_label}_f1_mean": summary["per_class_f1"][positive_label][0],
                f"{positive_label}_f1_std": summary["per_class_f1"][positive_label][1],
                "fire_mean": float(np.mean(fires)),
                "fire_total": int(np.sum(fires)),
                f"{negative_label}_as_{positive_label}": confusion[f"{negative_label}_as_{positive_label}"],
                f"{positive_label}_as_{negative_label}": confusion[f"{positive_label}_as_{negative_label}"],
                f"{negative_label}_correct": confusion[f"{negative_label}_correct"],
                f"{positive_label}_correct": confusion[f"{positive_label}_correct"],
                f"{negative_label}_support": confusion[f"{negative_label}_support"],
                f"{positive_label}_support": confusion[f"{positive_label}_support"],
                "negative_idx": neg_idx,
                "positive_idx": pos_idx,
            }
        )

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(output_metrics, index=False)
    chosen_margin: float | None = None
    if output_predictions is not None:
        if selected_margin is None:
            best = metrics_df.sort_values(
                ["lipid_macro_f1_mean", "macro_f1_mean"],
                ascending=[False, False],
            ).iloc[0]
            chosen_margin = float(best["margin"])
        else:
            chosen_margin = float(selected_margin)
        if chosen_margin not in augmented_by_margin:
            raise ValueError(
                f"selected_margin {chosen_margin} was not swept; available={sorted(augmented_by_margin)}"
            )
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        augmented_by_margin[chosen_margin].write_parquet(output_predictions)
    write_pair_experiment_report(
        output_report,
        negative_label=negative_label,
        positive_label=positive_label,
        rows=rows,
        base_summary=base_summary,
        pair_binary_f1_mean=float(np.mean(pair_binary_f1s)),
        pair_binary_f1_std=float(np.std(pair_binary_f1s)),
        scale_pos_weight_mean=float(np.mean(scale_pos_weights)),
        feature_importance=feature_importance,
    )
    return {
        "report": output_report,
        "metrics": output_metrics,
        "predictions": output_predictions,
        "selected_margin": chosen_margin,
        "rows": rows,
        "base_summary": base_summary,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone pairwise tiebreaker sweep.")
    parser.add_argument("--full-pockets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--selected-margin", type=float, default=None)
    parser.add_argument("--negative-label", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--margins", nargs="+", type=float, default=list(DEFAULT_MARGINS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_pair_tiebreaker_experiment(
        full_pockets_path=args.full_pockets,
        predictions_path=args.predictions,
        splits_dir=args.splits_dir,
        model_bundle_path=args.model_bundle,
        output_report=args.output_report,
        output_metrics=args.output_metrics,
        output_predictions=args.output_predictions,
        selected_margin=args.selected_margin,
        negative_label=args.negative_label,
        positive_label=args.positive_label,
        margins=list(args.margins),
        workers=args.workers,
        seed_base=args.seed_base,
    )
    print(f"wrote {result['report']}")
    print(f"wrote {result['metrics']}")


if __name__ == "__main__":
    main()
