"""Shared execution core for legacy pair-specific tiebreaker modules."""

from __future__ import annotations

import concurrent.futures as futures
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .boundary_head import (
    BoundaryRule,
    apply_boundary_head,
    build_boundary_training,
    gain_importance,
    train_boundary_head,
)
from .ensemble import average_softprobs, load_predictions
from .splits import load_split


def _rule_with_margin(rule: BoundaryRule, margin: float) -> BoundaryRule:
    return BoundaryRule(
        name=rule.name,
        positive_label=rule.positive_label,
        negative_labels=rule.negative_labels,
        margin=margin,
        max_rank=rule.max_rank,
        fired_column=rule.fired_column,
        score_column=rule.score_column,
    )


def _binary_f1(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    pred = (p_positive >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0


def _process_boundary_iteration(
    *,
    iteration: int,
    seed: int,
    full_pockets_path: Path,
    feature_columns: list[str],
    split_path: Path,
    iteration_pred_path: Path,
    margin: float,
    persist_importance: bool,
    rule: BoundaryRule,
    train_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    full = pd.read_parquet(full_pockets_path)
    X_tr, y_tr, X_te_pair, y_te_pair, _te_pair_idx = build_boundary_training(
        full,
        feature_columns,
        split_path,
        rule,
    )
    model = train_boundary_head(X_tr, y_tr, seed=seed, **(train_kwargs or {}))

    _, test_idx = load_split(split_path)
    X_all = full[feature_columns].to_numpy(dtype=np.float64)
    X_te_full = X_all[test_idx]
    proba_all = model.predict_proba(X_te_full)[:, 1]

    ensemble_iter = pl.read_parquet(iteration_pred_path)
    augmented = apply_boundary_head(
        ensemble_iter,
        proba_all,
        test_idx.astype(np.int64),
        _rule_with_margin(rule, margin),
    )

    importance: dict[str, float] | None = None
    if persist_importance:
        importance = gain_importance(model, feature_columns)

    p_te_pair = model.predict_proba(X_te_pair)[:, 1]
    return {
        "iteration": iteration,
        "augmented_frame": augmented.to_arrow(),
        "tiebreaker_fired": int(augmented[rule.fired_col].sum()),
        "n_test": augmented.shape[0],
        "binary_f1": float(_binary_f1(y_te_pair, p_te_pair)),
        "feature_importance": importance,
    }


def run_boundary_tiebreaker_iterations(
    *,
    full_pockets_path: Path,
    predictions_path: Path,
    splits_dir: Path,
    feature_columns: list[str],
    workers: int,
    margin: float,
    seed_base: int,
    rule: BoundaryRule,
    train_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a boundary tiebreaker across all persisted split iterations.

    Parameters
    ----------
    full_pockets_path
        Training feature parquet used to fit the local boundary head.
    predictions_path
        Base multiclass prediction parquet to augment.
    splits_dir
        Directory containing ``seed_*.parquet`` split files.
    feature_columns
        Feature columns used by the boundary head.
    workers
        Number of worker processes for per-iteration training.
    margin
        Boundary-rule firing margin applied during probability redistribution.
    seed_base
        Base seed; iteration ``i`` uses ``seed_base + i``.
    rule
        Boundary rule defining the positive and negative labels.
    train_kwargs
        Optional model-training keyword arguments passed to
        ``train_boundary_head``.

    Returns
    -------
    dict[str, Any]
        Base ensemble predictions, augmented predictions, per-iteration
        summaries, fire counts, pairwise binary F1 values, and iteration-0
        feature importance when available.
    """

    ensemble_pred = average_softprobs(load_predictions(predictions_path))
    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if not split_files:
        raise FileNotFoundError(f"no seed_*.parquet under {splits_dir}")

    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"{rule.name}_") as tmpd:
        tmp = Path(tmpd)
        per_iter_paths: dict[int, Path] = {}
        for it in ensemble_pred["iteration"].unique().to_list():
            path = tmp / f"ensemble_iter_{int(it):02d}.parquet"
            ensemble_pred.filter(pl.col("iteration") == it).write_parquet(path)
            per_iter_paths[int(it)] = path

        tasks = []
        with futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, split_path in enumerate(split_files):
                if idx not in per_iter_paths:
                    continue
                tasks.append(
                    ex.submit(
                        _process_boundary_iteration,
                        iteration=idx,
                        seed=seed_base + idx,
                        full_pockets_path=full_pockets_path,
                        feature_columns=list(feature_columns),
                        split_path=split_path,
                        iteration_pred_path=per_iter_paths[idx],
                        margin=margin,
                        persist_importance=(idx == 0),
                        rule=rule,
                        train_kwargs=train_kwargs,
                    )
                )

            per_iter_results: list[dict[str, Any]] = []
            for future in futures.as_completed(tasks):
                per_iter_results.append(future.result())

    per_iter_results.sort(key=lambda row: row["iteration"])
    augmented = pl.concat(
        [pl.from_arrow(row["augmented_frame"]) for row in per_iter_results]
    ).sort(["iteration", "row_index"])

    fires = [row["tiebreaker_fired"] for row in per_iter_results]
    binary_f1s = [row["binary_f1"] for row in per_iter_results]
    importance = next(
        (row["feature_importance"] for row in per_iter_results if row["feature_importance"]),
        None,
    )

    return {
        "ensemble_predictions": ensemble_pred,
        "augmented_predictions": augmented,
        "per_iter_results": per_iter_results,
        "fire_counts": fires,
        "binary_f1s": binary_f1s,
        "feature_importance": importance,
    }
