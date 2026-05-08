"""Reusable binary boundary heads for local class arbitration.

Boundary heads train on a narrow positive-vs-negative class contrast and then
redistribute probability mass only among the candidate classes. This preserves
unrelated softmax evidence while letting hard lipid boundaries use a local
objective instead of the global 10-class simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier

from .constants import CLASS_10
from .ensemble import PROBA_COLUMNS
from .splits import load_split


@dataclass(frozen=True)
class BoundaryRule:
    """Configuration for one positive-vs-negative boundary head."""

    name: str
    positive_label: str
    negative_labels: tuple[str, ...]
    margin: float = 0.20
    max_rank: int | None = None
    fired_column: str | None = None
    score_column: str | None = None

    @property
    def candidate_labels(self) -> tuple[str, ...]:
        return (*self.negative_labels, self.positive_label)

    @property
    def fired_col(self) -> str:
        return self.fired_column or f"{self.name}_fired"

    @property
    def score_col(self) -> str:
        return self.score_column or f"p_{self.positive_label}_{self.name}"


@dataclass(frozen=True)
class NeighborRescueRule:
    """Positive-vs-neighbors rescue with top-k routing and a probability threshold."""

    name: str
    positive_label: str
    neighbor_labels: tuple[str, ...]
    threshold: float = 0.50
    top_k: int = 4
    fired_column: str | None = None
    score_column: str | None = None

    @property
    def boundary_rule(self) -> BoundaryRule:
        return BoundaryRule(
            name=self.name,
            positive_label=self.positive_label,
            negative_labels=self.neighbor_labels,
        )

    @property
    def fired_col(self) -> str:
        return self.fired_column or f"{self.name}_fired"

    @property
    def score_col(self) -> str:
        return self.score_column or f"p_{self.positive_label}_{self.name}"


def validate_boundary_rule(rule: BoundaryRule) -> None:
    """Validate labels and thresholds for a binary boundary rule.

    Parameters
    ----------
    rule
        Boundary rule to validate against the canonical ``CLASS_10`` label set.

    Raises
    ------
    ValueError
        If labels are unknown, the positive label is also negative, no negative
        labels are configured, or routing thresholds are out of range.
    """

    labels = {rule.positive_label, *rule.negative_labels}
    unknown = sorted(labels - set(CLASS_10))
    if unknown:
        raise ValueError(f"unknown class labels in boundary rule {rule.name}: {unknown}")
    if rule.positive_label in rule.negative_labels:
        raise ValueError("positive_label cannot also be in negative_labels")
    if not rule.negative_labels:
        raise ValueError("negative_labels cannot be empty")
    if rule.margin < 0.0:
        raise ValueError("margin must be non-negative")
    if rule.max_rank is not None and not 2 <= rule.max_rank <= len(CLASS_10):
        raise ValueError("max_rank must be in [2, len(CLASS_10)]")


def validate_neighbor_rescue_rule(rule: NeighborRescueRule) -> None:
    """Validate a top-k neighbor rescue rule.

    Parameters
    ----------
    rule
        Neighbor rescue rule to validate.

    Raises
    ------
    ValueError
        If the embedded boundary rule is invalid or the rescue threshold/top-k
        settings are out of range.
    """

    validate_boundary_rule(rule.boundary_rule)
    if not 0.0 <= rule.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 2 <= rule.top_k <= len(CLASS_10):
        raise ValueError("top_k must be in [2, len(CLASS_10)]")


def build_boundary_training(
    full_pockets: pd.DataFrame,
    feature_columns: list[str],
    split_parquet: Path,
    rule: BoundaryRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build split-aligned train/test arrays for a boundary rule.

    Returns ``(X_tr, y_tr, X_te, y_te, test_row_index)``. Labels are binary:
    positive class is 1 and all configured negative classes are 0. The test
    arrays contain only rows whose true class belongs to the boundary.
    """

    validate_boundary_rule(rule)
    train_idx, test_idx = load_split(split_parquet)
    X_all = full_pockets[feature_columns].to_numpy(dtype=np.float64)
    y_all = full_pockets["class_10"].to_numpy()
    candidate_labels = set(rule.candidate_labels)

    tr_sel = train_idx[np.isin(y_all[train_idx], list(candidate_labels))]
    te_sel = test_idx[np.isin(y_all[test_idx], list(candidate_labels))]

    X_tr = X_all[tr_sel]
    y_tr = (y_all[tr_sel] == rule.positive_label).astype(np.int64)
    X_te = X_all[te_sel]
    y_te = (y_all[te_sel] == rule.positive_label).astype(np.int64)
    return X_tr, y_tr, X_te, y_te, te_sel.astype(np.int64)


def train_boundary_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    *,
    max_depth: int = 5,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    """Fit a class-balanced binary XGB boundary head."""

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "boundary head requires both positive and negative training rows "
            f"(n_pos={n_pos}, n_neg={n_neg})"
        )
    model = XGBClassifier(
        objective="binary:logistic",
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=n_neg / n_pos,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def _top_indices(proba: np.ndarray, k: int) -> np.ndarray:
    if k >= proba.shape[1]:
        return np.argsort(-proba, axis=1)
    unordered = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    row = np.arange(proba.shape[0])[:, None]
    order = np.argsort(-proba[row, unordered], axis=1)
    return np.take_along_axis(unordered, order, axis=1)


def apply_boundary_head(
    ensemble_df: pl.DataFrame,
    positive_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    rule: BoundaryRule,
) -> pl.DataFrame:
    """Apply a boundary head by redistributing candidate-class mass.

    A row fires when the top candidate labels are exactly the configured
    boundary labels and their internal probability spread is below
    ``rule.margin``. For a binary boundary this is equivalent to the historical
    PLM/STE "top-2 pair plus margin" rule.
    """

    validate_boundary_rule(rule)
    if len(positive_proba) != len(row_index_lookup):
        raise ValueError("positive_proba and row_index_lookup must align")
    if len(set(row_index_lookup.tolist())) != len(row_index_lookup):
        raise ValueError("row_index_lookup contains duplicate row indices")
    if np.any((positive_proba < 0.0) | (positive_proba > 1.0)):
        raise ValueError("positive_proba values must be in [0, 1]")

    candidate_labels = rule.candidate_labels
    candidate_idx = np.array([CLASS_10.index(label) for label in candidate_labels], dtype=np.int64)
    positive_idx = CLASS_10.index(rule.positive_label)
    negative_idx = np.array(
        [CLASS_10.index(label) for label in rule.negative_labels], dtype=np.int64
    )
    rank_k = rule.max_rank or len(candidate_idx)

    lookup = dict(zip(row_index_lookup.tolist(), positive_proba.tolist(), strict=True))
    proba = ensemble_df.select(PROBA_COLUMNS).to_numpy().astype(np.float64, copy=True)
    row_indices = ensemble_df["row_index"].to_numpy()
    n_rows = proba.shape[0]
    fired = np.zeros(n_rows, dtype=bool)
    score = np.full(n_rows, np.nan, dtype=np.float64)

    top = _top_indices(proba, rank_k)
    candidate_set = set(candidate_idx.tolist())
    for i in range(n_rows):
        top_candidate = [int(idx) for idx in top[i] if int(idx) in candidate_set]
        if set(top_candidate) != candidate_set:
            continue
        candidate_scores = proba[i, candidate_idx]
        if float(candidate_scores.max() - candidate_scores.min()) >= rule.margin:
            continue
        row_idx = int(row_indices[i])
        if row_idx not in lookup:
            continue

        p_positive = float(lookup[row_idx])
        mass = float(proba[i, candidate_idx].sum())
        if mass <= 0.0:
            continue
        proba[i, positive_idx] = mass * p_positive
        negative_mass = mass * (1.0 - p_positive)
        neg_scores = proba[i, negative_idx].astype(np.float64)
        denom = float(neg_scores.sum())
        if denom <= 1e-12:
            proba[i, negative_idx] = negative_mass / len(negative_idx)
        else:
            proba[i, negative_idx] = negative_mass * (neg_scores / denom)
        fired[i] = True
        score[i] = p_positive

    return ensemble_df.with_columns(
        *[pl.Series(c, proba[:, i]) for i, c in enumerate(PROBA_COLUMNS)],
        pl.Series("y_pred_int", proba.argmax(axis=1).astype(np.int64)),
        pl.Series(rule.fired_col, fired),
        pl.Series(rule.score_col, score),
    )


def apply_neighbor_rescue_head(
    ensemble_df: pl.DataFrame,
    positive_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    rule: NeighborRescueRule,
) -> pl.DataFrame:
    """Apply a positive-vs-neighbors rescue head.

    A row fires when the current top-1 class is a configured neighbor, the
    positive class is present in the row's top-k classes, and the binary head's
    positive probability is at least ``rule.threshold``. Only the positive
    class and current top neighbor exchange mass.
    """

    validate_neighbor_rescue_rule(rule)
    if len(positive_proba) != len(row_index_lookup):
        raise ValueError("positive_proba and row_index_lookup must align")
    if len(set(row_index_lookup.tolist())) != len(row_index_lookup):
        raise ValueError("row_index_lookup contains duplicate row indices")
    if np.any((positive_proba < 0.0) | (positive_proba > 1.0)):
        raise ValueError("positive_proba values must be in [0, 1]")

    positive_idx = CLASS_10.index(rule.positive_label)
    neighbor_idx = {CLASS_10.index(label) for label in rule.neighbor_labels}
    lookup = dict(zip(row_index_lookup.tolist(), positive_proba.tolist(), strict=True))

    proba = ensemble_df.select(PROBA_COLUMNS).to_numpy().astype(np.float64, copy=True)
    row_indices = ensemble_df["row_index"].to_numpy()
    n_rows = proba.shape[0]
    fired = np.zeros(n_rows, dtype=bool)
    score = np.full(n_rows, np.nan, dtype=np.float64)
    top = _top_indices(proba, rule.top_k)
    top1 = top[:, 0]

    for i in range(n_rows):
        current_top = int(top1[i])
        if current_top not in neighbor_idx:
            continue
        if positive_idx not in {int(v) for v in top[i]}:
            continue
        row_idx = int(row_indices[i])
        if row_idx not in lookup:
            continue
        p_positive = float(lookup[row_idx])
        if p_positive < rule.threshold:
            continue
        mass = float(proba[i, positive_idx] + proba[i, current_top])
        proba[i, positive_idx] = mass * p_positive
        proba[i, current_top] = mass * (1.0 - p_positive)
        fired[i] = True
        score[i] = p_positive

    return ensemble_df.with_columns(
        *[pl.Series(c, proba[:, i]) for i, c in enumerate(PROBA_COLUMNS)],
        pl.Series(rule.fired_col, fired),
        pl.Series(rule.score_col, score),
        pl.Series("y_pred_int", proba.argmax(axis=1).astype(np.int64)),
    )


def boundary_confusion(df: pl.DataFrame, rule: BoundaryRule) -> dict[str, int]:
    """Return positive-vs-negative confusion counts for a boundary rule."""

    validate_boundary_rule(rule)
    y_true = df["y_true_int"].to_numpy()
    y_pred = df["y_pred_int"].to_numpy()
    pos = CLASS_10.index(rule.positive_label)
    neg = np.array([CLASS_10.index(label) for label in rule.negative_labels], dtype=np.int64)

    true_pos = y_true == pos
    true_neg = np.isin(y_true, neg)
    pred_pos = y_pred == pos
    pred_neg = np.isin(y_pred, neg)
    return {
        "positive_as_positive": int((true_pos & pred_pos).sum()),
        "positive_as_negative": int((true_pos & pred_neg).sum()),
        "negative_as_positive": int((true_neg & pred_pos).sum()),
        "negative_as_negative": int((true_neg & pred_neg).sum()),
        "positive_support": int(true_pos.sum()),
        "negative_support": int(true_neg.sum()),
    }


def gain_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    """Map XGBoost f-index gain importance back to feature names."""

    gain_map = model.get_booster().get_score(importance_type="gain")
    return {feature: float(gain_map.get(f"f{i}", 0.0)) for i, feature in enumerate(feature_columns)}
