"""Reusable specialist-head postprocessing for hierarchical lipid experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .constants import CLASS_10
from .ensemble import PROBA_COLUMNS


@dataclass(frozen=True)
class OneVsNeighborsRule:
    """Gate and apply a binary positive-vs-neighbors specialist head."""

    name: str
    positive_label: str
    neighbor_labels: tuple[str, ...]
    top_k: int = 4
    min_positive_proba: float | None = None
    max_margin: float | None = None
    fired_column: str | None = None
    score_column: str | None = None

    @property
    def fired_col(self) -> str:
        return self.fired_column or f"{self.name}_fired"

    @property
    def score_col(self) -> str:
        return self.score_column or f"p_{self.positive_label}_{self.name}"


def _validate_rule(rule: OneVsNeighborsRule) -> None:
    labels = {rule.positive_label, *rule.neighbor_labels}
    unknown = sorted(labels - set(CLASS_10))
    if unknown:
        raise ValueError(f"unknown class labels in rule {rule.name}: {unknown}")
    if rule.positive_label in rule.neighbor_labels:
        raise ValueError("positive_label cannot also be a neighbor label")
    if not rule.neighbor_labels:
        raise ValueError("neighbor_labels cannot be empty")
    if rule.top_k < 2 or rule.top_k > len(CLASS_10):
        raise ValueError("top_k must be in [2, len(CLASS_10)]")
    if rule.min_positive_proba is not None and not 0.0 <= rule.min_positive_proba <= 1.0:
        raise ValueError("min_positive_proba must be in [0, 1]")
    if rule.max_margin is not None and rule.max_margin < 0.0:
        raise ValueError("max_margin must be non-negative")


def apply_one_vs_neighbors(
    ensemble_df: pl.DataFrame,
    positive_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    rule: OneVsNeighborsRule,
) -> pl.DataFrame:
    """Apply a binary specialist to rows routed by top-k multiclass context.

    Only the current top-1 neighbor and positive class exchange probability
    mass. This preserves total probability and avoids changing unrelated class
    evidence.
    """

    _validate_rule(rule)
    if len(positive_proba) != len(row_index_lookup):
        raise ValueError("positive_proba and row_index_lookup must align")
    if len(set(row_index_lookup.tolist())) != len(row_index_lookup):
        raise ValueError("row_index_lookup contains duplicate row indices")
    if np.any((positive_proba < 0.0) | (positive_proba > 1.0)):
        raise ValueError("positive_proba values must be in [0, 1]")

    positive_idx = CLASS_10.index(rule.positive_label)
    neighbor_idx = {CLASS_10.index(label) for label in rule.neighbor_labels}
    lookup = dict(zip(row_index_lookup.tolist(), positive_proba.tolist(), strict=True))

    proba = ensemble_df.select(PROBA_COLUMNS).to_numpy().copy()
    row_indices = ensemble_df["row_index"].to_numpy()
    n_rows = proba.shape[0]
    fired = np.zeros(n_rows, dtype=bool)
    score = np.full(n_rows, np.nan, dtype=np.float64)

    sorted_idx = np.argsort(-proba, axis=1)
    top1 = sorted_idx[:, 0]
    topk = sorted_idx[:, : rule.top_k]

    for i in range(n_rows):
        current_top = int(top1[i])
        if current_top not in neighbor_idx:
            continue
        if positive_idx not in topk[i]:
            continue
        if rule.max_margin is not None:
            positive_rank_score = float(proba[i, positive_idx])
            if float(proba[i, current_top] - positive_rank_score) > rule.max_margin:
                continue
        row_idx = int(row_indices[i])
        if row_idx not in lookup:
            continue
        p_positive = float(lookup[row_idx])
        if rule.min_positive_proba is not None and p_positive < rule.min_positive_proba:
            continue

        score[i] = p_positive
        mass = float(proba[i, positive_idx] + proba[i, current_top])
        proba[i, positive_idx] = mass * p_positive
        proba[i, current_top] = mass * (1.0 - p_positive)
        fired[i] = True

    new_y_pred = proba.argmax(axis=1).astype(np.int64)
    replacements = [pl.Series(c, proba[:, i]) for i, c in enumerate(PROBA_COLUMNS)]
    return ensemble_df.with_columns(
        *replacements,
        pl.Series(rule.fired_col, fired),
        pl.Series(rule.score_col, score),
        pl.Series("y_pred_int", new_y_pred),
    )
