"""Utility-aware gating for specialist post-processing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from .constants import CLASS_10
from .ensemble import PROBA_COLUMNS
from .hierarchical_postprocess import OneVsNeighborsRule


@dataclass(frozen=True)
class UtilityGateConfig:
    threshold_default: float = 0.50
    threshold_top1_plm: float | None = None


def _entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _top2(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-proba, axis=1)
    return order[:, 0].astype(np.int64), order[:, 1].astype(np.int64)


def build_utility_features(
    staged_df: pl.DataFrame,
    p_specialist: np.ndarray,
    rule: OneVsNeighborsRule,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-wise gate features and return (X, top1_idx)."""

    proba = staged_df.select(PROBA_COLUMNS).to_numpy().astype(np.float64)
    if len(proba) != len(p_specialist):
        raise ValueError("staged_df and p_specialist must align")
    ste_idx = CLASS_10.index(rule.positive_label)
    plm_idx = CLASS_10.index("PLM")
    top1, top2 = _top2(proba)
    neighbor_idx = np.array([CLASS_10.index(c) for c in rule.neighbor_labels], dtype=np.int64)
    is_top1_neighbor = np.isin(top1, neighbor_idx).astype(np.float64)
    x = np.column_stack(
        [
            p_specialist.astype(np.float64),
            proba[:, ste_idx],
            proba[np.arange(len(proba)), top1],
            proba[np.arange(len(proba)), top2],
            proba[np.arange(len(proba)), top1] - proba[np.arange(len(proba)), top2],
            proba[:, plm_idx] - proba[:, ste_idx],
            _entropy(proba),
            is_top1_neighbor,
            (top1 == plm_idx).astype(np.float64),
        ]
    )
    return x, top1


def fit_utility_gate(
    *,
    staged_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    p_specialist: np.ndarray,
    rule: OneVsNeighborsRule,
) -> LogisticRegression | None:
    """Fit logistic gate to predict when specialist rewrite helps."""

    if "y_true_int" not in staged_df.columns:
        return None
    if len(staged_df) == 0:
        return None
    x, _top1 = build_utility_features(staged_df, p_specialist, rule)
    y_true = staged_df["y_true_int"].to_numpy().astype(np.int64)
    y_base = staged_df["y_pred_int"].to_numpy().astype(np.int64)
    y_cand = candidate_df["y_pred_int"].to_numpy().astype(np.int64)
    fired = candidate_df[rule.fired_col].to_numpy().astype(bool)

    improve = (y_cand == y_true) & (y_base != y_true)
    target = (improve & fired).astype(np.int64)
    valid = fired
    if valid.sum() < 5:
        return None
    y = target[valid]
    if np.unique(y).size < 2:
        return None

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x[valid], y)
    return model


def apply_utility_gate(
    *,
    staged_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    p_specialist: np.ndarray,
    rule: OneVsNeighborsRule,
    utility_model: LogisticRegression | None,
    config: UtilityGateConfig,
) -> pl.DataFrame:
    """Keep specialist rewrite only when utility gate predicts positive gain."""

    if utility_model is None:
        return candidate_df
    x, top1 = build_utility_features(staged_df, p_specialist, rule)
    gate_scores = utility_model.predict_proba(x)[:, 1].astype(np.float64)
    fired = candidate_df[rule.fired_col].to_numpy().astype(bool)

    threshold = np.full(len(gate_scores), config.threshold_default, dtype=np.float64)
    if config.threshold_top1_plm is not None:
        plm_idx = CLASS_10.index("PLM")
        threshold[top1 == plm_idx] = float(config.threshold_top1_plm)

    keep = fired & (gate_scores >= threshold)
    base = staged_df.select(PROBA_COLUMNS).to_numpy()
    cand = candidate_df.select(PROBA_COLUMNS).to_numpy()
    merged = np.where(keep[:, None], cand, base)
    new_pred = merged.argmax(axis=1).astype(np.int64)

    score_col = f"{rule.name}_utility_score"
    keep_col = f"{rule.name}_utility_keep"
    return candidate_df.with_columns(
        *[pl.Series(c, merged[:, i]) for i, c in enumerate(PROBA_COLUMNS)],
        pl.Series("y_pred_int", new_pred),
        pl.Series(score_col, gate_scores),
        pl.Series(keep_col, keep),
        pl.Series(rule.fired_col, keep),
    )


def serialize_utility_model(model: LogisticRegression | None) -> dict[str, Any] | None:
    """Convert a fitted utility gate into a joblib-friendly payload.

    Parameters
    ----------
    model
        Fitted logistic regression gate, or ``None`` when no utility model was
        trainable.

    Returns
    -------
    dict[str, Any] | None
        Serializable coefficient/intercept/classes payload, or ``None``.
    """

    if model is None:
        return None
    return {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
    }


def deserialize_utility_model(payload: Mapping[str, Any] | None) -> LogisticRegression | None:
    """Rehydrate a utility gate from bundle metadata.

    Parameters
    ----------
    payload
        Serialized payload produced by ``serialize_utility_model``.

    Returns
    -------
    LogisticRegression | None
        Restored logistic regression model, or ``None`` when no payload is
        present.
    """

    if payload is None:
        return None
    coef = np.asarray(payload["coef"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)
    classes = np.asarray(payload["classes"], dtype=np.int64)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.classes_ = classes
    model.coef_ = coef
    model.intercept_ = intercept
    model.n_features_in_ = coef.shape[1]
    return model
