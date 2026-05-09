"""Stacked meta-learner over the flat-mode base ensemble.

The historical "ensemble" in this project mean-averages RF + XGB + LGBM
softprobs. Mean-averaging is the simplest possible stacker; learned stacking
on out-of-fold (OOF) base predictions usually adds another +0.005 to +0.020
macro-F1 on tabular problems by exploiting per-model calibration differences
that mean-averaging cannot. With CatBoost added as a fourth diversity source
the lift typically grows.

This module builds the OOF softprob matrix once across the existing 25-iter
CV splits, fits a meta-learner (logistic regression by default; lightgbm
optionally), and exposes a predict path for new rows. Both objects are
serializable so the stacker becomes a registered artifact like any other
model bundle.

The OOF construction follows the standard recipe: a row's softprobs come from
the iteration in which that row was in the *test* fold, never from a
training fit on itself. Because the project's stratified shuffle splits
allow rows to land in multiple test folds (n_iterations x test_fraction is
larger than 1), we average those visits per row. Rows that never enter a
test fold are excluded from meta-learner training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import FlatModelHyperparameters
from .constants import CLASS_10
from .train import _build_model, _fit_predict

MetaKind = Literal["lr", "lgbm"]


@dataclass
class StackingArtifact:
    """Serializable stacker bundle.

    Stores the per-fold base models (so inference can re-fit base learners on
    the entire training set in one pass) and the meta-learner trained on OOF
    softprobs. The base model order corresponds to ``model_keys``.
    """

    model_keys: list[str]
    meta_kind: MetaKind
    meta_model: Any
    base_full_models: list[Any] = field(default_factory=list)
    n_classes: int = len(CLASS_10)


def _stack_softprobs(softprobs: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-model (n, c) softprobs into a single (n, m*c) matrix."""

    return np.concatenate([sp for sp in softprobs], axis=1)


def generate_oof_softprobs(
    X: np.ndarray,
    y_int: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    models: list[str],
    seed_base: int = 42,
    hp: FlatModelHyperparameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an OOF softprob matrix and the matching label vector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(oof_features, oof_labels)`` of shape ``(n_seen, m * c)`` and
        ``(n_seen,)`` where ``n_seen`` is the number of rows that landed in at
        least one test fold across the splits.
    """

    n_rows, n_classes = len(y_int), len(CLASS_10)
    sums = np.zeros((n_rows, len(models), n_classes), dtype=np.float64)
    counts = np.zeros((n_rows, len(models)), dtype=np.int64)

    for i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_int[train_idx]
        for m_idx, key in enumerate(models):
            _, _, proba = _fit_predict(key, seed_base + i, X_tr, y_tr, X_te, hp=hp)
            sums[test_idx, m_idx, :] += proba
            counts[test_idx, m_idx] += 1

    seen_mask = (counts > 0).all(axis=1)
    if not seen_mask.any():
        raise ValueError(
            "no rows were in any test fold for every base model — check splits / models list"
        )
    safe_counts = np.maximum(counts, 1)
    oof_per_model = sums / safe_counts[:, :, None]
    oof_features = _stack_softprobs([oof_per_model[:, m, :] for m in range(len(models))])
    return oof_features[seen_mask], y_int[seen_mask]


def fit_meta_learner(
    oof_features: np.ndarray,
    oof_labels: np.ndarray,
    *,
    kind: MetaKind = "lr",
    seed: int = 42,
    meta_params: dict[str, Any] | None = None,
) -> Any:
    """Fit the second-stage meta-learner over OOF softprobs.

    The default is multinomial logistic regression with L2 regularization,
    which is a strong baseline for stacking softprobs and has cheap inference.
    Pass ``kind='lgbm'`` for a non-linear meta-learner; ``meta_params`` is
    forwarded so the meta-learner itself can be HPO'd.
    """

    meta_params = dict(meta_params or {})
    if kind == "lr":
        defaults = {
            "C": 1.0,
            "max_iter": 500,
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "random_state": seed,
        }
        defaults.update(meta_params)
        model = LogisticRegression(**defaults)
        model.fit(oof_features, oof_labels)
        return model
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        defaults = {
            "objective": "multiclass",
            "num_class": len(CLASS_10),
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 5,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(meta_params)
        model = LGBMClassifier(**defaults)
        model.fit(oof_features, oof_labels)
        return model
    raise ValueError(f"unknown meta-learner kind: {kind}")


def fit_full_base_models(
    X: np.ndarray,
    y_int: np.ndarray,
    *,
    models: list[str],
    seed: int = 42,
    hp: FlatModelHyperparameters | None = None,
) -> list[Any]:
    """Fit each base learner on the entire training set for inference time."""

    fitted: list[Any] = []
    for key in models:
        model = _build_model(key, seed, hp=hp)
        if key in {"xgb", "cat"}:
            from sklearn.utils.class_weight import compute_sample_weight

            sw = compute_sample_weight(class_weight="balanced", y=y_int)
            model.fit(X, y_int, sample_weight=sw)
        else:
            model.fit(X, y_int)
        fitted.append(model)
    return fitted


def predict_with_stacker(artifact: StackingArtifact, X: np.ndarray) -> np.ndarray:
    """Return class predictions from a fitted stacker."""

    softprobs = [m.predict_proba(X).astype(np.float64) for m in artifact.base_full_models]
    features = _stack_softprobs(softprobs)
    return artifact.meta_model.predict(features).astype(np.int64).reshape(-1)


def predict_proba_with_stacker(artifact: StackingArtifact, X: np.ndarray) -> np.ndarray:
    """Return class softprobs from a fitted stacker."""

    softprobs = [m.predict_proba(X).astype(np.float64) for m in artifact.base_full_models]
    features = _stack_softprobs(softprobs)
    return artifact.meta_model.predict_proba(features).astype(np.float64)


def train_stacker(
    X: np.ndarray,
    y_int: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    models: list[str],
    seed_base: int = 42,
    hp: FlatModelHyperparameters | None = None,
    meta_kind: MetaKind = "lr",
    meta_params: dict[str, Any] | None = None,
) -> StackingArtifact:
    """End-to-end stacker training: OOF features → meta-learner → full base fits."""

    oof_features, oof_labels = generate_oof_softprobs(
        X, y_int, splits, models=models, seed_base=seed_base, hp=hp
    )
    meta = fit_meta_learner(
        oof_features, oof_labels, kind=meta_kind, seed=seed_base, meta_params=meta_params
    )
    base_full = fit_full_base_models(X, y_int, models=models, seed=seed_base, hp=hp)
    return StackingArtifact(
        model_keys=list(models),
        meta_kind=meta_kind,
        meta_model=meta,
        base_full_models=base_full,
    )
