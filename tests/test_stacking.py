"""Tests for the stacked meta-learner module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from slipp_plus.constants import CLASS_10
from slipp_plus.stacking import (
    StackingArtifact,
    fit_full_base_models,
    fit_meta_learner,
    generate_oof_softprobs,
    predict_proba_with_stacker,
    predict_with_stacker,
    train_stacker,
)


def _data(n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=10,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.int64)


def _splits(n: int, n_iter: int = 4, test_frac: float = 0.25) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s in range(n_iter):
        perm = np.random.default_rng(s).permutation(n)
        n_test = max(1, int(n * test_frac))
        out.append((perm[n_test:], perm[:n_test]))
    return out


def test_generate_oof_softprobs_shape_and_seen_mask() -> None:
    X, y = _data(n=240)
    splits = _splits(len(y), n_iter=4, test_frac=0.4)  # broad coverage so most rows seen
    oof, oof_y = generate_oof_softprobs(
        X, y, splits, models=["rf", "lgbm"], seed_base=42
    )
    n_models, n_classes = 2, len(CLASS_10)
    assert oof.shape[1] == n_models * n_classes
    assert oof.shape[0] == len(oof_y)
    # Most rows should land in at least one test fold; some may be missed by a model
    # in some seed combinations, but it should not be > all rows
    assert oof.shape[0] <= len(y)
    assert oof.shape[0] > 0


def test_fit_meta_learner_lr_default_predicts_known_labels() -> None:
    X, y = _data(n=240)
    splits = _splits(len(y), n_iter=4, test_frac=0.4)
    oof, oof_y = generate_oof_softprobs(
        X, y, splits, models=["rf", "lgbm"], seed_base=42
    )
    meta = fit_meta_learner(oof, oof_y, kind="lr", seed=42)
    pred = meta.predict(oof)
    assert pred.shape == oof_y.shape
    # Predictions are valid class indices
    assert pred.min() >= 0 and pred.max() < len(CLASS_10)


def test_train_stacker_end_to_end_returns_serializable_artifact() -> None:
    X, y = _data(n=240)
    splits = _splits(len(y), n_iter=4, test_frac=0.4)
    artifact = train_stacker(
        X, y, splits, models=["rf", "lgbm"], seed_base=42, meta_kind="lr"
    )
    assert isinstance(artifact, StackingArtifact)
    assert artifact.model_keys == ["rf", "lgbm"]
    assert artifact.meta_kind == "lr"
    assert len(artifact.base_full_models) == 2

    # Inference path
    pred = predict_with_stacker(artifact, X[:50])
    assert pred.shape == (50,)
    proba = predict_proba_with_stacker(artifact, X[:50])
    assert proba.shape == (50, len(CLASS_10))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_train_stacker_lgbm_meta_learner() -> None:
    X, y = _data(n=240)
    splits = _splits(len(y), n_iter=4, test_frac=0.4)
    artifact = train_stacker(
        X,
        y,
        splits,
        models=["rf", "lgbm"],
        seed_base=42,
        meta_kind="lgbm",
        meta_params={"n_estimators": 30, "num_leaves": 7},
    )
    assert artifact.meta_kind == "lgbm"
    assert hasattr(artifact.meta_model, "predict_proba")
    pred = predict_with_stacker(artifact, X[:30])
    assert pred.shape == (30,)


def test_fit_full_base_models_includes_catboost_when_requested() -> None:
    pytest.importorskip("catboost")
    X, y = _data(n=180)
    fitted = fit_full_base_models(X, y, models=["rf", "cat"], seed=42)
    assert len(fitted) == 2
    # CatBoost has predict_proba like sklearn
    proba = fitted[1].predict_proba(X[:10])
    assert proba.shape == (10, len(CLASS_10))
