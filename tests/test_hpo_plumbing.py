"""Tests for the HPO plumbing: XGB / Flat hyperparameter overrides land at the
training call sites without breaking historical defaults.

These tests use synthetic data (sklearn ``make_classification``) so they run
in environments without the project's ``processed/`` parquet artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from slipp_plus.boundary_head import train_boundary_head
from slipp_plus.config import FlatModelHyperparameters, XGBHyperparameters
from slipp_plus.hierarchical_experiment import (
    train_lipid_family,
    train_lipid_gate,
    train_nonlipid_family,
    train_one_vs_neighbors,
)
from slipp_plus.train import _build_model, _fit_predict, cv_evaluate_flat


def _binary_data(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n,
        n_features=12,
        n_informative=6,
        n_classes=2,
        random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.int64)


def _multiclass_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=10,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X.astype(np.float64), y.astype(np.int64)


def test_xgb_hyperparameters_defaults_match_legacy_values() -> None:
    hp = XGBHyperparameters()
    kwargs = hp.to_xgb_kwargs()
    assert kwargs["max_depth"] == 5
    assert kwargs["n_estimators"] == 250
    assert kwargs["learning_rate"] == 0.05


def test_train_boundary_head_accepts_xgb_hyperparameters() -> None:
    X, y = _binary_data()
    hp = XGBHyperparameters(max_depth=2, n_estimators=20, learning_rate=0.2)
    model = train_boundary_head(X, y, seed=42, hyperparameters=hp)
    booster_kwargs = model.get_params()
    assert booster_kwargs["max_depth"] == 2
    assert booster_kwargs["n_estimators"] == 20
    assert booster_kwargs["learning_rate"] == pytest.approx(0.2)


def test_train_boundary_head_legacy_kwargs_still_work() -> None:
    X, y = _binary_data()
    model = train_boundary_head(X, y, seed=42, max_depth=3, n_estimators=10)
    booster_kwargs = model.get_params()
    assert booster_kwargs["max_depth"] == 3
    assert booster_kwargs["n_estimators"] == 10


def test_hierarchical_train_helpers_accept_hyperparameters() -> None:
    X, y_bin = _binary_data()
    hp = XGBHyperparameters(max_depth=3, n_estimators=15, learning_rate=0.1)

    gate = train_lipid_gate(X, y_bin, seed=42, hyperparameters=hp)
    assert gate.get_params()["n_estimators"] == 15

    spec = train_one_vs_neighbors(X, y_bin, seed=42, hyperparameters=hp)
    assert spec.get_params()["max_depth"] == 3

    # Multi-class helpers want a 5-way y for the lipid family head
    X5, y5 = make_classification(
        n_samples=200,
        n_features=12,
        n_informative=8,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=0,
    )
    family = train_lipid_family(X5.astype(np.float64), y5.astype(np.int64), seed=42, hyperparameters=hp)
    assert family.get_params()["learning_rate"] == pytest.approx(0.1)

    nonlipid = train_nonlipid_family(X5.astype(np.float64), y5.astype(np.int64), seed=42, hyperparameters=hp)
    assert nonlipid.get_params()["max_depth"] == 3


def test_flat_build_model_respects_per_family_hyperparameters() -> None:
    hp = FlatModelHyperparameters(
        rf_n_estimators=37,
        xgb_max_depth=4,
        lgbm_num_leaves=21,
    )
    rf = _build_model("rf", seed=0, hp=hp)
    assert rf.n_estimators == 37
    xgb = _build_model("xgb", seed=0, hp=hp)
    assert xgb.get_xgb_params()["max_depth"] == 4
    lgbm = _build_model("lgbm", seed=0, hp=hp)
    assert lgbm.num_leaves == 21


def test_flat_build_model_catboost_path() -> None:
    pytest.importorskip("catboost")
    hp = FlatModelHyperparameters(cat_depth=4, cat_iterations=50, cat_learning_rate=0.1)
    cat = _build_model("cat", seed=0, hp=hp)
    # CatBoostClassifier exposes params via .get_params()
    params = cat.get_params()
    assert params["depth"] == 4
    assert params["iterations"] == 50


def test_fit_predict_catboost_softprob_matches_class_count() -> None:
    pytest.importorskip("catboost")
    X, y = _multiclass_data()
    splits = [
        (np.arange(160), np.arange(160, 200)),
    ]
    train_idx, test_idx = splits[0]
    hp = FlatModelHyperparameters(cat_iterations=30, cat_depth=4)
    _, pred, proba = _fit_predict(
        "cat", seed=0, X_tr=X[train_idx], y_tr_int=y[train_idx], X_te=X[test_idx], hp=hp
    )
    assert pred.shape == (40,)
    assert proba.shape == (40, 10)
    assert pred.dtype == np.int64


def test_cv_evaluate_flat_smoke_three_iters_two_models() -> None:
    X, y = _multiclass_data(n=180)
    splits = []
    for s in range(3):
        perm = np.random.default_rng(s).permutation(len(y))
        n_test = len(y) // 5
        splits.append((perm[n_test:], perm[:n_test]))
    out = cv_evaluate_flat(X, y, splits, models=["rf", "lgbm"], seed_base=42)
    assert 0.0 <= out["macro_f1_mean"] <= 1.0
    assert 0.0 <= out["lipid_macro_f1_mean"] <= 1.0
    assert 0.0 <= out["binary_f1_mean"] <= 1.0
    # Hyperband intermediate keys are populated for each iteration
    assert "macro_f1_iter_0" in out and "macro_f1_iter_2" in out


def test_cv_evaluate_flat_progress_callback_invoked_per_iter() -> None:
    X, y = _multiclass_data(n=120)
    splits = []
    for s in range(2):
        perm = np.random.default_rng(s).permutation(len(y))
        n_test = len(y) // 4
        splits.append((perm[n_test:], perm[:n_test]))

    ticks: list[tuple[int, dict[str, float]]] = []

    def cb(i: int, partial: dict[str, float]) -> None:
        ticks.append((i, dict(partial)))

    cv_evaluate_flat(X, y, splits, models=["rf"], seed_base=42, progress_callback=cb)
    assert [t[0] for t in ticks] == [0, 1]
    for _, partial in ticks:
        assert "lipid_macro_f1_mean" in partial
