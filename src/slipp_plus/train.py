"""Train 3 model families x N split iterations.

Flat mode (``pipeline_mode=flat``): 10-class RF/XGB/LGBM, iteration-0 joblibs,
and ``test_predictions.parquet``.

Hierarchical/composite modes: staged lipid/MoE pipeline only;
see ``hierarchical_pipeline.run_hierarchical_training`` for artifacts
(including ``hierarchical_bundle.joblib`` and staged prediction parquet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from .artifact_schema import (
    build_feature_schema_metadata,
    write_artifact_schema_sidecar,
)
from .config import FlatModelHyperparameters, Settings
from .constants import CLASS_10
from .features import class10_labels, feature_matrix
from .run_metadata import write_run_metadata_sidecar
from .splits import load_split, make_splits, persist_splits


def _build_model(key: str, seed: int, hp: FlatModelHyperparameters | None = None) -> Any:
    """Construct a flat-mode base learner.

    ``hp`` overrides the historical hand-tuned defaults. When ``None`` the
    function reproduces the values used before HPO was introduced.
    """

    hp = hp or FlatModelHyperparameters()
    if key == "rf":
        rf_kwargs: dict[str, Any] = {
            "n_estimators": hp.rf_n_estimators,
            "min_samples_leaf": hp.rf_min_samples_leaf,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
        }
        if hp.rf_max_depth is not None:
            rf_kwargs["max_depth"] = hp.rf_max_depth
        if hp.rf_max_features is not None:
            rf_kwargs["max_features"] = hp.rf_max_features
        return RandomForestClassifier(**rf_kwargs)
    if key == "xgb":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_10),
            max_depth=hp.xgb_max_depth,
            n_estimators=hp.xgb_n_estimators,
            learning_rate=hp.xgb_learning_rate,
            subsample=hp.xgb_subsample,
            colsample_bytree=hp.xgb_colsample_bytree,
            min_child_weight=hp.xgb_min_child_weight,
            reg_alpha=hp.xgb_reg_alpha,
            reg_lambda=hp.xgb_reg_lambda,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
        )
    if key == "lgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_10),
            num_leaves=hp.lgbm_num_leaves,
            n_estimators=hp.lgbm_n_estimators,
            learning_rate=hp.lgbm_learning_rate,
            min_data_in_leaf=hp.lgbm_min_data_in_leaf,
            feature_fraction=hp.lgbm_feature_fraction,
            bagging_fraction=hp.lgbm_bagging_fraction,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if key == "cat":
        # CatBoost as a fourth diversity source. Imported lazily so optional
        # deployments without catboost still load.
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=hp.cat_iterations,
            depth=hp.cat_depth,
            learning_rate=hp.cat_learning_rate,
            l2_leaf_reg=hp.cat_l2_leaf_reg,
            loss_function="MultiClass",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    raise ValueError(f"unknown model key: {key}")


def _fit_predict(
    key: str,
    seed: int,
    X_tr: np.ndarray,
    y_tr_int: np.ndarray,
    X_te: np.ndarray,
    hp: FlatModelHyperparameters | None = None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit and return (model, pred_int, pred_proba) on test set."""
    model = _build_model(key, seed, hp=hp)
    if key in {"xgb", "cat"}:
        # Both XGB (softprob) and CatBoost (multi-class) benefit from sample
        # weights to keep parity with the lgbm/rf class_weight='balanced' path.
        sw = compute_sample_weight(class_weight="balanced", y=y_tr_int)
        model.fit(X_tr, y_tr_int, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr_int)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    # CatBoost.predict returns a 2-D array; flatten for downstream parity.
    pred_arr = np.asarray(pred).reshape(-1)
    return model, pred_arr.astype(np.int64), np.asarray(proba).astype(np.float64)


def _run_flat_training(settings: Settings) -> dict[str, Path]:
    paths = settings.paths
    proc = paths.processed_dir
    models_dir = paths.models_dir
    pred_dir = proc / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(proc / "full_pockets.parquet")
    feature_columns = settings.feature_columns()
    schema_metadata = build_feature_schema_metadata(
        feature_set=settings.feature_set,
        feature_columns=feature_columns,
    )
    X = feature_matrix(full, settings)
    y_str = class10_labels(full)
    class_to_int = {c: i for i, c in enumerate(CLASS_10)}
    y_int = np.array([class_to_int[c] for c in y_str], dtype=np.int64)
    group_labels: np.ndarray | None = None
    if settings.split_strategy == "grouped":
        group_column = settings.split_group_column
        if group_column not in full.columns:
            raise KeyError(f"group column not found in training parquet: {group_column}")
        group_labels = full[group_column].to_numpy()

    splits_dir = proc / "splits"
    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if len(split_files) != settings.n_iterations:
        splits = make_splits(
            class_labels=y_str,
            n_iterations=settings.n_iterations,
            test_fraction=settings.test_fraction,
            seed_base=settings.seed_base,
            strategy=settings.split_strategy,
            group_labels=group_labels,
        )
        split_files = persist_splits(splits, splits_dir)

    all_rows: list[pd.DataFrame] = []
    for i, split_path in enumerate(tqdm(split_files, desc="iterations")):
        train_idx, test_idx = load_split(split_path)
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_int[train_idx], y_int[test_idx]
        for key in settings.models:
            model, pred, proba = _fit_predict(
                key,
                settings.seed_base + i,
                X_tr,
                y_tr,
                X_te,
                hp=settings.flat_hyperparameters,
            )
            if i == 0:
                model_path = models_dir / f"{key}_multiclass.joblib"
                joblib.dump(
                    {
                        "model": model,
                        "class_order": CLASS_10,
                        "sklearn_version": sklearn.__version__,
                        "xgboost_version": getattr(__import__("xgboost"), "__version__", "unknown"),
                        "lightgbm_version": getattr(
                            __import__("lightgbm"), "__version__", "unknown"
                        ),
                        **schema_metadata,
                    },
                    model_path,
                )
                write_run_metadata_sidecar(
                    model_path,
                    settings=settings,
                    seed=settings.seed_base + i,
                    extra={"artifact_type": "flat_multiclass_model", "model_key": key},
                )
            df = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASS_10])
            df.insert(0, "iteration", i)
            df.insert(1, "model", key)
            df.insert(2, "row_index", test_idx)
            df.insert(3, "y_true_int", y_te)
            df.insert(4, "y_pred_int", pred)
            all_rows.append(df)

    preds = pd.concat(all_rows, ignore_index=True)
    preds_path = pred_dir / "test_predictions.parquet"
    preds.to_parquet(preds_path, index=False)
    write_artifact_schema_sidecar(
        preds_path,
        {
            "artifact_type": "test_predictions",
            "class_order": CLASS_10,
            "models": list(settings.models),
            **schema_metadata,
        },
    )

    return {"predictions": preds_path, "models_dir": models_dir}


def cv_evaluate_flat(
    X: np.ndarray,
    y_int: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    models: list[str],
    seed_base: int = 42,
    hp: FlatModelHyperparameters | None = None,
    lipid_indices: list[int] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, float]:
    """Pure-Python flat CV evaluator suitable for HPO inner loops.

    Parameters
    ----------
    X
        Feature matrix already aligned to ``y_int``.
    y_int
        Integer class labels (use the canonical ``CLASS_10`` order).
    splits
        Sequence of ``(train_idx, test_idx)`` arrays — pre-computed once and
        reused across trials so seeding is identical regardless of HP.
    models
        Subset of ``["rf", "xgb", "lgbm", "cat"]`` to ensemble.
    seed_base
        Seed offset; iteration ``i`` uses ``seed_base + i``.
    hp
        Optional hyperparameter overrides; defaults reproduce the historical
        values when ``None``.
    lipid_indices
        Optional list of class indices treated as "lipid" for the lipid macro-F1
        return value. Defaults to ``[CLASS_10.index(c) for c in LIPID_CODES]``.
    progress_callback
        Optional callable invoked after each iteration with
        ``(iter_idx, partial_metrics)``. Used by Optuna for intermediate
        reports + Hyperband pruning.

    Returns
    -------
    dict[str, float]
        ``macro_f1_mean``, ``lipid_macro_f1_mean``, ``binary_f1_mean`` and the
        per-iteration fold means (suffixed ``_iter_{i}``) for diagnostics.
    """

    from sklearn.metrics import f1_score

    from .constants import LIPID_CODES

    if lipid_indices is None:
        lipid_indices = [CLASS_10.index(c) for c in LIPID_CODES if c in CLASS_10]

    iter_macro: list[float] = []
    iter_lipid: list[float] = []
    iter_binary: list[float] = []

    for i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_int[train_idx], y_int[test_idx]
        # Mean-average ensembled softprobs across the configured base models.
        proba_sum: np.ndarray | None = None
        for key in models:
            _, _, proba = _fit_predict(key, seed_base + i, X_tr, y_tr, X_te, hp=hp)
            proba_sum = proba if proba_sum is None else proba_sum + proba
        assert proba_sum is not None
        proba_avg = proba_sum / len(models)
        y_pred = proba_avg.argmax(axis=1)
        macro = float(
            f1_score(y_te, y_pred, labels=list(range(len(CLASS_10))), average="macro", zero_division=0)
        )
        lipid_macro = float(
            f1_score(y_te, y_pred, labels=lipid_indices, average="macro", zero_division=0)
        )
        is_lipid_te = np.isin(y_te, lipid_indices).astype(np.int64)
        is_lipid_pred = np.isin(y_pred, lipid_indices).astype(np.int64)
        binary = float(f1_score(is_lipid_te, is_lipid_pred, zero_division=0))
        iter_macro.append(macro)
        iter_lipid.append(lipid_macro)
        iter_binary.append(binary)
        if progress_callback is not None:
            progress_callback(
                i,
                {
                    "macro_f1_mean": float(np.mean(iter_macro)),
                    "lipid_macro_f1_mean": float(np.mean(iter_lipid)),
                    "binary_f1_mean": float(np.mean(iter_binary)),
                },
            )

    out: dict[str, float] = {
        "macro_f1_mean": float(np.mean(iter_macro)),
        "macro_f1_std": float(np.std(iter_macro, ddof=0)),
        "lipid_macro_f1_mean": float(np.mean(iter_lipid)),
        "lipid_macro_f1_std": float(np.std(iter_lipid, ddof=0)),
        "binary_f1_mean": float(np.mean(iter_binary)),
        "binary_f1_std": float(np.std(iter_binary, ddof=0)),
    }
    for i, (m, lm, b) in enumerate(zip(iter_macro, iter_lipid, iter_binary, strict=True)):
        out[f"macro_f1_iter_{i}"] = m
        out[f"lipid_macro_f1_iter_{i}"] = lm
        out[f"binary_f1_iter_{i}"] = b
    return out


def run_training(settings: Settings) -> dict[str, Path]:
    """Train models for the configured pipeline mode.

    Parameters
    ----------
    settings
        Loaded experiment configuration. Flat mode trains RF/XGB/LGBM
        multiclass models; hierarchical and composite modes delegate to their
        staged training backends.

    Returns
    -------
    dict[str, Path]
        Paths to prediction artifacts and model directories or bundles.
    """

    if settings.pipeline_mode == "composite":
        from .composite.train import run_composite_training

        return run_composite_training(settings)
    if settings.pipeline_mode == "hierarchical":
        from .hierarchical_pipeline import run_hierarchical_training

        return run_hierarchical_training(settings)
    return _run_flat_training(settings)
