"""Train 3 model families x N stratified-shuffle iterations, 10-class softmax.

Iteration 0 models are persisted to ``models/`` for later interpretation and
holdout scoring. All iterations' predictions + metadata are written to
``processed/predictions/`` as parquet, consumed by ``evaluate.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from .config import Settings
from .constants import CLASS_10
from .features import class10_labels, feature_matrix
from .splits import load_split, make_splits, persist_splits


def _build_model(key: str, seed: int) -> Any:
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if key == "xgb":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_10),
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
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"unknown model key: {key}")


def _fit_predict(
    key: str, seed: int, X_tr: np.ndarray, y_tr_int: np.ndarray,
    X_te: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit and return (model, pred_int, pred_proba) on test set."""
    model = _build_model(key, seed)
    if key == "xgb":
        sw = compute_sample_weight(class_weight="balanced", y=y_tr_int)
        model.fit(X_tr, y_tr_int, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr_int)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    return model, pred.astype(np.int64), proba.astype(np.float64)


def run_training(settings: Settings) -> dict[str, Path]:
    paths = settings.paths
    proc = paths.processed_dir
    models_dir = paths.models_dir
    pred_dir = proc / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(proc / "full_pockets.parquet")
    X = feature_matrix(full, settings)
    y_str = class10_labels(full)
    class_to_int = {c: i for i, c in enumerate(CLASS_10)}
    y_int = np.array([class_to_int[c] for c in y_str], dtype=np.int64)

    splits_dir = proc / "splits"
    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if len(split_files) != settings.n_iterations:
        splits = make_splits(
            class_labels=y_str,
            n_iterations=settings.n_iterations,
            test_fraction=settings.test_fraction,
            seed_base=settings.seed_base,
        )
        split_files = persist_splits(splits, splits_dir)

    all_rows: list[pd.DataFrame] = []
    for i, split_path in enumerate(tqdm(split_files, desc="iterations")):
        train_idx, test_idx = load_split(split_path)
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_int[train_idx], y_int[test_idx]
        for key in settings.models:
            model, pred, proba = _fit_predict(key, settings.seed_base + i,
                                              X_tr, y_tr, X_te)
            if i == 0:
                joblib.dump(
                    {
                        "model": model,
                        "class_order": CLASS_10,
                        "feature_set": settings.feature_set,
                        "feature_columns": settings.feature_columns(),
                    },
                    models_dir / f"{key}_multiclass.joblib",
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

    return {"predictions": preds_path, "models_dir": models_dir}
