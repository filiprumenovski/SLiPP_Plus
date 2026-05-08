"""Train 3 model families x N split iterations.

Flat mode (``pipeline_mode=flat``): 10-class RF/XGB/LGBM, iteration-0 joblibs,
and ``test_predictions.parquet``.

Hierarchical mode (``pipeline_mode=hierarchical``): staged lipid pipeline only;
see ``hierarchical_pipeline.run_hierarchical_training`` for artifacts
(including ``hierarchical_bundle.joblib`` and staged prediction parquet).
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
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

from .__version__ import __version__
from .artifact_schema import (
    build_feature_schema_metadata,
    write_artifact_schema_sidecar,
)
from .config import Settings
from .constants import CLASS_10
from .features import class10_labels, feature_matrix
from .splits import load_split, make_splits, persist_splits


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_model_metadata_sidecar(
    model_path: Path,
    *,
    settings: Settings,
    seed: int,
) -> Path:
    metadata = {
        "slipp_plus_version": __version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(__import__("xgboost"), "__version__", "unknown"),
        "lightgbm_version": getattr(__import__("lightgbm"), "__version__", "unknown"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "config_path": str(settings.config_path) if settings.config_path else None,
        "config_sha256": settings.config_sha256,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
    }
    sidecar_path = model_path.with_suffix(f"{model_path.suffix}.metadata.json")
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


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
            model, pred, proba = _fit_predict(key, settings.seed_base + i, X_tr, y_tr, X_te)
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
                _write_model_metadata_sidecar(
                    model_path,
                    settings=settings,
                    seed=settings.seed_base + i,
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


def run_training(settings: Settings) -> dict[str, Path]:
    if settings.pipeline_mode == "hierarchical":
        from .hierarchical_pipeline import run_hierarchical_training

        return run_hierarchical_training(settings)
    return _run_flat_training(settings)
