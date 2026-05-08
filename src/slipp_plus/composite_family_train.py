"""Composite training backend for the PyTorch feature-family encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import sklearn

from .artifact_schema import build_feature_schema_metadata, write_artifact_schema_sidecar
from .backbone_family_encoder import (
    FamilyEncoderNet,
    FamilyEncoderTrainConfig,
    fit_family_encoder,
    predict_family_encoder_outputs,
    predict_family_encoder_proba,
)
from .composite_topology import composite_topology_metadata, resolve_composite_topology
from .config import Settings
from .constants import CLASS_10
from .ensemble import PROBA_COLUMNS
from .feature_families import (
    FamilyScaler,
    FeatureFamilySpec,
    fit_family_scalers,
    materialize_family_arrays,
    resolve_family_specs,
)
from .features import class10_labels
from .splits import load_split


def _test_prediction_frame(
    *,
    iteration: int,
    row_index: np.ndarray,
    y_true_int: np.ndarray,
    proba: np.ndarray,
    z: np.ndarray,
) -> pl.DataFrame:
    columns: dict[str, np.ndarray | int] = {
        "iteration": np.full(len(row_index), iteration, dtype=np.int64),
        "row_index": row_index.astype(np.int64, copy=False),
        "y_true_int": y_true_int.astype(np.int64, copy=False),
    }
    columns.update({name: proba[:, i] for i, name in enumerate(PROBA_COLUMNS)})
    columns["y_pred_int"] = proba.argmax(axis=1).astype(np.int64)
    columns.update({f"z_{i:03d}": z[:, i] for i in range(z.shape[1])})
    return pl.DataFrame(columns)


def _teacher_by_iteration(path: Path | None) -> dict[int, pd.DataFrame]:
    if path is None:
        return {}
    teacher = pd.read_parquet(path)
    if "model" in teacher.columns:
        proba_cols = [f"p_{label}" for label in CLASS_10]
        teacher = (
            teacher.groupby(["iteration", "row_index", "y_true_int"], as_index=False)[proba_cols]
            .mean()
            .assign(y_pred_int=lambda df: df[proba_cols].to_numpy().argmax(axis=1))
        )
    return {
        int(iteration): frame.reset_index(drop=True)
        for iteration, frame in teacher.groupby("iteration")
    }


def _oof_teacher_lookup(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    teacher = pd.read_parquet(path)
    if "model" in teacher.columns:
        teacher = (
            teacher.groupby(["iteration", "row_index", "y_true_int"], as_index=False)[PROBA_COLUMNS]
            .mean()
            .assign(y_pred_int=lambda df: df[PROBA_COLUMNS].to_numpy().argmax(axis=1))
        )
    return teacher.groupby("row_index", as_index=True)[PROBA_COLUMNS].mean()


def _train_val_split(train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    val_n = max(1, round(0.10 * len(shuffled)))
    return shuffled[val_n:], shuffled[:val_n]


def run_family_encoder_training(settings: Settings) -> dict[str, Path]:
    """Train and persist the composite family-encoder backbone.

    Parameters
    ----------
    settings
        Loaded experiment configuration. ``settings.pipeline_mode`` should be
        ``"composite"`` and ``settings.composite.backbone.kind`` should be
        ``"family_encoder"``.

    Returns
    -------
    dict[str, Path]
        Paths to prediction parquet files, model directory, and persisted
        family-encoder bundle.
    """

    topology = resolve_composite_topology(settings)
    specs = resolve_family_specs(tuple(topology.backbone.feature_families))
    proc = settings.paths.processed_dir
    pred_dir = proc / "predictions"
    reports_dir = settings.paths.reports_dir
    models_dir = settings.paths.models_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(proc / "full_pockets.parquet")
    y_str = class10_labels(full)
    class_to_int = {label: i for i, label in enumerate(CLASS_10)}
    y_int = np.array([class_to_int[str(label)] for label in y_str], dtype=np.int64)
    teacher_lookup = _oof_teacher_lookup(settings.composite.teacher_predictions_path)

    all_frames: list[pl.DataFrame] = []
    bundle_payload: dict[str, Any] | None = None
    cfg = FamilyEncoderTrainConfig()
    for iteration in range(settings.n_iterations):
        train_idx, test_idx = load_split(proc / "splits" / f"seed_{iteration:02d}.parquet")
        fit_idx, val_idx = _train_val_split(train_idx, settings.seed_base + iteration)
        scalers = fit_family_scalers(full, fit_idx, specs)
        arrays, masks = materialize_family_arrays(full, specs, scalers)
        teacher_train = None
        teacher_mask_train = None
        teacher_val = None
        teacher_mask_val = None
        if teacher_lookup is not None:
            available = np.isin(fit_idx, teacher_lookup.index.to_numpy())
            teacher_train = np.zeros((len(fit_idx), len(CLASS_10)), dtype=np.float32)
            if available.any():
                teacher_train[available] = teacher_lookup.loc[fit_idx[available]].to_numpy(
                    dtype=np.float32
                )
            teacher_mask_train = available.astype(np.float32)
            val_available = np.isin(val_idx, teacher_lookup.index.to_numpy())
            teacher_val = np.zeros((len(val_idx), len(CLASS_10)), dtype=np.float32)
            if val_available.any():
                teacher_val[val_available] = teacher_lookup.loc[val_idx[val_available]].to_numpy(
                    dtype=np.float32
                )
            teacher_mask_val = val_available.astype(np.float32)

        model = fit_family_encoder(
            {name: values[fit_idx] for name, values in arrays.items()},
            masks[fit_idx],
            y_int[fit_idx],
            val_arrays={name: values[val_idx] for name, values in arrays.items()},
            val_masks=masks[val_idx],
            y_val=y_int[val_idx],
            teacher_train=teacher_train,
            teacher_mask_train=teacher_mask_train,
            teacher_val=teacher_val,
            teacher_mask_val=teacher_mask_val,
            config=cfg,
            seed=settings.seed_base + iteration,
        )
        proba, z = predict_family_encoder_outputs(
            model,
            {name: values[test_idx] for name, values in arrays.items()},
            masks[test_idx],
        )
        all_frames.append(
            _test_prediction_frame(
                iteration=iteration,
                row_index=test_idx,
                y_true_int=y_int[test_idx],
                proba=proba,
                z=z,
            )
        )

        if iteration == 0:
            bundle_payload = {
                "backend": "family_encoder",
                "model_state_dict": {
                    key: value.detach().cpu().numpy()
                    for key, value in model.state_dict().items()
                },
                "train_config": cfg.__dict__,
                "family_specs": [
                    {
                        "name": spec.name,
                        "columns": list(spec.columns),
                        "optional": spec.optional,
                    }
                    for spec in specs
                ],
                "family_scalers": {
                    name: {"mean": scaler.mean, "scale": scaler.scale}
                    for name, scaler in scalers.items()
                },
            }

    predictions = pl.concat(all_frames, how="vertical")
    output_predictions = pred_dir / "hierarchical_lipid_predictions.parquet"
    predictions.write_parquet(output_predictions)
    family_predictions = pred_dir / "family_encoder_predictions.parquet"
    predictions.write_parquet(family_predictions)
    feature_columns = settings.feature_columns()
    schema_metadata = build_feature_schema_metadata(
        feature_set=settings.feature_set,
        feature_columns=feature_columns,
    )
    metadata = {
        "artifact_type": "family_encoder_predictions",
        "pipeline_mode": "composite",
        "class_order": CLASS_10,
        **schema_metadata,
        **composite_topology_metadata(settings),
    }
    write_artifact_schema_sidecar(output_predictions, metadata)
    write_artifact_schema_sidecar(family_predictions, metadata)

    bundle_path = models_dir / settings.hierarchical.bundle_name
    joblib.dump(
        {
            "bundle_version": 1,
            "pipeline_mode": "composite",
            "class_order": CLASS_10,
            "library_versions": {"sklearn": sklearn.__version__},
            **(bundle_payload or {}),
            **schema_metadata,
            **composite_topology_metadata(settings),
        },
        bundle_path,
    )
    write_artifact_schema_sidecar(bundle_path, metadata | {"artifact_type": "family_encoder_bundle"})

    return {
        "predictions": output_predictions,
        "family_encoder_predictions": family_predictions,
        "models_dir": models_dir,
        "hierarchical_bundle": bundle_path,
        "composite_predictions": output_predictions,
        "composite_bundle": bundle_path,
    }


def predict_family_encoder_holdout(
    holdout_df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Predict holdout probabilities from an iteration-0 family encoder bundle.

    Parameters
    ----------
    holdout_df
        Holdout feature table with the columns required by the bundle's family
        specifications.
    bundle
        Joblib-loaded family encoder bundle written by
        ``run_family_encoder_training``.

    Returns
    -------
    pd.DataFrame
        Prediction table containing probabilities in ``PROBA_COLUMNS`` order
        plus iteration, row index, placeholder true label, and predicted class.
    """

    spec_payload = bundle["family_specs"]
    specs = [
        FeatureFamilySpec(
            name=item["name"],
            columns=tuple(item["columns"]),
            optional=bool(item.get("optional", False)),
        )
        for item in spec_payload
    ]
    scalers = {
        name: FamilyScaler(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )
        for name, payload in bundle["family_scalers"].items()
    }
    import torch

    torch.set_num_threads(1)
    arrays, masks = materialize_family_arrays(holdout_df, specs, scalers)
    family_dims = {spec.name: len(spec.columns) for spec in specs}
    train_config = FamilyEncoderTrainConfig(**bundle.get("train_config", {}))
    model = FamilyEncoderNet(
        family_dims,
        token_dim=train_config.token_dim,
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
    )
    state_dict = {
        key: torch.as_tensor(np.asarray(value))
        for key, value in bundle["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    proba = predict_family_encoder_proba(model, arrays, masks)
    out = pd.DataFrame(proba, columns=PROBA_COLUMNS)
    out.insert(0, "iteration", 0)
    out.insert(1, "row_index", np.arange(len(holdout_df), dtype=np.int64))
    out.insert(2, "y_true_int", -1)
    out["y_pred_int"] = proba.argmax(axis=1).astype(np.int64)
    return out
