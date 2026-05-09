"""Learned pair-expert MoE stack over a teacher prediction surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from xgboost import __version__ as xgboost_version
except ImportError:
    xgboost_version = "unknown"

from ..artifact_schema import build_feature_schema_metadata, write_artifact_schema_sidecar
from ..config import Settings
from ..constants import CLASS_10, HIERARCHICAL_METRICS_NAME, HIERARCHICAL_PREDICTIONS_NAME
from ..ensemble import PROBA_COLUMNS
from ..features import class10_labels
from ..run_metadata import write_run_metadata_sidecar
from ..splits import load_split
from .config import CompositeExpertSettings
from .topology import composite_topology_metadata, resolve_composite_topology

PAIR_MOE_BUNDLE_VERSION = 1


def _train_pair_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
    )
    model.fit(X, y, sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    return model


def _train_local_multiclass_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    num_class: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
    )
    model.fit(X, y, sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    return model


def _train_utility_gate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> XGBClassifier | None:
    if len(np.unique(y)) < 2:
        return None
    model = XGBClassifier(
        objective="binary:logistic",
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        n_estimators=60,
        max_depth=2,
        learning_rate=0.08,
    )
    model.fit(X, y, sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    return model


def _apply_pair_expert(
    frame: pd.DataFrame,
    *,
    proba_positive: np.ndarray,
    negative_label: str,
    positive_label: str,
    margin: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = frame.copy()
    proba = out[PROBA_COLUMNS].to_numpy(dtype=np.float64)
    negative_idx = CLASS_10.index(negative_label)
    positive_idx = CLASS_10.index(positive_label)
    top2 = np.argsort(proba, axis=1)[:, -2:]
    in_scope = (top2 == negative_idx).any(axis=1) & (top2 == positive_idx).any(axis=1)
    pair_margin = np.abs(proba[:, negative_idx] - proba[:, positive_idx])
    fired = in_scope & (pair_margin <= margin)
    mass = proba[:, negative_idx] + proba[:, positive_idx]
    proba[fired, positive_idx] = mass[fired] * proba_positive[fired]
    proba[fired, negative_idx] = mass[fired] * (1.0 - proba_positive[fired])
    out[PROBA_COLUMNS] = proba
    out["y_pred_int"] = proba.argmax(axis=1).astype(np.int64)
    return out, fired


def _apply_local_multiclass_expert(
    frame: pd.DataFrame,
    *,
    local_proba: np.ndarray,
    labels: tuple[str, ...],
    min_confidence: float,
    max_rank: int | None,
    gate_proba: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = frame.copy()
    proba = out[PROBA_COLUMNS].to_numpy(dtype=np.float64)
    label_idx = np.array([CLASS_10.index(label) for label in labels], dtype=np.int64)
    rank = max_rank or len(labels)
    top = np.argsort(proba, axis=1)[:, -rank:]
    in_scope = np.isin(top, label_idx).any(axis=1)
    gate_score = local_proba.max(axis=1) if gate_proba is None else gate_proba
    confident = gate_score >= min_confidence
    fired = in_scope & confident
    mass = proba[:, label_idx].sum(axis=1)
    if fired.any():
        proba[np.ix_(fired, label_idx)] = mass[fired, None] * local_proba[fired]
    out[PROBA_COLUMNS] = proba
    out["y_pred_int"] = proba.argmax(axis=1).astype(np.int64)
    return out, fired


def _auxiliary_columns(base: pd.DataFrame) -> list[str]:
    z_cols = sorted(column for column in base.columns if column.startswith("z_"))
    return PROBA_COLUMNS + z_cols


def _utility_features(
    frame: pd.DataFrame,
    *,
    local_proba: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    base = frame[PROBA_COLUMNS].to_numpy(dtype=np.float64)
    label_idx = np.array([CLASS_10.index(label) for label in labels], dtype=np.int64)
    sorted_proba = np.sort(base, axis=1)
    top1 = sorted_proba[:, -1]
    top2 = sorted_proba[:, -2]
    entropy = -(base * np.log(np.clip(base, 1e-12, 1.0))).sum(axis=1)
    candidate_mass = base[:, label_idx].sum(axis=1)
    return np.column_stack(
        [
            base[:, label_idx],
            local_proba,
            local_proba.max(axis=1),
            local_proba.argmax(axis=1),
            candidate_mass,
            top1,
            top2,
            top1 - top2,
            entropy,
        ]
    ).astype(np.float64)


def _augmented_matrix(
    raw_X: np.ndarray,
    row_indices: np.ndarray,
    lookup: pd.DataFrame | None,
    aux_columns: list[str],
) -> np.ndarray:
    if lookup is None or not aux_columns:
        return raw_X[row_indices]
    aux = lookup.reindex(row_indices)[aux_columns].fillna(0.0).to_numpy(dtype=np.float64)
    return np.hstack([raw_X[row_indices], aux])


def _split_expert_fit_gate(
    train_idx: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    gate_n = max(1, round(0.20 * len(shuffled)))
    return shuffled[gate_n:], shuffled[:gate_n]


def _metric_row(name: str, frame: pd.DataFrame) -> dict[str, float | str]:
    labels = np.arange(len(CLASS_10))
    lipid_labels = np.array([CLASS_10.index(c) for c in ("CLR", "MYR", "OLA", "PLM", "STE")])
    y_true = frame["y_true_int"].to_numpy(dtype=np.int64)
    y_pred = frame["y_pred_int"].to_numpy(dtype=np.int64)
    row: dict[str, float | str] = {
        "condition": name,
        "macro_f1_10": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1_lipid5": float(
            f1_score(y_true, y_pred, labels=lipid_labels, average="macro", zero_division=0)
        ),
    }
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for label, value in zip(CLASS_10, per_class, strict=True):
        row[f"f1_{label}"] = float(value)
    return row


def _aggregate_iteration_metrics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = [_metric_row("composite_pair_moe", frame) for frame in frames]
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    base_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    experts: list[CompositeExpertSettings],
    fire_counts: dict[str, list[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_mean = base_metrics.mean(numeric_only=True)
    final_mean = final_metrics.mean(numeric_only=True)
    base_std = base_metrics.std(numeric_only=True)
    final_std = final_metrics.std(numeric_only=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Composite Pair-MoE Results\n\n")
        handle.write(
            "| condition | 10-class macro-F1 | 5-lipid macro-F1 | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, mean, std in (
            ("teacher baseline", base_mean, base_std),
            ("composite pair-MoE", final_mean, final_std),
        ):
            handle.write(
                f"| {label} "
                f"| {mean['macro_f1_10']:.3f} +/- {std['macro_f1_10']:.3f} "
                f"| {mean['macro_f1_lipid5']:.3f} +/- {std['macro_f1_lipid5']:.3f} "
                f"| {mean['f1_CLR']:.3f} "
                f"| {mean['f1_MYR']:.3f} "
                f"| {mean['f1_OLA']:.3f} "
                f"| {mean['f1_PLM']:.3f} "
                f"| {mean['f1_STE']:.3f} |\n"
            )
        handle.write("\n## Delta\n\n")
        handle.write(
            f"- 10-class macro-F1: {final_mean['macro_f1_10'] - base_mean['macro_f1_10']:+.4f}\n"
        )
        handle.write(
            f"- 5-lipid macro-F1: {final_mean['macro_f1_lipid5'] - base_mean['macro_f1_lipid5']:+.4f}\n"
        )
        handle.write("\n## Experts\n\n")
        handle.write("| expert | labels | margin | fired mean | fired total |\n")
        handle.write("|---|---|---:|---:|---:|\n")
        for expert in experts:
            counts = fire_counts.get(expert.name, [])
            handle.write(
                f"| {expert.name} | {', '.join(expert.labels)} | {expert.margin:.2f} "
                f"| {float(np.mean(counts)) if counts else 0.0:.1f} "
                f"| {int(np.sum(counts)) if counts else 0} |\n"
            )


def run_pair_moe_training(settings: Settings) -> dict[str, Path]:
    topology = resolve_composite_topology(settings)
    experts = [
        expert
        for expert in topology.experts
        if expert.kind in {"binary_boundary", "local_multiclass"}
    ]
    unsupported = [
        expert.name
        for expert in topology.experts
        if expert.kind not in {"binary_boundary", "local_multiclass"}
    ]
    if unsupported:
        raise NotImplementedError(
            "pair-MoE training only supports binary_boundary/local_multiclass experts; unsupported="
            f"{unsupported}"
        )
    if not settings.composite.teacher_predictions_path:
        raise ValueError("composite.teacher_predictions_path is required for pair-MoE training")

    proc = settings.paths.processed_dir
    pred_dir = proc / "predictions"
    reports_dir = settings.paths.reports_dir
    models_dir = settings.paths.models_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(proc / "full_pockets.parquet")
    feature_columns = settings.feature_columns()
    X = full[feature_columns].to_numpy(dtype=np.float64)
    y_str = class10_labels(full)
    base = pd.read_parquet(settings.composite.teacher_predictions_path)
    if "model" in base.columns:
        raise ValueError("teacher_predictions_path must contain one prediction row per test row")
    aux_columns = _auxiliary_columns(base)
    aux_lookup = (
        base.groupby("row_index", as_index=True)[aux_columns].mean() if aux_columns else None
    )

    final_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    fire_counts: dict[str, list[int]] = {expert.name: [] for expert in experts}
    bundle_experts: list[dict[str, Any]] = []

    for iteration in range(settings.n_iterations):
        split_path = proc / "splits" / f"seed_{iteration:02d}.parquet"
        train_idx, _test_idx = load_split(split_path)
        frame = base.loc[base["iteration"] == iteration].copy().reset_index(drop=True)
        base_frames.append(frame.copy())
        row_index = frame["row_index"].to_numpy(dtype=np.int64)

        iter_experts: list[dict[str, Any]] = []
        for expert in experts:
            expert_train_idx = train_idx[np.isin(y_str[train_idx], list(expert.labels))]
            if len(np.unique(y_str[expert_train_idx])) < 2:
                fire_counts[expert.name].append(0)
                continue
            X_train = _augmented_matrix(X, expert_train_idx, aux_lookup, aux_columns)
            X_test = _augmented_matrix(X, row_index, aux_lookup, aux_columns)
            if expert.kind == "binary_boundary":
                negative_label, positive_label = expert.labels[0], expert.labels[1]
                y_pair = (y_str[expert_train_idx] == positive_label).astype(np.int64)
                model = _train_pair_model(
                    X_train,
                    y_pair,
                    seed=settings.seed_base + iteration,
                )
                proba_positive = model.predict_proba(X_test)[:, 1].astype(np.float64)
                frame, fired = _apply_pair_expert(
                    frame,
                    proba_positive=proba_positive,
                    negative_label=negative_label,
                    positive_label=positive_label,
                    margin=expert.margin,
                )
            else:
                local_to_int = {label: i for i, label in enumerate(expert.labels)}
                gate_model = None
                if expert.gate == "utility" and aux_lookup is not None:
                    fit_idx, gate_idx = _split_expert_fit_gate(
                        expert_train_idx,
                        seed=settings.seed_base + iteration,
                    )
                    gate_idx = gate_idx[np.isin(gate_idx, aux_lookup.index.to_numpy())]
                    if (
                        len(gate_idx) > 0
                        and len(np.unique(y_str[fit_idx])) >= 2
                        and len(np.unique(y_str[gate_idx])) >= 2
                    ):
                        fit_y = np.array(
                            [local_to_int[str(label)] for label in y_str[fit_idx]],
                            dtype=np.int64,
                        )
                        fit_model = _train_local_multiclass_model(
                            _augmented_matrix(X, fit_idx, aux_lookup, aux_columns),
                            fit_y,
                            seed=settings.seed_base + iteration,
                            num_class=len(expert.labels),
                        )
                        gate_frame = aux_lookup.loc[gate_idx].reset_index()
                        gate_frame.insert(0, "iteration", iteration)
                        gate_frame.insert(
                            2,
                            "y_true_int",
                            [CLASS_10.index(str(label)) for label in y_str[gate_idx]],
                        )
                        gate_base_proba = gate_frame[PROBA_COLUMNS].to_numpy(dtype=np.float64)
                        gate_frame["y_pred_int"] = gate_base_proba.argmax(axis=1).astype(np.int64)
                        gate_local_proba = fit_model.predict_proba(
                            _augmented_matrix(X, gate_idx, aux_lookup, aux_columns)
                        ).astype(np.float64)
                        candidate_frame, gate_candidate_fired = _apply_local_multiclass_expert(
                            gate_frame,
                            local_proba=gate_local_proba,
                            labels=tuple(expert.labels),
                            min_confidence=0.0,
                            max_rank=expert.max_rank,
                        )
                        gate_target = (
                            candidate_frame["y_pred_int"].to_numpy(dtype=np.int64)
                            == gate_frame["y_true_int"].to_numpy(dtype=np.int64)
                        ).astype(np.int64)
                        gate_features = _utility_features(
                            gate_frame,
                            local_proba=gate_local_proba,
                            labels=tuple(expert.labels),
                        )
                        gate_model = _train_utility_gate(
                            gate_features[gate_candidate_fired],
                            gate_target[gate_candidate_fired],
                            seed=settings.seed_base + iteration,
                        )

                y_local = np.array(
                    [local_to_int[str(label)] for label in y_str[expert_train_idx]],
                    dtype=np.int64,
                )
                model = _train_local_multiclass_model(
                    X_train,
                    y_local,
                    seed=settings.seed_base + iteration,
                    num_class=len(expert.labels),
                )
                local_proba = model.predict_proba(X_test).astype(np.float64)
                gate_proba = None
                if gate_model is not None:
                    gate_features_test = _utility_features(
                        frame,
                        local_proba=local_proba,
                        labels=tuple(expert.labels),
                    )
                    gate_proba = gate_model.predict_proba(gate_features_test)[:, 1].astype(
                        np.float64
                    )
                frame, fired = _apply_local_multiclass_expert(
                    frame,
                    local_proba=local_proba,
                    labels=tuple(expert.labels),
                    min_confidence=expert.margin,
                    max_rank=expert.max_rank,
                    gate_proba=gate_proba,
                )
            frame[f"moe_{expert.name}_fired"] = fired
            fire_counts[expert.name].append(int(fired.sum()))
            if iteration == 0:
                iter_experts.append(
                    {
                        "name": expert.name,
                        "model": model,
                        "labels": tuple(expert.labels),
                        "margin": expert.margin,
                        "kind": expert.kind,
                        "feature_columns": feature_columns + aux_columns,
                    }
                )
        if iteration == 0:
            bundle_experts = iter_experts
        final_frames.append(frame)

    predictions = pd.concat(final_frames, ignore_index=True)
    output_predictions = pred_dir / HIERARCHICAL_PREDICTIONS_NAME
    predictions.to_parquet(output_predictions, index=False)

    schema_metadata = build_feature_schema_metadata(
        feature_set=settings.feature_set,
        feature_columns=feature_columns,
    )
    metadata = {
        "artifact_type": "composite_pair_moe_predictions",
        "pipeline_mode": "composite",
        "class_order": CLASS_10,
        **schema_metadata,
        **composite_topology_metadata(settings),
    }
    write_artifact_schema_sidecar(output_predictions, metadata)

    base_metrics = _aggregate_iteration_metrics(base_frames)
    final_metrics = _aggregate_iteration_metrics(final_frames)
    output_metrics = reports_dir / HIERARCHICAL_METRICS_NAME
    final_metrics.to_parquet(output_metrics, index=False)
    report_path = reports_dir / "composite_pair_moe_report.md"
    _write_report(
        report_path,
        base_metrics=base_metrics,
        final_metrics=final_metrics,
        experts=experts,
        fire_counts=fire_counts,
    )

    bundle_path = models_dir / settings.hierarchical.bundle_name
    joblib.dump(
        {
            "bundle_version": PAIR_MOE_BUNDLE_VERSION,
            "pipeline_mode": "composite",
            "composite_backend": "pair_moe",
            "class_order": CLASS_10,
            "experts": bundle_experts,
            "settings_snapshot": settings.model_dump(mode="python"),
            "library_versions": {
                "sklearn": sklearn.__version__,
                "xgboost": str(xgboost_version),
            },
            **schema_metadata,
            **composite_topology_metadata(settings),
        },
        bundle_path,
    )
    write_artifact_schema_sidecar(
        bundle_path, metadata | {"artifact_type": "composite_pair_moe_bundle"}
    )
    write_run_metadata_sidecar(
        bundle_path,
        settings,
        seed=settings.seed_base,
        extra={"artifact_type": "composite_pair_moe_bundle"},
    )

    return {
        "predictions": output_predictions,
        "models_dir": models_dir,
        "hierarchical_metrics": output_metrics,
        "hierarchical_report": report_path,
        "hierarchical_bundle": bundle_path,
        "composite_predictions": output_predictions,
        "composite_bundle": bundle_path,
    }
