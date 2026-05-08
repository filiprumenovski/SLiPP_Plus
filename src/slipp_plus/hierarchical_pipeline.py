"""Primary hierarchical lipid pipeline: train, persist bundle, predict."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

try:
    from xgboost import __version__ as xgboost_version
except ImportError:
    xgboost_version = "unknown"

from .artifact_schema import (
    build_feature_schema_metadata,
    validate_feature_schema_metadata,
    write_artifact_schema_sidecar,
)
from .boundary_head import (
    BoundaryRule,
    apply_boundary_head,
    build_boundary_training,
    gain_importance,
    train_boundary_head,
)
from .config import Settings
from .constants import (
    CLASS_10,
    FEATURE_SETS,
    HIERARCHICAL_METRICS_NAME,
    HIERARCHICAL_PREDICTIONS_NAME,
    HIERARCHICAL_REPORT_NAME,
)
from .ensemble import PROBA_COLUMNS, score_summary
from .features import class10_labels
from .hierarchical_experiment import (
    LIPID_LABELS,
    NONLIPID_LABELS,
    _binary_f1,
    _gain_importance,
    _metric_row,
    _write_report,
    build_specialist_training,
    combine_hierarchical_softprobs,
    predict_lipid_binary_heads,
    train_lipid_binary_heads,
    train_lipid_family,
    train_lipid_gate,
    train_nonlipid_family,
    train_one_vs_neighbors,
)
from .hierarchical_postprocess import OneVsNeighborsRule, apply_one_vs_neighbors
from .specialist_utility_gate import (
    UtilityGateConfig,
    apply_utility_gate,
    deserialize_utility_model,
    fit_utility_gate,
    serialize_utility_model,
)
from .splits import load_split, make_splits, persist_splits
from .train import _fit_predict

BUNDLE_VERSION = 1

_LIPID_CLASS_IDX = np.array([CLASS_10.index(c) for c in LIPID_LABELS], dtype=np.int64)


def _feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"frame missing feature columns: {missing}")
    return df[feature_columns].to_numpy(dtype=np.float64)


def _resolve_feature_columns(
    primary_feature_columns: list[str],
    feature_set: str | None,
) -> list[str]:
    if feature_set is None:
        return list(primary_feature_columns)
    return list(FEATURE_SETS[feature_set])


def _union_feature_columns(bundle: Mapping[str, Any]) -> list[str]:
    columns: list[str] = list(bundle["feature_columns"])
    for key in ("stage2_feature_columns", "nonlipid_feature_columns", "stage3_feature_columns"):
        for column in bundle.get(key, []) or []:
            if column not in columns:
                columns.append(column)
    for item in bundle.get("boundary_heads", []):
        for column in item.get("feature_columns", []) or []:
            if column not in columns:
                columns.append(column)
    return columns


def _library_versions() -> dict[str, str]:
    return {
        "sklearn": sklearn.__version__,
        "xgboost": str(xgboost_version),
    }


def combine_hierarchical_softprobs_from_bundle(
    *,
    df: pd.DataFrame,
    bundle: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_lipid, lipid_family_proba, nonlipid_family_proba) for feature rows."""

    stage1_source = bundle["stage1_source"]
    primary_feature_columns: list[str] = list(bundle["feature_columns"])
    X_stage1 = _feature_matrix(df, primary_feature_columns)
    if stage1_source == "ensemble":
        models: dict[str, Any] = bundle["stage1_ensemble_models"]
        keys: tuple[str, ...] = bundle["flat_model_keys"]
        probas = np.stack([models[k].predict_proba(X_stage1) for k in keys], axis=0)
        mean = probas.mean(axis=0).astype(np.float64)
        p_lipid = mean[:, _LIPID_CLASS_IDX].sum(axis=1)
    elif stage1_source == "trained":
        gate = bundle["stage1_model"]
        if gate is None:
            raise ValueError("hierarchical bundle missing stage1_model for trained gate")
        p_lipid = gate.predict_proba(X_stage1)[:, 1].astype(np.float64)
    else:
        raise ValueError(f"unknown stage1_source: {stage1_source!r}")

    lipid_family_mode = str(bundle.get("stage2_lipid_family_mode", "softmax"))
    X_stage2 = _feature_matrix(
        df,
        list(bundle.get("stage2_feature_columns") or primary_feature_columns),
    )
    if lipid_family_mode == "binary_ovr":
        family_models: dict[str, Any] = bundle["stage2_models"]
        lipid_family_proba = predict_lipid_binary_heads(family_models, X_stage2)
    else:
        family = bundle["stage2_model"]
        lipid_family_proba = family.predict_proba(X_stage2).astype(np.float64)
    nonlipid = bundle["nonlipid_model"]
    X_nonlipid = _feature_matrix(
        df,
        list(bundle.get("nonlipid_feature_columns") or primary_feature_columns),
    )
    nonlipid_family_proba = nonlipid.predict_proba(X_nonlipid).astype(np.float64)
    return p_lipid, lipid_family_proba, nonlipid_family_proba


def predict_hierarchical_holdout(
    *,
    holdout_df: pd.DataFrame,
    bundle: Mapping[str, Any],
    expected_feature_columns: list[str] | None = None,
    expected_feature_set: str | None = None,
) -> pd.DataFrame:
    """Apply iteration-0 hierarchical bundle to holdout rows (no retraining)."""

    if bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"unsupported hierarchical bundle_version {bundle.get('bundle_version')!r}; "
            f"expected {BUNDLE_VERSION}"
        )
    if bundle.get("pipeline_mode") != "hierarchical":
        raise ValueError("artifact is not a hierarchical bundle")

    if expected_feature_columns is not None:
        validate_feature_schema_metadata(
            bundle,
            expected_feature_columns=expected_feature_columns,
            expected_feature_set=expected_feature_set,
            artifact_label="hierarchical_bundle",
        )

    feature_columns = _union_feature_columns(bundle)
    missing = [c for c in feature_columns if c not in holdout_df.columns]
    if missing:
        raise KeyError(f"holdout missing feature columns: {missing}")

    p_lipid, lipid_family_proba, nonlipid_proba = combine_hierarchical_softprobs_from_bundle(
        df=holdout_df, bundle=bundle
    )
    staged_np = combine_hierarchical_softprobs(p_lipid, lipid_family_proba, nonlipid_proba)
    n = len(holdout_df)
    row_index_lookup = np.arange(n, dtype=np.int64)
    p_lipid_clip = np.clip(p_lipid.astype(np.float64), 0.0, 1.0)

    staged_pl = pl.DataFrame(
        {
            "iteration": np.zeros(n, dtype=np.int64),
            "row_index": row_index_lookup,
            "y_true_int": np.full(n, -1, dtype=np.int64),
            **{PROBA_COLUMNS[i]: staged_np[:, i] for i in range(len(PROBA_COLUMNS))},
            "y_pred_int": staged_np.argmax(axis=1).astype(np.int64),
            "stage1_p_lipid": p_lipid_clip,
            "stage2_y_pred_lipid": lipid_family_proba.argmax(axis=1).astype(np.int64),
        }
    )

    specialist: XGBClassifier = bundle["stage3_model"]
    rule: OneVsNeighborsRule = bundle["stage3_rule"]
    X_stage3 = _feature_matrix(
        holdout_df,
        list(bundle.get("stage3_feature_columns") or list(bundle["feature_columns"])),
    )
    p_specialist = specialist.predict_proba(X_stage3)[:, 1].astype(np.float64)
    candidate_frame = apply_one_vs_neighbors(
        staged_pl,
        p_specialist,
        row_index_lookup,
        rule,
    )
    gate_mode = str(bundle.get("stage3_specialist_gate_mode", "heuristic"))
    if gate_mode == "utility":
        utility_model = deserialize_utility_model(bundle.get("stage3_utility_model"))
        utility_config = UtilityGateConfig(
            threshold_default=float(bundle.get("stage3_utility_threshold_default", 0.50)),
            threshold_top1_plm=(
                None
                if bundle.get("stage3_utility_threshold_top1_plm") is None
                else float(bundle.get("stage3_utility_threshold_top1_plm"))
            ),
        )
        specialist_frame = apply_utility_gate(
            staged_df=staged_pl,
            candidate_df=candidate_frame,
            p_specialist=p_specialist,
            rule=rule,
            utility_model=utility_model,
            config=utility_config,
        )
    else:
        specialist_frame = candidate_frame
    for item in bundle.get("boundary_heads", []):
        boundary_rule: BoundaryRule = item["rule"]
        boundary_model = item["model"]
        boundary_feature_columns = list(item.get("feature_columns") or list(bundle["feature_columns"]))
        X_boundary = _feature_matrix(holdout_df, boundary_feature_columns)
        p_boundary = boundary_model.predict_proba(X_boundary)[:, 1].astype(np.float64)
        specialist_frame = apply_boundary_head(
            specialist_frame,
            p_boundary,
            row_index_lookup,
            boundary_rule,
        )
    return specialist_frame.to_pandas()


def load_hierarchical_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def save_hierarchical_bundle(bundle: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), path)
    return path


def _primary_training_worker(
    *,
    iteration: int,
    split_path: Path,
    full_pockets_path: Path,
    feature_columns: list[str],
    seed: int,
    specialist_rule: OneVsNeighborsRule,
    stage1_source: str,
    persist_importance: bool,
    flat_model_keys: tuple[str, ...],
    nonlipid_source: str,
    specialist_gate: str,
    utility_threshold_default: float,
    lipid_family_mode: str,
    utility_threshold_top1_plm: float | None,
    lipid_family_feature_columns: list[str],
    specialist_feature_columns: list[str],
    nonlipid_feature_columns: list[str],
    boundary_specs: tuple[tuple[BoundaryRule, list[str], str | None], ...],
) -> dict[str, Any]:
    if stage1_source not in {"ensemble", "trained"}:
        raise ValueError("stage1_source must be 'ensemble' or 'trained'")
    if nonlipid_source != "dedicated_head":
        raise ValueError("only nonlipid_source='dedicated_head' is implemented")

    full = pd.read_parquet(full_pockets_path)
    train_idx, test_idx = load_split(split_path)
    X_all = full[feature_columns].to_numpy(dtype=np.float64)
    X_lipid_family = _feature_matrix(full, lipid_family_feature_columns)
    X_specialist = _feature_matrix(full, specialist_feature_columns)
    X_nonlipid = _feature_matrix(full, nonlipid_feature_columns)
    y_str = full["class_10"].to_numpy()
    class_to_int = {c: i for i, c in enumerate(CLASS_10)}
    y_int = np.array([class_to_int[str(c)] for c in y_str], dtype=np.int64)

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr_int, y_te_int = y_int[train_idx], y_int[test_idx]
    X_te_stage2 = X_lipid_family[test_idx]
    X_te_stage3 = X_specialist[test_idx]

    flat_probas: list[np.ndarray] = []
    flat_models: dict[str, Any] = {}
    for key in flat_model_keys:
        model, _pred, proba = _fit_predict(key, seed, X_tr, y_tr_int, X_te)
        flat_models[key] = model
        flat_probas.append(proba)
    ensemble_mean_te = np.mean(np.stack(flat_probas, axis=0), axis=0).astype(np.float64)
    ensemble_pred_te = ensemble_mean_te.argmax(axis=1).astype(np.int64)

    ensemble_frame = pl.DataFrame(
        {
            "iteration": pl.Series([iteration] * len(test_idx), dtype=pl.Int64),
            "row_index": test_idx.astype(np.int64),
            "y_true_int": y_te_int,
            **{
                PROBA_COLUMNS[i]: ensemble_mean_te[:, i]
                for i in range(len(PROBA_COLUMNS))
            },
            "y_pred_int": ensemble_pred_te,
        }
    )

    y_bin_all = np.isin(y_str, list(LIPID_LABELS)).astype(np.int64)
    gate = None
    p_lipid: np.ndarray | None = None
    stage1_binary_f1: float | None = None
    if stage1_source == "trained":
        gate = train_lipid_gate(X_all[train_idx], y_bin_all[train_idx], seed=seed)
        p_lipid = gate.predict_proba(X_te)[:, 1].astype(np.float64)
        stage1_binary_f1 = _binary_f1(y_bin_all[test_idx], p_lipid)
    else:
        p_lipid = ensemble_mean_te[:, _LIPID_CLASS_IDX].sum(axis=1)

    lipid_to_int = {label: i for i, label in enumerate(LIPID_LABELS)}
    lipid_train = train_idx[np.isin(y_str[train_idx], list(LIPID_LABELS))]
    y_family = np.array(
        [lipid_to_int[str(label)] for label in y_str[lipid_train]], dtype=np.int64
    )
    family_models: dict[str, Any] | None = None
    family: Any = None
    if lipid_family_mode == "binary_ovr":
        family_models = train_lipid_binary_heads(X_lipid_family[lipid_train], y_family, seed=seed)
        lipid_family_proba = predict_lipid_binary_heads(family_models, X_te_stage2)
    else:
        family = train_lipid_family(X_lipid_family[lipid_train], y_family, seed=seed)
        lipid_family_proba = family.predict_proba(X_te_stage2).astype(np.float64)

    nonlipid_to_int = {label: i for i, label in enumerate(NONLIPID_LABELS)}
    nonlipid_train = train_idx[np.isin(y_str[train_idx], list(NONLIPID_LABELS))]
    if len(nonlipid_train) == 0:
        raise ValueError("no non-lipid training rows for nonlipid head")
    y_nonlipid = np.array(
        [nonlipid_to_int[str(label)] for label in y_str[nonlipid_train]], dtype=np.int64
    )
    nonlipid_model = train_nonlipid_family(
        X_nonlipid[nonlipid_train], y_nonlipid, seed=seed
    )
    nonlipid_proba_te = nonlipid_model.predict_proba(X_nonlipid[test_idx]).astype(np.float64)

    X_spec, y_spec, _ = build_specialist_training(
        full,
        specialist_feature_columns,
        split_path,
        specialist_rule,
    )
    specialist = train_one_vs_neighbors(X_spec, y_spec, seed=seed)
    p_specialist = specialist.predict_proba(X_te_stage3)[:, 1].astype(np.float64)

    if stage1_source == "trained":
        p_lipid_tr = gate.predict_proba(X_tr)[:, 1].astype(np.float64) if gate is not None else np.zeros(len(X_tr))
    else:
        ensemble_mean_tr = np.mean(
            np.stack([flat_models[key].predict_proba(X_tr) for key in flat_model_keys], axis=0),
            axis=0,
        ).astype(np.float64)
        p_lipid_tr = ensemble_mean_tr[:, _LIPID_CLASS_IDX].sum(axis=1)
    if lipid_family_mode == "binary_ovr":
        lipid_family_proba_tr = predict_lipid_binary_heads(family_models, X_lipid_family[train_idx])
    else:
        lipid_family_proba_tr = family.predict_proba(X_lipid_family[train_idx]).astype(np.float64)
    nonlipid_proba_tr = nonlipid_model.predict_proba(X_nonlipid[train_idx]).astype(np.float64)
    staged_np_tr = combine_hierarchical_softprobs(
        p_lipid_tr,
        lipid_family_proba_tr,
        nonlipid_proba_tr,
    )
    staged_pl_tr = pl.DataFrame(
        {
            "iteration": pl.Series([iteration] * len(train_idx), dtype=pl.Int64),
            "row_index": train_idx.astype(np.int64),
            "y_true_int": y_tr_int,
            **{PROBA_COLUMNS[i]: staged_np_tr[:, i] for i in range(len(PROBA_COLUMNS))},
            "y_pred_int": staged_np_tr.argmax(axis=1).astype(np.int64),
        }
    )
    p_specialist_tr = specialist.predict_proba(X_specialist[train_idx])[:, 1].astype(np.float64)
    candidate_tr = apply_one_vs_neighbors(
        staged_pl_tr,
        p_specialist_tr,
        train_idx.astype(np.int64),
        specialist_rule,
    )

    staged_np = combine_hierarchical_softprobs(p_lipid, lipid_family_proba, nonlipid_proba_te)
    p_lipid_clip = np.clip(p_lipid.astype(np.float64), 0.0, 1.0)
    staged_pl = pl.DataFrame(
        {
            "iteration": pl.Series([iteration] * len(test_idx), dtype=pl.Int64),
            "row_index": test_idx.astype(np.int64),
            "y_true_int": y_te_int,
            **{PROBA_COLUMNS[i]: staged_np[:, i] for i in range(len(PROBA_COLUMNS))},
            "y_pred_int": staged_np.argmax(axis=1).astype(np.int64),
            "stage1_p_lipid": p_lipid_clip,
            "stage2_y_pred_lipid": lipid_family_proba.argmax(axis=1).astype(np.int64),
        }
    )
    candidate_frame = apply_one_vs_neighbors(
        staged_pl,
        p_specialist,
        test_idx.astype(np.int64),
        specialist_rule,
    )
    utility_model = fit_utility_gate(
        staged_df=staged_pl_tr,
        candidate_df=candidate_tr,
        p_specialist=p_specialist_tr,
        rule=specialist_rule,
    )
    utility_config = UtilityGateConfig(
        threshold_default=float(utility_threshold_default),
        threshold_top1_plm=utility_threshold_top1_plm,
    )
    if specialist_gate == "utility":
        specialist_frame = apply_utility_gate(
            staged_df=staged_pl,
            candidate_df=candidate_frame,
            p_specialist=p_specialist,
            rule=specialist_rule,
            utility_model=utility_model,
            config=utility_config,
        )
    else:
        specialist_frame = candidate_frame

    boundary_models: list[dict[str, Any]] = []
    boundary_frame = specialist_frame
    for boundary_rule, boundary_feature_columns, boundary_feature_set in boundary_specs:
        X_boundary, y_boundary, _X_te_boundary, _y_te_boundary, _te_boundary_idx = (
            build_boundary_training(
                full,
                boundary_feature_columns,
                split_path,
                boundary_rule,
            )
        )
        boundary_model = train_boundary_head(
            X_boundary,
            y_boundary,
            seed=seed,
        )
        X_te_boundary = _feature_matrix(full.iloc[test_idx], boundary_feature_columns)
        p_boundary = boundary_model.predict_proba(X_te_boundary)[:, 1].astype(np.float64)
        boundary_frame = apply_boundary_head(
            boundary_frame,
            p_boundary,
            test_idx.astype(np.int64),
            boundary_rule,
        )
        boundary_models.append(
            {
                "rule": boundary_rule,
                "model": boundary_model,
                "feature_columns": list(boundary_feature_columns),
                "feature_set": boundary_feature_set,
            }
        )

    staged_scoring = staged_pl.select(
        ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    )

    lipid_test_mask = np.isin(y_str[test_idx], list(LIPID_LABELS))
    y_family_test = np.array(
        [lipid_to_int[str(label)] for label in y_str[test_idx][lipid_test_mask]],
        dtype=np.int64,
    )
    p_family_test = lipid_family_proba[lipid_test_mask].argmax(axis=1)

    importance = None
    if persist_importance:
        importance = {
            specialist_rule.name: _gain_importance(specialist, specialist_feature_columns),
            "stage3_nonlipid_family": _gain_importance(nonlipid_model, nonlipid_feature_columns),
        }
        if family is not None:
            importance["stage2_lipid_family"] = _gain_importance(family, lipid_family_feature_columns)
        if family_models is not None:
            for label, model in family_models.items():
                importance[f"stage2_lipid_{label}"] = _gain_importance(model, lipid_family_feature_columns)
        for item in boundary_models:
            importance[f"boundary_{item['rule'].name}"] = gain_importance(
                item["model"],
                item["feature_columns"],
            )
        if gate is not None:
            importance["stage1_lipid_gate"] = _gain_importance(gate, feature_columns)

    bundle_payload: dict[str, Any] | None = None
    if iteration == 0:
        schema_meta = build_feature_schema_metadata(
            feature_set="",  # filled by caller
            feature_columns=feature_columns,
        )
        bundle_payload = {
            "bundle_version": BUNDLE_VERSION,
            "pipeline_mode": "hierarchical",
            **schema_meta,
            "class_order": CLASS_10,
            "lipid_labels": list(LIPID_LABELS),
            "nonlipid_labels": list(NONLIPID_LABELS),
            "stage1_model": gate,
            "stage1_ensemble_models": flat_models if stage1_source == "ensemble" else None,
            "flat_model_keys": flat_model_keys,
            "stage2_model": family,
            "stage2_models": family_models,
            "stage2_lipid_family_mode": lipid_family_mode,
            "stage2_feature_columns": list(lipid_family_feature_columns),
            "nonlipid_model": nonlipid_model,
            "nonlipid_feature_columns": list(nonlipid_feature_columns),
            "stage3_model": specialist,
            "stage3_rule": specialist_rule,
            "stage3_feature_columns": list(specialist_feature_columns),
            "stage3_utility_model": serialize_utility_model(utility_model),
            "stage3_specialist_gate_mode": specialist_gate,
            "stage3_utility_threshold_default": float(utility_threshold_default),
            "stage3_utility_threshold_top1_plm": utility_threshold_top1_plm,
            "boundary_heads": boundary_models,
            "nonlipid_source": nonlipid_source,
            "stage1_source": stage1_source,
            "library_versions": _library_versions(),
        }

    return {
        "iteration": iteration,
        "ensemble_frame": ensemble_frame,
        "staged_scoring": staged_scoring,
        "specialist_frame": specialist_frame,
        "boundary_frame": boundary_frame,
        "stage1_binary_f1": stage1_binary_f1,
        "stage2_lipid_family_macro_f1": float(
            f1_score(
                y_family_test,
                p_family_test,
                labels=np.arange(len(LIPID_LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
        "importance": importance,
        "bundle_payload": bundle_payload,
    }


def run_hierarchical_training(settings: Settings) -> dict[str, Path]:
    """Train all split iterations, persist predictions, metrics, report, and iter-0 bundle."""

    hierarchy = settings.hierarchical
    if hierarchy.nonlipid_source != "dedicated_head":
        raise ValueError(
            "hierarchical.nonlipid_source must be 'dedicated_head' "
            f"(got {hierarchy.nonlipid_source!r})"
        )

    proc = settings.paths.processed_dir
    models_dir = settings.paths.models_dir
    reports_dir = settings.paths.reports_dir
    pred_dir = proc / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = settings.feature_columns()
    schema_metadata = build_feature_schema_metadata(
        feature_set=settings.feature_set,
        feature_columns=feature_columns,
    )

    rule = hierarchy.resolved_specialist_rule()

    splits_dir = proc / "splits"
    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if len(split_files) != settings.n_iterations:
        full_split = pd.read_parquet(proc / "full_pockets.parquet")
        y_str = class10_labels(full_split)
        group_labels: np.ndarray | None = None
        if settings.split_strategy == "grouped":
            group_column = settings.split_group_column
            if group_column not in full_split.columns:
                raise KeyError(
                    f"group column not found in training parquet: {group_column}"
                )
            group_labels = full_split[group_column].to_numpy()
        splits = make_splits(
            class_labels=y_str,
            n_iterations=settings.n_iterations,
            test_fraction=settings.test_fraction,
            seed_base=settings.seed_base,
            strategy=settings.split_strategy,
            group_labels=group_labels,
        )
        split_files = persist_splits(splits, splits_dir)
    if not split_files:
        raise FileNotFoundError(f"no split files found in {splits_dir}")

    flat_keys = tuple(settings.models)
    boundary_rules = tuple(item.to_boundary_rule() for item in hierarchy.boundary_heads)
    lipid_family_feature_columns = _resolve_feature_columns(
        feature_columns,
        hierarchy.lipid_family_feature_set,
    )
    specialist_feature_columns = _resolve_feature_columns(
        feature_columns,
        hierarchy.specialist_feature_set,
    )
    nonlipid_feature_columns = _resolve_feature_columns(
        feature_columns,
        hierarchy.nonlipid_feature_set,
    )
    boundary_specs = tuple(
        (
            item.to_boundary_rule(),
            _resolve_feature_columns(feature_columns, item.feature_set),
            item.feature_set,
        )
        for item in hierarchy.boundary_heads
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=hierarchy.workers) as ex:
        futures = [
            ex.submit(
                _primary_training_worker,
                iteration=i,
                split_path=split_path,
                full_pockets_path=proc / "full_pockets.parquet",
                feature_columns=feature_columns,
                seed=settings.seed_base + i,
                specialist_rule=rule,
                stage1_source=hierarchy.stage1_source,
                persist_importance=(i == 0),
                flat_model_keys=flat_keys,
                nonlipid_source=hierarchy.nonlipid_source,
                specialist_gate=hierarchy.specialist_gate,
                utility_threshold_default=hierarchy.utility_threshold_default,
                lipid_family_mode=hierarchy.lipid_family_mode,
                utility_threshold_top1_plm=hierarchy.utility_threshold_top1_plm,
                lipid_family_feature_columns=lipid_family_feature_columns,
                specialist_feature_columns=specialist_feature_columns,
                nonlipid_feature_columns=nonlipid_feature_columns,
                boundary_specs=boundary_specs,
            )
            for i, split_path in enumerate(split_files)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["iteration"])

    bundle_path = models_dir / hierarchy.bundle_name
    bundle_saved: Path | None = None
    for row in results:
        payload = row.get("bundle_payload")
        if payload is None:
            continue
        payload.update(schema_metadata)
        payload["settings_snapshot"] = {
            "feature_set": settings.feature_set,
            "pipeline_mode": settings.pipeline_mode,
            "hierarchical": hierarchy.model_dump(),
            "models": list(settings.models),
        }
        bundle_saved = save_hierarchical_bundle(payload, bundle_path)
        write_artifact_schema_sidecar(
            bundle_path,
            {
                "artifact_type": "hierarchical_bundle",
                "class_order": CLASS_10,
                **schema_metadata,
            },
        )
        break
    if bundle_saved is None:
        raise RuntimeError("internal error: missing iteration-0 hierarchical bundle payload")

    ensemble_frames = [row["ensemble_frame"] for row in results]
    specialist_frames = [row["specialist_frame"] for row in results]
    final_frames = [
        row["boundary_frame"] if boundary_rules else row["specialist_frame"]
        for row in results
    ]
    ensemble_all = pl.concat(ensemble_frames, how="diagonal_relaxed").sort(["iteration", "row_index"])
    specialist_all = pl.concat(specialist_frames, how="diagonal_relaxed").sort(["iteration", "row_index"])
    final_all = pl.concat(final_frames, how="diagonal_relaxed").sort(["iteration", "row_index"])

    base_summary = score_summary(ensemble_all)
    staged_scoring = pl.concat([row["staged_scoring"] for row in results]).sort(
        ["iteration", "row_index"]
    )
    specialist_scoring = specialist_all.select(
        ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    )
    final_scoring = final_all.select(
        ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    )
    staged_summary = score_summary(staged_scoring)
    specialist_summary = score_summary(specialist_scoring)
    final_summary = score_summary(final_scoring)

    rows = [
        _metric_row("v_sterol ensemble", base_summary),
        _metric_row("stage1+stage2 hierarchy", staged_summary),
        _metric_row(f"hierarchy + {rule.name}", specialist_summary),
    ]
    if boundary_rules:
        rows.append(_metric_row("hierarchy + configured boundaries", final_summary))
    output_metrics = reports_dir / HIERARCHICAL_METRICS_NAME
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_metrics, index=False)

    output_predictions = pred_dir / HIERARCHICAL_PREDICTIONS_NAME
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    final_all.write_parquet(output_predictions)
    write_artifact_schema_sidecar(
        output_predictions,
        {
            "artifact_type": "hierarchical_lipid_predictions",
            "class_order": CLASS_10,
            "pipeline_mode": "hierarchical",
            **schema_metadata,
        },
    )

    output_report = reports_dir / HIERARCHICAL_REPORT_NAME
    importance = next((row["importance"] for row in results if row["importance"]), {})
    _write_report(
        output_report,
        base_summary=base_summary,
        staged_summary=staged_summary,
        specialist_summary=specialist_summary,
        specialist_rule=rule,
        specialist_frame=final_all,
        stage1_source=hierarchy.stage1_source,
        stage1_f1=[
            row["stage1_binary_f1"] for row in results if row["stage1_binary_f1"] is not None
        ],
        stage2_f1=[row["stage2_lipid_family_macro_f1"] for row in results],
        importance=importance,
    )

    return {
        "predictions": output_predictions,
        "models_dir": models_dir,
        "hierarchical_report": output_report,
        "hierarchical_metrics": output_metrics,
        "hierarchical_bundle": bundle_saved,
    }
