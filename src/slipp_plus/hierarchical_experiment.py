"""Experimental staged lipid-class hierarchy.

Architecture:
1. binary lipid-vs-rest gate;
2. five-way lipid-family classifier for CLR/MYR/OLA/PLM/STE;
3. gated one-vs-neighbors specialist heads for known hard boundaries.

The non-lipid subtype distribution is inherited from the existing 10-class
ensemble, so this experiment can answer whether the lipid-specific hierarchy
helps without rebuilding the rest of the Day-1 classifier stack.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .constants import CLASS_10, LIPID_CODES
from .ensemble import (
    PROBA_COLUMNS,
    average_softprobs,
    load_predictions,
    score_summary,
)
from .hierarchical_postprocess import OneVsNeighborsRule, apply_one_vs_neighbors
from .splits import load_split

LIPID_LABELS: tuple[str, ...] = tuple(c for c in CLASS_10 if c in LIPID_CODES)
NONLIPID_LABELS: tuple[str, ...] = tuple(c for c in CLASS_10 if c not in LIPID_CODES)
DEFAULT_STE_NEIGHBORS: tuple[str, ...] = ("PLM", "COA", "OLA", "MYR")
DEFAULT_STE_RULE = OneVsNeighborsRule(
    name="ste_specialist",
    positive_label="STE",
    neighbor_labels=DEFAULT_STE_NEIGHBORS,
    top_k=4,
    min_positive_proba=0.40,
)


_DEFAULT_XGB_KWARGS: dict[str, float | int] = {
    "max_depth": 5,
    "n_estimators": 250,
    "learning_rate": 0.05,
}


def _resolve_xgb_kwargs(hyperparameters: Any | None) -> dict[str, float | int]:
    """Return XGB kwargs from a hyperparameters object or the legacy defaults."""

    if hyperparameters is None:
        return dict(_DEFAULT_XGB_KWARGS)
    return hyperparameters.to_xgb_kwargs()


def train_lipid_gate(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    *,
    hyperparameters: Any | None = None,
) -> XGBClassifier:
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    kwargs = _resolve_xgb_kwargs(hyperparameters)
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        **kwargs,
    )
    model.fit(X_tr, y_tr)
    return model


def train_lipid_family(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    *,
    hyperparameters: Any | None = None,
) -> XGBClassifier:
    kwargs = _resolve_xgb_kwargs(hyperparameters)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(LIPID_LABELS),
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
        **kwargs,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    model.fit(X_tr, y_tr, sample_weight=weights)
    return model


def train_lipid_binary_heads(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    *,
    hyperparameters: Any | None = None,
) -> dict[str, XGBClassifier]:
    """Train 5 independent binary one-vs-rest heads for lipid family classification.

    Each head learns P(class_k | lipid pocket) independently, eliminating the
    softmax simplex constraint that forces mutual exclusivity between lipid
    subclasses. This lets the model express "this pocket is equally compatible
    with PLM and OLA" rather than suppressing one to raise the other.

    Parameters
    ----------
    X_tr:
        Feature matrix for lipid-only training rows.
    y_tr:
        Integer labels in [0, len(LIPID_LABELS)), mapping to LIPID_LABELS order.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict mapping lipid label string to its fitted binary XGB model.
    """
    models: dict[str, XGBClassifier] = {}
    kwargs = _resolve_xgb_kwargs(hyperparameters)
    for i, label in enumerate(LIPID_LABELS):
        y_binary = (y_tr == i).astype(np.int64)
        n_pos = int(y_binary.sum())
        n_neg = int((y_binary == 0).sum())
        model = XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=(n_neg / n_pos) if n_pos > 0 else 1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
            **kwargs,
        )
        model.fit(X_tr, y_binary)
        models[label] = model
    return models


def predict_lipid_binary_heads(
    models: dict[str, XGBClassifier],
    X: np.ndarray,
) -> np.ndarray:
    """Return (n, 5) normalized probability matrix from independent binary heads.

    Raw posteriors are normalized to sum to 1 so the result is a drop-in
    replacement for the softmax family head's output in
    ``combine_hierarchical_softprobs``.
    """
    n = X.shape[0]
    raw = np.zeros((n, len(LIPID_LABELS)), dtype=np.float64)
    for i, label in enumerate(LIPID_LABELS):
        raw[:, i] = models[label].predict_proba(X)[:, 1].astype(np.float64)
    denom = raw.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return raw / denom


def train_nonlipid_family(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    *,
    hyperparameters: Any | None = None,
) -> XGBClassifier:
    kwargs = _resolve_xgb_kwargs(hyperparameters)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(NONLIPID_LABELS),
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
        **kwargs,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    model.fit(X_tr, y_tr, sample_weight=weights)
    return model


def combine_hierarchical_softprobs(
    p_lipid: np.ndarray,
    lipid_family_proba: np.ndarray,
    nonlipid_family_proba: np.ndarray,
) -> np.ndarray:
    """Fuse lipid gate, lipid-family head, and non-lipid head into a 10-class simplex."""

    if len(p_lipid) != len(lipid_family_proba) or len(p_lipid) != len(nonlipid_family_proba):
        raise ValueError("p_lipid, lipid_family_proba, and nonlipid_family_proba must align")
    if lipid_family_proba.shape[1] != len(LIPID_LABELS):
        raise ValueError(
            f"lipid_family_proba has {lipid_family_proba.shape[1]} columns; "
            f"expected {len(LIPID_LABELS)}"
        )
    if nonlipid_family_proba.shape[1] != len(NONLIPID_LABELS):
        raise ValueError(
            f"nonlipid_family_proba has {nonlipid_family_proba.shape[1]} columns; "
            f"expected {len(NONLIPID_LABELS)}"
        )

    n = len(p_lipid)
    staged = np.zeros((n, len(CLASS_10)), dtype=np.float64)
    lipid_indices = [CLASS_10.index(label) for label in LIPID_LABELS]
    nonlipid_indices = [CLASS_10.index(label) for label in NONLIPID_LABELS]

    p_lipid_clipped = np.clip(p_lipid.astype(np.float64), 0.0, 1.0)
    staged[:, lipid_indices] = lipid_family_proba.astype(np.float64) * p_lipid_clipped[:, None]

    denom = nonlipid_family_proba.sum(axis=1, keepdims=True).astype(np.float64)
    nonlipid_dist = np.full(
        (n, len(nonlipid_indices)),
        1.0 / len(nonlipid_indices),
        dtype=np.float64,
    )
    np.divide(
        nonlipid_family_proba.astype(np.float64),
        denom,
        out=nonlipid_dist,
        where=denom > 1e-12,
    )
    staged[:, nonlipid_indices] = nonlipid_dist * (1.0 - p_lipid_clipped[:, None])
    return staged


def train_one_vs_neighbors(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    *,
    hyperparameters: Any | None = None,
) -> XGBClassifier:
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    kwargs = _resolve_xgb_kwargs(hyperparameters)
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        **kwargs,
    )
    model.fit(X_tr, y_tr)
    return model


def build_specialist_training(
    full_pockets: pd.DataFrame,
    feature_columns: list[str],
    split_parquet: Path,
    rule: OneVsNeighborsRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, test_idx = load_split(split_parquet)
    labels = {rule.positive_label, *rule.neighbor_labels}
    y_all = full_pockets["class_10"].to_numpy()
    X_all = full_pockets[feature_columns].to_numpy(dtype=np.float64)
    tr_sel = train_idx[np.isin(y_all[train_idx], list(labels))]
    y_tr = (y_all[tr_sel] == rule.positive_label).astype(np.int64)
    return X_all[tr_sel], y_tr, test_idx


def build_staged_probabilities(
    baseline_iter_df: pl.DataFrame,
    *,
    row_index_lookup: np.ndarray,
    p_lipid: np.ndarray,
    lipid_family_proba: np.ndarray,
) -> pl.DataFrame:
    """Combine the binary gate and lipid-family head into 10-class softprobs."""

    if len(row_index_lookup) != len(p_lipid) or len(row_index_lookup) != len(lipid_family_proba):
        raise ValueError("row_index_lookup, p_lipid, and lipid_family_proba must align")

    baseline = baseline_iter_df.sort("row_index")
    baseline_rows = baseline["row_index"].to_numpy().astype(np.int64)
    position = {int(row): i for i, row in enumerate(row_index_lookup.tolist())}
    if set(baseline_rows.tolist()) != set(position):
        raise ValueError("baseline rows do not align with row_index_lookup")
    order = np.array([position[int(row)] for row in baseline_rows], dtype=np.int64)
    p_lipid = p_lipid[order]
    lipid_family_proba = lipid_family_proba[order]

    base_proba = baseline.select(PROBA_COLUMNS).to_numpy()
    staged = np.zeros_like(base_proba, dtype=np.float64)
    lipid_indices = [CLASS_10.index(label) for label in LIPID_LABELS]
    nonlipid_indices = [CLASS_10.index(label) for label in NONLIPID_LABELS]

    p_lipid_clipped = np.clip(p_lipid.astype(np.float64), 0.0, 1.0)
    staged[:, lipid_indices] = lipid_family_proba * p_lipid_clipped[:, None]

    nonlipid_base = base_proba[:, nonlipid_indices].astype(np.float64)
    denom = nonlipid_base.sum(axis=1, keepdims=True)
    fallback = np.full_like(nonlipid_base, 1.0 / len(nonlipid_indices))
    nonlipid_dist = np.divide(
        nonlipid_base,
        denom,
        out=fallback,
        where=denom > 1e-12,
    )
    staged[:, nonlipid_indices] = nonlipid_dist * (1.0 - p_lipid_clipped[:, None])

    y_pred = staged.argmax(axis=1).astype(np.int64)
    return baseline.with_columns(
        *[pl.Series(c, staged[:, i]) for i, c in enumerate(PROBA_COLUMNS)],
        pl.Series("stage1_p_lipid", p_lipid_clipped),
        pl.Series("stage2_y_pred_lipid", lipid_family_proba.argmax(axis=1).astype(np.int64)),
        pl.Series("y_pred_int", y_pred),
    )


def _binary_f1(y_true: np.ndarray, p_positive: np.ndarray) -> float:
    return float(f1_score(y_true, (p_positive >= 0.5).astype(np.int64), zero_division=0))


def _gain_importance(model: XGBClassifier, feature_columns: list[str]) -> dict[str, float]:
    gain_map = model.get_booster().get_score(importance_type="gain")
    return {feature: float(gain_map.get(f"f{i}", 0.0)) for i, feature in enumerate(feature_columns)}


def _worker(
    *,
    iteration: int,
    split_path: Path,
    full_pockets_path: Path,
    feature_columns: list[str],
    seed: int,
    specialist_rule: OneVsNeighborsRule,
    stage1_source: str,
    persist_importance: bool,
) -> dict[str, Any]:
    full = pd.read_parquet(full_pockets_path)
    train_idx, test_idx = load_split(split_path)
    X_all = full[feature_columns].to_numpy(dtype=np.float64)
    y_str = full["class_10"].to_numpy()

    y_bin_all = np.isin(y_str, list(LIPID_LABELS)).astype(np.int64)
    gate = None
    p_lipid = None
    stage1_binary_f1 = None
    if stage1_source == "trained":
        gate = train_lipid_gate(X_all[train_idx], y_bin_all[train_idx], seed=seed)
        p_lipid = gate.predict_proba(X_all[test_idx])[:, 1].astype(np.float64)
        stage1_binary_f1 = _binary_f1(y_bin_all[test_idx], p_lipid)

    lipid_to_int = {label: i for i, label in enumerate(LIPID_LABELS)}
    lipid_train = train_idx[np.isin(y_str[train_idx], list(LIPID_LABELS))]
    y_family = np.array([lipid_to_int[label] for label in y_str[lipid_train]], dtype=np.int64)
    family = train_lipid_family(X_all[lipid_train], y_family, seed=seed)
    lipid_family_proba = family.predict_proba(X_all[test_idx]).astype(np.float64)

    X_spec, y_spec, _ = build_specialist_training(
        full,
        feature_columns,
        split_path,
        specialist_rule,
    )
    specialist = train_one_vs_neighbors(X_spec, y_spec, seed=seed)
    p_specialist = specialist.predict_proba(X_all[test_idx])[:, 1].astype(np.float64)

    lipid_test_mask = np.isin(y_str[test_idx], list(LIPID_LABELS))
    y_family_test = np.array(
        [lipid_to_int[label] for label in y_str[test_idx][lipid_test_mask]],
        dtype=np.int64,
    )
    p_family_test = lipid_family_proba[lipid_test_mask].argmax(axis=1)

    importance = None
    if persist_importance:
        importance = {
            "stage2_lipid_family": _gain_importance(family, feature_columns),
            specialist_rule.name: _gain_importance(specialist, feature_columns),
        }
        if gate is not None:
            importance["stage1_lipid_gate"] = _gain_importance(gate, feature_columns)

    return {
        "iteration": iteration,
        "row_index_lookup": test_idx.astype(np.int64),
        "p_lipid": p_lipid,
        "lipid_family_proba": lipid_family_proba,
        "p_specialist": p_specialist,
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
    }


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def _metric_row(condition: str, summary: dict[str, Any]) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "condition": condition,
        "macro_f1_mean": summary["macro_f1_mean"],
        "macro_f1_std": summary["macro_f1_std"],
        "lipid_macro_f1_mean": summary["lipid_macro_f1_mean"],
        "lipid_macro_f1_std": summary["lipid_macro_f1_std"],
        "binary_f1_mean": summary["binary_f1_mean"],
        "binary_f1_std": summary["binary_f1_std"],
        "auroc_mean": summary["auroc_mean"],
        "auroc_std": summary["auroc_std"],
    }
    for label in LIPID_LABELS:
        mean, std = summary["per_class_f1"][label]
        row[f"{label}_f1_mean"] = mean
        row[f"{label}_f1_std"] = std
    return row


def _ste_confusion(df: pl.DataFrame) -> dict[str, int]:
    y_true = df["y_true_int"].to_numpy()
    y_pred = df["y_pred_int"].to_numpy()
    ste = CLASS_10.index("STE")
    out = {
        "STE_support": int((y_true == ste).sum()),
        "STE_correct": int(((y_true == ste) & (y_pred == ste)).sum()),
    }
    for label in ("PLM", "COA", "OLA", "MYR", "PP", "CLR"):
        idx = CLASS_10.index(label)
        out[f"STE_as_{label}"] = int(((y_true == ste) & (y_pred == idx)).sum())
        out[f"{label}_as_STE"] = int(((y_true == idx) & (y_pred == ste)).sum())
    return out


def _write_top_features(
    handle: Any,
    title: str,
    importance: dict[str, float],
    n: int = 12,
) -> None:
    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:n]
    if not top:
        return
    handle.write(f"\n## {title}\n\n")
    handle.write("| rank | feature | gain |\n|---:|---|---:|\n")
    for rank, (name, gain) in enumerate(top, start=1):
        handle.write(f"| {rank} | {name} | {gain:.4f} |\n")


def _write_report(
    output_report: Path,
    *,
    base_summary: dict[str, Any],
    staged_summary: dict[str, Any],
    specialist_summary: dict[str, Any],
    specialist_rule: OneVsNeighborsRule,
    specialist_frame: pl.DataFrame,
    stage1_source: str,
    stage1_f1: list[float],
    stage2_f1: list[float],
    importance: dict[str, dict[str, float]],
) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    fire_col = specialist_rule.fired_col
    fire_total = (
        int(specialist_frame[fire_col].sum()) if fire_col in specialist_frame.columns else 0
    )
    fire_mean = fire_total / max(1, specialist_frame["iteration"].n_unique())
    base_ste = base_summary["per_class_f1"]["STE"]
    staged_ste = staged_summary["per_class_f1"]["STE"]
    spec_ste = specialist_summary["per_class_f1"]["STE"]

    with output_report.open("w", encoding="utf-8") as handle:
        handle.write("# Hierarchical lipid-class experiment\n\n")
        handle.write(
            f"_Stage 1: `{stage1_source}` lipid-vs-rest gate. "
            "Stage 2: five-way lipid-family XGB. "
            "Stage 3: gated one-vs-neighbors specialist head._\n\n"
        )
        handle.write("## Headline metrics\n\n")
        handle.write(
            "| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | "
            "CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for name, summary in [
            ("v_sterol ensemble", base_summary),
            ("stage1+stage2 hierarchy", staged_summary),
            (f"hierarchy + {specialist_rule.name}", specialist_summary),
        ]:
            lipid_cells = " | ".join(
                _fmt(*summary["per_class_f1"][label]) for label in LIPID_LABELS
            )
            handle.write(
                f"| {name} "
                f"| {_fmt(summary['macro_f1_mean'], summary['macro_f1_std'])} "
                f"| {_fmt(summary['lipid_macro_f1_mean'], summary['lipid_macro_f1_std'])} "
                f"| {_fmt(summary['binary_f1_mean'], summary['binary_f1_std'])} "
                f"| {_fmt(summary['auroc_mean'], summary['auroc_std'])} "
                f"| {lipid_cells} |\n"
            )

        handle.write("\n## Stage diagnostics\n\n")
        if stage1_f1:
            handle.write(
                f"- Stage-1 lipid gate binary F1: "
                f"{_fmt(float(np.mean(stage1_f1)), float(np.std(stage1_f1)))}.\n"
            )
        else:
            handle.write(
                "- Stage-1 lipid gate uses the existing ensemble lipid probability mass "
                "to preserve binary parity.\n"
            )
        handle.write(
            f"- Stage-2 lipid-family macro-F1 on true lipid test rows: "
            f"{_fmt(float(np.mean(stage2_f1)), float(np.std(stage2_f1)))}.\n"
        )
        handle.write(
            f"- Specialist `{specialist_rule.name}`: positive `{specialist_rule.positive_label}`, "
            f"neighbors `{', '.join(specialist_rule.neighbor_labels)}`, "
            f"threshold `{specialist_rule.min_positive_proba}`, top-k `{specialist_rule.top_k}`.\n"
        )
        handle.write(f"- Specialist fires: {fire_total} total, {fire_mean:.1f} per iteration.\n")
        handle.write(
            f"- STE F1 delta vs ensemble: {spec_ste[0] - base_ste[0]:+.3f}; "
            f"vs stage1+stage2 only: {spec_ste[0] - staged_ste[0]:+.3f}.\n"
        )

        confusion = _ste_confusion(specialist_frame)
        handle.write("\n## STE-focused confusion\n\n")
        handle.write("| metric | count |\n|---|---:|\n")
        for key, value in confusion.items():
            handle.write(f"| {key} | {value} |\n")

        for title, values in importance.items():
            _write_top_features(handle, f"{title} top features", values)


def run_hierarchical_experiment(
    *,
    full_pockets_path: Path,
    predictions_path: Path,
    splits_dir: Path,
    model_bundle_path: Path,
    output_report: Path,
    output_metrics: Path,
    output_predictions: Path | None = None,
    specialist_rule: OneVsNeighborsRule = DEFAULT_STE_RULE,
    stage1_source: str = "ensemble",
    workers: int = 8,
    seed_base: int = 42,
) -> dict[str, Any]:
    if stage1_source not in {"ensemble", "trained"}:
        raise ValueError("stage1_source must be 'ensemble' or 'trained'")
    bundle = joblib.load(model_bundle_path)
    feature_columns = list(bundle["feature_columns"])
    ensemble_df = average_softprobs(load_predictions(predictions_path))
    base_summary = score_summary(ensemble_df)

    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if not split_files:
        raise FileNotFoundError(f"no split files found in {splits_dir}")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _worker,
                iteration=i,
                split_path=split_path,
                full_pockets_path=full_pockets_path,
                feature_columns=feature_columns,
                seed=seed_base + i,
                specialist_rule=specialist_rule,
                stage1_source=stage1_source,
                persist_importance=(i == 0),
            )
            for i, split_path in enumerate(split_files)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["iteration"])

    staged_frames: list[pl.DataFrame] = []
    specialist_frames: list[pl.DataFrame] = []
    for result in results:
        sub = ensemble_df.filter(pl.col("iteration") == result["iteration"])
        p_lipid = result["p_lipid"]
        if stage1_source == "ensemble":
            lipid_cols = [f"p_{label}" for label in LIPID_LABELS]
            by_row = dict(
                zip(
                    sub["row_index"].to_list(),
                    sub.select(pl.sum_horizontal(lipid_cols).alias("p_lipid"))["p_lipid"].to_list(),
                    strict=True,
                )
            )
            p_lipid = np.array(
                [float(by_row[int(row)]) for row in result["row_index_lookup"]],
                dtype=np.float64,
            )
        staged = build_staged_probabilities(
            sub,
            row_index_lookup=result["row_index_lookup"],
            p_lipid=p_lipid,
            lipid_family_proba=result["lipid_family_proba"],
        )
        staged_frames.append(staged)
        specialist_frames.append(
            apply_one_vs_neighbors(
                staged,
                result["p_specialist"],
                result["row_index_lookup"],
                specialist_rule,
            )
        )

    staged_all = pl.concat(staged_frames).sort(["iteration", "row_index"])
    specialist_all = pl.concat(specialist_frames).sort(["iteration", "row_index"])
    staged_scoring = staged_all.select(
        ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    )
    specialist_scoring = specialist_all.select(
        ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    )
    staged_summary = score_summary(staged_scoring)
    specialist_summary = score_summary(specialist_scoring)

    rows = [
        _metric_row("v_sterol ensemble", base_summary),
        _metric_row("stage1+stage2 hierarchy", staged_summary),
        _metric_row(f"hierarchy + {specialist_rule.name}", specialist_summary),
    ]
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_metrics, index=False)
    if output_predictions is not None:
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        specialist_all.write_parquet(output_predictions)

    importance = next((row["importance"] for row in results if row["importance"]), {})
    _write_report(
        output_report,
        base_summary=base_summary,
        staged_summary=staged_summary,
        specialist_summary=specialist_summary,
        specialist_rule=specialist_rule,
        specialist_frame=specialist_all,
        stage1_source=stage1_source,
        stage1_f1=[
            row["stage1_binary_f1"] for row in results if row["stage1_binary_f1"] is not None
        ],
        stage2_f1=[row["stage2_lipid_family_macro_f1"] for row in results],
        importance=importance,
    )
    return {
        "report": output_report,
        "metrics": output_metrics,
        "predictions": output_predictions,
        "base_summary": base_summary,
        "staged_summary": staged_summary,
        "specialist_summary": specialist_summary,
        "fire_total": int(specialist_all[specialist_rule.fired_col].sum()),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged lipid-class hierarchy.")
    parser.add_argument("--full-pockets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--stage1-source", choices=["ensemble", "trained"], default="ensemble")
    parser.add_argument("--ste-threshold", type=float, default=0.40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=DEFAULT_STE_NEIGHBORS,
        top_k=4,
        min_positive_proba=args.ste_threshold,
    )
    result = run_hierarchical_experiment(
        full_pockets_path=args.full_pockets,
        predictions_path=args.predictions,
        splits_dir=args.splits_dir,
        model_bundle_path=args.model_bundle,
        output_report=args.output_report,
        output_metrics=args.output_metrics,
        output_predictions=args.output_predictions,
        specialist_rule=rule,
        stage1_source=args.stage1_source,
        workers=args.workers,
        seed_base=args.seed_base,
    )
    print(f"wrote {result['report']}")
    print(f"wrote {result['metrics']}")
    if result["predictions"] is not None:
        print(f"wrote {result['predictions']}")


if __name__ == "__main__":
    main()
