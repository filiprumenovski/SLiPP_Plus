"""Metrics + holdout evaluation + paper comparison."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .artifact_schema import (
    read_artifact_schema_sidecar,
    validate_feature_schema_metadata,
)
from .config import Settings
from .constants import (
    CLASS_10,
    HIERARCHICAL_PREDICTIONS_NAME,
    LIPID_CODES,
)

LIPID_IDX = np.array([i for i, c in enumerate(CLASS_10) if c in LIPID_CODES])
LIPID_CLASSES = [c for c in CLASS_10 if c in LIPID_CODES]


def binary_collapse(
    y_true_int: np.ndarray, y_pred_int: np.ndarray, proba: np.ndarray
) -> dict[str, float]:
    """Collapse 10-class prediction to lipid vs non-lipid and compute paper metrics.

    Paper Methods p.19 uses:
        TP = lipid correctly -> lipid, FP = non-lipid -> lipid,
        TN = non-lipid -> non-lipid, FN = lipid -> non-lipid.
    F1 is the harmonic mean of sensitivity (recall) and precision.
    """
    true_bin = np.isin(y_true_int, LIPID_IDX).astype(int)
    pred_bin = np.isin(y_pred_int, LIPID_IDX).astype(int)
    p_lipid = proba[:, LIPID_IDX].sum(axis=1)

    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    try:
        auroc = roc_auc_score(true_bin, p_lipid)
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def multiclass_metrics(
    y_true_int: np.ndarray, y_pred_int: np.ndarray
) -> dict[str, float]:
    """Compute 10-class and per-class metrics for integer predictions.

    Parameters
    ----------
    y_true_int
        True class indices in ``CLASS_10`` order.
    y_pred_int
        Predicted class indices in ``CLASS_10`` order.

    Returns
    -------
    dict[str, float]
        Accuracy, 10-class macro-F1, lipid-class macro-F1, and per-class
        precision, recall, and F1 values.
    """

    p, r, f, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int,
        labels=np.arange(len(CLASS_10)),
        average=None,
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true_int, y_pred_int, average="macro",
        labels=np.arange(len(CLASS_10)), zero_division=0,
    )
    lipid_mask = np.isin(np.arange(len(CLASS_10)), LIPID_IDX)
    lipid_macro_f1 = float(np.mean(f[lipid_mask]))
    acc = accuracy_score(y_true_int, y_pred_int)
    out = {
        "accuracy": float(acc),
        "macro_f1_10": float(macro_f1),
        "macro_f1_lipid5": lipid_macro_f1,
    }
    for i, c in enumerate(CLASS_10):
        out[f"precision_{c}"] = float(p[i])
        out[f"recall_{c}"] = float(r[i])
        out[f"f1_{c}"] = float(f[i])
    return out


def _aggregate(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    num_cols = rows.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in group_cols]
    agg = rows.groupby(group_cols)[num_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    out = agg.reset_index()
    for col in num_cols:
        out[f"{col}_ci95"] = [
            _ci95_half_width(std, count)
            for std, count in zip(out[f"{col}_std"], out[f"{col}_count"], strict=True)
        ]
    return out


def _t_critical_975(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if df in table:
        return table[df]
    if df < 30:
        return table[max(k for k in table if k < df)]
    if df < 40:
        return table[30]
    if df < 60:
        return table[40]
    if df < 120:
        return table[60]
    return 1.960


def _ci95_half_width(std: float, count: int) -> float:
    if count <= 1 or np.isnan(std):
        return float("nan")
    return float(_t_critical_975(count - 1) * std / np.sqrt(count))


def evaluate_test_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(iteration, model) metrics; return long DataFrame."""
    rows: list[dict] = []
    proba_cols = [f"p_{c}" for c in CLASS_10]
    for (i, model_key), sub in preds.groupby(["iteration", "model"]):
        y_true = sub["y_true_int"].to_numpy()
        y_pred = sub["y_pred_int"].to_numpy()
        proba = sub[proba_cols].to_numpy()
        binary = binary_collapse(y_true, y_pred, proba)
        multi = multiclass_metrics(y_true, y_pred)
        rows.append(
            {"iteration": int(i), "model": model_key,
             **{f"binary_{k}": v for k, v in binary.items()},
             **multi}
        )
    return pd.DataFrame(rows)


def evaluate_holdout(
    model_bundle: dict,
    holdout_df: pd.DataFrame,
    _settings: Settings,
) -> dict[str, float]:
    """Score a holdout using the iteration-0 fitted model bundle.

    Holdouts carry ``class_binary`` only (no class_10), so we collapse the
    10-class predicted softmax to a lipid probability via LIPID_IDX.sum.

    The lipid indices are derived from the bundle's ``class_order`` metadata
    to guard against silent misalignment if class ordering ever changes.
    """
    # Derive lipid indices from the bundle's class_order, falling back to the
    # module constant if the bundle predates the class_order field.
    bundle_class_order = model_bundle.get("class_order", CLASS_10)
    if bundle_class_order != CLASS_10:
        raise ValueError(
            f"model bundle class_order {bundle_class_order} does not match "
            f"the canonical CLASS_10 {CLASS_10}. Cannot safely evaluate holdout."
        )
    lipid_idx = np.array(
        [i for i, c in enumerate(bundle_class_order) if c in LIPID_CODES]
    )

    cols = model_bundle["feature_columns"]
    missing = [c for c in cols if c not in holdout_df.columns]
    if missing:
        raise KeyError(
            f"holdout missing feature columns: {missing}. "
            f"Holdouts only support feature_set=v14. Retrain with feature_set=v14 "
            f"to evaluate holdouts."
        )
    X = holdout_df[cols].to_numpy(dtype=np.float64)
    true_bin = holdout_df["class_binary"].to_numpy(dtype=int)
    proba = model_bundle["model"].predict_proba(X)
    p_lipid = proba[:, lipid_idx].sum(axis=1)
    pred_bin = (p_lipid >= 0.5).astype(int)

    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / len(true_bin)
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    try:
        auroc = roc_auc_score(true_bin, p_lipid)
    except ValueError:
        auroc = float("nan")
    return {
        "n": len(true_bin),
        "n_lipid": int(true_bin.sum()),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
    }


def evaluate_staged_holdout_predictions(
    holdout_preds: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate staged or composite holdout predictions after binary collapse.

    Parameters
    ----------
    holdout_preds
        Prediction table with probability columns named ``p_<CLASS>``.
    holdout_df
        Holdout feature table containing ``class_binary``.

    Returns
    -------
    dict[str, float]
        Binary-collapse metrics, sample counts, and confusion counts.
    """

    proba_cols = [f"p_{c}" for c in CLASS_10]
    proba = holdout_preds[proba_cols].to_numpy(dtype=np.float64)
    true_bin = holdout_df["class_binary"].to_numpy(dtype=int)
    p_lipid = proba[:, LIPID_IDX].sum(axis=1)
    pred_bin = (p_lipid >= 0.5).astype(int)

    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / len(true_bin) if len(true_bin) else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    try:
        auroc = roc_auc_score(true_bin, p_lipid)
    except ValueError:
        auroc = float("nan")

    return {
        "n": len(true_bin),
        "n_lipid": int(true_bin.sum()),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_hierarchical_holdouts(
    settings: Settings,
    *,
    feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Score configured apo-PDB and AlphaFold holdouts for staged pipelines.

    Parameters
    ----------
    settings
        Experiment settings pointing at processed holdout parquets and the
        persisted staged/composite bundle.
    feature_columns
        Expected model feature columns used for schema validation.

    Returns
    -------
    dict[str, dict[str, float]]
        Metrics keyed by holdout name.

    Raises
    ------
    ValueError
        If the persisted bundle class order differs from ``CLASS_10``.
    """

    from .hierarchical_pipeline import load_hierarchical_bundle, predict_hierarchical_holdout

    proc = settings.paths.processed_dir
    holdout_dir = proc / "predictions" / "holdouts"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = settings.paths.models_dir / settings.hierarchical.bundle_name
    bundle = load_hierarchical_bundle(bundle_path)
    if list(bundle.get("class_order", CLASS_10)) != list(CLASS_10):
        raise ValueError(
            "hierarchical bundle class_order does not match canonical CLASS_10"
        )
    if bundle.get("backend") == "family_encoder":
        from .composite_family_train import predict_family_encoder_holdout

        outputs: dict[str, dict[str, float]] = {}
        for holdout_name in ("apo_pdb", "alphafold"):
            holdout_df = pd.read_parquet(proc / f"{holdout_name}_holdout.parquet")
            holdout_preds = predict_family_encoder_holdout(holdout_df, bundle)
            holdout_preds.to_parquet(
                holdout_dir / f"family_encoder_{holdout_name}_predictions.parquet",
                index=False,
            )
            outputs[holdout_name] = evaluate_staged_holdout_predictions(
                holdout_preds,
                holdout_df,
            )
        return outputs

    outputs: dict[str, dict[str, float]] = {}
    for holdout_name in ("apo_pdb", "alphafold"):
        holdout_df = pd.read_parquet(proc / f"{holdout_name}_holdout.parquet")
        holdout_preds = predict_hierarchical_holdout(
            holdout_df=holdout_df,
            bundle=bundle,
            expected_feature_columns=feature_columns,
            expected_feature_set=settings.feature_set,
        )
        holdout_preds.to_parquet(
            holdout_dir / f"hierarchical_{holdout_name}_predictions.parquet",
            index=False,
        )
        outputs[holdout_name] = evaluate_staged_holdout_predictions(holdout_preds, holdout_df)
    return outputs


def _fmt(v: float | int) -> str:
    if isinstance(v, int):
        return str(v)
    if np.isnan(v):
        return "nan"
    return f"{v:.3f}"


def run_evaluation(settings: Settings) -> dict[str, Path]:
    """Evaluate persisted predictions and write metric reports.

    Parameters
    ----------
    settings
        Loaded experiment configuration.

    Returns
    -------
    dict[str, Path]
        Paths to raw metrics parquet and markdown metrics table, plus holdout
        metrics in the ``holdouts`` entry.
    """

    paths = settings.paths
    proc = paths.processed_dir
    reports = paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    feature_columns = settings.feature_columns()
    staged_modes = {"hierarchical", "composite"}
    preds_name = (
        HIERARCHICAL_PREDICTIONS_NAME
        if settings.pipeline_mode in staged_modes
        else "test_predictions.parquet"
    )
    preds_path = proc / "predictions" / preds_name
    preds_schema = read_artifact_schema_sidecar(preds_path)
    if preds_schema is not None:
        validate_feature_schema_metadata(
            preds_schema,
            expected_feature_columns=feature_columns,
            expected_feature_set=settings.feature_set,
            artifact_label=f"prediction artifact {preds_path.name}",
        )

    preds = pd.read_parquet(preds_path)
    if "model" not in preds.columns:
        preds = preds.copy()
        preds["model"] = settings.pipeline_mode
    metrics = evaluate_test_predictions(preds)
    raw_path = reports / "raw_metrics.parquet"
    metrics.to_parquet(raw_path, index=False)

    summary = _aggregate(metrics, group_cols=["model"])

    holdout_rows: dict[str, dict[str, dict[str, float]]] = {}
    if settings.pipeline_mode not in staged_modes:
        apo = pd.read_parquet(proc / "apo_pdb_holdout.parquet")
        af = pd.read_parquet(proc / "alphafold_holdout.parquet")
        for key in settings.models:
            bundle_path = paths.models_dir / f"{key}_multiclass.joblib"
            if not bundle_path.exists():
                continue
            bundle = joblib.load(bundle_path)
            validate_feature_schema_metadata(
                bundle,
                expected_feature_columns=feature_columns,
                expected_feature_set=settings.feature_set,
                artifact_label=f"model bundle {bundle_path.name}",
            )
            try:
                apo_m = evaluate_holdout(bundle, apo, settings)
                af_m = evaluate_holdout(bundle, af, settings)
                holdout_rows[key] = {"apo_pdb": apo_m, "alphafold": af_m}
            except KeyError as e:
                holdout_rows[key] = {"apo_pdb": {"error": str(e)},
                                     "alphafold": {"error": str(e)}}
    else:
        try:
            holdout_rows["hierarchical"] = evaluate_hierarchical_holdouts(
                settings,
                feature_columns=feature_columns,
            )
        except Exception as e:
            holdout_rows[settings.pipeline_mode] = {
                "apo_pdb": {"error": str(e)},
                "alphafold": {"error": str(e)},
            }

    md_path = reports / "metrics_table.md"
    gt = settings.ground_truth
    with md_path.open("w") as f:
        f.write("# SLiPP++ Day 1 metrics\n\n")
        f.write(f"_Feature set: `{settings.feature_set}`, "
                f"{settings.n_iterations} stratified shuffle iterations, "
                f"pipeline mode: `{settings.pipeline_mode}`._\n\n")

        # --- Section 1: test split, binary-collapsed ---
        f.write("## 1. Binary-collapsed on test split (paper Table 1 line 1)\n\n")
        f.write("| model | F1 mean±std | F1 95% CI | AUROC mean±std | AUROC 95% CI | accuracy | precision | sensitivity | specificity |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        f.write(f"| paper (RF) | {gt.test.f1:.3f} | - | {gt.test.auroc:.3f} | - | "
                f"{gt.test.accuracy or float('nan'):.3f} | "
                f"{gt.test.precision or float('nan'):.3f} | "
                f"{gt.test.sensitivity or float('nan'):.3f} | - |\n")
        for _, row in summary.iterrows():
            mk = row["model"]
            f.write(
                f"| {mk} | {row['binary_f1_mean']:.3f} \u00b1 {row['binary_f1_std']:.3f} "
                f"| \u00b1 {row['binary_f1_ci95']:.3f} "
                f"| {row['binary_auroc_mean']:.3f} \u00b1 {row['binary_auroc_std']:.3f} "
                f"| \u00b1 {row['binary_auroc_ci95']:.3f} "
                f"| {row['binary_accuracy_mean']:.3f} "
                f"| {row['binary_precision_mean']:.3f} "
                f"| {row['binary_sensitivity_mean']:.3f} "
                f"| {row['binary_specificity_mean']:.3f} |\n"
            )

        # --- Section 2: holdouts ---
        f.write("\n## 2. Holdouts (paper Table 1 lines 2-3)\n\n")
        f.write("### apo-PDB holdout\n\n")
        f.write("| model | F1 | AUROC | precision | sensitivity |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| paper (RF) | {gt.apo_pdb.f1:.3f} | {gt.apo_pdb.auroc:.3f} | - | - |\n")
        for key, spl in holdout_rows.items():
            m = spl["apo_pdb"]
            if "error" in m:
                f.write(f"| {key} | n/a ({m['error']}) | | | |\n")
                continue
            f.write(f"| {key} | {_fmt(m['f1'])} | {_fmt(m['auroc'])} "
                    f"| {_fmt(m['precision'])} | {_fmt(m['sensitivity'])} |\n")
        f.write("\n### AlphaFold holdout\n\n")
        f.write("| model | F1 | AUROC | precision | sensitivity |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| paper (RF) | {gt.alphafold.f1:.3f} | {gt.alphafold.auroc:.3f} | - | - |\n")
        for key, spl in holdout_rows.items():
            m = spl["alphafold"]
            if "error" in m:
                f.write(f"| {key} | n/a ({m['error']}) | | | |\n")
                continue
            f.write(f"| {key} | {_fmt(m['f1'])} | {_fmt(m['auroc'])} "
                    f"| {_fmt(m['precision'])} | {_fmt(m['sensitivity'])} |\n")

        # --- Section 3: multi-class ---
        f.write("\n## 3. Multi-class (the headline Day 1 result)\n\n")
        f.write("| model | macro-F1 (10) mean±std | macro-F1 (10) 95% CI | macro-F1 (5 lipids) mean±std | macro-F1 (5 lipids) 95% CI | accuracy |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(
                f"| {row['model']} "
                f"| {row['macro_f1_10_mean']:.3f} \u00b1 {row['macro_f1_10_std']:.3f} "
                f"| \u00b1 {row['macro_f1_10_ci95']:.3f} "
                f"| {row['macro_f1_lipid5_mean']:.3f} \u00b1 {row['macro_f1_lipid5_std']:.3f} "
                f"| \u00b1 {row['macro_f1_lipid5_ci95']:.3f} "
                f"| {row['accuracy_mean']:.3f} |\n"
            )

        f.write("\n### Per-class F1 (mean across iterations)\n\n")
        f.write("| model |" + "".join(f" {c} |" for c in CLASS_10) + "\n")
        f.write("|---|" + "---|" * len(CLASS_10) + "\n")
        for _, row in summary.iterrows():
            cells = " | ".join(f"{row[f'f1_{c}_mean']:.3f}" for c in CLASS_10)
            f.write(f"| {row['model']} | {cells} |\n")

        f.write("\n### Per-class precision (mean across iterations)\n\n")
        f.write("| model |" + "".join(f" {c} |" for c in CLASS_10) + "\n")
        f.write("|---|" + "---|" * len(CLASS_10) + "\n")
        for _, row in summary.iterrows():
            cells = " | ".join(f"{row[f'precision_{c}_mean']:.3f}" for c in CLASS_10)
            f.write(f"| {row['model']} | {cells} |\n")

        f.write("\n### Per-class recall (mean across iterations)\n\n")
        f.write("| model |" + "".join(f" {c} |" for c in CLASS_10) + "\n")
        f.write("|---|" + "---|" * len(CLASS_10) + "\n")
        for _, row in summary.iterrows():
            cells = " | ".join(f"{row[f'recall_{c}_mean']:.3f}" for c in CLASS_10)
            f.write(f"| {row['model']} | {cells} |\n")

    return {
        "raw_metrics": raw_path,
        "metrics_table": md_path,
        "holdouts": holdout_rows,  # type: ignore[dict-item]
    }
