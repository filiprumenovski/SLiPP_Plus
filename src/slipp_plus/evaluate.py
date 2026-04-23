"""Metrics + holdout evaluation + paper comparison."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Settings
from .constants import CLASS_10, LIPID_CODES
from .features import feature_matrix


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
    agg = rows.groupby(group_cols)[num_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


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
    settings: Settings,
) -> dict[str, float]:
    """Score a holdout using the iteration-0 fitted model bundle.

    Holdouts carry ``class_binary`` only (no class_10), so we collapse the
    10-class predicted softmax to a lipid probability via LIPID_IDX.sum.
    """
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
    p_lipid = proba[:, LIPID_IDX].sum(axis=1)
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
        "n": int(len(true_bin)),
        "n_lipid": int(true_bin.sum()),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
    }


def _fmt(v: float | int) -> str:
    if isinstance(v, int):
        return str(v)
    if np.isnan(v):
        return "nan"
    return f"{v:.3f}"


def run_evaluation(settings: Settings) -> dict[str, Path]:
    paths = settings.paths
    proc = paths.processed_dir
    reports = paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(proc / "predictions" / "test_predictions.parquet")
    metrics = evaluate_test_predictions(preds)
    raw_path = reports / "raw_metrics.parquet"
    metrics.to_parquet(raw_path, index=False)

    summary = _aggregate(metrics, group_cols=["model"])

    apo = pd.read_parquet(proc / "apo_pdb_holdout.parquet")
    af = pd.read_parquet(proc / "alphafold_holdout.parquet")
    holdout_rows: dict[str, dict[str, dict[str, float]]] = {}
    for key in settings.models:
        bundle_path = paths.models_dir / f"{key}_multiclass.joblib"
        if not bundle_path.exists():
            continue
        bundle = joblib.load(bundle_path)
        try:
            apo_m = evaluate_holdout(bundle, apo, settings)
            af_m = evaluate_holdout(bundle, af, settings)
            holdout_rows[key] = {"apo_pdb": apo_m, "alphafold": af_m}
        except KeyError as e:
            holdout_rows[key] = {"apo_pdb": {"error": str(e)},
                                 "alphafold": {"error": str(e)}}

    md_path = reports / "metrics_table.md"
    gt = settings.ground_truth
    with md_path.open("w") as f:
        f.write("# SLiPP++ Day 1 metrics\n\n")
        f.write(f"_Feature set: `{settings.feature_set}`, "
                f"{settings.n_iterations} stratified shuffle iterations._\n\n")

        # --- Section 1: test split, binary-collapsed ---
        f.write("## 1. Binary-collapsed on test split (paper Table 1 line 1)\n\n")
        f.write("| model | F1 | AUROC | accuracy | precision | sensitivity | specificity |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        f.write(f"| paper (RF) | {gt.test.f1:.3f} | {gt.test.auroc:.3f} | "
                f"{gt.test.accuracy or float('nan'):.3f} | "
                f"{gt.test.precision or float('nan'):.3f} | "
                f"{gt.test.sensitivity or float('nan'):.3f} | - |\n")
        for _, row in summary.iterrows():
            mk = row["model"]
            f.write(
                f"| {mk} | {row['binary_f1_mean']:.3f} \u00b1 {row['binary_f1_std']:.3f} "
                f"| {row['binary_auroc_mean']:.3f} \u00b1 {row['binary_auroc_std']:.3f} "
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
        f.write("| model | macro-F1 (10) | macro-F1 (5 lipids) | accuracy |\n")
        f.write("|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(
                f"| {row['model']} "
                f"| {row['macro_f1_10_mean']:.3f} \u00b1 {row['macro_f1_10_std']:.3f} "
                f"| {row['macro_f1_lipid5_mean']:.3f} \u00b1 {row['macro_f1_lipid5_std']:.3f} "
                f"| {row['accuracy_mean']:.3f} |\n"
            )

        f.write("\n### Per-class F1 (mean across iterations)\n\n")
        f.write("| model |" + "".join(f" {c} |" for c in CLASS_10) + "\n")
        f.write("|---|" + "---|" * len(CLASS_10) + "\n")
        for _, row in summary.iterrows():
            cells = " | ".join(f"{row[f'f1_{c}_mean']:.3f}" for c in CLASS_10)
            f.write(f"| {row['model']} | {cells} |\n")

    return {
        "raw_metrics": raw_path,
        "metrics_table": md_path,
        "holdouts": holdout_rows,  # type: ignore[dict-item]
    }
