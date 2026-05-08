"""Day 1 figures: confusion, per-class ROC, PCA, comparison bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .constants import CLASS_10, HIERARCHICAL_PREDICTIONS_NAME, LIPID_CODES
from .features import class10_labels, feature_matrix


def _headline_model(settings: Settings) -> str:
    # Prefer XGB > LGBM > RF for figures if configured; else fall back to the first.
    for pref in ("xgb", "lgbm", "rf"):
        if pref in settings.models:
            return pref
    return settings.models[0]


def plot_confusion_matrix(
    preds_iter0: pd.DataFrame, reports_dir: Path, model_key: str
) -> Path:
    y_true = preds_iter0["y_true_int"].to_numpy()
    y_pred = preds_iter0["y_pred_int"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_10)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_10,
        yticklabels=CLASS_10,
        cbar_kws={"label": "row-normalized rate"},
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"10-class confusion matrix (iter 0, {model_key})")
    fig.tight_layout()
    out = reports_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_class_roc(
    preds_iter0: pd.DataFrame, reports_dir: Path, model_key: str
) -> Path:
    proba_cols = [f"p_{c}" for c in CLASS_10]
    y_true = preds_iter0["y_true_int"].to_numpy()
    proba = preds_iter0[proba_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(CLASS_10):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        roc_auc = auc(fpr, tpr)
        lw = 2.0 if c in LIPID_CODES else 1.0
        ax.plot(fpr, tpr, lw=lw, label=f"{c} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"One-vs-rest ROC ({model_key}, iter 0)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = reports_dir / "per_class_roc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pca_colored_by_pred(
    full: pd.DataFrame, preds_iter0: pd.DataFrame, settings: Settings,
    reports_dir: Path, model_key: str,
) -> Path:
    X = feature_matrix(full, settings)
    X_std = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=settings.seed_base).fit_transform(X_std)

    test_idx = preds_iter0["row_index"].to_numpy()
    y_pred = preds_iter0["y_pred_int"].to_numpy()
    colors_by_class = {c: plt.cm.tab10(i / 10) for i, c in enumerate(CLASS_10)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    truth = class10_labels(full)
    for c in CLASS_10:
        mask = truth == c
        axes[0].scatter(
            pcs[mask, 0], pcs[mask, 1], s=3, alpha=0.3,
            color=colors_by_class[c], label=c,
        )
    axes[0].set_title("PCA colored by true class (all 15,219 pockets)")
    axes[0].legend(markerscale=3, fontsize=8, loc="best")

    for i, c in enumerate(CLASS_10):
        mask = y_pred == i
        axes[1].scatter(
            pcs[test_idx[mask], 0], pcs[test_idx[mask], 1], s=6, alpha=0.6,
            color=colors_by_class[c], label=c,
        )
    axes[1].set_title(f"PCA colored by predicted class ({model_key}, test iter 0)")
    axes[1].legend(markerscale=2, fontsize=8, loc="best")

    for a in axes:
        a.set_xlabel("PC1")
        a.set_ylabel("PC2")
    fig.tight_layout()
    out = reports_dir / "pca_colored_by_pred.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_metrics_comparison(settings: Settings, reports_dir: Path) -> Path:
    raw = pd.read_parquet(reports_dir / "raw_metrics.parquet")
    summary = raw.groupby("model").agg(
        f1_mean=("binary_f1", "mean"),
        f1_std=("binary_f1", "std"),
        auroc_mean=("binary_auroc", "mean"),
        auroc_std=("binary_auroc", "std"),
        lipid5_mean=("macro_f1_lipid5", "mean"),
        lipid5_std=("macro_f1_lipid5", "std"),
    ).reset_index()

    gt = settings.ground_truth
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(summary))
    width = 0.25
    ax.bar(xs - width, summary["f1_mean"], width,
           yerr=summary["f1_std"], label="Binary F1", capsize=3)
    ax.bar(xs, summary["auroc_mean"], width,
           yerr=summary["auroc_std"], label="Binary AUROC", capsize=3)
    ax.bar(xs + width, summary["lipid5_mean"], width,
           yerr=summary["lipid5_std"], label="5-lipid macro-F1", capsize=3)
    ax.axhline(gt.test.f1, color="tab:blue", ls="--", lw=1,
               label=f"paper F1 ({gt.test.f1:.3f})")
    ax.axhline(gt.test.auroc, color="tab:orange", ls="--", lw=1,
               label=f"paper AUROC ({gt.test.auroc:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(summary["model"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("metric")
    ax.set_title(
        f"Day 1 vs paper Table 1 | feature_set={settings.feature_set} "
        f"| {settings.n_iterations} iters"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = reports_dir / "metrics_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_figures(settings: Settings) -> dict[str, Path]:
    proc = settings.paths.processed_dir
    reports = settings.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    staged_mode = settings.pipeline_mode in {"hierarchical", "composite"}
    preds_name = HIERARCHICAL_PREDICTIONS_NAME if staged_mode else "test_predictions.parquet"
    preds = pd.read_parquet(proc / "predictions" / preds_name)
    model_key = settings.pipeline_mode if staged_mode else _headline_model(settings)
    if "model" in preds.columns:
        iter0 = preds[(preds["iteration"] == 0) & (preds["model"] == model_key)]
    else:
        iter0 = preds[preds["iteration"] == 0]
    full = pd.read_parquet(proc / "full_pockets.parquet")

    out_paths: dict[str, Path] = {}
    out_paths["confusion"] = plot_confusion_matrix(iter0, reports, model_key)
    out_paths["roc"] = plot_per_class_roc(iter0, reports, model_key)
    out_paths["pca"] = plot_pca_colored_by_pred(full, iter0, settings, reports, model_key)
    out_paths["comparison"] = plot_metrics_comparison(settings, reports)
    return out_paths
