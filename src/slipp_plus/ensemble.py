"""Probability-averaging ensemble over the v49 RF/XGB/LGBM heads.

Consumes the ``test_predictions.parquet`` emitted by :mod:`slipp_plus.train`
(one row per ``(iteration, model, row_index)`` with ``p_<CLASS>`` softprobs)
and produces an ensemble by averaging the softprob matrices across the three
base models for each ``(iteration, row_index)``.

The per-class/summary metrics match the conventions used in
:mod:`slipp_plus.evaluate` so that ensemble numbers are directly comparable
to the single-model numbers in ``reports/v49/metrics_table.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .constants import CLASS_10, LIPID_CODES

DEFAULT_MODELS: list[str] = ["rf", "xgb", "lgbm"]
PROBA_COLUMNS: list[str] = [f"p_{c}" for c in CLASS_10]
LIPID_IDX: np.ndarray = np.array(
    [i for i, c in enumerate(CLASS_10) if c in LIPID_CODES], dtype=np.int64
)


# ---------------------------------------------------------------------------
# Loading / averaging
# ---------------------------------------------------------------------------
def load_predictions(parquet_path: Path) -> pl.DataFrame:
    """Read the per-(iteration, model, row_index) soft-probability table."""
    return pl.read_parquet(parquet_path)


def average_softprobs(
    df: pl.DataFrame,
    models: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Average the soft-probability matrices across ``models`` per (iter, row).

    Parameters
    ----------
    df:
        Long-format predictions as emitted by :mod:`slipp_plus.train`.
    models:
        Which base models to include. Default: RF + XGB + LGBM.
    weights:
        Optional per-model weights (will be renormalized). Default: equal.

    Returns
    -------
    pl.DataFrame with columns ``iteration, row_index, y_true_int,
    p_<CLASS>... , y_pred_int`` (argmax of the averaged softprobs).
    """
    if models is None:
        models = DEFAULT_MODELS
    models = list(models)

    filtered = df.filter(pl.col("model").is_in(models))
    present = set(filtered["model"].unique().to_list())
    missing = [m for m in models if m not in present]
    if missing:
        raise ValueError(f"predictions frame is missing models: {missing}")

    if weights is None:
        w = {m: 1.0 / len(models) for m in models}
    else:
        total = sum(weights.get(m, 0.0) for m in models)
        if total <= 0:
            raise ValueError("weights must sum to > 0 across selected models")
        w = {m: weights.get(m, 0.0) / total for m in models}

    # Multiply each model's probs by its weight, then sum per (iteration,row_index).
    weighted = filtered.with_columns(
        [
            (pl.col(c) * pl.col("model").replace_strict(w, default=0.0)).alias(c)
            for c in PROBA_COLUMNS
        ]
    )
    averaged = weighted.group_by(["iteration", "row_index", "y_true_int"]).agg(
        [pl.col(c).sum().alias(c) for c in PROBA_COLUMNS]
    )

    proba_np = averaged.select(PROBA_COLUMNS).to_numpy()
    y_pred_int = proba_np.argmax(axis=1).astype(np.int64)

    return averaged.with_columns(pl.Series("y_pred_int", y_pred_int)).sort(
        ["iteration", "row_index"]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _per_iteration_class_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """One row per ``(class_10, iteration)`` with precision/recall/F1."""
    rows: list[dict[str, Any]] = []
    all_labels = np.arange(len(CLASS_10))
    for iter_id, sub in df.sort("iteration").group_by("iteration", maintain_order=True):
        y_true = sub["y_true_int"].to_numpy()
        y_pred = sub["y_pred_int"].to_numpy()
        p, r, f, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        it = int(iter_id[0] if isinstance(iter_id, tuple) else iter_id)
        for i, c in enumerate(CLASS_10):
            rows.append(
                {
                    "class_10": c,
                    "iteration": it,
                    "precision": float(p[i]),
                    "recall": float(r[i]),
                    "f1": float(f[i]),
                    "support": int(support[i]),
                }
            )
    return pl.DataFrame(rows)


def score_by_class(ensemble_df: pl.DataFrame) -> pl.DataFrame:
    """Per-class P/R/F1 with per-iteration rows + a summary row per class.

    Summary rows carry ``iteration = null`` and populate ``*_std`` columns with
    the across-iteration standard deviation. Per-iteration rows leave the
    ``*_std`` columns null so a consumer can filter on either partition.
    """
    per_iter = _per_iteration_class_metrics(ensemble_df)

    summary = (
        per_iter.group_by("class_10", maintain_order=True)
        .agg(
            pl.col("precision").mean().alias("precision"),
            pl.col("precision").std(ddof=0).alias("precision_std"),
            pl.col("recall").mean().alias("recall"),
            pl.col("recall").std(ddof=0).alias("recall_std"),
            pl.col("f1").mean().alias("f1"),
            pl.col("f1").std(ddof=0).alias("f1_std"),
            pl.col("support").sum().alias("support"),
        )
        .with_columns(pl.lit(None, dtype=pl.Int64).alias("iteration"))
    )

    per_iter_padded = per_iter.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("precision_std"),
        pl.lit(None, dtype=pl.Float64).alias("recall_std"),
        pl.lit(None, dtype=pl.Float64).alias("f1_std"),
    )

    ordered = [
        "class_10",
        "iteration",
        "precision",
        "precision_std",
        "recall",
        "recall_std",
        "f1",
        "f1_std",
        "support",
    ]
    return pl.concat([per_iter_padded.select(ordered), summary.select(ordered)])


def _iteration_summary_metrics(df: pl.DataFrame) -> dict[str, list[float]]:
    """Collect macro / lipid / binary / auroc per iteration + per-class F1."""
    out: dict[str, list[float]] = {
        "macro_f1": [],
        "lipid_macro_f1": [],
        "binary_f1": [],
        "auroc": [],
        "accuracy": [],
    }
    per_class_f1: dict[str, list[float]] = {c: [] for c in CLASS_10}
    all_labels = np.arange(len(CLASS_10))

    for _, sub in df.sort("iteration").group_by("iteration", maintain_order=True):
        y_true = sub["y_true_int"].to_numpy()
        y_pred = sub["y_pred_int"].to_numpy()
        proba = sub.select(PROBA_COLUMNS).to_numpy()

        _, _, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        out["macro_f1"].append(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=all_labels,
                    average="macro",
                    zero_division=0,
                )
            )
        )
        lipid_mask = np.isin(all_labels, LIPID_IDX)
        out["lipid_macro_f1"].append(float(np.mean(f[lipid_mask])))
        out["accuracy"].append(float(accuracy_score(y_true, y_pred)))

        for i, c in enumerate(CLASS_10):
            per_class_f1[c].append(float(f[i]))

        true_bin = np.isin(y_true, LIPID_IDX).astype(int)
        pred_bin = np.isin(y_pred, LIPID_IDX).astype(int)
        tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
        fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
        fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        bf1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        out["binary_f1"].append(float(bf1))
        p_lipid = proba[:, LIPID_IDX].sum(axis=1)
        try:
            out["auroc"].append(float(roc_auc_score(true_bin, p_lipid)))
        except ValueError:
            out["auroc"].append(float("nan"))

    out["_per_class_f1"] = per_class_f1  # type: ignore[assignment]
    return out


def score_summary(ensemble_df: pl.DataFrame) -> dict[str, Any]:
    """Return mean ± std across iterations for the headline metrics."""
    agg = _iteration_summary_metrics(ensemble_df)
    per_class = agg.pop("_per_class_f1")

    def ms(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    macro_mean, macro_std = ms(agg["macro_f1"])
    lipid_mean, lipid_std = ms(agg["lipid_macro_f1"])
    bin_mean, bin_std = ms(agg["binary_f1"])
    auroc_mean, auroc_std = ms(agg["auroc"])
    acc_mean, acc_std = ms(agg["accuracy"])

    per_class_f1 = {c: ms(per_class[c]) for c in CLASS_10}

    return {
        "macro_f1_mean": macro_mean,
        "macro_f1_std": macro_std,
        "lipid_macro_f1_mean": lipid_mean,
        "lipid_macro_f1_std": lipid_std,
        "binary_f1_mean": bin_mean,
        "binary_f1_std": bin_std,
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "per_class_f1": per_class_f1,
    }


# ---------------------------------------------------------------------------
# Confusion helpers used by the report
# ---------------------------------------------------------------------------
def clr_ste_confusion(df: pl.DataFrame) -> dict[str, int]:
    """Count CLR<->STE confusions across all rows of ``df``."""
    clr = CLASS_10.index("CLR")
    ste = CLASS_10.index("STE")
    y_true = df["y_true_int"].to_numpy()
    y_pred = df["y_pred_int"].to_numpy()
    return {
        "CLR_as_STE": int(((y_true == clr) & (y_pred == ste)).sum()),
        "STE_as_CLR": int(((y_true == ste) & (y_pred == clr)).sum()),
        "CLR_correct": int(((y_true == clr) & (y_pred == clr)).sum()),
        "STE_correct": int(((y_true == ste) & (y_pred == ste)).sum()),
        "CLR_support": int((y_true == clr).sum()),
        "STE_support": int((y_true == ste).sum()),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_ms(mean: float, std: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.{digits}f} \u00b1 {std:.{digits}f}"


def _per_model_summary(df: pl.DataFrame, model_key: str) -> dict[str, Any]:
    """Produce a score_summary equivalent for a single base model."""
    sub = df.filter(pl.col("model") == model_key).select(
        ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    )
    return score_summary(sub)


def write_markdown_report(
    df: pl.DataFrame,
    ensemble_df: pl.DataFrame,
    output_path: Path,
    *,
    title: str = "SLiPP++ v49 ensemble metrics",
) -> Path:
    """Write the ensemble-only markdown report to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_model: dict[str, dict[str, Any]] = {}
    for m in DEFAULT_MODELS:
        if (df["model"] == m).any():
            per_model[m] = _per_model_summary(df, m)
    ensemble = score_summary(ensemble_df)

    with output_path.open("w") as f:
        f.write(f"# {title}\n\n")
        f.write(
            "_Probability-averaging ensemble of RF + XGB + LGBM, "
            "25 stratified shuffle iterations._\n\n"
        )
        f.write("## Headline metrics (mean \u00b1 std across 25 iterations)\n\n")
        f.write(
            "| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | AUROC | CLR F1 | STE F1 |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for m, s in per_model.items():
            clr_m, clr_s = s["per_class_f1"]["CLR"]
            ste_m, ste_s = s["per_class_f1"]["STE"]
            f.write(
                f"| v49 {m} only "
                f"| {_fmt_ms(s['macro_f1_mean'], s['macro_f1_std'])} "
                f"| {_fmt_ms(s['lipid_macro_f1_mean'], s['lipid_macro_f1_std'])} "
                f"| {_fmt_ms(s['binary_f1_mean'], s['binary_f1_std'])} "
                f"| {_fmt_ms(s['auroc_mean'], s['auroc_std'])} "
                f"| {_fmt_ms(clr_m, clr_s)} "
                f"| {_fmt_ms(ste_m, ste_s)} |\n"
            )
        clr_m, clr_s = ensemble["per_class_f1"]["CLR"]
        ste_m, ste_s = ensemble["per_class_f1"]["STE"]
        f.write(
            f"| v49 ensemble (mean prob) "
            f"| {_fmt_ms(ensemble['macro_f1_mean'], ensemble['macro_f1_std'])} "
            f"| {_fmt_ms(ensemble['lipid_macro_f1_mean'], ensemble['lipid_macro_f1_std'])} "
            f"| {_fmt_ms(ensemble['binary_f1_mean'], ensemble['binary_f1_std'])} "
            f"| {_fmt_ms(ensemble['auroc_mean'], ensemble['auroc_std'])} "
            f"| {_fmt_ms(clr_m, clr_s)} "
            f"| {_fmt_ms(ste_m, ste_s)} |\n"
        )

        f.write("\n## Per-class F1 (mean across iterations)\n\n")
        f.write("| condition |" + "".join(f" {c} |" for c in CLASS_10) + "\n")
        f.write("|---|" + "---:|" * len(CLASS_10) + "\n")
        for m, s in per_model.items():
            cells = " | ".join(f"{s['per_class_f1'][c][0]:.3f}" for c in CLASS_10)
            f.write(f"| v49 {m} only | {cells} |\n")
        cells = " | ".join(f"{ensemble['per_class_f1'][c][0]:.3f}" for c in CLASS_10)
        f.write(f"| v49 ensemble | {cells} |\n")

        f.write("\n## CLR vs STE confusion counts (summed over 25 iterations)\n\n")
        f.write("| condition | CLR\u2192STE | STE\u2192CLR | CLR correct | STE correct |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for m in per_model:
            sub = df.filter(pl.col("model") == m)
            c = clr_ste_confusion(sub)
            f.write(
                f"| v49 {m} only | {c['CLR_as_STE']} | {c['STE_as_CLR']} "
                f"| {c['CLR_correct']} | {c['STE_correct']} |\n"
            )
        c = clr_ste_confusion(ensemble_df)
        f.write(
            f"| v49 ensemble | {c['CLR_as_STE']} | {c['STE_as_CLR']} "
            f"| {c['CLR_correct']} | {c['STE_correct']} |\n"
        )

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Average RF/XGB/LGBM softprobs and compute ensemble metrics.",
    )
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Which base models to average (default: rf xgb lgbm).",
    )
    p.add_argument(
        "--ensemble-predictions",
        type=Path,
        default=None,
        help="If set, write the averaged predictions parquet to this path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    df = load_predictions(args.predictions)
    ensemble_df = average_softprobs(df, models=args.models)
    write_markdown_report(df, ensemble_df, args.output)
    if args.ensemble_predictions is not None:
        args.ensemble_predictions.parent.mkdir(parents=True, exist_ok=True)
        ensemble_df.write_parquet(args.ensemble_predictions)
    summary = score_summary(ensemble_df)
    print(
        "ensemble macro-F1 = "
        f"{summary['macro_f1_mean']:.3f} \u00b1 {summary['macro_f1_std']:.3f}, "
        f"CLR F1 = {summary['per_class_f1']['CLR'][0]:.3f}, "
        f"STE F1 = {summary['per_class_f1']['STE'][0]:.3f}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
