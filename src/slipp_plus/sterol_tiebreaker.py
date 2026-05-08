"""CLR vs STE binary tie-breaker head applied on top of the ensemble.

The multiclass ensemble frequently ends up with CLR and STE as its two most
probable classes with a tiny margin between them. This module trains a
dedicated ``XGBClassifier`` on CLR-vs-STE rows only (per iteration, honoring
the existing train/test split) and uses it to arbitrate those ties:

* if the ensemble's top-1 and top-2 predictions are exactly ``{CLR, STE}``
* and the top-1/top-2 margin is below ``TIEBREAKER_MARGIN``

then the combined ``p_CLR + p_STE`` mass is split between CLR and STE in the
ratio given by the binary head; probabilities for the other 8 classes are
unchanged. The argmax is then recomputed.

The tie-breaker operates purely on the CSV descriptor features (``v49``) and
never retrains the base models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier

from .boundary_head import (
    BoundaryRule,
    apply_boundary_head,
    build_boundary_training,
    train_boundary_head,
)
from .constants import CLASS_10
from .ensemble import (
    DEFAULT_MODELS,
    PROBA_COLUMNS,
    clr_ste_confusion,
    load_predictions,
    score_summary,
)
from .pair_tiebreaker_pipeline import run_boundary_tiebreaker_iterations

TIEBREAKER_MARGIN_DEFAULT: float = 0.15
CLR_IDX: int = CLASS_10.index("CLR")
STE_IDX: int = CLASS_10.index("STE")
CLR_STE_RULE = BoundaryRule(
    name="clr_ste_tiebreaker",
    positive_label="STE",
    negative_labels=("CLR",),
    margin=TIEBREAKER_MARGIN_DEFAULT,
    max_rank=2,
    fired_column="tiebreaker_fired",
    score_column="p_STE_binary",
)


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------
def build_clr_vs_ste_training(
    full_pockets: pd.DataFrame,
    feature_columns: list[str],
    split_parquet: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X_tr, y_tr, X_te, y_te, test_row_index)`` for one split.

    ``y`` is binary: ``1`` if class_10 is ``STE`` else ``0`` (for CLR).
    The test tuple keeps only rows whose true class is CLR or STE; we also
    return their row indices so the caller can align test predictions back to
    the ensemble frame's ``row_index`` column.
    """
    return build_boundary_training(
        full_pockets,
        feature_columns,
        split_parquet,
        CLR_STE_RULE,
    )


def train_sterol_tiebreaker(
    X_tr: np.ndarray, y_tr: np.ndarray, seed: int
) -> XGBClassifier:
    """Fit a CLR-vs-STE binary XGB head with class-balanced ``scale_pos_weight``."""
    return train_boundary_head(
        X_tr,
        y_tr,
        seed=seed,
        n_estimators=200,
    )


# ---------------------------------------------------------------------------
# Tie-break application
# ---------------------------------------------------------------------------
def apply_tiebreaker(
    ensemble_df: pl.DataFrame,
    tiebreaker_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    margin: float = TIEBREAKER_MARGIN_DEFAULT,
) -> pl.DataFrame:
    """Redistribute ``p_CLR``/``p_STE`` for close sterol calls.

    Parameters
    ----------
    ensemble_df:
        The frame returned by :func:`slipp_plus.ensemble.average_softprobs`.
    tiebreaker_proba:
        ``P(STE)`` predicted by the binary head, one entry per row in
        ``row_index_lookup``.
    row_index_lookup:
        Row indices (matching ``ensemble_df['row_index']``) corresponding
        element-wise to ``tiebreaker_proba``.
    margin:
        Fire the tiebreaker only when the ensemble's top-1/top-2 gap is
        ``< margin`` *and* those two classes are exactly CLR and STE.

    Returns
    -------
    A new polars frame with the same columns as ``ensemble_df`` plus
    ``tiebreaker_fired`` (Bool) and ``p_STE_binary`` (Float64, ``NaN`` for
    rows with no tiebreaker score).
    """
    rule = BoundaryRule(
        name=CLR_STE_RULE.name,
        positive_label=CLR_STE_RULE.positive_label,
        negative_labels=CLR_STE_RULE.negative_labels,
        margin=margin,
        max_rank=CLR_STE_RULE.max_rank,
        fired_column=CLR_STE_RULE.fired_column,
        score_column=CLR_STE_RULE.score_column,
    )
    return apply_boundary_head(ensemble_df, tiebreaker_proba, row_index_lookup, rule)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_tiebreaker_pipeline(
    full_pockets_path: Path,
    predictions_path: Path,
    splits_dir: Path,
    feature_columns: list[str],
    output_path: Path | None,
    tiebreaker_predictions_path: Path | None = None,
    workers: int = 6,
    margin: float = TIEBREAKER_MARGIN_DEFAULT,
    seed_base: int = 42,
    ensemble_predictions_path: Path | None = None,
) -> dict[str, Any]:
    """End-to-end: ensemble average, train 25 binary heads, apply, report.

    Returns a dict with the summary dicts + paths + diagnostics.
    """
    core = run_boundary_tiebreaker_iterations(
        full_pockets_path=full_pockets_path,
        predictions_path=predictions_path,
        splits_dir=splits_dir,
        feature_columns=feature_columns,
        workers=workers,
        margin=margin,
        seed_base=seed_base,
        rule=CLR_STE_RULE,
        train_kwargs={"n_estimators": 200},
    )
    ensemble_pred = core["ensemble_predictions"]
    augmented = core["augmented_predictions"]

    # ensure tiebreaker_fired & p_STE_binary columns are preserved through concat
    if tiebreaker_predictions_path is not None:
        tiebreaker_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        augmented.write_parquet(tiebreaker_predictions_path)

    if ensemble_predictions_path is not None:
        ensemble_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble_pred.write_parquet(ensemble_predictions_path)

    fires = core["fire_counts"]
    sterol_bin_f1s = core["binary_f1s"]
    importance = core["feature_importance"]

    summaries: dict[str, dict[str, Any]] = {}
    base_df = load_predictions(predictions_path)
    for m in DEFAULT_MODELS:
        if (base_df["model"] == m).any():
            sub = base_df.filter(pl.col("model") == m).select(
                ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
            )
            summaries[f"v49 {m} only"] = score_summary(sub)
    summaries["v49 ensemble (mean prob)"] = score_summary(ensemble_pred)
    summaries["v49 ensemble + tiebreaker"] = score_summary(
        augmented.select(
            ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
        )
    )

    confusions: dict[str, dict[str, int]] = {}
    for m in DEFAULT_MODELS:
        if (base_df["model"] == m).any():
            confusions[f"v49 {m} only"] = clr_ste_confusion(
                base_df.filter(pl.col("model") == m)
            )
    confusions["v49 ensemble (mean prob)"] = clr_ste_confusion(ensemble_pred)
    confusions["v49 ensemble + tiebreaker"] = clr_ste_confusion(augmented)

    diagnostics = {
        "fire_counts_per_iter": fires,
        "fire_mean": float(np.mean(fires)) if fires else 0.0,
        "fire_std": float(np.std(fires)) if fires else 0.0,
        "fire_total": int(np.sum(fires)) if fires else 0,
        "sterol_binary_f1_mean": float(np.mean(sterol_bin_f1s)),
        "sterol_binary_f1_std": float(np.std(sterol_bin_f1s)),
        "feature_importance_iter0": importance or {},
    }

    ste_breakdown = _ste_error_breakdown(augmented)

    report_paths: dict[str, Path] = {}
    if output_path is not None:
        report_paths["metrics_md"] = write_tiebreaker_report(
            output_path,
            summaries=summaries,
            confusions=confusions,
            diagnostics=diagnostics,
            margin=margin,
            ste_error_breakdown=ste_breakdown,
        )

    return {
        "summaries": summaries,
        "confusions": confusions,
        "diagnostics": diagnostics,
        "augmented_predictions": augmented,
        "ensemble_predictions": ensemble_pred,
        "report_paths": report_paths,
    }


# ---------------------------------------------------------------------------
# Report writer (5-row comparison table)
# ---------------------------------------------------------------------------
def _fmt_ms(mean: float, std: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.{digits}f} \u00b1 {std:.{digits}f}"


CONDITION_ORDER: list[str] = [
    "v49 rf only",
    "v49 xgb only",
    "v49 lgbm only",
    "v49 ensemble (mean prob)",
    "v49 ensemble + tiebreaker",
]


def _ste_error_breakdown(
    augmented: pl.DataFrame | None,
) -> list[tuple[str, int]]:
    """Where do STE true rows get mispredicted? (Diagnostic.)"""
    if augmented is None:
        return []
    y_true = augmented["y_true_int"].to_numpy()
    y_pred = augmented["y_pred_int"].to_numpy()
    ste_mask = y_true == STE_IDX
    miss = y_pred[ste_mask & (y_pred != STE_IDX)]
    counts: dict[int, int] = {}
    for v in miss.tolist():
        counts[int(v)] = counts.get(int(v), 0) + 1
    out = [(CLASS_10[k], v) for k, v in counts.items()]
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


def write_tiebreaker_report(
    output_path: Path,
    *,
    summaries: dict[str, dict[str, Any]],
    confusions: dict[str, dict[str, int]],
    diagnostics: dict[str, Any],
    margin: float,
    ste_error_breakdown: list[tuple[str, int]] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("# SLiPP++ v49 ensemble + CLR/STE tiebreaker metrics\n\n")
        f.write(
            f"_Probability-averaging ensemble of RF + XGB + LGBM with a binary "
            f"CLR-vs-STE tiebreaker (margin < {margin:.2f}), "
            f"25 stratified shuffle iterations._\n\n"
        )
        f.write(
            "| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | "
            "AUROC | CLR F1 | STE F1 | CLR recall | STE recall |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for cond in CONDITION_ORDER:
            if cond not in summaries:
                continue
            s = summaries[cond]
            clr_m, clr_s = s["per_class_f1"]["CLR"]
            ste_m, ste_s = s["per_class_f1"]["STE"]
            # per-class recall means: recompute from the summary — we didn't
            # stash per-class recall, so derive a coarse estimate from the
            # confusion-counts (summed across iters).
            f.write(
                f"| {cond} "
                f"| {_fmt_ms(s['macro_f1_mean'], s['macro_f1_std'])} "
                f"| {_fmt_ms(s['lipid_macro_f1_mean'], s['lipid_macro_f1_std'])} "
                f"| {_fmt_ms(s['binary_f1_mean'], s['binary_f1_std'])} "
                f"| {_fmt_ms(s['auroc_mean'], s['auroc_std'])} "
                f"| {_fmt_ms(clr_m, clr_s)} "
                f"| {_fmt_ms(ste_m, ste_s)} "
                f"| {_recall_from_confusion(confusions[cond], 'CLR'):.3f} "
                f"| {_recall_from_confusion(confusions[cond], 'STE'):.3f} |\n"
            )

        f.write("\n## CLR vs STE confusion counts (summed over 25 iterations)\n\n")
        f.write("| condition | CLR correct | STE correct | CLR\u2192STE | STE\u2192CLR |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cond in CONDITION_ORDER:
            if cond not in confusions:
                continue
            c = confusions[cond]
            f.write(
                f"| {cond} | {c['CLR_correct']} | {c['STE_correct']} "
                f"| {c['CLR_as_STE']} | {c['STE_as_CLR']} |\n"
            )

        f.write("\n## Tiebreaker diagnostics\n\n")
        fires = diagnostics["fire_counts_per_iter"]
        f.write(
            f"- Tiebreaker fired: mean = {diagnostics['fire_mean']:.1f} "
            f"rows/iter (std {diagnostics['fire_std']:.1f}), "
            f"total = {diagnostics['fire_total']} over 25 iterations\n"
        )
        f.write(
            "- Per-iteration fire counts: "
            + ", ".join(str(x) for x in fires)
            + "\n"
        )
        f.write(
            f"- Tiebreaker sterol-only binary F1 (CLR vs STE, on true sterol "
            f"test rows): {diagnostics['sterol_binary_f1_mean']:.3f} "
            f"\u00b1 {diagnostics['sterol_binary_f1_std']:.3f}\n"
        )
        if ste_error_breakdown:
            f.write(
                "\n### Where do true STE rows get misclassified? (ensemble, "
                "summed over 25 iters)\n\n"
            )
            f.write("| predicted class | count |\n|---|---:|\n")
            for cls, cnt in ste_error_breakdown:
                f.write(f"| {cls} | {cnt} |\n")
            f.write(
                "\n_Note: the tiebreaker only fires when both CLR and STE are "
                "in the ensemble's top-2. If STE loses to PP, COA, or OLA "
                "instead of CLR, it is outside the tiebreaker's scope._\n"
            )

        importance = diagnostics.get("feature_importance_iter0") or {}
        if importance:
            top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:15]
            f.write(
                "\n## Tiebreaker iteration-0 top-15 features (gain)\n\n"
                "| rank | feature | gain |\n|---:|---|---:|\n"
            )
            for r, (name, g) in enumerate(top, start=1):
                f.write(f"| {r} | {name} | {g:.4f} |\n")

    return output_path


def _recall_from_confusion(c: dict[str, int], cls: str) -> float:
    if cls == "CLR":
        num = c["CLR_correct"]
        den = c.get("CLR_support") or (c["CLR_correct"] + c["CLR_as_STE"])
    else:
        num = c["STE_correct"]
        den = c.get("STE_support") or (c["STE_correct"] + c["STE_as_CLR"])
    return (num / den) if den else 0.0


def compose_overall_report(
    output_path: Path,
    *,
    summaries: dict[str, dict[str, Any]],
    confusions: dict[str, dict[str, int]],
    diagnostics: dict[str, Any],
    margin: float,
    ste_error_breakdown: list[tuple[str, int]] | None = None,
) -> Path:
    """Compose the canonical ``reports/ensemble/metrics.md`` comparison."""
    return write_tiebreaker_report(
        output_path,
        summaries=summaries,
        confusions=confusions,
        diagnostics=diagnostics,
        margin=margin,
        ste_error_breakdown=ste_error_breakdown,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLR-vs-STE binary tiebreaker head on top of the ensemble.",
    )
    p.add_argument("--full-pockets", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tiebreaker-predictions", type=Path, default=None)
    p.add_argument("--ensemble-predictions", type=Path, default=None)
    p.add_argument(
        "--overall-report",
        type=Path,
        default=None,
        help="If set, also write a consolidated comparison to this path "
        "(use reports/ensemble/metrics.md).",
    )
    p.add_argument("--model-bundle", type=Path, default=None,
                   help="Any v49 iteration-0 joblib bundle (used to infer feature_columns).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--margin", type=float, default=TIEBREAKER_MARGIN_DEFAULT)
    p.add_argument("--seed-base", type=int, default=42)
    return p.parse_args(argv)


def _infer_feature_columns(args: argparse.Namespace) -> list[str]:
    if args.model_bundle is not None and args.model_bundle.exists():
        bundle = joblib.load(args.model_bundle)
        return list(bundle["feature_columns"])
    # Default: look next to the full_pockets parquet for a sibling model.
    candidates = [
        args.full_pockets.parent.parent / "models" / "v49" / "xgb_multiclass.joblib",
        Path("models/v49/xgb_multiclass.joblib"),
        Path("models/xgb_multiclass.joblib"),
    ]
    for c in candidates:
        if c.exists():
            bundle = joblib.load(c)
            return list(bundle["feature_columns"])
    # Final fallback: constants-level v49 feature set.
    from .constants import FEATURE_SETS

    return list(FEATURE_SETS["v49"])


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    feature_columns = _infer_feature_columns(args)
    result = run_tiebreaker_pipeline(
        full_pockets_path=args.full_pockets,
        predictions_path=args.predictions,
        splits_dir=args.splits_dir,
        feature_columns=feature_columns,
        output_path=args.output,
        tiebreaker_predictions_path=args.tiebreaker_predictions,
        ensemble_predictions_path=args.ensemble_predictions,
        workers=args.workers,
        margin=args.margin,
        seed_base=args.seed_base,
    )
    if args.overall_report is not None:
        compose_overall_report(
            args.overall_report,
            summaries=result["summaries"],
            confusions=result["confusions"],
            diagnostics=result["diagnostics"],
            margin=args.margin,
            ste_error_breakdown=_ste_error_breakdown(
                result["augmented_predictions"]
            ),
        )
        print(f"wrote {args.overall_report}")

    s_tb = result["summaries"]["v49 ensemble + tiebreaker"]
    s_en = result["summaries"]["v49 ensemble (mean prob)"]
    print(
        "ensemble: "
        f"CLR F1={s_en['per_class_f1']['CLR'][0]:.3f}, "
        f"STE F1={s_en['per_class_f1']['STE'][0]:.3f}, "
        f"macro-F1={s_en['macro_f1_mean']:.3f}"
    )
    print(
        "ensemble + tiebreaker: "
        f"CLR F1={s_tb['per_class_f1']['CLR'][0]:.3f}, "
        f"STE F1={s_tb['per_class_f1']['STE'][0]:.3f}, "
        f"macro-F1={s_tb['macro_f1_mean']:.3f}"
    )
    print(
        f"tiebreaker fired on average "
        f"{result['diagnostics']['fire_mean']:.1f} rows/iter, "
        f"{result['diagnostics']['fire_total']} total"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
