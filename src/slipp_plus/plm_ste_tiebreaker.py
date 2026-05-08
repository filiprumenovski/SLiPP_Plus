"""PLM vs STE binary tie-breaker head applied on top of the ensemble.

STE's dominant failure mode in the multiclass ensemble is confusion with PLM
(palmitate), not CLR: across 25 iterations XGB miscalls ~141 STE-as-PLM rows
vs. ~3 STE-as-CLR. Models are confidently wrong (avg margin ~0.7), so a
margin alone will not rescue STE; but combined with a richer feature set
(``v_plm_ste``) and a class-balanced binary head, redistributing the top-2
mass between PLM and STE should recover STE recall on close calls.

This module mirrors :mod:`slipp_plus.sterol_tiebreaker` in shape. It trains a
dedicated ``XGBClassifier`` on PLM-vs-STE rows only (per iteration, honoring
the existing train/test split) and uses it to arbitrate those ties:

* if the ensemble's top-1 and top-2 predictions are exactly ``{PLM, STE}``
* and the top-1/top-2 margin is below ``TIEBREAKER_MARGIN_DEFAULT``

then the combined ``p_PLM + p_STE`` mass is split between PLM and STE in the
ratio given by the binary head; probabilities for the other 8 classes are
unchanged. The argmax is then recomputed.

The tie-breaker operates on whatever feature columns the caller supplies
(typically ``v_plm_ste``) and never retrains the base models.
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
    boundary_confusion,
    build_boundary_training,
    train_boundary_head,
)
from .confusion_mining import candidate_boundary_rules, mine_confusion_edges
from .constants import CLASS_10, FEATURE_SETS
from .ensemble import (
    DEFAULT_MODELS,
    PROBA_COLUMNS,
    load_predictions,
    score_summary,
)
from .pair_tiebreaker_pipeline import run_boundary_tiebreaker_iterations

# Slightly wider than the CLR-STE margin (0.15) because the base models are
# "confidently wrong" on PLM-vs-STE ties. We want to catch more candidate rows.
TIEBREAKER_MARGIN_DEFAULT: float = 0.20
PLM_IDX: int = CLASS_10.index("PLM")
STE_IDX: int = CLASS_10.index("STE")

PLM_STE_RULE = BoundaryRule(
    name="plm_ste_tiebreaker",
    positive_label="STE",
    negative_labels=("PLM",),
    margin=TIEBREAKER_MARGIN_DEFAULT,
    max_rank=2,
    fired_column="tiebreaker_fired",
    score_column="p_STE_binary",
)


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------
def build_plm_vs_ste_training(
    full_pockets: pd.DataFrame,
    feature_columns: list[str],
    split_parquet: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X_tr, y_tr, X_te, y_te, test_row_index)`` for one split.

    ``y`` is binary: ``1`` if class_10 is ``STE`` else ``0`` (for PLM).
    The test tuple keeps only rows whose true class is PLM or STE; we also
    return their row indices so the caller can align test predictions back to
    the ensemble frame's ``row_index`` column.
    """
    return build_boundary_training(
        full_pockets,
        feature_columns,
        split_parquet,
        PLM_STE_RULE,
    )


def train_plm_ste_tiebreaker(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> XGBClassifier:
    """Fit a PLM-vs-STE binary XGB head with class-balanced ``scale_pos_weight``.

    With PLM=718, STE=152 the class imbalance is ~4.72x; ``scale_pos_weight``
    is computed per-iteration from the actual training fold to stay exact.
    """
    return train_boundary_head(X_tr, y_tr, seed=seed)


# ---------------------------------------------------------------------------
# Tie-break application
# ---------------------------------------------------------------------------
def apply_tiebreaker(
    ensemble_df: pl.DataFrame,
    tiebreaker_proba: np.ndarray,
    row_index_lookup: np.ndarray,
    margin: float = TIEBREAKER_MARGIN_DEFAULT,
) -> pl.DataFrame:
    """Redistribute ``p_PLM``/``p_STE`` for close PLM-vs-STE calls.

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
        ``< margin`` *and* those two classes are exactly PLM and STE.

    Returns
    -------
    A new polars frame with the same columns as ``ensemble_df`` plus
    ``tiebreaker_fired`` (Bool) and ``p_STE_binary`` (Float64, ``NaN`` for
    rows with no tiebreaker score).
    """
    rule = BoundaryRule(
        name=PLM_STE_RULE.name,
        positive_label=PLM_STE_RULE.positive_label,
        negative_labels=PLM_STE_RULE.negative_labels,
        margin=margin,
        max_rank=PLM_STE_RULE.max_rank,
        fired_column=PLM_STE_RULE.fired_column,
        score_column=PLM_STE_RULE.score_column,
    )
    return apply_boundary_head(ensemble_df, tiebreaker_proba, row_index_lookup, rule)


# ---------------------------------------------------------------------------
# Confusion helper (PLM <-> STE)
# ---------------------------------------------------------------------------
def plm_ste_confusion(df: pl.DataFrame) -> dict[str, int]:
    """Count PLM<->STE confusions across all rows of ``df``."""
    generic = boundary_confusion(df, PLM_STE_RULE)
    return {
        "PLM_as_STE": generic["negative_as_positive"],
        "STE_as_PLM": generic["positive_as_negative"],
        "PLM_correct": generic["negative_as_negative"],
        "STE_correct": generic["positive_as_positive"],
        "PLM_support": generic["negative_support"],
        "STE_support": generic["positive_support"],
    }


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
    prefix: str = "v_plm_ste",
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
        rule=PLM_STE_RULE,
    )
    ensemble_pred = core["ensemble_predictions"]
    augmented = core["augmented_predictions"]

    if tiebreaker_predictions_path is not None:
        tiebreaker_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        augmented.write_parquet(tiebreaker_predictions_path)

    if ensemble_predictions_path is not None:
        ensemble_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble_pred.write_parquet(ensemble_predictions_path)

    fires = core["fire_counts"]
    plm_ste_bin_f1s = core["binary_f1s"]
    importance = core["feature_importance"]

    summaries: dict[str, dict[str, Any]] = {}
    base_df = load_predictions(predictions_path)
    for m in DEFAULT_MODELS:
        if (base_df["model"] == m).any():
            sub = base_df.filter(pl.col("model") == m).select(
                ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
            )
            summaries[f"{prefix} {m} only"] = score_summary(sub)
    summaries[f"{prefix} ensemble (mean prob)"] = score_summary(ensemble_pred)
    summaries[f"{prefix} ensemble + plm_ste_tiebreaker"] = score_summary(
        augmented.select(["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS])
    )

    confusions: dict[str, dict[str, int]] = {}
    for m in DEFAULT_MODELS:
        if (base_df["model"] == m).any():
            confusions[f"{prefix} {m} only"] = plm_ste_confusion(
                base_df.filter(pl.col("model") == m)
            )
    confusions[f"{prefix} ensemble (mean prob)"] = plm_ste_confusion(ensemble_pred)
    confusions[f"{prefix} ensemble + plm_ste_tiebreaker"] = plm_ste_confusion(augmented)

    diagnostics = {
        "fire_counts_per_iter": fires,
        "fire_mean": float(np.mean(fires)) if fires else 0.0,
        "fire_std": float(np.std(fires)) if fires else 0.0,
        "fire_total": int(np.sum(fires)) if fires else 0,
        "plm_ste_binary_f1_mean": float(np.mean(plm_ste_bin_f1s)) if plm_ste_bin_f1s else 0.0,
        "plm_ste_binary_f1_std": float(np.std(plm_ste_bin_f1s)) if plm_ste_bin_f1s else 0.0,
        "feature_importance_iter0": importance or {},
    }
    residual_edges = mine_confusion_edges(
        augmented.select(["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]),
        average_models=False,
        lipid_only=True,
        min_count=1,
    )
    residual_rules = candidate_boundary_rules(
        residual_edges,
        top_n=5,
        min_count=1,
        margin=0.99,
    )
    diagnostics["residual_lipid_confusions"] = residual_edges.head(10).to_dict(orient="records")
    diagnostics["candidate_boundary_rules"] = [
        {
            "name": rule.name,
            "positive_label": rule.positive_label,
            "negative_labels": list(rule.negative_labels),
            "margin": rule.margin,
            "max_rank": rule.max_rank,
        }
        for rule in residual_rules
    ]

    ste_breakdown = _ste_error_breakdown(augmented)

    report_paths: dict[str, Path] = {}
    if output_path is not None:
        report_paths["metrics_md"] = write_plm_ste_tiebreaker_report(
            output_path,
            summaries=summaries,
            confusions=confusions,
            diagnostics=diagnostics,
            margin=margin,
            ste_error_breakdown=ste_breakdown,
            prefix=prefix,
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
# Report writer
# ---------------------------------------------------------------------------
def _fmt_ms(mean: float, std: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.{digits}f} \u00b1 {std:.{digits}f}"


def _condition_order(prefix: str) -> list[str]:
    return [
        f"{prefix} rf only",
        f"{prefix} xgb only",
        f"{prefix} lgbm only",
        f"{prefix} ensemble (mean prob)",
        f"{prefix} ensemble + plm_ste_tiebreaker",
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


def _recall_from_confusion(c: dict[str, int], cls: str) -> float:
    if cls == "PLM":
        num = c["PLM_correct"]
        den = c.get("PLM_support") or (c["PLM_correct"] + c["PLM_as_STE"])
    else:
        num = c["STE_correct"]
        den = c.get("STE_support") or (c["STE_correct"] + c["STE_as_PLM"])
    return (num / den) if den else 0.0


def write_plm_ste_tiebreaker_report(
    output_path: Path,
    *,
    summaries: dict[str, dict[str, Any]],
    confusions: dict[str, dict[str, int]],
    diagnostics: dict[str, Any],
    margin: float,
    ste_error_breakdown: list[tuple[str, int]] | None = None,
    prefix: str = "v_plm_ste",
) -> Path:
    condition_order = _condition_order(prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(f"# SLiPP++ {prefix} ensemble + PLM/STE tiebreaker metrics\n\n")
        f.write(
            f"_Probability-averaging ensemble of RF + XGB + LGBM with a binary "
            f"PLM-vs-STE tiebreaker (margin < {margin:.2f}), "
            f"25 stratified shuffle iterations._\n\n"
        )
        f.write(
            "| condition | 10-class macro-F1 | 5-lipid macro-F1 | binary F1 | "
            "AUROC | PLM F1 | STE F1 | PLM recall | STE recall |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for cond in condition_order:
            if cond not in summaries:
                continue
            s = summaries[cond]
            plm_m, plm_s = s["per_class_f1"]["PLM"]
            ste_m, ste_s = s["per_class_f1"]["STE"]
            f.write(
                f"| {cond} "
                f"| {_fmt_ms(s['macro_f1_mean'], s['macro_f1_std'])} "
                f"| {_fmt_ms(s['lipid_macro_f1_mean'], s['lipid_macro_f1_std'])} "
                f"| {_fmt_ms(s['binary_f1_mean'], s['binary_f1_std'])} "
                f"| {_fmt_ms(s['auroc_mean'], s['auroc_std'])} "
                f"| {_fmt_ms(plm_m, plm_s)} "
                f"| {_fmt_ms(ste_m, ste_s)} "
                f"| {_recall_from_confusion(confusions[cond], 'PLM'):.3f} "
                f"| {_recall_from_confusion(confusions[cond], 'STE'):.3f} |\n"
            )

        f.write("\n## PLM vs STE confusion counts (summed over 25 iterations)\n\n")
        f.write("| condition | PLM correct | STE correct | PLM\u2192STE | STE\u2192PLM |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cond in condition_order:
            if cond not in confusions:
                continue
            c = confusions[cond]
            f.write(
                f"| {cond} | {c['PLM_correct']} | {c['STE_correct']} "
                f"| {c['PLM_as_STE']} | {c['STE_as_PLM']} |\n"
            )

        f.write("\n## Tiebreaker diagnostics\n\n")
        fires = diagnostics["fire_counts_per_iter"]
        f.write(
            f"- Tiebreaker fired: mean = {diagnostics['fire_mean']:.1f} "
            f"rows/iter (std {diagnostics['fire_std']:.1f}), "
            f"total = {diagnostics['fire_total']} over {len(fires)} iterations\n"
        )
        f.write("- Per-iteration fire counts: " + ", ".join(str(x) for x in fires) + "\n")
        f.write(
            f"- Tiebreaker PLM/STE-only binary F1 (STE=positive, on true PLM+STE "
            f"test rows): {diagnostics['plm_ste_binary_f1_mean']:.3f} "
            f"\u00b1 {diagnostics['plm_ste_binary_f1_std']:.3f}\n"
        )
        if ste_error_breakdown:
            f.write(
                "\n### Where do true STE rows get misclassified? (ensemble + "
                "tiebreaker, summed over all iterations)\n\n"
            )
            f.write("| predicted class | count |\n|---|---:|\n")
            for cls, cnt in ste_error_breakdown:
                f.write(f"| {cls} | {cnt} |\n")
            f.write(
                "\n_Note: the tiebreaker only fires when both PLM and STE are "
                "in the ensemble's top-2. If STE loses to CLR, OLA, or another "
                "class instead of PLM, it is outside this tiebreaker's scope._\n"
            )

        residual_confusions = diagnostics.get("residual_lipid_confusions") or []
        if residual_confusions:
            f.write("\n## Residual lipid confusion candidates\n\n")
            f.write(
                "| rank | true | predicted | count | error share | "
                "top-2 recoverable | mean margin |\n"
            )
            f.write("|---:|---|---|---:|---:|---:|---:|\n")
            for rank, row in enumerate(residual_confusions, start=1):
                f.write(
                    f"| {rank} | {row['true_label']} | {row['pred_label']} "
                    f"| {int(row['count'])} "
                    f"| {float(row['error_fraction_of_true']):.3f} "
                    f"| {float(row['top2_recoverable_fraction']):.3f} "
                    f"| {float(row['mean_margin']):.3f} |\n"
                )

        candidate_rules = diagnostics.get("candidate_boundary_rules") or []
        if candidate_rules:
            f.write("\n## Candidate next boundary heads\n\n")
            f.write("| rank | name | positive | negative | margin | max rank |\n")
            f.write("|---:|---|---|---|---:|---:|\n")
            for rank, rule in enumerate(candidate_rules, start=1):
                f.write(
                    f"| {rank} | {rule['name']} | {rule['positive_label']} "
                    f"| {', '.join(rule['negative_labels'])} "
                    f"| {float(rule['margin']):.2f} | {rule['max_rank']} |\n"
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


def compose_overall_report(
    output_path: Path,
    *,
    summaries: dict[str, dict[str, Any]],
    confusions: dict[str, dict[str, int]],
    diagnostics: dict[str, Any],
    margin: float,
    ste_error_breakdown: list[tuple[str, int]] | None = None,
    prefix: str = "v_plm_ste",
) -> Path:
    """Compose the canonical consolidated comparison report."""
    return write_plm_ste_tiebreaker_report(
        output_path,
        summaries=summaries,
        confusions=confusions,
        diagnostics=diagnostics,
        margin=margin,
        ste_error_breakdown=ste_error_breakdown,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PLM-vs-STE binary tiebreaker head on top of the ensemble.",
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
        "(e.g. reports/v_plm_ste/overall_metrics.md).",
    )
    p.add_argument(
        "--model-bundle",
        type=Path,
        default=None,
        help="Any v_plm_ste iteration-0 joblib bundle (used to infer feature_columns).",
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--margin", type=float, default=TIEBREAKER_MARGIN_DEFAULT)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument(
        "--prefix",
        default="v_plm_ste",
        help="Label prefix for conditions in the report (use 'v_sterol' "
        "when the tiebreaker runs on the v_sterol ensemble).",
    )
    return p.parse_args(argv)


def _infer_feature_columns(args: argparse.Namespace) -> list[str]:
    """Resolve feature columns by probing bundles, then falling back to constants."""
    if args.model_bundle is not None and args.model_bundle.exists():
        bundle = joblib.load(args.model_bundle)
        return list(bundle["feature_columns"])
    candidates = [
        Path("models/v_plm_ste/xgb_multiclass.joblib"),
        Path("models/v_sterol/xgb_multiclass.joblib"),
    ]
    for c in candidates:
        if c.exists():
            bundle = joblib.load(c)
            return list(bundle["feature_columns"])
    if "v_plm_ste" in FEATURE_SETS:
        return list(FEATURE_SETS["v_plm_ste"])
    # Conservative fallback: use the richest available chem-aware feature set.
    return list(FEATURE_SETS["v_sterol"])


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
        prefix=args.prefix,
    )
    if args.overall_report is not None:
        compose_overall_report(
            args.overall_report,
            summaries=result["summaries"],
            confusions=result["confusions"],
            diagnostics=result["diagnostics"],
            margin=args.margin,
            ste_error_breakdown=_ste_error_breakdown(result["augmented_predictions"]),
            prefix=args.prefix,
        )
        print(f"wrote {args.overall_report}")

    s_tb = result["summaries"][f"{args.prefix} ensemble + plm_ste_tiebreaker"]
    s_en = result["summaries"][f"{args.prefix} ensemble (mean prob)"]
    print(
        "ensemble: "
        f"PLM F1={s_en['per_class_f1']['PLM'][0]:.3f}, "
        f"STE F1={s_en['per_class_f1']['STE'][0]:.3f}, "
        f"macro-F1={s_en['macro_f1_mean']:.3f}"
    )
    print(
        "ensemble + plm_ste_tiebreaker: "
        f"PLM F1={s_tb['per_class_f1']['PLM'][0]:.3f}, "
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
