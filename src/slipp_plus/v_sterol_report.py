"""Build the ``reports/v_sterol/metrics.md`` report.

Loads the iteration-wise predictions for v49 and v_sterol, computes per-class
F1 deltas (CLR / STE highlighted), feature importance for the iteration-0 XGB
model, and CLR<->STE confusion counts on the test split.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .constants import CLASS_10, LIPID_CODES

LIPID_IDX = np.array([i for i, c in enumerate(CLASS_10) if c in LIPID_CODES])


def _per_class_f1_mean(preds: pd.DataFrame, model_key: str) -> dict[str, float]:
    sub = preds[preds["model"] == model_key]
    f1_per_iter: dict[str, list[float]] = {c: [] for c in CLASS_10}
    for _, iter_df in sub.groupby("iteration"):
        y_true = iter_df["y_true_int"].to_numpy()
        y_pred = iter_df["y_pred_int"].to_numpy()
        for cls_idx, cls_name in enumerate(CLASS_10):
            mask_true = y_true == cls_idx
            if not mask_true.any() and not (y_pred == cls_idx).any():
                continue
            f1_per_iter[cls_name].append(
                f1_score(y_true == cls_idx, y_pred == cls_idx, zero_division=0.0)
            )
    return {c: float(np.mean(f1_per_iter[c])) if f1_per_iter[c] else float("nan") for c in CLASS_10}


def _macro_f1(preds: pd.DataFrame, model_key: str) -> tuple[float, float, float, float]:
    sub = preds[preds["model"] == model_key]
    macro10: list[float] = []
    lipid5: list[float] = []
    binary_f1: list[float] = []
    binary_auroc: list[float] = []
    proba_cols = [f"p_{c}" for c in CLASS_10]
    for _, iter_df in sub.groupby("iteration"):
        y_true = iter_df["y_true_int"].to_numpy()
        y_pred = iter_df["y_pred_int"].to_numpy()
        proba = iter_df[proba_cols].to_numpy()
        macro10.append(
            f1_score(
                y_true, y_pred, labels=np.arange(len(CLASS_10)), average="macro", zero_division=0.0
            )
        )
        lipid_mask = np.isin(np.arange(len(CLASS_10)), LIPID_IDX)
        _, _, f_per, _ = _per_class_support(y_true, y_pred)
        lipid5.append(float(np.mean(f_per[lipid_mask])))
        true_bin = np.isin(y_true, LIPID_IDX).astype(int)
        pred_bin = np.isin(y_pred, LIPID_IDX).astype(int)
        binary_f1.append(f1_score(true_bin, pred_bin, zero_division=0.0))
        p_lipid = proba[:, LIPID_IDX].sum(axis=1)
        try:
            binary_auroc.append(roc_auc_score(true_bin, p_lipid))
        except ValueError:
            binary_auroc.append(float("nan"))
    return (
        float(np.mean(macro10)),
        float(np.mean(lipid5)),
        float(np.mean(binary_f1)),
        float(np.nanmean(binary_auroc)),
    )


def _per_class_support(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.metrics import precision_recall_fscore_support

    return precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASS_10)), average=None, zero_division=0
    )


def _clr_ste_confusion(preds: pd.DataFrame, model_key: str) -> dict[str, float]:
    clr_idx = CLASS_10.index("CLR")
    ste_idx = CLASS_10.index("STE")
    sub = preds[preds["model"] == model_key]
    total_clr_as_ste = 0
    total_ste_as_clr = 0
    total_clr = 0
    total_ste = 0
    for _, iter_df in sub.groupby("iteration"):
        y_true = iter_df["y_true_int"].to_numpy()
        y_pred = iter_df["y_pred_int"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_10)))
        total_clr_as_ste += int(cm[clr_idx, ste_idx])
        total_ste_as_clr += int(cm[ste_idx, clr_idx])
        total_clr += int(cm[clr_idx, :].sum())
        total_ste += int(cm[ste_idx, :].sum())
    return {
        "total_clr_as_ste": total_clr_as_ste,
        "total_ste_as_clr": total_ste_as_clr,
        "total_clr_rows": total_clr,
        "total_ste_rows": total_ste,
        "clr_as_ste_rate": total_clr_as_ste / total_clr if total_clr else float("nan"),
        "ste_as_clr_rate": total_ste_as_clr / total_ste if total_ste else float("nan"),
    }


def _holdout_eval(bundle_path: Path, holdout_df: pd.DataFrame) -> dict[str, float]:
    bundle = joblib.load(bundle_path)
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in holdout_df.columns]
    if missing:
        return {"error": f"missing columns: {missing[:3]}..."}
    X = holdout_df[cols].to_numpy(dtype=np.float64)
    true_bin = holdout_df["class_binary"].to_numpy(dtype=int)
    proba = bundle["model"].predict_proba(X)
    p_lipid = proba[:, LIPID_IDX].sum(axis=1)
    pred_bin = (p_lipid >= 0.5).astype(int)
    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    try:
        auroc = roc_auc_score(true_bin, p_lipid)
    except ValueError:
        auroc = float("nan")
    return {"n": len(true_bin), "f1": f1, "auroc": auroc, "precision": prec, "sensitivity": sens}


def _xgb_feature_importance(bundle_path: Path, top_n: int = 15) -> list[tuple[str, float]]:
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    cols = bundle["feature_columns"]
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    items: list[tuple[str, float]] = []
    for feature_id, value in gain.items():
        if feature_id.startswith("f"):
            index = int(feature_id[1:])
            if 0 <= index < len(cols):
                items.append((cols[index], float(value)))
        elif feature_id in cols:
            items.append((feature_id, float(value)))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:top_n]


def run_report(v49_dir: Path, v_sterol_dir: Path, output_path: Path) -> Path:
    v49_preds = pd.read_parquet(v49_dir / "predictions" / "test_predictions.parquet")
    vst_preds = pd.read_parquet(v_sterol_dir / "predictions" / "test_predictions.parquet")

    v49_apo = pd.read_parquet(v49_dir / "apo_pdb_holdout.parquet")
    v49_af = pd.read_parquet(v49_dir / "alphafold_holdout.parquet")
    vst_apo = pd.read_parquet(v_sterol_dir / "apo_pdb_holdout.parquet")
    vst_af = pd.read_parquet(v_sterol_dir / "alphafold_holdout.parquet")

    v49_models_dir = Path("models/v49")
    vst_models_dir = Path("models/v_sterol")

    model_keys = ["rf", "xgb", "lgbm"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# v_sterol vs v49 — sterol-targeted feature set\n\n")
        handle.write(
            "_38 new features layered on top of v49 "
            "(17 fpocket descriptors + 20 AA counts + 12 aromatic/aliphatic shells)._\n\n"
        )
        handle.write(
            "New columns:\n"
            "- 28 chemistry-shell counts (7 groups x 4 shells)\n"
            "- 4 polar/hydrophobic ratios per shell\n"
            "- 5 alpha-sphere PCA features (lam1/2/3 + elongation + planarity)\n"
            "- 1 pocket burial (distance to protein CA centroid / max CA spread)\n\n"
        )

        # --- Per-class F1 with deltas ---
        handle.write("## Per-class F1 on test split (mean across 25 iterations)\n\n")
        handle.write(
            "| model |"
            + "".join(f" {c} |" for c in CLASS_10)
            + " macro-F1 (10) | macro-F1 (5 lipids) | binary F1 | binary AUROC |\n"
        )
        handle.write("|---|" + "---|" * len(CLASS_10) + "---|---|---|---|\n")
        for key in model_keys:
            v49_f1 = _per_class_f1_mean(v49_preds, key)
            vst_f1 = _per_class_f1_mean(vst_preds, key)
            v49_m10, v49_l5, v49_bf1, v49_au = _macro_f1(v49_preds, key)
            vst_m10, vst_l5, vst_bf1, vst_au = _macro_f1(vst_preds, key)
            handle.write(f"| {key} (v49) |" + "".join(f" {v49_f1[c]:.3f} |" for c in CLASS_10))
            handle.write(f" {v49_m10:.3f} | {v49_l5:.3f} | {v49_bf1:.3f} | {v49_au:.3f} |\n")
            handle.write(f"| {key} (v_sterol) |" + "".join(f" {vst_f1[c]:.3f} |" for c in CLASS_10))
            handle.write(f" {vst_m10:.3f} | {vst_l5:.3f} | {vst_bf1:.3f} | {vst_au:.3f} |\n")
            handle.write("| **" + key + " Δ** |")
            for c in CLASS_10:
                delta = vst_f1[c] - v49_f1[c]
                marker = " ⭑" if c in {"CLR", "STE"} else ""
                handle.write(f" **{delta:+.3f}**{marker} |")
            handle.write(
                f" **{vst_m10 - v49_m10:+.3f}** | **{vst_l5 - v49_l5:+.3f}** | "
                f"**{vst_bf1 - v49_bf1:+.3f}** | **{vst_au - v49_au:+.3f}** |\n"
            )
        handle.write("\n_⭑ CLR and STE — the sterol classes that motivated this sprint._\n\n")

        # --- CLR / STE focused ---
        handle.write("## CLR / STE headline (what we are trying to move)\n\n")
        handle.write(
            "| model | CLR v49 | CLR v_sterol | CLR Δ | STE v49 | STE v_sterol | STE Δ |\n"
        )
        handle.write("|---|---|---|---|---|---|---|\n")
        for key in model_keys:
            v49_f1 = _per_class_f1_mean(v49_preds, key)
            vst_f1 = _per_class_f1_mean(vst_preds, key)
            handle.write(
                f"| {key} | {v49_f1['CLR']:.3f} | {vst_f1['CLR']:.3f} | "
                f"{vst_f1['CLR'] - v49_f1['CLR']:+.3f} | {v49_f1['STE']:.3f} | "
                f"{vst_f1['STE']:.3f} | {vst_f1['STE'] - v49_f1['STE']:+.3f} |\n"
            )
        handle.write("\n")

        # --- CLR <-> STE confusion ---
        handle.write("## CLR ↔ STE confusion (summed across all 25 test folds)\n\n")
        handle.write(
            "| model | variant | CLR rows | CLR → STE | rate | STE rows | STE → CLR | rate |\n"
        )
        handle.write("|---|---|---|---|---|---|---|---|\n")
        for key in model_keys:
            v49_cm = _clr_ste_confusion(v49_preds, key)
            vst_cm = _clr_ste_confusion(vst_preds, key)
            handle.write(
                f"| {key} | v49 | {v49_cm['total_clr_rows']} | {v49_cm['total_clr_as_ste']} "
                f"| {v49_cm['clr_as_ste_rate']:.3f} | {v49_cm['total_ste_rows']} "
                f"| {v49_cm['total_ste_as_clr']} | {v49_cm['ste_as_clr_rate']:.3f} |\n"
            )
            handle.write(
                f"| {key} | v_sterol | {vst_cm['total_clr_rows']} | {vst_cm['total_clr_as_ste']} "
                f"| {vst_cm['clr_as_ste_rate']:.3f} | {vst_cm['total_ste_rows']} "
                f"| {vst_cm['total_ste_as_clr']} | {vst_cm['ste_as_clr_rate']:.3f} |\n"
            )
        handle.write("\n")

        # --- Holdouts ---
        handle.write("## Holdouts (binary, iteration-0 model)\n\n")
        handle.write("### apo-PDB holdout\n\n")
        handle.write("| model | variant | F1 | AUROC | precision | sensitivity |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for key in model_keys:
            v49_bundle = v49_models_dir / f"{key}_multiclass.joblib"
            vst_bundle = vst_models_dir / f"{key}_multiclass.joblib"
            for variant, bundle, holdout in [
                ("v49", v49_bundle, v49_apo),
                ("v_sterol", vst_bundle, vst_apo),
            ]:
                if not bundle.exists():
                    handle.write(f"| {key} | {variant} | missing | | | |\n")
                    continue
                result = _holdout_eval(bundle, holdout)
                if "error" in result:
                    handle.write(f"| {key} | {variant} | n/a ({result['error']}) | | | |\n")
                    continue
                handle.write(
                    f"| {key} | {variant} | {result['f1']:.3f} | {result['auroc']:.3f} "
                    f"| {result['precision']:.3f} | {result['sensitivity']:.3f} |\n"
                )
        handle.write("\n### AlphaFold holdout\n\n")
        handle.write("| model | variant | F1 | AUROC | precision | sensitivity |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for key in model_keys:
            v49_bundle = v49_models_dir / f"{key}_multiclass.joblib"
            vst_bundle = vst_models_dir / f"{key}_multiclass.joblib"
            for variant, bundle, holdout in [
                ("v49", v49_bundle, v49_af),
                ("v_sterol", vst_bundle, vst_af),
            ]:
                if not bundle.exists():
                    handle.write(f"| {key} | {variant} | missing | | | |\n")
                    continue
                result = _holdout_eval(bundle, holdout)
                if "error" in result:
                    handle.write(f"| {key} | {variant} | n/a ({result['error']}) | | | |\n")
                    continue
                handle.write(
                    f"| {key} | {variant} | {result['f1']:.3f} | {result['auroc']:.3f} "
                    f"| {result['precision']:.3f} | {result['sensitivity']:.3f} |\n"
                )

        # --- Feature importance ---
        handle.write("\n## Top XGB feature importance (gain) — iteration-0 v_sterol model\n\n")
        xgb_bundle = vst_models_dir / "xgb_multiclass.joblib"
        if xgb_bundle.exists():
            top = _xgb_feature_importance(xgb_bundle, top_n=15)
            handle.write("| rank | feature | gain |\n|---|---|---|\n")
            for rank, (feature, gain_value) in enumerate(top, start=1):
                new_marker = ""
                sterol_cols = {
                    *(
                        f"{g}_count_shell{s}"
                        for g in [
                            "aromatic_pi",
                            "aromatic_polar",
                            "bulky_hydrophobic",
                            "small_special",
                            "polar_neutral",
                            "cationic",
                            "anionic",
                        ]
                        for s in (1, 2, 3, 4)
                    ),
                    *(f"polar_hydrophobic_ratio_shell{s}" for s in (1, 2, 3, 4)),
                    "pocket_lam1",
                    "pocket_lam2",
                    "pocket_lam3",
                    "pocket_elongation",
                    "pocket_planarity",
                    "pocket_burial",
                }
                if feature in sterol_cols:
                    new_marker = " ⬥"
                handle.write(f"| {rank} | `{feature}`{new_marker} | {gain_value:.2f} |\n")
            handle.write("\n_⬥ new sterol feature (not in v49)._\n")
        else:
            handle.write("_XGB bundle missing; skipped importance._\n")

    return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v49-dir", type=Path, default=Path("processed/v49"))
    parser.add_argument("--v-sterol-dir", type=Path, default=Path("processed/v_sterol"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v_sterol/metrics.md"),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = run_report(args.v49_dir, args.v_sterol_dir, args.output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
