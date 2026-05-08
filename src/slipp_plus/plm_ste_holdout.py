"""Holdout validation for the v_sterol ensemble plus PLM/STE tiebreaker.

This path is intentionally separate from the main Day 1 evaluation command.
It reuses the already-trained iteration-0 multiclass bundles, averages their
soft probabilities on the apo-PDB and AlphaFold holdouts, then trains the
existing PLM-vs-STE binary head on the iteration-0 train fold (seed_00) and
applies it to the holdout ensemble probabilities.

The holdouts only carry ``class_binary``, so validation is limited to the
paper's binary metrics: F1, AUROC, precision, sensitivity, specificity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from .artifact_schema import validate_feature_schema_metadata
from .boundary_head import BoundaryRule, apply_boundary_head, build_boundary_training, train_boundary_head
from .config import Settings, load_settings
from .constants import CLASS_10, LIG_TO_CLASS
from .ensemble import DEFAULT_MODELS, LIPID_IDX, PROBA_COLUMNS
from .plm_ste_tiebreaker import PLM_STE_RULE


PLM = "PLM"
STE = "STE"
PLM_IDX = CLASS_10.index(PLM)
STE_IDX = CLASS_10.index(STE)


def binary_metrics_from_probabilities(
    true_bin: np.ndarray,
    p_lipid: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the paper's binary metrics from lipid probabilities."""
    pred_bin = (p_lipid >= threshold).astype(int)

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
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(true_bin, p_lipid))
    except ValueError:
        auroc = float("nan")

    return {
        "n": int(len(true_bin)),
        "n_lipid": int(true_bin.sum()),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": auroc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _load_model_bundles(
    models_dir: Path,
    *,
    expected_feature_columns: list[str],
    expected_feature_set: str,
) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for key in DEFAULT_MODELS:
        bundle_path = models_dir / f"{key}_multiclass.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"missing model bundle: {bundle_path}")
        bundle = joblib.load(bundle_path)
        validate_feature_schema_metadata(
            bundle,
            expected_feature_columns=expected_feature_columns,
            expected_feature_set=expected_feature_set,
            artifact_label=f"model bundle {bundle_path.name}",
        )
        bundles[key] = bundle
    return bundles


def _predict_bundle_probabilities(
    bundles: dict[str, dict[str, Any]],
    holdout_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    probas: dict[str, np.ndarray] = {}
    for key, bundle in bundles.items():
        cols = list(bundle["feature_columns"])
        missing = [c for c in cols if c not in holdout_df.columns]
        if missing:
            raise KeyError(f"holdout missing feature columns for {key}: {missing}")
        X = holdout_df[cols].to_numpy(dtype=np.float64)
        probas[key] = bundle["model"].predict_proba(X)
    return probas


def ensemble_holdout_frame(
    holdout_df: pd.DataFrame,
    probas_by_model: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Build an ensemble-style frame so the existing tiebreaker can be reused."""
    keys = list(probas_by_model)
    stacked = np.stack([probas_by_model[k] for k in keys], axis=0)
    proba = stacked.mean(axis=0)
    y_pred = proba.argmax(axis=1).astype(np.int64)
    row_index = np.arange(len(holdout_df), dtype=np.int64)
    y_true_placeholder = np.full(len(holdout_df), -1, dtype=np.int64)

    cols: dict[str, Any] = {
        "iteration": np.zeros(len(holdout_df), dtype=np.int64),
        "row_index": row_index,
        "y_true_int": y_true_placeholder,
        "y_pred_int": y_pred,
    }
    for i, c in enumerate(PROBA_COLUMNS):
        cols[c] = proba[:, i]
    return pl.DataFrame(cols)


def train_iteration0_tiebreaker(
    full_pockets_path: Path,
    split_path: Path,
    feature_columns: list[str],
    seed: int,
):
    full = pd.read_parquet(full_pockets_path)
    X_tr, y_tr, _, _, _ = build_boundary_training(
        full,
        feature_columns,
        split_path,
        PLM_STE_RULE,
    )
    return train_boundary_head(X_tr, y_tr, seed=seed)


def _holdout_rule(margin: float) -> BoundaryRule:
    return BoundaryRule(
        name=PLM_STE_RULE.name,
        positive_label=PLM_STE_RULE.positive_label,
        negative_labels=PLM_STE_RULE.negative_labels,
        margin=margin,
        max_rank=PLM_STE_RULE.max_rank,
        fired_column=PLM_STE_RULE.fired_column,
        score_column=PLM_STE_RULE.score_column,
    )


def score_holdout_condition(frame: pl.DataFrame, true_bin: np.ndarray) -> dict[str, float]:
    proba = frame.select(PROBA_COLUMNS).to_numpy()
    p_lipid = proba[:, LIPID_IDX].sum(axis=1)
    return binary_metrics_from_probabilities(true_bin, p_lipid)


def infer_holdout_classes(holdout_df: pd.DataFrame) -> np.ndarray:
    ligands = holdout_df["ligand"].fillna("").astype(str).str.upper()
    return ligands.map(LIG_TO_CLASS).to_numpy(dtype=object)


def pair_confusion_metrics(
    frame: pl.DataFrame,
    true_classes: np.ndarray,
) -> dict[str, float]:
    pair_mask = np.isin(true_classes, [PLM, STE])
    if not pair_mask.any():
        return {
            "n_pair": 0,
            "plm_f1": float("nan"),
            "ste_f1": float("nan"),
            "plm_correct": 0,
            "ste_correct": 0,
            "plm_as_ste": 0,
            "ste_as_plm": 0,
        }

    y_true = true_classes[pair_mask]
    y_pred_idx = frame.filter(pl.Series("pair_mask", pair_mask))["y_pred_int"].to_numpy()
    y_pred = np.array([CLASS_10[int(v)] for v in y_pred_idx], dtype=object)

    def _f1_for(label: str) -> float:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "n_pair": int(pair_mask.sum()),
        "plm_f1": float(_f1_for(PLM)),
        "ste_f1": float(_f1_for(STE)),
        "plm_correct": int(((y_true == PLM) & (y_pred == PLM)).sum()),
        "ste_correct": int(((y_true == STE) & (y_pred == STE)).sum()),
        "plm_as_ste": int(((y_true == PLM) & (y_pred == STE)).sum()),
        "ste_as_plm": int(((y_true == STE) & (y_pred == PLM)).sum()),
    }


def write_holdout_report(
    output_path: Path,
    *,
    metrics: dict[str, dict[str, dict[str, float]]],
    margin: float,
    split_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(v: float | int) -> str:
        if isinstance(v, int):
            return str(v)
        if np.isnan(v):
            return "nan"
        return f"{v:.3f}"

    with output_path.open("w") as f:
        f.write("# v_sterol holdout validation — ensemble + PLM/STE tiebreaker\n\n")
        f.write(
            "_Iteration-0 holdout validation using the saved v_sterol RF/XGB/LGBM "
            f"multiclass bundles, mean-probability ensembling, and a PLM-vs-STE "
            f"binary head trained on {split_path.name} with margin < {margin:.2f}._\n\n"
        )
        f.write(
            "| Holdout | Condition | N | N lipid | F1 | AUROC | precision | sensitivity | specificity | ΔF1 vs base | ΔAUROC vs base |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for holdout in ["apo_pdb", "alphafold"]:
            base = metrics[holdout]["ensemble"]
            tb = metrics[holdout]["ensemble_plus_tiebreaker"]
            for label, row in [
                ("ensemble", base),
                ("ensemble + plm_ste_tiebreaker", tb),
            ]:
                delta_f1 = row["f1"] - base["f1"]
                delta_auroc = row["auroc"] - base["auroc"]
                f.write(
                    f"| {holdout} | {label} | {_fmt(row['n'])} | {_fmt(row['n_lipid'])} "
                    f"| {_fmt(row['f1'])} | {_fmt(row['auroc'])} | {_fmt(row['precision'])} "
                    f"| {_fmt(row['sensitivity'])} | {_fmt(row['specificity'])} "
                    f"| {delta_f1:+.3f} | {delta_auroc:+.3f} |\n"
                )
        f.write(
            "\n_Binary holdout F1/AUROC are invariant here by construction: the PLM/STE "
            "arbiter only redistributes mass within lipid subclasses, so summed lipid "
            "probability is unchanged. The discriminating holdout check is the pair-only "
            "PLM/STE section below._\n"
        )

        f.write("\n## PLM/STE pair-only holdout metrics\n\n")
        f.write(
            "| Holdout | Condition | Pair rows | PLM F1 | STE F1 | PLM correct | STE correct | PLM→STE | STE→PLM | Fired rows |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for holdout in ["apo_pdb", "alphafold"]:
            for label in ["ensemble", "ensemble + plm_ste_tiebreaker"]:
                key = "ensemble" if label == "ensemble" else "ensemble_plus_tiebreaker"
                row = metrics[holdout][key]
                pair = row["pair_metrics"]
                f.write(
                    f"| {holdout} | {label} | {_fmt(pair['n_pair'])} | {_fmt(pair['plm_f1'])} "
                    f"| {_fmt(pair['ste_f1'])} | {_fmt(pair['plm_correct'])} | {_fmt(pair['ste_correct'])} "
                    f"| {_fmt(pair['plm_as_ste'])} | {_fmt(pair['ste_as_plm'])} "
                    f"| {_fmt(row['tiebreaker_fired'])} |\n"
                )

        f.write("\n## Interpretation\n\n")
        apo_delta = metrics["apo_pdb"]["ensemble_plus_tiebreaker"]["f1"] - metrics["apo_pdb"]["ensemble"]["f1"]
        af_delta = metrics["alphafold"]["ensemble_plus_tiebreaker"]["f1"] - metrics["alphafold"]["ensemble"]["f1"]
        apo_ste_delta = (
            metrics["apo_pdb"]["ensemble_plus_tiebreaker"]["pair_metrics"]["ste_f1"]
            - metrics["apo_pdb"]["ensemble"]["pair_metrics"]["ste_f1"]
        )
        af_ste_delta = (
            metrics["alphafold"]["ensemble_plus_tiebreaker"]["pair_metrics"]["ste_f1"]
            - metrics["alphafold"]["ensemble"]["pair_metrics"]["ste_f1"]
        )
        if apo_ste_delta >= 0.0 and af_ste_delta >= 0.0:
            verdict = "On the pair-only holdout rows, the PLM/STE tiebreaker does not reduce STE F1 on either holdout."
        elif apo_ste_delta < 0.0 and af_ste_delta < 0.0:
            verdict = "On the pair-only holdout rows, the PLM/STE tiebreaker reduces STE F1 on both holdouts, consistent with test-split overfitting."
        else:
            verdict = "On the pair-only holdout rows, the PLM/STE tiebreaker has mixed effects: one holdout improves while the other degrades."
        f.write(verdict + "\n\n")
        f.write(
            f"Binary metrics are listed for completeness and remain unchanged "
            f"(apo ΔF1={apo_delta:+.3f}, AlphaFold ΔF1={af_delta:+.3f}); the real signal is "
            f"STE F1 on pair rows (apo Δ={apo_ste_delta:+.3f}, AlphaFold Δ={af_ste_delta:+.3f}).\n"
        )
    return output_path


def run_holdout_validation(
    settings: Settings,
    *,
    full_pockets_path: Path,
    splits_dir: Path,
    output_path: Path,
    margin: float = 0.99,
    seed: int | None = None,
    predictions_dir: Path | None = None,
) -> dict[str, Any]:
    bundles = _load_model_bundles(
        settings.paths.models_dir,
        expected_feature_columns=settings.feature_columns(),
        expected_feature_set=settings.feature_set,
    )
    xgb_bundle = bundles["xgb"]
    seed = settings.seed_base if seed is None else seed
    split_path = splits_dir / "seed_00.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"missing holdout training split: {split_path}")

    tiebreaker_model = train_iteration0_tiebreaker(
        full_pockets_path=full_pockets_path,
        split_path=split_path,
        feature_columns=list(xgb_bundle["feature_columns"]),
        seed=seed,
    )

    processed_dir = Path(settings.paths.processed_dir)
    holdout_files = {
        "apo_pdb": processed_dir / "apo_pdb_holdout.parquet",
        "alphafold": processed_dir / "alphafold_holdout.parquet",
    }

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    outputs: dict[str, Path] = {}

    if predictions_dir is None:
        predictions_dir = output_path.parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for holdout, parquet_path in holdout_files.items():
        holdout_df = pd.read_parquet(parquet_path)
        true_bin = holdout_df["class_binary"].to_numpy(dtype=int)
        true_classes = infer_holdout_classes(holdout_df)
        probas_by_model = _predict_bundle_probabilities(bundles, holdout_df)
        ensemble_df = ensemble_holdout_frame(holdout_df, probas_by_model)

        X_holdout = holdout_df[list(xgb_bundle["feature_columns"])].to_numpy(dtype=np.float64)
        p_ste = tiebreaker_model.predict_proba(X_holdout)[:, 1]
        row_lookup = np.arange(len(holdout_df), dtype=np.int64)
        augmented = apply_boundary_head(
            ensemble_df,
            p_ste,
            row_lookup,
            _holdout_rule(margin),
        )

        base_metrics = score_holdout_condition(ensemble_df, true_bin)
        tb_metrics = score_holdout_condition(augmented, true_bin)
        base_metrics["pair_metrics"] = pair_confusion_metrics(ensemble_df, true_classes)
        tb_metrics["pair_metrics"] = pair_confusion_metrics(augmented, true_classes)
        base_metrics["tiebreaker_fired"] = 0
        tb_metrics["tiebreaker_fired"] = int(augmented["tiebreaker_fired"].sum())

        metrics[holdout] = {
            "ensemble": base_metrics,
            "ensemble_plus_tiebreaker": tb_metrics,
        }

        ensemble_path = predictions_dir / f"{holdout}_ensemble_predictions.parquet"
        augmented_path = predictions_dir / f"{holdout}_plm_ste_tiebreaker_predictions.parquet"
        ensemble_df.write_parquet(ensemble_path)
        augmented.write_parquet(augmented_path)
        outputs[f"{holdout}_ensemble_predictions"] = ensemble_path
        outputs[f"{holdout}_tiebreaker_predictions"] = augmented_path

    report_path = write_holdout_report(
        output_path,
        metrics=metrics,
        margin=margin,
        split_path=split_path,
    )
    outputs["report"] = report_path
    return {
        "metrics": metrics,
        "outputs": outputs,
        "split_path": split_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Holdout validation for the v_sterol ensemble + PLM/STE tiebreaker.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/v_sterol.yaml"))
    parser.add_argument("--full-pockets", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("processed/splits"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = load_settings(args.config)
    result = run_holdout_validation(
        settings,
        full_pockets_path=args.full_pockets,
        splits_dir=args.splits_dir,
        output_path=args.output,
        predictions_dir=args.predictions_dir,
        margin=args.margin,
        seed=args.seed,
    )
    print(f"wrote {result['outputs']['report']}")
    for holdout in ["apo_pdb", "alphafold"]:
        base = result["metrics"][holdout]["ensemble"]
        tb = result["metrics"][holdout]["ensemble_plus_tiebreaker"]
        print(
            f"{holdout}: ensemble F1={base['f1']:.3f}, AUROC={base['auroc']:.3f}; "
            f"tiebreaker F1={tb['f1']:.3f}, AUROC={tb['auroc']:.3f}, "
            f"fires={tb['tiebreaker_fired']}"
        )


if __name__ == "__main__":
    main()