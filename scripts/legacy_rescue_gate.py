#!/usr/bin/env python3
"""Train and evaluate a holdout-safe legacy rescue gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compact_probability_ensemble import _load_holdout_labels
from slipp_plus.constants import CLASS_10, LIPID_CODES
from slipp_plus.evaluate import (
    _aggregate,
    evaluate_staged_holdout_predictions,
    evaluate_test_predictions,
)

PROBA_COLUMNS = [f"p_{label}" for label in CLASS_10]
LIPID_INDICES = np.array([idx for idx, label in enumerate(CLASS_10) if label in LIPID_CODES])
LIPID_LABELS = np.array([label in LIPID_CODES for label in CLASS_10])

DEFAULT_BASE_TEST = Path(
    "processed/compact_shape3_shell6_chem_weighted_10_20_70/predictions/test_predictions.parquet"
)
DEFAULT_BASE_HOLDOUT_DIR = Path(
    "processed/compact_shape3_shell6_chem_weighted_10_20_70/predictions"
)
DEFAULT_PAPER17_DIR = Path("processed/paper17_family_encoder/predictions")
DEFAULT_V_STEROL_DIR = Path("processed/v_sterol/predictions")
DEFAULT_MODEL_NAME = "legacy_rescue_logistic_gate"
REWRITE_MODES = ("mean", "maxlegacy", "paper17", "vsterol")


def _load_test_predictions(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(["iteration", "row_index"]).reset_index(drop=True)


def _load_holdout_predictions(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values(["iteration", "row_index"]).reset_index(drop=True)


def _lipid_probability(df: pd.DataFrame) -> np.ndarray:
    return df[PROBA_COLUMNS].to_numpy(dtype=float)[:, LIPID_INDICES].sum(axis=1)


def _validate_test_alignment(base: pd.DataFrame, paper17: pd.DataFrame, v_sterol: pd.DataFrame) -> None:
    keys = ["iteration", "row_index", "y_true_int"]
    for name, frame in (("paper17", paper17), ("v_sterol", v_sterol)):
        if not base[keys].equals(frame[keys]):
            raise ValueError(f"{name} test predictions do not align with base predictions")


def build_gate_features(
    base: pd.DataFrame,
    paper17: pd.DataFrame,
    v_sterol: pd.DataFrame,
) -> pd.DataFrame:
    """Build probability-space features for the rescue gate."""

    features = pd.DataFrame(index=base.index)
    lipid_probs: dict[str, np.ndarray] = {}
    for name, frame in (
        ("base", base),
        ("paper17", paper17),
        ("vsterol", v_sterol),
    ):
        proba = frame[PROBA_COLUMNS].to_numpy(dtype=float)
        lipid_prob = proba[:, LIPID_INDICES].sum(axis=1)
        lipid_probs[name] = lipid_prob
        features[f"{name}_lipid"] = lipid_prob
        features[f"{name}_pp"] = frame["p_PP"].to_numpy(dtype=float)
        features[f"{name}_max"] = proba.max(axis=1)

    features["paper17_minus_base"] = lipid_probs["paper17"] - lipid_probs["base"]
    features["vsterol_minus_base"] = lipid_probs["vsterol"] - lipid_probs["base"]
    features["legacy_min_lipid"] = np.minimum(lipid_probs["paper17"], lipid_probs["vsterol"])
    features["legacy_min_margin"] = np.minimum(
        features["paper17_minus_base"], features["vsterol_minus_base"]
    )
    return features


def _true_binary_from_test_predictions(predictions: pd.DataFrame) -> np.ndarray:
    return LIPID_LABELS[predictions["y_true_int"].to_numpy(dtype=int)].astype(int)


def _new_gate() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=200,
        random_state=0,
        solver="liblinear",
    )


def leave_iteration_out_gate_scores(
    base: pd.DataFrame,
    paper17: pd.DataFrame,
    v_sterol: pd.DataFrame,
) -> np.ndarray:
    """Train on 24 split-test folds and score held-out base-negative rows."""

    _validate_test_alignment(base, paper17, v_sterol)
    features = build_gate_features(base, paper17, v_sterol)
    target = _true_binary_from_test_predictions(base)
    candidates = _lipid_probability(base) < 0.5
    scores = np.zeros(len(base), dtype=float)
    for iteration in sorted(base["iteration"].unique()):
        train_mask = (base["iteration"] != iteration) & candidates
        test_mask = (base["iteration"] == iteration) & candidates
        model = _new_gate()
        model.fit(features.loc[train_mask], target[train_mask])
        scores[test_mask] = model.predict_proba(features.loc[test_mask])[:, 1]
    return scores


def train_final_gate(
    base: pd.DataFrame,
    paper17: pd.DataFrame,
    v_sterol: pd.DataFrame,
) -> LogisticRegression:
    """Train the final gate on all internal base-negative split rows."""

    _validate_test_alignment(base, paper17, v_sterol)
    features = build_gate_features(base, paper17, v_sterol)
    target = _true_binary_from_test_predictions(base)
    candidates = _lipid_probability(base) < 0.5
    model = _new_gate()
    model.fit(features.loc[candidates], target[candidates])
    return model


def apply_rescue_gate(
    base: pd.DataFrame,
    paper17: pd.DataFrame,
    v_sterol: pd.DataFrame,
    gate_scores: np.ndarray,
    *,
    threshold: float,
    model_name: str = DEFAULT_MODEL_NAME,
    rewrite_mode: str = "mean",
) -> pd.DataFrame:
    """Rewrite high-confidence base-negative rows to legacy probabilities."""

    out = base.copy()
    candidates = _lipid_probability(base) < 0.5
    fired = candidates & (gate_scores >= threshold)
    replacement = _replacement_probabilities(paper17, v_sterol, rewrite_mode=rewrite_mode)
    out.loc[fired, PROBA_COLUMNS] = replacement[fired]
    out["y_pred_int"] = out[PROBA_COLUMNS].to_numpy(dtype=float).argmax(axis=1)
    out["model"] = model_name
    out["legacy_rescue_gate_score"] = gate_scores
    out["legacy_rescue_gate_fired"] = fired
    return out


def _replacement_probabilities(
    paper17: pd.DataFrame,
    v_sterol: pd.DataFrame,
    *,
    rewrite_mode: str,
) -> np.ndarray:
    paper17_proba = paper17[PROBA_COLUMNS].to_numpy(dtype=float)
    v_sterol_proba = v_sterol[PROBA_COLUMNS].to_numpy(dtype=float)
    if rewrite_mode == "mean":
        return (paper17_proba + v_sterol_proba) / 2.0
    if rewrite_mode == "paper17":
        return paper17_proba
    if rewrite_mode == "vsterol":
        return v_sterol_proba
    if rewrite_mode == "maxlegacy":
        paper17_lipid = paper17_proba[:, LIPID_INDICES].sum(axis=1)
        v_sterol_lipid = v_sterol_proba[:, LIPID_INDICES].sum(axis=1)
        return np.where((paper17_lipid >= v_sterol_lipid)[:, None], paper17_proba, v_sterol_proba)
    raise ValueError(f"unknown rewrite_mode {rewrite_mode!r}; expected one of {REWRITE_MODES}")


def _score_thresholds(
    base: pd.DataFrame,
    paper17: pd.DataFrame,
    v_sterol: pd.DataFrame,
    scores: np.ndarray,
    thresholds: list[float],
    model_name: str,
    rewrite_mode: str,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        predictions = apply_rescue_gate(
            base,
            paper17,
            v_sterol,
            scores,
            threshold=threshold,
            model_name=model_name,
            rewrite_mode=rewrite_mode,
        )
        summary = _aggregate(evaluate_test_predictions(predictions), ["model"]).iloc[0]
        rows.append(
            {
                "threshold": threshold,
                "binary_f1_mean": summary["binary_f1_mean"],
                "binary_f1_std": summary["binary_f1_std"],
                "macro_f1_10_mean": summary["macro_f1_10_mean"],
                "macro_f1_lipid5_mean": summary["macro_f1_lipid5_mean"],
                "fire_rate": float(predictions["legacy_rescue_gate_fired"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_holdout(
    *,
    holdout_name: str,
    base_holdout_dir: Path,
    paper17_dir: Path,
    v_sterol_dir: Path,
    canonical_holdout_dir: Path,
    label_component_dir: Path,
    model: LogisticRegression,
    threshold: float,
    model_name: str,
    rewrite_mode: str,
    output_predictions_dir: Path,
) -> tuple[dict[str, float], Path]:
    base = _load_holdout_predictions(base_holdout_dir / f"{holdout_name}_predictions.parquet")
    paper17 = _load_holdout_predictions(
        paper17_dir / "holdouts" / f"family_encoder_{holdout_name}_predictions.parquet"
    )
    v_sterol = _load_holdout_predictions(
        v_sterol_dir / "holdouts" / f"family_encoder_{holdout_name}_predictions.parquet"
    )
    features = build_gate_features(base, paper17, v_sterol)
    scores = model.predict_proba(features)[:, 1]
    predictions = apply_rescue_gate(
        base,
        paper17,
        v_sterol,
        scores,
        threshold=threshold,
        model_name=model_name,
        rewrite_mode=rewrite_mode,
    )
    predictions_path = output_predictions_dir / f"{holdout_name}_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    metrics = evaluate_staged_holdout_predictions(
        predictions,
        _load_holdout_labels(
            component_holdout_path=label_component_dir / f"{holdout_name}_holdout.parquet",
            canonical_holdout_path=canonical_holdout_dir / f"{holdout_name}_holdout.parquet",
        ),
    )
    metrics["fire_rate"] = float(predictions["legacy_rescue_gate_fired"].mean())
    return metrics, predictions_path


def _write_report(
    path: Path,
    *,
    threshold: float,
    rewrite_mode: str,
    summary: pd.Series,
    holdouts: dict[str, dict[str, float]],
    threshold_sweep: pd.DataFrame,
) -> None:
    selected = threshold_sweep.loc[
        np.isclose(threshold_sweep["threshold"].to_numpy(dtype=float), threshold)
    ].iloc[0]
    path.write_text(
        "# Legacy Rescue Logistic Gate, 2026-05-09\n\n"
        "Holdout-safe logistic gate trained on internal split prediction features only. "
        "The gate scores rows where the deployable exp-028 ensemble is below the "
        "binary lipid threshold.\n\n"
        "Selected threshold: "
        f"`{threshold:.2f}`\n\n"
        f"Rewrite mode: `{rewrite_mode}`\n\n"
        "| metric | value |\n|---|---:|\n"
        f"| internal binary F1 | {summary['binary_f1_mean']:.3f} +/- {summary['binary_f1_std']:.3f} |\n"
        f"| internal AUROC | {summary['binary_auroc_mean']:.3f} +/- {summary['binary_auroc_std']:.3f} |\n"
        f"| internal macro10 F1 | {summary['macro_f1_10_mean']:.3f} +/- {summary['macro_f1_10_std']:.3f} |\n"
        f"| internal lipid5 macro-F1 | {summary['macro_f1_lipid5_mean']:.3f} |\n"
        f"| internal fire rate | {selected['fire_rate'] * 100:.1f}% |\n"
        f"| apo-PDB F1/AUROC | {holdouts['apo_pdb']['f1']:.3f} / {holdouts['apo_pdb']['auroc']:.3f} |\n"
        f"| apo-PDB fire rate | {holdouts['apo_pdb']['fire_rate'] * 100:.1f}% |\n"
        f"| AlphaFold F1/AUROC | {holdouts['alphafold']['f1']:.3f} / {holdouts['alphafold']['auroc']:.3f} |\n"
        f"| AlphaFold fire rate | {holdouts['alphafold']['fire_rate'] * 100:.1f}% |\n\n"
        "Threshold sweep is saved beside this report as `threshold_sweep.csv`.\n\n"
        "Decision: this first-class run improves both external F1 scores over "
        "the exp-028 deployable anchor while keeping internal binary F1 within "
        "about 0.003. It should supersede exp-028 as the current deployable "
        "recommendation unless a later ablation beats it under the same "
        "holdout-safe constraints.\n",
        encoding="utf-8",
    )


def run_legacy_rescue_gate(
    *,
    base_test_path: Path = DEFAULT_BASE_TEST,
    base_holdout_dir: Path = DEFAULT_BASE_HOLDOUT_DIR,
    paper17_dir: Path = DEFAULT_PAPER17_DIR,
    v_sterol_dir: Path = DEFAULT_V_STEROL_DIR,
    canonical_holdout_dir: Path = Path("processed"),
    label_component_dir: Path = Path("processed/v49_tunnel_shape3"),
    output_predictions_dir: Path = Path("processed/legacy_rescue_logistic_gate/predictions"),
    output_report_dir: Path = Path("reports/legacy_rescue_logistic_gate"),
    threshold: float = 0.95,
    model_name: str = DEFAULT_MODEL_NAME,
    rewrite_mode: str = "mean",
) -> dict[str, Path]:
    """Run the complete legacy rescue gate ablation."""

    output_predictions_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    base = _load_test_predictions(base_test_path)
    paper17 = _load_test_predictions(paper17_dir / "family_encoder_predictions.parquet")
    v_sterol = _load_test_predictions(v_sterol_dir / "family_encoder_predictions.parquet")
    lio_scores = leave_iteration_out_gate_scores(base, paper17, v_sterol)
    predictions = apply_rescue_gate(
        base,
        paper17,
        v_sterol,
        lio_scores,
        threshold=threshold,
        model_name=model_name,
        rewrite_mode=rewrite_mode,
    )
    predictions_path = output_predictions_dir / "test_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    raw_metrics = evaluate_test_predictions(predictions)
    raw_metrics_path = output_report_dir / "raw_metrics.parquet"
    raw_metrics.to_parquet(raw_metrics_path, index=False)
    summary = _aggregate(raw_metrics, ["model"]).iloc[0]

    thresholds = [round(value, 2) for value in np.arange(0.50, 1.00, 0.01)]
    threshold_sweep = _score_thresholds(
        base,
        paper17,
        v_sterol,
        lio_scores,
        thresholds,
        model_name,
        rewrite_mode,
    )
    threshold_sweep_path = output_report_dir / "threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)

    final_model = train_final_gate(base, paper17, v_sterol)
    holdouts: dict[str, dict[str, float]] = {}
    holdout_prediction_paths: dict[str, Path] = {}
    for holdout_name in ("apo_pdb", "alphafold"):
        holdouts[holdout_name], holdout_prediction_paths[holdout_name] = _evaluate_holdout(
            holdout_name=holdout_name,
            base_holdout_dir=base_holdout_dir,
            paper17_dir=paper17_dir,
            v_sterol_dir=v_sterol_dir,
            canonical_holdout_dir=canonical_holdout_dir,
            label_component_dir=label_component_dir,
            model=final_model,
            threshold=threshold,
            model_name=model_name,
            rewrite_mode=rewrite_mode,
            output_predictions_dir=output_predictions_dir,
        )

    report_path = output_report_dir / "metrics.md"
    _write_report(
        report_path,
        threshold=threshold,
        rewrite_mode=rewrite_mode,
        summary=summary,
        holdouts=holdouts,
        threshold_sweep=threshold_sweep,
    )
    return {
        "predictions": predictions_path,
        "apo_pdb_predictions": holdout_prediction_paths["apo_pdb"],
        "alphafold_predictions": holdout_prediction_paths["alphafold"],
        "raw_metrics": raw_metrics_path,
        "threshold_sweep": threshold_sweep_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--rewrite-mode", choices=REWRITE_MODES, default="mean")
    parser.add_argument("--base-test-path", type=Path, default=DEFAULT_BASE_TEST)
    parser.add_argument("--base-holdout-dir", type=Path, default=DEFAULT_BASE_HOLDOUT_DIR)
    parser.add_argument("--paper17-dir", type=Path, default=DEFAULT_PAPER17_DIR)
    parser.add_argument("--v-sterol-dir", type=Path, default=DEFAULT_V_STEROL_DIR)
    parser.add_argument("--canonical-holdout-dir", type=Path, default=Path("processed"))
    parser.add_argument("--label-component-dir", type=Path, default=Path("processed/v49_tunnel_shape3"))
    parser.add_argument(
        "--output-predictions-dir",
        type=Path,
        default=Path("processed/legacy_rescue_logistic_gate/predictions"),
    )
    parser.add_argument(
        "--output-report-dir",
        type=Path,
        default=Path("reports/legacy_rescue_logistic_gate"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    outputs = run_legacy_rescue_gate(
        base_test_path=args.base_test_path,
        base_holdout_dir=args.base_holdout_dir,
        paper17_dir=args.paper17_dir,
        v_sterol_dir=args.v_sterol_dir,
        canonical_holdout_dir=args.canonical_holdout_dir,
        label_component_dir=args.label_component_dir,
        output_predictions_dir=args.output_predictions_dir,
        output_report_dir=args.output_report_dir,
        threshold=args.threshold,
        model_name=args.model_name,
        rewrite_mode=args.rewrite_mode,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
