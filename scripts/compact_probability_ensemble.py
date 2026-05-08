#!/usr/bin/env python3
"""Average compact model probabilities and write an ablation report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from slipp_plus.constants import CLASS_10
from slipp_plus.evaluate import (
    _aggregate,
    evaluate_staged_holdout_predictions,
    evaluate_test_predictions,
)

PROBA_COLUMNS = [f"p_{label}" for label in CLASS_10]


def _average_prediction_frames(paths: list[Path], model_name: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path).sort_values(["iteration", "row_index"]).reset_index(drop=True)
        for path in paths
    ]
    if not frames:
        raise ValueError("at least one prediction parquet is required")
    base = frames[0].copy()
    keys = ["iteration", "row_index", "y_true_int"]
    for path, frame in zip(paths[1:], frames[1:], strict=False):
        if not base[keys].equals(frame[keys]):
            raise ValueError(f"prediction keys do not align: {path}")
    averaged = sum(frame[PROBA_COLUMNS].to_numpy(dtype=float) for frame in frames) / len(frames)
    base[PROBA_COLUMNS] = averaged
    base["y_pred_int"] = averaged.argmax(axis=1)
    base["model"] = model_name
    return base


def _write_report(
    path: Path,
    *,
    summary: pd.Series,
    holdouts: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Compact shape3/shape6 probability ensemble\n\n"
        "| metric | value |\n|---|---:|\n"
        f"| binary_f1 | {summary['binary_f1_mean']:.3f} +/- {summary['binary_f1_std']:.3f} |\n"
        f"| binary_auroc | {summary['binary_auroc_mean']:.3f} +/- {summary['binary_auroc_std']:.3f} |\n"
        f"| macro_f1_10 | {summary['macro_f1_10_mean']:.3f} +/- {summary['macro_f1_10_std']:.3f} |\n"
        f"| lipid_macro_f1 | {summary['macro_f1_lipid5_mean']:.3f} +/- {summary['macro_f1_lipid5_std']:.3f} |\n"
        f"| apo_pdb_f1 | {holdouts['apo_pdb']['f1']:.3f} |\n"
        f"| apo_pdb_auroc | {holdouts['apo_pdb']['auroc']:.3f} |\n"
        f"| alphafold_f1 | {holdouts['alphafold']['f1']:.3f} |\n"
        f"| alphafold_auroc | {holdouts['alphafold']['auroc']:.3f} |\n\n"
        "Per-class F1 means:\n\n"
        "| class | F1 |\n|---|---:|\n"
        + "".join(f"| {label} | {summary[f'f1_{label}_mean']:.3f} |\n" for label in CLASS_10),
        encoding="utf-8",
    )


def run_compact_probability_ensemble(
    *,
    shape3_dir: Path,
    shape6_dir: Path,
    output_predictions_dir: Path,
    output_report_dir: Path,
) -> dict[str, Path]:
    """Average shape3 and shape6 compact predictions and write metrics."""

    output_predictions_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)
    averaged = _average_prediction_frames(
        [
            shape3_dir / "predictions" / "hierarchical_lipid_predictions.parquet",
            shape6_dir / "predictions" / "hierarchical_lipid_predictions.parquet",
        ],
        model_name="shape3_shape6_mean",
    )
    predictions_path = output_predictions_dir / "test_predictions.parquet"
    averaged.to_parquet(predictions_path, index=False)

    metrics = evaluate_test_predictions(averaged)
    raw_metrics_path = output_report_dir / "raw_metrics.parquet"
    metrics.to_parquet(raw_metrics_path, index=False)
    summary = _aggregate(metrics, ["model"]).iloc[0]

    holdouts: dict[str, dict[str, float]] = {}
    for holdout_name, holdout_file in (
        ("apo_pdb", "apo_pdb_holdout.parquet"),
        ("alphafold", "alphafold_holdout.parquet"),
    ):
        holdout_preds = _average_prediction_frames(
            [
                shape3_dir
                / "predictions"
                / "holdouts"
                / f"family_encoder_{holdout_name}_predictions.parquet",
                shape6_dir
                / "predictions"
                / "holdouts"
                / f"family_encoder_{holdout_name}_predictions.parquet",
            ],
            model_name="shape3_shape6_mean",
        )
        holdout_preds_path = output_predictions_dir / f"{holdout_name}_predictions.parquet"
        holdout_preds.to_parquet(holdout_preds_path, index=False)
        holdouts[holdout_name] = evaluate_staged_holdout_predictions(
            holdout_preds,
            pd.read_parquet(shape3_dir / holdout_file),
        )

    report_path = output_report_dir / "metrics.md"
    _write_report(report_path, summary=summary, holdouts=holdouts)
    return {
        "predictions": predictions_path,
        "raw_metrics": raw_metrics_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape3-dir", type=Path, default=Path("processed/v49_tunnel_shape3"))
    parser.add_argument("--shape6-dir", type=Path, default=Path("processed/v49_tunnel_shape"))
    parser.add_argument(
        "--output-predictions-dir",
        type=Path,
        default=Path("processed/compact_shape3_shape6_ensemble/predictions"),
    )
    parser.add_argument(
        "--output-report-dir",
        type=Path,
        default=Path("reports/compact_shape3_shape6_ensemble"),
    )
    args = parser.parse_args()
    outputs = run_compact_probability_ensemble(
        shape3_dir=args.shape3_dir,
        shape6_dir=args.shape6_dir,
        output_predictions_dir=args.output_predictions_dir,
        output_report_dir=args.output_report_dir,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
