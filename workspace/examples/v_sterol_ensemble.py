"""Inspect the recommended v_sterol ensemble's residual STE confusions.

Run the v_sterol pipeline first:

    uv run slipp_plus all --config configs/v_sterol.yaml

Then run this example from the repository root:

    uv run python examples/v_sterol_ensemble.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from slipp_plus.confusion_mining import mine_confusion_edges
from slipp_plus.ensemble import load_predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("processed/v_sterol/predictions/test_predictions.parquet"),
        help="Parquet predictions written by `slipp_plus train`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v_sterol/ste_confusion_mining.csv"),
        help="CSV table for STE residual confusion rows.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of STE confusion rows to print.",
    )
    return parser.parse_args()


def _format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No STE confusions found."
    columns = [
        "pred_label",
        "count",
        "true_support",
        "error_fraction_of_true",
        "top2_recoverable_fraction",
        "mean_margin",
    ]
    shown = frame.loc[:, columns].copy()
    for column in ("error_fraction_of_true", "top2_recoverable_fraction", "mean_margin"):
        shown[column] = shown[column].map(lambda value: f"{float(value):.3f}")
    return shown.to_string(index=False)


def main() -> None:
    args = _parse_args()
    if not args.predictions.exists():
        raise SystemExit(
            f"{args.predictions} does not exist. Run "
            "`uv run slipp_plus all --config configs/v_sterol.yaml` first."
        )

    predictions = load_predictions(args.predictions)
    lipid_edges = mine_confusion_edges(
        predictions,
        average_models=True,
        lipid_only=True,
        min_count=1,
    )
    ste_edges = lipid_edges[lipid_edges["true_label"] == "STE"].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ste_edges.to_csv(args.output, index=False)

    print("v_sterol ensemble residual STE confusions")
    print(f"predictions={args.predictions}")
    print(f"rows={len(ste_edges)}")
    print(_format_table(ste_edges.head(args.top)))
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
