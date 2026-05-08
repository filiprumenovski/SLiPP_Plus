"""Minimal Day 1 reproduction smoke run.

Run from the repository root:

    uv run python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from slipp_plus.config import Settings, load_settings
from slipp_plus.evaluate import run_evaluation
from slipp_plus.ingest import run_ingest
from slipp_plus.train import run_training


def main() -> None:
    with TemporaryDirectory(prefix="slipp_plus_quickstart_") as tmp:
        tmp_path = Path(tmp)

        # 1. Load config.
        raw = load_settings(Path("configs/day1.yaml")).model_dump(mode="python")
        raw["n_iterations"] = 1
        raw["models"] = ["xgb"]
        raw["paths"]["processed_dir"] = tmp_path / "processed"
        raw["paths"]["models_dir"] = tmp_path / "models"
        raw["paths"]["reports_dir"] = tmp_path / "reports"
        settings = Settings.model_validate(raw)

        # 2. Ingest.
        run_ingest(settings)

        # 3. Train one iteration and one model.
        run_training(settings)

        # 4. Evaluate.
        outputs = run_evaluation(settings)

        # 5. Print binary F1.
        metrics = pd.read_parquet(outputs["raw_metrics"])
        row = metrics.loc[(metrics["iteration"] == 0) & (metrics["model"] == "xgb")].iloc[0]
        print(f"binary_f1={row['binary_f1']:.3f}")


if __name__ == "__main__":
    main()
