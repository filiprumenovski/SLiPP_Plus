from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from slipp_plus.config import Settings, load_settings
from slipp_plus.evaluate import run_evaluation
from slipp_plus.ingest import run_ingest
from slipp_plus.train import run_training


def _day1_smoke_settings(tmp_path: Path) -> Settings:
    raw = load_settings(Path("configs/day1.yaml")).model_dump(mode="python")
    raw["n_iterations"] = 1
    raw["models"] = ["xgb"]
    raw["paths"]["processed_dir"] = tmp_path / "processed"
    raw["paths"]["models_dir"] = tmp_path / "models"
    raw["paths"]["reports_dir"] = tmp_path / "reports"
    return Settings.model_validate(raw)


@pytest.mark.slow
def test_day1_xgb_binary_f1_regression(tmp_path: Path) -> None:
    settings = _day1_smoke_settings(tmp_path)

    run_ingest(settings)
    run_training(settings)
    outputs = run_evaluation(settings)

    metrics = pd.read_parquet(outputs["raw_metrics"])
    row = metrics.loc[(metrics["iteration"] == 0) & (metrics["model"] == "xgb")].iloc[0]
    assert row["binary_f1"] == pytest.approx(settings.ground_truth.test.f1, abs=0.05)
