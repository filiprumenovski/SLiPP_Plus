from __future__ import annotations

from pathlib import Path

import pandas as pd

import slipp_plus.hierarchical_pipeline as hierarchical_pipeline
from slipp_plus.config import Settings, load_settings
from slipp_plus.constants import CLASS_10, HIERARCHICAL_PREDICTIONS_NAME
from slipp_plus.evaluate import evaluate_staged_holdout_predictions, run_evaluation


def _settings_with_paths(tmp_path: Path, *, pipeline_mode: str = "flat") -> Settings:
    raw = load_settings(Path("configs/day1.yaml")).model_dump(mode="python")
    raw["pipeline_mode"] = pipeline_mode
    raw["paths"]["processed_dir"] = tmp_path / "processed"
    raw["paths"]["reports_dir"] = tmp_path / "reports"
    raw["paths"]["models_dir"] = tmp_path / "models"
    return Settings.model_validate(raw)


def test_settings_accept_hierarchical_pipeline_config(tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path)
    assert settings.pipeline_mode == "flat"
    assert settings.hierarchical.stage1_source == "ensemble"

    raw = settings.model_dump(mode="python")
    raw["pipeline_mode"] = "hierarchical"
    raw["hierarchical"] = {
        "stage1_source": "trained",
        "ste_threshold": 0.55,
        "workers": 2,
        "lipid_family_feature_set": "v49",
        "specialist_feature_set": "v_sterol",
        "nonlipid_feature_set": "v14",
        "boundary_heads": [
            {
                "name": "ste_vs_plm",
                "positive_label": "STE",
                "negative_labels": ["PLM"],
                "margin": 0.99,
                "max_rank": 2,
                "feature_set": "v_sterol",
            }
        ],
    }

    validated = Settings.model_validate(raw)
    assert validated.pipeline_mode == "hierarchical"
    assert validated.hierarchical.stage1_source == "trained"
    assert validated.hierarchical.ste_threshold == 0.55
    assert validated.hierarchical.workers == 2
    assert validated.hierarchical.lipid_family_feature_set == "v49"
    assert validated.hierarchical.specialist_feature_set == "v_sterol"
    assert validated.hierarchical.nonlipid_feature_set == "v14"
    assert validated.hierarchical.boundary_heads[0].name == "ste_vs_plm"
    assert validated.hierarchical.boundary_heads[0].positive_label == "STE"
    assert validated.hierarchical.boundary_heads[0].negative_labels == ("PLM",)
    assert validated.hierarchical.boundary_heads[0].to_boundary_rule().max_rank == 2


def test_hierarchical_training_calls_primary_pipeline(monkeypatch, tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="hierarchical")
    processed = settings.paths.processed_dir
    models_dir = settings.paths.models_dir
    reports_dir = settings.paths.reports_dir
    (processed / "predictions").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def fake_run_hierarchical_training(incoming: Settings) -> dict[str, Path]:
        captured["settings"] = incoming
        pred = processed / "predictions" / HIERARCHICAL_PREDICTIONS_NAME
        pred.touch()
        bundle = models_dir / incoming.hierarchical.bundle_name
        bundle.touch()
        return {
            "predictions": pred,
            "models_dir": models_dir,
            "hierarchical_report": reports_dir / "hierarchical_lipid_report.md",
            "hierarchical_metrics": reports_dir / "hierarchical_lipid_metrics.parquet",
            "hierarchical_bundle": bundle,
        }

    monkeypatch.setattr(
        hierarchical_pipeline,
        "run_hierarchical_training",
        fake_run_hierarchical_training,
    )

    from slipp_plus.train import run_training

    out = run_training(settings)

    assert captured["settings"] == settings
    assert out["predictions"] == processed / "predictions" / HIERARCHICAL_PREDICTIONS_NAME
    assert out["hierarchical_bundle"] == models_dir / settings.hierarchical.bundle_name


def test_hierarchical_evaluation_reads_staged_predictions(tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="hierarchical")
    processed = settings.paths.processed_dir
    reports_dir = settings.paths.reports_dir
    (processed / "predictions").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    probs = {f"p_{label}": 0.0 for label in CLASS_10}
    probs["p_PLM"] = 1.0
    pd.DataFrame(
        [
            {
                "iteration": 0,
                "row_index": 7,
                "y_true_int": CLASS_10.index("PLM"),
                "y_pred_int": CLASS_10.index("PLM"),
                **probs,
            }
        ]
    ).to_parquet(processed / "predictions" / HIERARCHICAL_PREDICTIONS_NAME, index=False)

    captured: dict[str, object] = {}

    def fake_evaluate_hierarchical_holdouts(
        incoming_settings: Settings,
        *,
        feature_columns: list[str],
    ) -> dict[str, dict[str, float]]:
        captured["settings"] = incoming_settings
        captured["feature_columns"] = feature_columns
        return {
            "apo_pdb": {
                "f1": 0.70,
                "auroc": 0.80,
                "precision": 0.75,
                "sensitivity": 0.66,
            },
            "alphafold": {
                "f1": 0.60,
                "auroc": 0.78,
                "precision": 0.62,
                "sensitivity": 0.58,
            },
        }

    import slipp_plus.evaluate as evaluate

    original = evaluate.evaluate_hierarchical_holdouts
    evaluate.evaluate_hierarchical_holdouts = fake_evaluate_hierarchical_holdouts
    try:
        out = run_evaluation(settings)
    finally:
        evaluate.evaluate_hierarchical_holdouts = original

    assert captured["settings"] == settings
    assert captured["feature_columns"] == settings.feature_columns()

    raw_metrics = pd.read_parquet(out["raw_metrics"])
    assert raw_metrics["model"].tolist() == ["hierarchical"]
    text = out["metrics_table"].read_text(encoding="utf-8")
    assert "pipeline mode: `hierarchical`" in text
    assert "Holdout evaluation is not available" not in text
    assert "| hierarchical | 0.700 | 0.800 | 0.750 | 0.660 |" in text
    assert "| hierarchical | 0.600 | 0.780 | 0.620 | 0.580 |" in text


def test_staged_holdout_predictions_use_lipid_probability_threshold() -> None:
    probs = {f"p_{label}": 0.0 for label in CLASS_10}
    probs["p_PLM"] = 0.20
    probs["p_STE"] = 0.20
    probs["p_CLR"] = 0.09
    holdout_preds = pd.DataFrame(
        [
            {
                "y_pred_int": CLASS_10.index("PP"),
                **probs,
            }
        ]
    )
    holdout_df = pd.DataFrame(
        [
            {
                "ligand": "HEM",
                "class_binary": 1,
            }
        ]
    )

    metrics = evaluate_staged_holdout_predictions(holdout_preds, holdout_df)

    assert metrics["n"] == 1
    assert metrics["n_lipid"] == 1
    assert metrics["tp"] == 0
    assert metrics["fn"] == 1
