from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import slipp_plus.cli as cli
import slipp_plus.hierarchical_pipeline as hierarchical_pipeline
from slipp_plus.composite_topology import resolve_composite_topology
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
        "specialist_rule": {
            "name": "clr_specialist",
            "positive_label": "CLR",
            "neighbor_labels": ["STE", "OLA"],
            "top_k": 3,
            "min_positive_proba": 0.42,
            "max_margin": 0.10,
        },
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
    assert validated.hierarchical.specialist_rule is not None
    assert validated.hierarchical.specialist_rule.name == "clr_specialist"
    assert validated.hierarchical.specialist_rule.positive_label == "CLR"
    assert validated.hierarchical.specialist_rule.neighbor_labels == ("STE", "OLA")
    assert validated.hierarchical.specialist_rule.top_k == 3
    assert validated.hierarchical.specialist_rule.min_positive_proba == 0.42
    assert validated.hierarchical.specialist_rule.max_margin == 0.10
    assert validated.hierarchical.lipid_family_feature_set == "v49"
    assert validated.hierarchical.specialist_feature_set == "v_sterol"
    assert validated.hierarchical.nonlipid_feature_set == "v14"
    assert validated.hierarchical.boundary_heads[0].name == "ste_vs_plm"
    assert validated.hierarchical.boundary_heads[0].positive_label == "STE"
    assert validated.hierarchical.boundary_heads[0].negative_labels == ("PLM",)
    assert validated.hierarchical.boundary_heads[0].to_boundary_rule().max_rank == 2


def test_boundary_refactor_config_validates() -> None:
    settings = load_settings(Path("configs/v_sterol_boundary_refactor.yaml"))

    assert settings.pipeline_mode == "hierarchical"
    assert settings.feature_set == "v_sterol"
    assert settings.hierarchical.stage1_source == "ensemble"
    assert settings.hierarchical.ste_threshold == 0.50
    assert settings.hierarchical.specialist_rule is not None
    assert settings.hierarchical.specialist_rule.positive_label == "STE"
    assert settings.hierarchical.specialist_rule.neighbor_labels == ("PLM", "COA", "OLA", "MYR")
    assert settings.hierarchical.resolved_specialist_rule().min_positive_proba == 0.50
    assert settings.hierarchical.boundary_heads[0].name == "ola_vs_plm"
    assert settings.hierarchical.boundary_heads[0].positive_label == "OLA"
    assert settings.hierarchical.boundary_heads[0].negative_labels == ("PLM",)
    assert settings.hierarchical.boundary_heads[0].margin == 0.05
    assert settings.hierarchical.boundary_heads[0].to_boundary_rule().max_rank == 2


def test_moe_config_validates_explicit_topology() -> None:
    settings = load_settings(Path("configs/v_sterol_moe.yaml"))

    assert settings.pipeline_mode == "composite"
    assert settings.composite.backbone.kind == "teacher_ensemble"
    assert [item.name for item in settings.composite.experts] == [
        "ste_neighbors_expert",
        "ola_plm_expert",
    ]
    assert settings.composite.execution_order[-2:] == [
        "ste_neighbors_expert",
        "ola_plm_expert",
    ]


def test_pair_moe_config_validates_teacher_stack() -> None:
    settings = load_settings(Path("configs/v_sterol_pair_moe.yaml"))

    assert settings.pipeline_mode == "composite"
    assert settings.composite.teacher_predictions_path is not None
    assert [item.name for item in settings.composite.experts] == [
        "plm_ste_pair_expert",
        "clr_ola_pair_expert",
        "myr_plm_pair_expert",
        "coa_adn_pair_expert",
        "coa_b12_pair_expert",
    ]
    assert all(item.kind == "binary_boundary" for item in settings.composite.experts)


def test_pair_moe_combiner_preserves_non_candidate_mass() -> None:
    from slipp_plus.composite_pair_moe import _apply_pair_expert

    probs = {f"p_{label}": 0.0 for label in CLASS_10}
    probs["p_PLM"] = 0.35
    probs["p_STE"] = 0.30
    probs["p_PP"] = 0.20
    probs["p_COA"] = 0.15
    frame = pd.DataFrame(
        [
            {
                "iteration": 0,
                "row_index": 1,
                "y_true_int": CLASS_10.index("STE"),
                "y_pred_int": CLASS_10.index("PLM"),
                **probs,
            }
        ]
    )

    out, fired = _apply_pair_expert(
        frame,
        proba_positive=pd.Series([0.80]).to_numpy(),
        negative_label="PLM",
        positive_label="STE",
        margin=0.99,
    )

    assert fired.tolist() == [True]
    assert out["p_PP"].item() == frame["p_PP"].item()
    assert out["p_COA"].item() == frame["p_COA"].item()
    assert out["p_PLM"].item() + out["p_STE"].item() == pytest.approx(0.65)
    assert out["y_pred_int"].item() == CLASS_10.index("STE")


def test_local_moe_combiner_preserves_outside_mass() -> None:
    from slipp_plus.composite_pair_moe import _apply_local_multiclass_expert

    probs = {f"p_{label}": 0.0 for label in CLASS_10}
    probs["p_CLR"] = 0.20
    probs["p_OLA"] = 0.25
    probs["p_PLM"] = 0.10
    probs["p_COA"] = 0.10
    probs["p_PP"] = 0.15
    probs["p_STE"] = 0.20
    frame = pd.DataFrame(
        [
            {
                "iteration": 0,
                "row_index": 1,
                "y_true_int": CLASS_10.index("CLR"),
                "y_pred_int": CLASS_10.index("OLA"),
                **probs,
            }
        ]
    )

    out, fired = _apply_local_multiclass_expert(
        frame,
        local_proba=pd.DataFrame([[0.80, 0.05, 0.05, 0.05, 0.05]]).to_numpy(),
        labels=("CLR", "OLA", "PLM", "COA", "PP"),
        min_confidence=0.50,
        max_rank=3,
    )

    assert fired.tolist() == [True]
    assert out["p_STE"].item() == pytest.approx(0.20)
    local_mass = sum(out[f"p_{label}"].item() for label in ("CLR", "OLA", "PLM", "COA", "PP"))
    assert local_mass == pytest.approx(0.80)
    assert out["y_pred_int"].item() == CLASS_10.index("CLR")


def test_default_composite_topology_maps_hierarchical_settings(tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="composite")
    raw = settings.model_dump(mode="python")
    raw["hierarchical"]["specialist_gate"] = "utility"
    raw["hierarchical"]["boundary_heads"] = [
        {
            "name": "ola_vs_plm",
            "positive_label": "OLA",
            "negative_labels": ["PLM"],
            "margin": 0.05,
            "max_rank": 2,
            "feature_set": "v_sterol",
        }
    ]
    settings = Settings.model_validate(raw)

    topology = resolve_composite_topology(settings)

    assert topology.backbone.kind == "teacher_ensemble"
    assert [head.name for head in topology.heads] == [
        "global_10",
        "binary_lipid",
        "lipid_family",
        "nonlipid_family",
    ]
    assert topology.experts[0].kind == "one_vs_neighbors"
    assert topology.experts[0].gate == "utility"
    assert topology.experts[1].name == "ola_vs_plm"
    assert topology.experts[1].combine == "pair_swap"


def test_composite_phase_a_translates_topology_to_hierarchical_settings() -> None:
    settings = load_settings(Path("configs/v_sterol_moe.yaml"))

    from slipp_plus.composite_train import _phase_a_hierarchical_settings

    translated = _phase_a_hierarchical_settings(settings)

    assert translated.pipeline_mode == "hierarchical"
    assert translated.hierarchical.specialist_gate == "utility"
    assert translated.hierarchical.specialist_rule is not None
    assert translated.hierarchical.specialist_rule.name == "ste_neighbors_expert"
    assert translated.hierarchical.specialist_rule.positive_label == "STE"
    assert translated.hierarchical.specialist_rule.neighbor_labels == (
        "PLM",
        "COA",
        "OLA",
        "MYR",
    )
    assert translated.hierarchical.boundary_heads[0].name == "ola_plm_expert"
    assert translated.hierarchical.boundary_heads[0].positive_label == "OLA"


def test_hierarchical_settings_resolve_default_specialist_from_legacy_threshold(tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="hierarchical")

    rule = settings.hierarchical.resolved_specialist_rule()

    assert rule.name == "ste_specialist"
    assert rule.positive_label == "STE"
    assert rule.neighbor_labels == ("PLM", "COA", "OLA", "MYR")
    assert rule.top_k == 4
    assert rule.min_positive_proba == settings.hierarchical.ste_threshold


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


def test_composite_training_calls_composite_pipeline(monkeypatch, tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="composite")
    processed = settings.paths.processed_dir
    models_dir = settings.paths.models_dir
    reports_dir = settings.paths.reports_dir
    (processed / "predictions").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def fake_run_composite_training(incoming: Settings) -> dict[str, Path]:
        captured["settings"] = incoming
        pred = processed / "predictions" / HIERARCHICAL_PREDICTIONS_NAME
        pred.touch()
        bundle = models_dir / incoming.hierarchical.bundle_name
        bundle.touch()
        return {
            "predictions": pred,
            "models_dir": models_dir,
            "composite_bundle": bundle,
        }

    import slipp_plus.composite_train as composite_train

    monkeypatch.setattr(
        composite_train,
        "run_composite_training",
        fake_run_composite_training,
    )

    from slipp_plus.train import run_training

    out = run_training(settings)

    assert captured["settings"] == settings
    assert out["predictions"] == processed / "predictions" / HIERARCHICAL_PREDICTIONS_NAME
    assert out["composite_bundle"] == models_dir / settings.hierarchical.bundle_name


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


def test_hierarchical_cli_uses_config_driven_pipeline(monkeypatch, tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="hierarchical")
    config_path = tmp_path / "hierarchical.yaml"
    captured: dict[str, object] = {}

    def fake_load_settings(path: Path) -> Settings:
        captured["config"] = path
        return settings

    def fake_run_training(incoming: Settings) -> dict[str, Path]:
        captured["train_settings"] = incoming
        return {"predictions": tmp_path / "predictions.parquet"}

    def fake_run_evaluation(incoming: Settings) -> dict[str, Path]:
        captured["eval_settings"] = incoming
        return {
            "metrics_table": tmp_path / "metrics_table.md",
            "raw_metrics": tmp_path / "raw_metrics.parquet",
        }

    monkeypatch.setattr(cli, "load_settings", fake_load_settings)
    monkeypatch.setattr("slipp_plus.train.run_training", fake_run_training)
    monkeypatch.setattr("slipp_plus.evaluate.run_evaluation", fake_run_evaluation)

    cli.hierarchical_lipid_cmd(config=config_path)

    assert captured["config"] == config_path
    assert captured["train_settings"] == settings
    assert captured["eval_settings"] == settings


def test_hierarchical_cli_rejects_flat_configs(monkeypatch, tmp_path: Path) -> None:
    settings = _settings_with_paths(tmp_path, pipeline_mode="flat")
    monkeypatch.setattr(cli, "load_settings", lambda _path: settings)

    with pytest.raises(Exception, match="pipeline_mode: hierarchical"):
        cli.hierarchical_lipid_cmd(config=tmp_path / "flat.yaml")
