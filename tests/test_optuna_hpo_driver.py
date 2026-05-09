"""Smoke tests for the Optuna HPO driver.

The full multi-objective study runs against real data; here we exercise
the search loop on a synthetic context so the test stays fast and self-
contained. Two regressions matter most:

1. Multi-objective NSGA-II returns a Pareto front (no ``trial.report`` crash).
2. Single-objective TPE + Hyperband prunes correctly via ``progress_callback``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make ``tools/`` importable regardless of where the test runner is launched.
_HERE = Path(__file__).resolve().parent.parent
_TOOLS = _HERE / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

pytest.importorskip("optuna")


@pytest.fixture
def optuna_hpo_module():
    return importlib.import_module("optuna_hpo")


def test_multi_objective_nsga2_returns_pareto_front(optuna_hpo_module) -> None:
    ctx = optuna_hpo_module._smoke_test_context(seed=42)
    study = optuna_hpo_module.run_study(
        ctx,
        study_name="test_nsga2",
        storage=None,
        n_trials=3,
        sampler_name="nsga2",
        pruner_name="none",
        multi_objective=True,
    )
    assert len(study.trials) == 3
    assert len(study.directions) == 2
    pareto = study.best_trials
    assert len(pareto) >= 1
    for t in pareto:
        # values is (lipid_macro_f1, holdout_f1)
        assert len(t.values) == 2
        assert all(0.0 <= v <= 1.0 for v in t.values)


def test_single_objective_tpe_with_hyperband_prunes_or_completes(optuna_hpo_module) -> None:
    ctx = optuna_hpo_module._smoke_test_context(seed=42)
    study = optuna_hpo_module.run_study(
        ctx,
        study_name="test_tpe",
        storage=None,
        n_trials=3,
        sampler_name="tpe",
        pruner_name="hyperband",
        multi_objective=False,
    )
    assert len(study.trials) == 3
    # All completed trials must have a numeric value; pruned trials show state PRUNED
    for t in study.trials:
        if t.value is not None:
            assert 0.0 <= t.value <= 1.0


def test_suggest_flat_hp_covers_all_families(optuna_hpo_module) -> None:
    """The suggest_flat_hp helper must populate every field of FlatModelHyperparameters."""

    import optuna

    from slipp_plus.config import FlatModelHyperparameters

    fields = set(FlatModelHyperparameters.model_fields.keys())
    captured: dict[str, object] = {}

    def objective(trial: optuna.trial.Trial) -> float:
        hp = optuna_hpo_module._suggest_flat_hp(trial)
        captured.update(hp.model_dump())
        return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=1)
    assert set(captured.keys()) == fields
