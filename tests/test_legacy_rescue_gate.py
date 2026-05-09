from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.legacy_rescue_gate import apply_rescue_gate, build_gate_features
from slipp_plus.constants import CLASS_10


def _frame(*, pp: float, ste: float, model: str = "m") -> pd.DataFrame:
    probabilities = {f"p_{label}": 0.0 for label in CLASS_10}
    probabilities["p_PP"] = pp
    probabilities["p_STE"] = ste
    return pd.DataFrame(
        [
            {
                "iteration": 0,
                "row_index": 0,
                "y_true_int": CLASS_10.index("STE"),
                "y_pred_int": CLASS_10.index("PP"),
                "model": model,
                **probabilities,
            }
        ]
    )


def test_build_gate_features_tracks_legacy_lipid_margins() -> None:
    base = _frame(pp=0.8, ste=0.2)
    paper17 = _frame(pp=0.3, ste=0.7)
    v_sterol = _frame(pp=0.4, ste=0.6)

    features = build_gate_features(base, paper17, v_sterol)

    assert features["base_lipid"].item() == pytest.approx(0.2)
    assert features["paper17_lipid"].item() == pytest.approx(0.7)
    assert features["vsterol_lipid"].item() == pytest.approx(0.6)
    assert features["legacy_min_lipid"].item() == pytest.approx(0.6)
    assert features["legacy_min_margin"].item() == pytest.approx(0.4)


def test_apply_rescue_gate_rewrites_base_negative_to_mean_legacy_probabilities() -> None:
    base = _frame(pp=0.8, ste=0.2)
    paper17 = _frame(pp=0.3, ste=0.7)
    v_sterol = _frame(pp=0.5, ste=0.5)

    out = apply_rescue_gate(
        base,
        paper17,
        v_sterol,
        np.array([0.96]),
        threshold=0.95,
        model_name="gate",
    )

    assert out["legacy_rescue_gate_fired"].item() is True
    assert out["p_PP"].item() == pytest.approx(0.4)
    assert out["p_STE"].item() == pytest.approx(0.6)
    assert out["y_pred_int"].item() == CLASS_10.index("STE")
    assert out["model"].item() == "gate"


def test_apply_rescue_gate_does_not_rewrite_base_positive_rows() -> None:
    base = _frame(pp=0.4, ste=0.6)
    paper17 = _frame(pp=0.9, ste=0.1)
    v_sterol = _frame(pp=0.9, ste=0.1)

    out = apply_rescue_gate(
        base,
        paper17,
        v_sterol,
        np.array([0.99]),
        threshold=0.95,
    )

    assert out["legacy_rescue_gate_fired"].item() is False
    assert out["p_PP"].item() == pytest.approx(0.4)
    assert out["p_STE"].item() == pytest.approx(0.6)
