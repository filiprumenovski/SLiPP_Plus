from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slipp_plus.composite.pair_moe import predict_pair_moe_holdout
from slipp_plus.constants import CLASS_10


class _FixedModel:
    def __init__(self, proba: list[list[float]]) -> None:
        self._proba = np.asarray(proba, dtype=np.float64)

    def predict_proba(self, _x: np.ndarray) -> np.ndarray:
        return self._proba


def _teacher_frame() -> pd.DataFrame:
    probs = {f"p_{label}": 0.0 for label in CLASS_10}
    probs["p_PLM"] = 0.40
    probs["p_STE"] = 0.35
    probs["p_PP"] = 0.25
    return pd.DataFrame([{**probs}])


def test_predict_pair_moe_holdout_applies_saved_binary_expert() -> None:
    holdout = pd.DataFrame([{"feature": 1.0}])
    bundle = {
        "experts": [
            {
                "name": "plm_ste_pair_expert",
                "kind": "binary_boundary",
                "labels": ("PLM", "STE"),
                "margin": 0.99,
                "feature_columns": ["feature", "p_PLM", "p_STE"],
                "model": _FixedModel([[0.10, 0.90]]),
            }
        ]
    }

    out = predict_pair_moe_holdout(holdout, bundle, _teacher_frame())

    assert out["p_STE"].item() == pytest.approx(0.75 * 0.90)
    assert out["p_PLM"].item() == pytest.approx(0.75 * 0.10)
    assert out["y_pred_int"].item() == CLASS_10.index("STE")


def test_predict_pair_moe_holdout_uses_topology_max_rank_for_legacy_local_expert() -> None:
    holdout = pd.DataFrame([{"feature": 1.0}])
    bundle = {
        "topology": {
            "experts": [
                {
                    "name": "local",
                    "max_rank": 1,
                }
            ]
        },
        "experts": [
            {
                "name": "local",
                "kind": "local_multiclass",
                "labels": ("PLM", "STE"),
                "margin": 0.0,
                "feature_columns": ["feature", "p_PLM", "p_STE"],
                "model": _FixedModel([[0.10, 0.90]]),
            }
        ],
    }

    out = predict_pair_moe_holdout(holdout, bundle, _teacher_frame())

    assert out["p_STE"].item() == pytest.approx(0.75 * 0.90)
    assert out["p_PLM"].item() == pytest.approx(0.75 * 0.10)
