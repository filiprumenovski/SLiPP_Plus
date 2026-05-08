from __future__ import annotations

import numpy as np
import pandas as pd

from slipp_plus.boundary_head import BoundaryRule
from slipp_plus.constants import CLASS_10
from slipp_plus.hierarchical_pipeline import (
    _resolve_feature_columns,
    _union_feature_columns,
    combine_hierarchical_softprobs_from_bundle,
)


class _PredictModel:
    def __init__(self, expected_width: int, output: np.ndarray) -> None:
        self.expected_width = expected_width
        self.output = output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.expected_width:
            raise AssertionError(f"expected width {self.expected_width}, got {X.shape[1]}")
        return self.output


def test_resolve_feature_columns_honors_override() -> None:
    primary = ["f0", "f1"]
    assert _resolve_feature_columns(primary, None) == primary
    assert len(_resolve_feature_columns(primary, "v14")) == 17


def test_union_feature_columns_includes_stage_specific_columns() -> None:
    bundle = {
        "feature_columns": ["f0", "f1"],
        "stage2_feature_columns": ["f1", "f2"],
        "nonlipid_feature_columns": ["f3"],
        "stage3_feature_columns": ["f4"],
        "boundary_heads": [{"rule": BoundaryRule("b", "STE", ("PLM",)), "feature_columns": ["f5", "f2"]}],
    }
    assert _union_feature_columns(bundle) == ["f0", "f1", "f2", "f3", "f4", "f5"]


def test_combine_hierarchical_softprobs_from_bundle_uses_stage_specific_feature_widths() -> None:
    n = 2
    df = pd.DataFrame(
        {
            "f0": [1.0, 2.0],
            "f1": [3.0, 4.0],
            "f2": [5.0, 6.0],
            "f3": [7.0, 8.0],
        }
    )
    flat = _PredictModel(
        2,
        np.full((n, len(CLASS_10)), 1.0 / len(CLASS_10), dtype=np.float64),
    )
    family = _PredictModel(
        1,
        np.full((n, 5), 0.2, dtype=np.float64),
    )
    nonlipid = _PredictModel(
        1,
        np.full((n, 5), 0.2, dtype=np.float64),
    )
    bundle = {
        "stage1_source": "ensemble",
        "stage1_ensemble_models": {"rf": flat},
        "flat_model_keys": ("rf",),
        "feature_columns": ["f0", "f1"],
        "stage2_model": family,
        "stage2_models": None,
        "stage2_lipid_family_mode": "softmax",
        "stage2_feature_columns": ["f2"],
        "nonlipid_model": nonlipid,
        "nonlipid_feature_columns": ["f3"],
    }

    p_lipid, lipid_family_proba, nonlipid_family_proba = combine_hierarchical_softprobs_from_bundle(
        df=df,
        bundle=bundle,
    )

    assert p_lipid.shape == (n,)
    assert lipid_family_proba.shape == (n, 5)
    assert nonlipid_family_proba.shape == (n, 5)
