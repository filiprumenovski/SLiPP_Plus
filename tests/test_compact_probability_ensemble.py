from __future__ import annotations

import pandas as pd
import pytest
from scripts.compact_probability_ensemble import (
    _average_prediction_frames,
    _load_holdout_labels,
)

from slipp_plus.constants import CLASS_10


def _prediction_frame(p_pp: float, p_ste: float) -> pd.DataFrame:
    probs = {f"p_{label}": 0.0 for label in CLASS_10}
    probs["p_PP"] = p_pp
    probs["p_STE"] = p_ste
    return pd.DataFrame(
        [
            {
                "iteration": 0,
                "row_index": 1,
                "y_true_int": CLASS_10.index("STE"),
                "y_pred_int": CLASS_10.index("PP"),
                **probs,
            }
        ]
    )


def test_average_prediction_frames_applies_component_weights(tmp_path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _prediction_frame(0.8, 0.2).to_parquet(path_a, index=False)
    _prediction_frame(0.2, 0.8).to_parquet(path_b, index=False)

    out = _average_prediction_frames(
        [path_a, path_b],
        "weighted",
        weights=[0.25, 0.75],
    )

    assert out["p_PP"].item() == pytest.approx(0.35)
    assert out["p_STE"].item() == pytest.approx(0.65)
    assert out["y_pred_int"].item() == CLASS_10.index("STE")


def test_load_holdout_labels_prefers_canonical_class_binary(tmp_path) -> None:
    component_path = tmp_path / "component.parquet"
    canonical_path = tmp_path / "canonical.parquet"
    component = pd.DataFrame(
        {
            "structure_id": ["A", "B"],
            "ligand": ["COA", "CLR"],
            "class_binary": [1, 0],
            "feature": [10.0, 20.0],
        }
    )
    canonical = pd.DataFrame(
        {
            "structure_id": ["A", "B"],
            "ligand": ["COA", "CLR"],
            "class_binary": [0, 1],
        }
    )
    component.to_parquet(component_path, index=False)
    canonical.to_parquet(canonical_path, index=False)

    out = _load_holdout_labels(
        component_holdout_path=component_path,
        canonical_holdout_path=canonical_path,
    )

    assert out["class_binary"].to_list() == [0, 1]
    assert out["feature"].to_list() == [10.0, 20.0]


def test_load_holdout_labels_aligns_canonical_rows_by_identity(tmp_path) -> None:
    component_path = tmp_path / "component.parquet"
    canonical_path = tmp_path / "canonical.parquet"
    pd.DataFrame(
        {
            "structure_id": ["A", "B"],
            "ligand": ["COA", "CLR"],
            "class_binary": [1, 0],
        }
    ).to_parquet(component_path, index=False)
    pd.DataFrame(
        {
            "structure_id": ["B", "A"],
            "ligand": ["CLR", "COA"],
            "class_binary": [1, 0],
        }
    ).to_parquet(canonical_path, index=False)

    out = _load_holdout_labels(
        component_holdout_path=component_path,
        canonical_holdout_path=canonical_path,
    )

    assert out["class_binary"].to_list() == [0, 1]
