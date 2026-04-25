from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS
from slipp_plus.plm_ste_holdout import (
    binary_metrics_from_probabilities,
    ensemble_holdout_frame,
    pair_confusion_metrics,
    score_holdout_condition,
)


def test_binary_metrics_from_probabilities_matches_manual_counts() -> None:
    true_bin = np.array([1, 1, 0, 0], dtype=int)
    p_lipid = np.array([0.9, 0.4, 0.7, 0.2], dtype=float)

    metrics = binary_metrics_from_probabilities(true_bin, p_lipid)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5


def test_ensemble_holdout_frame_averages_model_probabilities() -> None:
    holdout_df = pd.DataFrame({"class_binary": [1, 0]})
    p1 = np.zeros((2, len(CLASS_10)), dtype=float)
    p2 = np.zeros((2, len(CLASS_10)), dtype=float)
    p1[:, CLASS_10.index("PLM")] = [0.6, 0.1]
    p1[:, CLASS_10.index("STE")] = [0.2, 0.1]
    p1[:, CLASS_10.index("PP")] = [0.2, 0.8]
    p2[:, CLASS_10.index("PLM")] = [0.2, 0.2]
    p2[:, CLASS_10.index("STE")] = [0.4, 0.1]
    p2[:, CLASS_10.index("PP")] = [0.4, 0.7]

    frame = ensemble_holdout_frame(holdout_df, {"rf": p1, "xgb": p2})
    proba = frame.select(PROBA_COLUMNS).to_numpy()

    assert frame["row_index"].to_list() == [0, 1]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba[0, CLASS_10.index("PLM")] == pytest.approx(0.4)
    assert proba[0, CLASS_10.index("STE")] == pytest.approx(0.3)
    assert proba[0, CLASS_10.index("PP")] == pytest.approx(0.3)


def test_score_holdout_condition_uses_lipid_probability_sum() -> None:
    frame = pl.DataFrame(
        {
            "iteration": [0, 0],
            "row_index": [0, 1],
            "y_true_int": [-1, -1],
            **{c: [0.0, 0.0] for c in PROBA_COLUMNS},
            "y_pred_int": [0, 0],
        }
    )
    lipid_cols = [f"p_{c}" for c in ["CLR", "MYR", "OLA", "PLM", "STE"]]
    updates = [pl.Series(name, [0.1, 0.0]) for name in lipid_cols]
    updates.append(pl.Series("p_PP", [0.5, 0.9]))
    frame = frame.with_columns(*updates, pl.Series("y_pred_int", [0, 8]))

    metrics = score_holdout_condition(frame, np.array([1, 0], dtype=int))
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["sensitivity"] == 1.0


def test_pair_confusion_metrics_tracks_plm_ste_confusions() -> None:
    frame = pl.DataFrame(
        {
            "iteration": [0, 0, 0],
            "row_index": [0, 1, 2],
            "y_true_int": [-1, -1, -1],
            **{c: [0.0, 0.0, 0.0] for c in PROBA_COLUMNS},
            "y_pred_int": [CLASS_10.index("PLM"), CLASS_10.index("STE"), CLASS_10.index("STE")],
        }
    )
    metrics = pair_confusion_metrics(frame, np.array(["PLM", "STE", "STE"], dtype=object))

    assert metrics["n_pair"] == 3
    assert metrics["plm_correct"] == 1
    assert metrics["ste_correct"] == 2
    assert metrics["plm_as_ste"] == 0
    assert metrics["ste_as_plm"] == 0
    assert metrics["plm_f1"] == 1.0
    assert metrics["ste_f1"] == 1.0