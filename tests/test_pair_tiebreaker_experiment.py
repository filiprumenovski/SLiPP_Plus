from __future__ import annotations

import numpy as np
import polars as pl

from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS
from slipp_plus.pair_tiebreaker_experiment import apply_pair_tiebreaker


def _base_frame() -> pl.DataFrame:
    rows = {
        "iteration": [0, 0, 0],
        "row_index": [0, 1, 2],
        "y_true_int": [CLASS_10.index("OLA"), CLASS_10.index("PLM"), CLASS_10.index("OLA")],
        "y_pred_int": [CLASS_10.index("PLM"), CLASS_10.index("PLM"), CLASS_10.index("PP")],
    }
    for col in PROBA_COLUMNS:
        rows[col] = [0.0, 0.0, 0.0]
    rows["p_PLM"] = [0.45, 0.60, 0.20]
    rows["p_OLA"] = [0.40, 0.30, 0.10]
    rows["p_PP"] = [0.10, 0.10, 0.65]
    rows["p_COA"] = [0.05, 0.00, 0.05]
    return pl.DataFrame(rows)


def test_apply_pair_tiebreaker_only_fires_on_requested_pair() -> None:
    out = apply_pair_tiebreaker(
        _base_frame(),
        positive_proba=np.array([0.9, 0.9, 0.9]),
        row_index_lookup=np.array([0, 1, 2]),
        negative_label="PLM",
        positive_label="OLA",
        margin=0.20,
    )

    fired = out["tiebreaker_fired"].to_list()
    assert fired == [True, False, False]
    proba = out.select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert out["y_pred_int"].to_list()[0] == CLASS_10.index("OLA")
    assert out["y_pred_int"].to_list()[1] == CLASS_10.index("PLM")
    assert out["y_pred_int"].to_list()[2] == CLASS_10.index("PP")
