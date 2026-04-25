"""Tests for slipp_plus.ensemble."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import (
    PROBA_COLUMNS,
    average_softprobs,
    clr_ste_confusion,
    score_summary,
)


def _toy_ten_class_row(iteration: int, row_index: int, model: str,
                       probs: dict[str, float], y_true: int) -> dict:
    d = {
        "iteration": iteration,
        "model": model,
        "row_index": row_index,
        "y_true_int": y_true,
        "y_pred_int": int(np.argmax(
            [probs.get(c, 0.0) for c in CLASS_10]
        )),
    }
    for c in CLASS_10:
        d[f"p_{c}"] = float(probs.get(c, 0.0))
    return d


def test_average_softprobs_is_mean_of_models() -> None:
    """Averaging two models on a 2-row toy must equal column-wise mean."""
    rows: list[dict] = []
    rows.append(_toy_ten_class_row(0, 0, "rf",
                                   {"CLR": 0.6, "STE": 0.2, "PP": 0.2}, y_true=3))
    rows.append(_toy_ten_class_row(0, 0, "xgb",
                                   {"CLR": 0.2, "STE": 0.6, "PP": 0.2}, y_true=3))
    rows.append(_toy_ten_class_row(0, 1, "rf",
                                   {"PP": 1.0}, y_true=8))
    rows.append(_toy_ten_class_row(0, 1, "xgb",
                                   {"PP": 1.0}, y_true=8))
    df = pl.DataFrame(rows)
    ens = average_softprobs(df, models=["rf", "xgb"]).sort("row_index")
    r0 = ens.filter(pl.col("row_index") == 0).row(0, named=True)
    assert r0["p_CLR"] == pytest.approx(0.4)
    assert r0["p_STE"] == pytest.approx(0.4)
    assert r0["p_PP"] == pytest.approx(0.2)
    r1 = ens.filter(pl.col("row_index") == 1).row(0, named=True)
    assert r1["p_PP"] == pytest.approx(1.0)
    # Ensemble argmax breaks tie deterministically by going to lowest class idx
    assert r0["y_pred_int"] in {CLASS_10.index("CLR"), CLASS_10.index("STE")}


def test_average_softprobs_sums_to_one() -> None:
    """Mean of two distributions that each sum to 1 must also sum to 1."""
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for ri in range(5):
        for model in ("rf", "xgb", "lgbm"):
            probs_arr = rng.dirichlet(np.ones(len(CLASS_10)))
            probs = dict(zip(CLASS_10, probs_arr, strict=True))
            rows.append(_toy_ten_class_row(0, ri, model, probs, y_true=0))
    df = pl.DataFrame(rows)
    ens = average_softprobs(df)
    matrix = ens.select(PROBA_COLUMNS).to_numpy()
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9)


def test_score_summary_has_required_keys() -> None:
    """score_summary should expose the headline metrics and per-class F1 dict."""
    rows: list[dict] = []
    for ri in range(4):
        true_c = CLASS_10[ri % len(CLASS_10)]
        probs = {c: 0.0 for c in CLASS_10}
        probs[true_c] = 1.0
        for model in ("rf", "xgb", "lgbm"):
            rows.append(_toy_ten_class_row(0, ri, model, probs,
                                           y_true=CLASS_10.index(true_c)))
    df = pl.DataFrame(rows)
    ens = average_softprobs(df)
    s = score_summary(ens)
    for key in (
        "macro_f1_mean", "macro_f1_std",
        "lipid_macro_f1_mean", "lipid_macro_f1_std",
        "binary_f1_mean", "binary_f1_std",
        "auroc_mean", "auroc_std",
        "per_class_f1",
    ):
        assert key in s
    for c in CLASS_10:
        assert c in s["per_class_f1"]


def test_clr_ste_confusion_counts() -> None:
    """Confusion helper counts CLR->STE and STE->CLR misses separately."""
    clr = CLASS_10.index("CLR")
    ste = CLASS_10.index("STE")
    df = pl.DataFrame(
        {
            "y_true_int": [clr, clr, ste, ste],
            "y_pred_int": [clr, ste, clr, ste],
        }
    )
    c = clr_ste_confusion(df)
    assert c["CLR_correct"] == 1
    assert c["STE_correct"] == 1
    assert c["CLR_as_STE"] == 1
    assert c["STE_as_CLR"] == 1
    assert c["CLR_support"] == 2
    assert c["STE_support"] == 2
