from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from slipp_plus.boundary_head import (
    BoundaryRule,
    NeighborRescueRule,
    apply_boundary_head,
    apply_neighbor_rescue_head,
    boundary_confusion,
    build_boundary_training,
    train_boundary_head,
)
from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS


def _row(row_index: int, y_true: str, probs: dict[str, float]) -> dict[str, float | int]:
    values = [probs.get(label, 0.0) for label in CLASS_10]
    row: dict[str, float | int] = {
        "iteration": 0,
        "row_index": row_index,
        "y_true_int": CLASS_10.index(y_true),
        "y_pred_int": int(np.argmax(values)),
    }
    for label in CLASS_10:
        row[f"p_{label}"] = float(probs.get(label, 0.0))
    return row


def _frame(rows: list[dict[str, float | int]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select(
        ["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"]
    )


def test_build_boundary_training_and_fit(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    labels = np.array(["PLM"] * 18 + ["STE"] * 8 + ["OLA"] * 10 + ["PP"] * 12)
    features = rng.normal(size=(len(labels), 3))
    features[labels == "STE"] += 2.0
    full = pd.DataFrame(features, columns=["f0", "f1", "f2"])
    full["class_10"] = labels

    split = pd.DataFrame(
        {
            "index": np.arange(len(full)),
            "split": ["train"] * 36 + ["test"] * 12,
        }
    )
    split_path = tmp_path / "seed_00.parquet"
    split.to_parquet(split_path, index=False)

    rule = BoundaryRule(
        name="ste_vs_acyl",
        positive_label="STE",
        negative_labels=("PLM", "OLA"),
    )
    X_tr, y_tr, X_te, y_te, te_idx = build_boundary_training(
        full,
        ["f0", "f1", "f2"],
        split_path,
        rule,
    )

    assert X_tr.shape[1] == 3
    assert set(np.unique(y_tr).tolist()) == {0, 1}
    assert set(full.iloc[te_idx]["class_10"].unique()).issubset({"STE", "PLM", "OLA"})
    assert np.array_equal((full.iloc[te_idx]["class_10"].to_numpy() == "STE").astype(int), y_te)

    model = train_boundary_head(X_tr, y_tr, seed=3)
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_apply_boundary_head_redistributes_only_candidate_mass() -> None:
    rule = BoundaryRule(
        name="ste_vs_acyl",
        positive_label="STE",
        negative_labels=("PLM", "OLA"),
        margin=0.12,
        max_rank=3,
    )
    probs_a = {label: 0.0 for label in CLASS_10}
    probs_a["PLM"] = 0.34
    probs_a["STE"] = 0.30
    probs_a["OLA"] = 0.28
    probs_a["PP"] = 0.08

    probs_b = {label: 0.0 for label in CLASS_10}
    probs_b["PLM"] = 0.52
    probs_b["STE"] = 0.20
    probs_b["OLA"] = 0.18
    probs_b["PP"] = 0.10

    df = _frame(
        [
            _row(10, "STE", probs_a),
            _row(11, "PLM", probs_b),
        ]
    )
    out = apply_boundary_head(
        df,
        positive_proba=np.array([0.75, 0.75], dtype=np.float64),
        row_index_lookup=np.array([10, 11], dtype=np.int64),
        rule=rule,
    )

    fired = out[rule.fired_col].to_list()
    assert fired == [True, False]

    proba = out.select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    mass = probs_a["PLM"] + probs_a["STE"] + probs_a["OLA"]
    assert proba[0, CLASS_10.index("STE")] == pytest.approx(mass * 0.75)
    negative_mass = mass * 0.25
    negative_original = probs_a["PLM"] + probs_a["OLA"]
    assert proba[0, CLASS_10.index("PLM")] == pytest.approx(
        negative_mass * probs_a["PLM"] / negative_original
    )
    assert proba[0, CLASS_10.index("OLA")] == pytest.approx(
        negative_mass * probs_a["OLA"] / negative_original
    )
    assert proba[0, CLASS_10.index("PP")] == pytest.approx(0.08)
    assert out["y_pred_int"][0] == CLASS_10.index("STE")

    assert proba[1, CLASS_10.index("PLM")] == pytest.approx(0.52)
    assert np.isnan(out[rule.score_col].to_numpy()[1])


def test_boundary_confusion_counts_positive_vs_negative() -> None:
    rule = BoundaryRule(
        name="ste_vs_plm",
        positive_label="STE",
        negative_labels=("PLM",),
    )
    rows = [
        _row(1, "STE", {"STE": 0.8, "PLM": 0.2}),
        _row(2, "STE", {"PLM": 0.7, "STE": 0.3}),
        _row(3, "PLM", {"STE": 0.6, "PLM": 0.4}),
        _row(4, "PLM", {"PLM": 0.9, "STE": 0.1}),
        _row(5, "PP", {"PP": 0.9, "STE": 0.1}),
    ]
    counts = boundary_confusion(_frame(rows), rule)
    assert counts == {
        "positive_as_positive": 1,
        "positive_as_negative": 1,
        "negative_as_positive": 1,
        "negative_as_negative": 1,
        "positive_support": 2,
        "negative_support": 2,
    }


def test_apply_neighbor_rescue_head_uses_top_neighbor_and_threshold() -> None:
    rule = NeighborRescueRule(
        name="ste_rescue",
        positive_label="STE",
        neighbor_labels=("PLM", "COA", "OLA", "MYR"),
        threshold=0.50,
        top_k=4,
        fired_column="ste_rescue_fired",
        score_column="p_STE_rescue",
    )
    probs_a = {label: 0.0 for label in CLASS_10}
    probs_a["PLM"] = 0.48
    probs_a["COA"] = 0.18
    probs_a["STE"] = 0.16
    probs_a["OLA"] = 0.12
    probs_a["PP"] = 0.06

    probs_b = dict(probs_a)
    probs_b["PLM"] = 0.50
    probs_b["STE"] = 0.14

    probs_c = {label: 0.0 for label in CLASS_10}
    probs_c["PP"] = 0.60
    probs_c["PLM"] = 0.20
    probs_c["STE"] = 0.15
    probs_c["COA"] = 0.05

    df = _frame(
        [
            _row(10, "STE", probs_a),
            _row(11, "STE", probs_b),
            _row(12, "STE", probs_c),
        ]
    )
    out = apply_neighbor_rescue_head(
        df,
        positive_proba=np.array([0.80, 0.40, 0.90]),
        row_index_lookup=np.array([10, 11, 12], dtype=np.int64),
        rule=rule,
    )

    assert out["ste_rescue_fired"].to_list() == [True, False, False]
    proba = out.select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    mass = probs_a["PLM"] + probs_a["STE"]
    assert proba[0, CLASS_10.index("STE")] == pytest.approx(mass * 0.80)
    assert proba[0, CLASS_10.index("PLM")] == pytest.approx(mass * 0.20)
    assert out["y_pred_int"][0] == CLASS_10.index("STE")
    assert np.isnan(out["p_STE_rescue"].to_numpy()[1])
