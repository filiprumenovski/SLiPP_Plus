from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS
from slipp_plus.hierarchical_experiment import build_specialist_training, build_staged_probabilities
from slipp_plus.hierarchical_postprocess import OneVsNeighborsRule, apply_one_vs_neighbors


def _row(row_index: int, y_true: str, probs: dict[str, float]) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "iteration": 0,
        "row_index": row_index,
        "y_true_int": CLASS_10.index(y_true),
        "y_pred_int": int(np.argmax([probs.get(label, 0.0) for label in CLASS_10])),
    }
    for label in CLASS_10:
        row[f"p_{label}"] = float(probs.get(label, 0.0))
    return row


def _frame(rows: list[dict[str, float | int]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select(
        ["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"]
    )


def test_one_vs_neighbors_routes_ste_from_any_configured_neighbor() -> None:
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=("PLM", "COA", "OLA", "MYR"),
        top_k=4,
        min_positive_proba=0.40,
    )
    rows = []
    for i, neighbor in enumerate(rule.neighbor_labels):
        probs = {label: 0.01 for label in CLASS_10}
        probs[neighbor] = 0.46
        probs["STE"] = 0.14
        probs["PP"] = 0.12
        probs["CLR"] = 0.10
        remainder = 1.0 - sum(probs.values())
        probs["ADN"] += remainder
        rows.append(_row(i, "STE", probs))

    out = apply_one_vs_neighbors(
        _frame(rows),
        positive_proba=np.array([0.90, 0.80, 0.70, 0.60]),
        row_index_lookup=np.array([0, 1, 2, 3]),
        rule=rule,
    )

    assert out["ste_specialist_fired"].to_list() == [True, True, True, True]
    proba = out.select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert out["y_pred_int"].to_list() == [CLASS_10.index("STE")] * 4


def test_one_vs_neighbors_redistributes_only_positive_and_top_neighbor() -> None:
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=("PLM", "COA", "OLA", "MYR"),
        top_k=4,
        min_positive_proba=0.40,
    )
    probs = {label: 0.0 for label in CLASS_10}
    probs["PLM"] = 0.46
    probs["COA"] = 0.20
    probs["OLA"] = 0.15
    probs["STE"] = 0.14
    probs["PP"] = 0.05
    df = _frame([_row(10, "STE", probs)])

    out = apply_one_vs_neighbors(
        df,
        positive_proba=np.array([0.90]),
        row_index_lookup=np.array([10]),
        rule=rule,
    )
    proba = out.select(PROBA_COLUMNS).to_numpy()[0]
    mass = 0.46 + 0.14
    assert proba[CLASS_10.index("STE")] == pytest.approx(mass * 0.90)
    assert proba[CLASS_10.index("PLM")] == pytest.approx(mass * 0.10)
    assert proba[CLASS_10.index("COA")] == pytest.approx(0.20)
    assert proba[CLASS_10.index("OLA")] == pytest.approx(0.15)
    assert proba[CLASS_10.index("PP")] == pytest.approx(0.05)
    assert out["y_pred_int"][0] == CLASS_10.index("STE")


def test_one_vs_neighbors_no_fire_gates_and_threshold_inclusive() -> None:
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=("PLM", "COA", "OLA", "MYR"),
        top_k=4,
        min_positive_proba=0.40,
    )
    cases = [
        ("STE top1", {"STE": 0.45, "PLM": 0.30, "COA": 0.10, "OLA": 0.08, "PP": 0.07}, 0.90),
        ("PP top1", {"PP": 0.45, "PLM": 0.25, "STE": 0.15, "COA": 0.10, "OLA": 0.05}, 0.90),
        (
            "STE rank5",
            {"PLM": 0.35, "COA": 0.25, "OLA": 0.18, "MYR": 0.12, "STE": 0.09, "PP": 0.01},
            0.90,
        ),
        ("below threshold", {"PLM": 0.46, "COA": 0.20, "OLA": 0.15, "STE": 0.14, "PP": 0.05}, 0.39),
        ("at threshold", {"PLM": 0.46, "COA": 0.20, "OLA": 0.15, "STE": 0.14, "PP": 0.05}, 0.40),
    ]
    df = _frame([_row(i, "STE", probs) for i, (_, probs, _) in enumerate(cases)])
    original = df.select(PROBA_COLUMNS).to_numpy()
    out = apply_one_vs_neighbors(
        df,
        positive_proba=np.array([score for _, _, score in cases]),
        row_index_lookup=np.arange(len(cases)),
        rule=rule,
    )
    assert out["ste_specialist_fired"].to_list() == [False, False, False, False, True]
    updated = out.select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(updated[:4], original[:4], atol=1e-12)
    assert out["p_STE_ste_specialist"][4] == pytest.approx(0.40)


def test_one_vs_neighbors_alignment_errors() -> None:
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=("PLM",),
        top_k=2,
    )
    df = _frame([_row(1, "STE", {"PLM": 0.6, "STE": 0.4})])
    with pytest.raises(ValueError, match="must align"):
        apply_one_vs_neighbors(df, np.array([0.9, 0.8]), np.array([1]), rule)
    with pytest.raises(ValueError, match="duplicate"):
        apply_one_vs_neighbors(df, np.array([0.9, 0.8]), np.array([1, 1]), rule)


def test_build_specialist_training_filters_target_and_neighbors(tmp_path: Path) -> None:
    labels = np.array(["STE", "PLM", "COA", "OLA", "MYR", "PP", "CLR", "ADN"])
    full = pd.DataFrame(
        {
            "f0": np.arange(len(labels), dtype=float),
            "f1": np.arange(len(labels), dtype=float) + 1.0,
            "class_10": labels,
        }
    )
    split = pd.DataFrame(
        {
            "index": np.arange(len(labels)),
            "split": ["train", "train", "train", "train", "test", "test", "test", "test"],
        }
    )
    split_path = tmp_path / "seed_00.parquet"
    split.to_parquet(split_path, index=False)
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=("PLM", "COA", "OLA", "MYR"),
    )

    X_tr, y_tr, test_idx = build_specialist_training(full, ["f0", "f1"], split_path, rule)

    assert X_tr.shape == (4, 2)
    assert y_tr.tolist() == [1, 0, 0, 0]
    assert test_idx.tolist() == [4, 5, 6, 7]


def test_build_staged_probabilities_preserves_lipid_mass_and_normalization() -> None:
    baseline = _frame(
        [
            _row(5, "PLM", {"PLM": 0.30, "STE": 0.20, "COA": 0.25, "PP": 0.25}),
            _row(2, "PP", {"CLR": 0.10, "MYR": 0.10, "COA": 0.50, "PP": 0.30}),
        ]
    )
    out = build_staged_probabilities(
        baseline,
        row_index_lookup=np.array([2, 5]),
        p_lipid=np.array([0.20, 0.75]),
        lipid_family_proba=np.array(
            [
                [0.50, 0.10, 0.10, 0.20, 0.10],
                [0.05, 0.05, 0.10, 0.60, 0.20],
            ]
        ),
    )
    proba = out.sort("row_index").select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    lipid_idx = [CLASS_10.index(label) for label in ("CLR", "MYR", "OLA", "PLM", "STE")]
    np.testing.assert_allclose(proba[:, lipid_idx].sum(axis=1), [0.20, 0.75], atol=1e-9)


def test_existing_ste_rescue_artifact_if_present() -> None:
    path = Path("processed/v_sterol/predictions/ste_rescue_predictions.parquet")
    if not path.exists():
        pytest.skip("existing STE rescue artifact not present")
    df = pl.read_parquet(path)
    proba = df.select(PROBA_COLUMNS).to_numpy()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert {"ste_rescue_fired", "p_STE_rescue"}.issubset(df.columns)
    assert int(df["ste_rescue_fired"].sum()) == 160
