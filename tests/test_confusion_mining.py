from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from slipp_plus.confusion_mining import (
    candidate_boundary_rules,
    mine_confusion_edges,
    run_confusion_mining,
)
from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS


def _row(iteration: int, row_index: int, true_label: str, probs: dict[str, float]) -> dict:
    values = [probs.get(label, 0.0) for label in CLASS_10]
    row = {
        "iteration": iteration,
        "row_index": row_index,
        "y_true_int": CLASS_10.index(true_label),
        "y_pred_int": int(np.argmax(values)),
    }
    for label in CLASS_10:
        row[f"p_{label}"] = float(probs.get(label, 0.0))
    return row


def _norm(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    return {key: value / total for key, value in probs.items()}


def test_mine_confusion_edges_ranks_and_reports_recoverability() -> None:
    rows = [
        _row(0, 1, "STE", _norm({"PLM": 0.55, "STE": 0.35, "PP": 0.10})),
        _row(0, 2, "STE", _norm({"PLM": 0.50, "STE": 0.40, "PP": 0.10})),
        _row(0, 3, "OLA", _norm({"PLM": 0.48, "OLA": 0.42, "PP": 0.10})),
        _row(0, 4, "CLR", _norm({"CLR": 0.75, "PLM": 0.15, "PP": 0.10})),
        _row(0, 5, "PP", _norm({"PP": 0.80, "STE": 0.20})),
    ]
    edges = mine_confusion_edges(pl.DataFrame(rows), lipid_only=True)

    assert edges.iloc[0]["true_label"] == "STE"
    assert edges.iloc[0]["pred_label"] == "PLM"
    assert int(edges.iloc[0]["count"]) == 2
    assert int(edges.iloc[0]["true_support"]) == 2
    assert float(edges.iloc[0]["error_fraction_of_true"]) == 1.0
    assert float(edges.iloc[0]["top2_recoverable_fraction"]) == 1.0

    rules = candidate_boundary_rules(edges, top_n=2, margin=0.99)
    assert [rule.name for rule in rules] == ["ste_vs_plm", "ola_vs_plm"]
    assert rules[0].positive_label == "STE"
    assert rules[0].negative_labels == ("PLM",)
    assert rules[0].max_rank == 2


def test_run_confusion_mining_writes_report_and_table(tmp_path: Path) -> None:
    rows = [
        _row(0, 1, "STE", _norm({"PLM": 0.55, "STE": 0.35, "PP": 0.10})),
        _row(0, 2, "OLA", _norm({"PLM": 0.48, "OLA": 0.42, "PP": 0.10})),
    ]
    predictions_path = tmp_path / "predictions.parquet"
    report_path = tmp_path / "confusions.md"
    table_path = tmp_path / "confusions.parquet"
    pl.DataFrame(rows).select(
        ["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"]
    ).write_parquet(predictions_path)

    result = run_confusion_mining(
        predictions_path=predictions_path,
        output_report=report_path,
        output_table=table_path,
        lipid_only=True,
        candidate_count=1,
    )

    assert report_path.exists()
    assert table_path.exists()
    assert "Candidate Boundary Rules" in report_path.read_text()
    assert len(result["rules"]) == 1
    table = pl.read_parquet(table_path)
    assert table.height == 2
