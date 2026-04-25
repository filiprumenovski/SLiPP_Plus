from __future__ import annotations

import numpy as np
import polars as pl

from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS
from slipp_plus.hierarchical_postprocess import OneVsNeighborsRule, apply_one_vs_neighbors
from slipp_plus.specialist_utility_gate import (
    UtilityGateConfig,
    apply_utility_gate,
    fit_utility_gate,
)


def _frame_from_rows(rows: list[dict[str, float | int]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select(
        ["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"]
    )


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


def test_utility_gate_does_not_increase_firing() -> None:
    rule = OneVsNeighborsRule(
        name="ste_specialist",
        positive_label="STE",
        neighbor_labels=("PLM", "COA", "OLA", "MYR"),
        top_k=4,
        min_positive_proba=0.30,
    )
    rows: list[dict[str, float | int]] = []
    p_spec: list[float] = []
    for i in range(24):
        probs = {label: 0.01 for label in CLASS_10}
        probs["PLM"] = 0.46
        probs["STE"] = 0.14
        probs["PP"] = 0.20
        probs["COA"] = 0.10
        probs["OLA"] = 0.08
        probs["MYR"] = 0.07
        probs["ADN"] += 1.0 - sum(probs.values())
        y_true = "STE" if i % 2 == 0 else "PLM"
        rows.append(_row(i, y_true, probs))
        p_spec.append(0.9 if i % 2 == 0 else 0.2)

    staged = _frame_from_rows(rows)
    p_specialist = np.asarray(p_spec, dtype=np.float64)
    candidate = apply_one_vs_neighbors(
        staged,
        positive_proba=p_specialist,
        row_index_lookup=np.arange(len(rows), dtype=np.int64),
        rule=rule,
    )
    model = fit_utility_gate(
        staged_df=staged,
        candidate_df=candidate,
        p_specialist=p_specialist,
        rule=rule,
    )
    out = apply_utility_gate(
        staged_df=staged,
        candidate_df=candidate,
        p_specialist=p_specialist,
        rule=rule,
        utility_model=model,
        config=UtilityGateConfig(threshold_default=0.5, threshold_top1_plm=0.7),
    )
    fired_candidate = int(candidate[rule.fired_col].sum())
    fired_out = int(out[rule.fired_col].sum())
    assert fired_out <= fired_candidate
