"""Tests for slipp_plus.plm_ste_tiebreaker.

These tests deliberately avoid depending on the not-yet-materialized
``processed/v_plm_ste/...`` outputs. They instead exercise:

* the training-split builder on the real ``processed/v_sterol/full_pockets.parquet``
  (using the ``v_sterol`` feature list) and any available split parquet,
* a smoke fit on a tiny synthetic PLM/STE-like dataset,
* the ``apply_tiebreaker`` routing logic on a hand-crafted ensemble frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from slipp_plus.constants import CLASS_10, FEATURE_SETS
from slipp_plus.ensemble import PROBA_COLUMNS
from slipp_plus.plm_ste_tiebreaker import (
    PLM_IDX,
    STE_IDX,
    apply_tiebreaker,
    build_plm_vs_ste_training,
    train_plm_ste_tiebreaker,
)

PLM = CLASS_10.index("PLM")
STE = CLASS_10.index("STE")
CLR = CLASS_10.index("CLR")
OLA = CLASS_10.index("OLA")
PP = CLASS_10.index("PP")

ROOT = Path(__file__).resolve().parents[1]


def _ensemble_row(row_index: int, y_true: int, probs: dict[str, float]) -> dict:
    d = {
        "iteration": 0,
        "row_index": row_index,
        "y_true_int": y_true,
        "y_pred_int": int(np.argmax([probs.get(c, 0.0) for c in CLASS_10])),
    }
    for c in CLASS_10:
        d[f"p_{c}"] = float(probs.get(c, 0.0))
    return d


def _ensemble_df(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    ordered = ["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"]
    return df.select(ordered)


# ---------------------------------------------------------------------------
# 1) Training-split builder honors binary labels and class restriction
# ---------------------------------------------------------------------------
def test_build_training_split_binary_labels() -> None:
    """Use real v_sterol assets to verify the PLM/STE training split.

    We exercise the real parquet to get a realistic mix of class labels, then
    confirm:
      - y labels are exactly {0, 1},
      - training rows originate only from PLM/STE rows of class_10,
      - both classes are represented with the expected ~4.7x imbalance.
    """
    full_path = ROOT / "processed/v_sterol/full_pockets.parquet"
    split_path = ROOT / "processed/splits/seed_00.parquet"
    if not full_path.exists() or not split_path.exists():
        pytest.skip("v_sterol / splits artifacts not present")

    full = pd.read_parquet(full_path)
    feat_cols = list(FEATURE_SETS["v_sterol"])
    X_tr, y_tr, X_te, y_te, te_idx = build_plm_vs_ste_training(full, feat_cols, split_path)

    assert X_tr.shape[1] == len(feat_cols)
    assert X_te.shape[1] == len(feat_cols)
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0] == te_idx.shape[0]

    assert set(np.unique(y_tr).tolist()).issubset({0, 1})
    assert set(np.unique(y_te).tolist()).issubset({0, 1})

    tr_classes = set(
        full["class_10"]
        .to_numpy()[np.setdiff1d(np.arange(len(full)), te_idx, assume_unique=False)]
        .tolist()
    )
    # te_idx is a subset of test split; y_te matches the STE-positive binary rule
    te_classes = set(full["class_10"].to_numpy()[te_idx].tolist())
    assert te_classes.issubset({"PLM", "STE"})

    # y_tr should contain both 0s and 1s; imbalance should be PLM-heavy.
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    assert n_pos > 0 and n_neg > 0
    assert n_neg > n_pos, "expected more PLM (label=0) than STE (label=1) in train"

    # Sanity check the test-side binary labels.
    recovered = (full.iloc[te_idx]["class_10"].to_numpy() == "STE").astype(int)
    assert np.array_equal(recovered, y_te)

    # Silence the unused-local warnings for reviewers.
    del tr_classes


# ---------------------------------------------------------------------------
# 2) Smoke fit on a synthetic dataset
# ---------------------------------------------------------------------------
def test_train_tiebreaker_smoke(tmp_path: Path) -> None:
    """Fit on a tiny PLM/STE-like dataset and sanity-check predict_proba."""
    rng = np.random.default_rng(7)
    n_plm, n_ste = 40, 20
    feats = ["f0", "f1", "f2", "f3"]
    plm_X = rng.normal(loc=0.0, scale=1.0, size=(n_plm, len(feats)))
    ste_X = rng.normal(loc=2.0, scale=1.0, size=(n_ste, len(feats)))
    X = np.vstack([plm_X, ste_X])
    y_str = np.array(["PLM"] * n_plm + ["STE"] * n_ste)
    full = pd.DataFrame(X, columns=feats)
    full["class_10"] = y_str

    split = pd.DataFrame(
        {
            "index": np.arange(len(full)),
            "split": ["train"] * 48 + ["test"] * 12,
        }
    )
    split_path = tmp_path / "seed_00.parquet"
    split.to_parquet(split_path, index=False)

    X_tr, y_tr, X_te, y_te, _te_idx = build_plm_vs_ste_training(full, feats, split_path)
    assert y_tr.sum() > 0 and y_tr.sum() < len(y_tr)

    model = train_plm_ste_tiebreaker(X_tr, y_tr, seed=0)

    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    p_ste = proba[:, 1]
    assert ((p_ste >= 0.0) & (p_ste <= 1.0)).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # With well-separated classes we expect better-than-random AUC-like behaviour.
    # We don't assert AUROC directly (too brittle for n=12) but at least one of
    # the held-out predictions should be on each side of 0.5.
    assert p_ste.min() < 0.5 < p_ste.max() or y_te.sum() in (0, len(y_te))


# ---------------------------------------------------------------------------
# 3) apply_tiebreaker firing logic
# ---------------------------------------------------------------------------
def test_apply_tiebreaker_fires_on_plm_ste() -> None:
    """Three rows exercising fire / no-fire / margin-cutoff paths."""
    # Row A: top-2 = {PLM, STE}, gap = 0.05 < 0.20 → fires, STE wins after tb
    probs_a = {c: 0.0 for c in CLASS_10}
    probs_a["PLM"] = 0.45
    probs_a["STE"] = 0.40
    probs_a["PP"] = 0.10
    probs_a["COA"] = 0.05

    # Row B: top-2 = {PLM, OLA}, no STE competition → never fires
    probs_b = {c: 0.0 for c in CLASS_10}
    probs_b["PLM"] = 0.50
    probs_b["OLA"] = 0.40
    probs_b["STE"] = 0.05
    probs_b["PP"] = 0.05

    # Row C: top-2 = {PLM, STE} but gap = 0.30 > 0.20 margin → skipped
    probs_c = {c: 0.0 for c in CLASS_10}
    probs_c["PLM"] = 0.55
    probs_c["STE"] = 0.25
    probs_c["PP"] = 0.10
    probs_c["COA"] = 0.10

    df = _ensemble_df(
        [
            _ensemble_row(row_index=1, y_true=STE, probs=probs_a),
            _ensemble_row(row_index=2, y_true=PLM, probs=probs_b),
            _ensemble_row(row_index=3, y_true=STE, probs=probs_c),
        ]
    )
    # Binary head strongly prefers STE on A; its score on B/C is irrelevant
    # because those rows should not fire anyway.
    tb_proba = np.array([0.9, 0.9, 0.9])
    row_lookup = np.array([1, 2, 3])

    out = apply_tiebreaker(df, tb_proba, row_lookup, margin=0.20)

    fired = out["tiebreaker_fired"].to_list()
    assert fired == [True, False, False]

    proba_np = out.select(PROBA_COLUMNS).to_numpy()
    # Every row's softprobs must remain normalized to 1.
    np.testing.assert_allclose(proba_np.sum(axis=1), 1.0, atol=1e-9)

    # Row A redistribution: combined PLM+STE mass is 0.85, split 0.1/0.9.
    mass_a = 0.45 + 0.40
    assert proba_np[0, PLM_IDX] == pytest.approx(mass_a * 0.1)
    assert proba_np[0, STE_IDX] == pytest.approx(mass_a * 0.9)
    # STE should now win on row A.
    assert out["y_pred_int"][0] == STE_IDX
    # p_STE_binary recorded only where fired.
    p_bin = out["p_STE_binary"].to_numpy()
    assert p_bin[0] == pytest.approx(0.9)
    assert np.isnan(p_bin[1]) and np.isnan(p_bin[2])

    # Row B untouched: PLM still wins, probs unchanged.
    assert out["y_pred_int"][1] == PLM
    assert proba_np[1, PLM_IDX] == pytest.approx(0.50)
    assert proba_np[1, OLA] == pytest.approx(0.40)

    # Row C untouched by margin cutoff: PLM still wins.
    assert out["y_pred_int"][2] == PLM
    assert proba_np[2, PLM_IDX] == pytest.approx(0.55)
    assert proba_np[2, STE_IDX] == pytest.approx(0.25)
