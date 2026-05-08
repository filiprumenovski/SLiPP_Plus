"""Tests for slipp_plus.sterol_tiebreaker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from slipp_plus.constants import CLASS_10
from slipp_plus.ensemble import PROBA_COLUMNS
from slipp_plus.sterol_tiebreaker import (
    CLR_IDX,
    STE_IDX,
    apply_tiebreaker,
    build_clr_vs_ste_training,
    run_tiebreaker_pipeline,
    train_sterol_tiebreaker,
)

CLR = CLASS_10.index("CLR")
STE = CLASS_10.index("STE")
PP = CLASS_10.index("PP")


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
    # Match the column order / types produced by average_softprobs().
    ordered = ["iteration", "row_index", "y_true_int", *PROBA_COLUMNS, "y_pred_int"]
    return df.select(ordered)


def test_tiebreaker_does_not_fire_when_top_is_not_sterol_pair() -> None:
    """Predicted class is PP; tiebreaker must leave the row alone."""
    probs = {c: 0.0 for c in CLASS_10}
    probs["PP"] = 0.9
    probs["COA"] = 0.05
    probs["CLR"] = 0.03
    probs["STE"] = 0.02
    df = _ensemble_df([_ensemble_row(row_index=7, y_true=PP, probs=probs)])
    tb_proba = np.array([0.9])  # would push to STE if it fired
    row_lookup = np.array([7])
    out = apply_tiebreaker(df, tb_proba, row_lookup, margin=0.15)
    # Nothing changed.
    assert out["y_pred_int"].to_list() == [PP]
    assert out["p_CLR"][0] == pytest.approx(0.03)
    assert out["p_STE"][0] == pytest.approx(0.02)
    assert not out["tiebreaker_fired"][0]


def test_tiebreaker_preserves_sum_to_one_when_firing() -> None:
    """Redistribution of CLR+STE mass must keep the softprob row normalized."""
    probs = {c: 0.0 for c in CLASS_10}
    probs["CLR"] = 0.42
    probs["STE"] = 0.38
    probs["PP"] = 0.10
    probs["COA"] = 0.10
    df = _ensemble_df([_ensemble_row(row_index=3, y_true=STE, probs=probs)])
    tb_proba = np.array([0.8])
    row_lookup = np.array([3])
    out = apply_tiebreaker(df, tb_proba, row_lookup, margin=0.15)
    row_probs = out.select(PROBA_COLUMNS).to_numpy()[0]
    assert row_probs.sum() == pytest.approx(1.0, abs=1e-9)
    # Mass moved from CLR to STE.
    mass = 0.42 + 0.38
    assert row_probs[CLR_IDX] == pytest.approx(mass * 0.2)
    assert row_probs[STE_IDX] == pytest.approx(mass * 0.8)
    assert out["tiebreaker_fired"][0]
    assert out["y_pred_int"][0] == STE_IDX


def test_tiebreaker_respects_margin_cutoff() -> None:
    """When the top-1 lead exceeds the margin, we do not overwrite."""
    probs = {c: 0.0 for c in CLASS_10}
    probs["CLR"] = 0.70
    probs["STE"] = 0.20
    probs["PP"] = 0.05
    probs["COA"] = 0.05
    df = _ensemble_df([_ensemble_row(row_index=1, y_true=CLR, probs=probs)])
    out = apply_tiebreaker(df, np.array([0.99]), np.array([1]), margin=0.15)
    assert not out["tiebreaker_fired"][0]
    assert out["y_pred_int"][0] == CLR_IDX
    assert out["p_CLR"][0] == pytest.approx(0.70)


def test_train_sterol_tiebreaker_smoke(tmp_path) -> None:
    """Tiebreaker trains on CLR/STE rows and gives a per-test-row P(STE)."""
    # Build a minimal synthetic sterol-only dataset.
    rng = np.random.default_rng(42)
    n_clr, n_ste = 40, 20
    feats = ["f0", "f1", "f2"]
    clr_X = rng.normal(loc=0.0, scale=1.0, size=(n_clr, 3))
    ste_X = rng.normal(loc=2.5, scale=1.0, size=(n_ste, 3))
    X = np.vstack([clr_X, ste_X])
    y_str = np.array(["CLR"] * n_clr + ["STE"] * n_ste)
    full = pd.DataFrame(X, columns=feats)
    full["class_10"] = y_str

    # 80/20 split
    split = pd.DataFrame(
        {
            "index": np.arange(len(full)),
            "split": ["train"] * 48 + ["test"] * 12,
        }
    )
    split_path = tmp_path / "seed_00.parquet"
    split.to_parquet(split_path, index=False)

    X_tr, y_tr, X_te, _y_te, _te_idx = build_clr_vs_ste_training(full, feats, split_path)
    assert y_tr.sum() > 0 and y_tr.sum() < len(y_tr)
    model = train_sterol_tiebreaker(X_tr, y_tr, seed=0)
    p = model.predict_proba(X_te)[:, 1]
    assert p.shape == (len(X_te),)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_end_to_end_smoke_on_real_predictions(tmp_path) -> None:
    """End-to-end on 3 test rows of 1 iteration using the real v49 assets.

    Skips if the v49 processed artifacts are not checked out locally.
    """
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    full_path = root / "processed/v49/full_pockets.parquet"
    pred_path = root / "processed/v49/predictions/test_predictions.parquet"
    splits_dir = root / "processed/v49/splits"
    if not (full_path.exists() and pred_path.exists() and splits_dir.exists()):
        pytest.skip("v49 processed artifacts not available")

    # Make a mini splits dir containing just seed_00.parquet.
    mini_splits = tmp_path / "splits"
    mini_splits.mkdir()
    import shutil

    shutil.copy(splits_dir / "seed_00.parquet", mini_splits / "seed_00.parquet")

    # Slice the predictions to only iteration 0 and a handful of rows to keep
    # the test fast. We keep all three models for row_indices that appear for
    # all three (they do by construction).
    base = pl.read_parquet(pred_path).filter(pl.col("iteration") == 0)
    picked_rows = base["row_index"].unique().to_list()[:20]
    mini_pred = base.filter(pl.col("row_index").is_in(picked_rows))
    mini_pred_path = tmp_path / "test_predictions.parquet"
    mini_pred.write_parquet(mini_pred_path)

    from slipp_plus.constants import FEATURE_SETS

    os.environ["LOKY_MAX_CPU_COUNT"] = "2"
    result = run_tiebreaker_pipeline(
        full_pockets_path=full_path,
        predictions_path=mini_pred_path,
        splits_dir=mini_splits,
        feature_columns=FEATURE_SETS["v49"],
        output_path=None,
        workers=1,
    )
    # Augmented should cover only iteration 0.
    aug = result["augmented_predictions"]
    assert aug["iteration"].unique().to_list() == [0]
    # Every row's softprobs still sum to 1.
    sums = aug.select(PROBA_COLUMNS).to_numpy().sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    # Tiebreaker metadata present.
    assert "tiebreaker_fired" in aug.columns
    assert "p_STE_binary" in aug.columns
