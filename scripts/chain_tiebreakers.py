"""Chain CLR/STE and PLM/STE tiebreakers on the v_sterol ensemble.

Disjoint top-2 pairs → the two binary heads operate on non-overlapping rows,
so stacking them is safe. Applies the PLM/STE head first, then the CLR/STE
head, and reports headline metrics for each stage.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from slipp_plus.ensemble import (
    PROBA_COLUMNS,
    average_softprobs,
    load_predictions,
    score_summary,
)
from slipp_plus.experiments.plm_ste_tiebreaker import (
    apply_tiebreaker as apply_plm_ste_tb,
)
from slipp_plus.experiments.plm_ste_tiebreaker import (
    build_plm_vs_ste_training,
    train_plm_ste_tiebreaker,
)
from slipp_plus.splits import load_split
from slipp_plus.experiments.sterol_tiebreaker import (
    apply_tiebreaker as apply_clr_ste_tb,
)
from slipp_plus.experiments.sterol_tiebreaker import (
    build_clr_vs_ste_training,
    train_sterol_tiebreaker,
)

FULL_POCKETS = Path("processed/v_sterol/full_pockets.parquet")
PREDICTIONS = Path("processed/v_sterol/predictions/test_predictions.parquet")
BUNDLE = Path("models/v_sterol/xgb_multiclass.joblib")
SPLITS_DIR = Path("processed/splits")
OUTPUT = Path("processed/v_sterol/predictions/chained_tiebreaker_predictions.parquet")
CLR_STE_MARGIN = 0.99
PLM_STE_MARGIN = 0.99


def process_iter(args):
    i, split_path_str, full_path_str, feature_columns = args
    split_path = Path(split_path_str)
    full_pockets = pd.read_parquet(full_path_str)
    X_tr_c, y_tr_c, _, _, _ = build_clr_vs_ste_training(full_pockets, feature_columns, split_path)
    m_c = train_sterol_tiebreaker(X_tr_c, y_tr_c, seed=42 + i)
    X_tr_p, y_tr_p, _, _, _ = build_plm_vs_ste_training(full_pockets, feature_columns, split_path)
    m_p = train_plm_ste_tiebreaker(X_tr_p, y_tr_p, seed=42 + i)
    _, test_idx = load_split(split_path)
    X_te = full_pockets[feature_columns].to_numpy(dtype=np.float64)[test_idx]
    p_ste_clr = m_c.predict_proba(X_te)[:, 1]
    p_ste_plm = m_p.predict_proba(X_te)[:, 1]
    return i, test_idx.astype(np.int64), p_ste_clr, p_ste_plm


def _score(df: pl.DataFrame, name: str) -> dict:
    cols = ["iteration", "row_index", "y_true_int", "y_pred_int", *PROBA_COLUMNS]
    s = score_summary(df.select(cols))
    pcf = s["per_class_f1"]
    print(
        f"{name}: macro-F1={s['macro_f1_mean']:.3f}\u00b1{s['macro_f1_std']:.3f}, "
        f"5-lipid={s['lipid_macro_f1_mean']:.3f}\u00b1{s['lipid_macro_f1_std']:.3f}, "
        f"binary-F1={s['binary_f1_mean']:.3f}\u00b1{s['binary_f1_std']:.3f}, "
        f"AUROC={s['auroc_mean']:.3f}\u00b1{s['auroc_std']:.3f}, "
        f"PLM={pcf['PLM'][0]:.3f}, CLR={pcf['CLR'][0]:.3f}, STE={pcf['STE'][0]:.3f}"
    )
    return s


def main() -> None:
    bundle = joblib.load(BUNDLE)
    feature_columns = list(bundle["feature_columns"])

    ensemble_df = average_softprobs(load_predictions(PREDICTIONS))

    split_files = sorted(SPLITS_DIR.glob("seed_*.parquet"))
    if len(split_files) != 25:
        raise RuntimeError(f"expected 25 splits, got {len(split_files)}")

    tasks = [(i, str(sp), str(FULL_POCKETS), feature_columns) for i, sp in enumerate(split_files)]
    results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_iter, t) for t in tasks]
        for fut in as_completed(futs):
            i, idx, pc, pp = fut.result()
            results[i] = (idx, pc, pp)

    chained_frames: list[pl.DataFrame] = []
    only_plm_frames: list[pl.DataFrame] = []
    only_clr_frames: list[pl.DataFrame] = []
    for i in sorted(results):
        idx, pc, pp = results[i]
        sub = ensemble_df.filter(pl.col("iteration") == i)

        after_plm = apply_plm_ste_tb(sub, pp, idx, margin=PLM_STE_MARGIN)
        chained = apply_clr_ste_tb(
            after_plm.drop(["tiebreaker_fired", "p_STE_binary"]),
            pc,
            idx,
            margin=CLR_STE_MARGIN,
        )
        chained_frames.append(chained)

        only_plm = apply_plm_ste_tb(sub, pp, idx, margin=PLM_STE_MARGIN)
        only_plm_frames.append(only_plm)

        only_clr = apply_clr_ste_tb(sub, pc, idx, margin=CLR_STE_MARGIN)
        only_clr_frames.append(only_clr)

    chained = pl.concat(chained_frames).sort(["iteration", "row_index"])
    only_plm = pl.concat(only_plm_frames).sort(["iteration", "row_index"])
    only_clr = pl.concat(only_clr_frames).sort(["iteration", "row_index"])

    _score(ensemble_df, "baseline ensemble        ")
    _score(only_plm, "+ PLM/STE only           ")
    _score(only_clr, "+ CLR/STE only           ")
    _score(chained, "+ chained PLM/STE+CLR/STE")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    chained.write_parquet(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
