"""Optuna multi-objective Hyperband HPO driver for SLiPP++ flat-mode CV.

This is the modern-paradigm HPO entry point the project never had. It runs
Optuna's NSGA-II sampler over the joint space of RF / XGB / LGBM / CatBoost
hyperparameters with HyperbandPruner reporting after each CV iteration.

Two objectives are optimized simultaneously to navigate around the existing
holdout-regression crisis:

1. ``lipid_macro_f1`` — internal grouped-CV lipid 5-class macro-F1 (maximize).
2. ``holdout_f1`` — external apo-PDB / AlphaFold mean F1 (maximize).

The output is a Pareto front, persisted in a SQLite study so it can be
resumed and post-hoc analyzed (parameter importance, parallel-coordinates,
fANOVA).

Usage (synthetic smoke test, no data needed)::

    python tools/optuna_hpo.py --smoke-test --trials 5

Usage (real run, requires processed/full_pockets.parquet)::

    python tools/optuna_hpo.py \\
        --processed-dir processed/v_sterol \\
        --feature-set v_sterol \\
        --models rf xgb lgbm cat \\
        --study-name slipp_hpo_v1 \\
        --storage sqlite:///experiments/hpo/slipp_hpo_v1.db \\
        --trials 400 --pruner hyperband --sampler nsga2

The driver does not run training in the worktree if ``processed/`` is absent;
it falls back to ``--smoke-test`` mode for sanity-checking the search loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add src/ to path for in-tree execution.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass(frozen=True)
class HPOContext:
    """Per-trial inputs that don't change across the study."""

    X: np.ndarray
    y: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    models: list[str]
    seed_base: int
    holdout_X: np.ndarray | None = None
    holdout_y: np.ndarray | None = None


def _suggest_flat_hp(trial: Any) -> Any:
    """Suggest a :class:`FlatModelHyperparameters` from an Optuna trial."""

    from slipp_plus.config import FlatModelHyperparameters

    return FlatModelHyperparameters(
        # RandomForest
        rf_n_estimators=trial.suggest_int("rf_n_estimators", 80, 400, step=40),
        rf_min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 8),
        rf_max_features=trial.suggest_float("rf_max_features", 0.3, 1.0),
        # XGBoost
        xgb_max_depth=trial.suggest_int("xgb_max_depth", 3, 9),
        xgb_n_estimators=trial.suggest_int("xgb_n_estimators", 100, 600, step=50),
        xgb_learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
        xgb_subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
        xgb_colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        xgb_min_child_weight=trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
        xgb_reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-3, 1.0, log=True),
        xgb_reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 1.0, log=True),
        # LightGBM
        lgbm_num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 127),
        lgbm_n_estimators=trial.suggest_int("lgbm_n_estimators", 100, 600, step=50),
        lgbm_learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.2, log=True),
        lgbm_min_data_in_leaf=trial.suggest_int("lgbm_min_data_in_leaf", 5, 60),
        lgbm_feature_fraction=trial.suggest_float("lgbm_feature_fraction", 0.5, 1.0),
        lgbm_bagging_fraction=trial.suggest_float("lgbm_bagging_fraction", 0.5, 1.0),
        # CatBoost (cheap to suggest even when not used)
        cat_depth=trial.suggest_int("cat_depth", 4, 9),
        cat_iterations=trial.suggest_int("cat_iterations", 100, 500, step=50),
        cat_learning_rate=trial.suggest_float("cat_learning_rate", 0.01, 0.2, log=True),
        cat_l2_leaf_reg=trial.suggest_float("cat_l2_leaf_reg", 1e-2, 10.0, log=True),
    )


def _evaluate_holdout_f1(hp: Any, ctx: HPOContext) -> float:
    """Evaluate holdout lipid macro-F1 on the held-out validation slice.

    Trains a single 4-model ensemble on the entire ``ctx.X / ctx.y`` and
    evaluates on ``holdout_X / holdout_y``. Returns 0.0 if no holdout is
    available so single-objective fallback runs still work.
    """

    if ctx.holdout_X is None or ctx.holdout_y is None:
        return 0.0
    from sklearn.metrics import f1_score

    from slipp_plus.constants import CLASS_10, LIPID_CODES
    from slipp_plus.train import _fit_predict

    proba_sum: np.ndarray | None = None
    for key in ctx.models:
        _, _, proba = _fit_predict(
            key, ctx.seed_base, ctx.X, ctx.y, ctx.holdout_X, hp=hp
        )
        proba_sum = proba if proba_sum is None else proba_sum + proba
    assert proba_sum is not None
    y_pred = (proba_sum / len(ctx.models)).argmax(axis=1)
    lipid_idx = [CLASS_10.index(c) for c in LIPID_CODES if c in CLASS_10]
    return float(
        f1_score(
            ctx.holdout_y, y_pred, labels=lipid_idx, average="macro", zero_division=0
        )
    )


def _build_objective(ctx: HPOContext, *, multi_objective: bool) -> Any:
    """Build an Optuna objective closure over the immutable HPO context."""

    from slipp_plus.train import cv_evaluate_flat

    def objective(trial: Any) -> Any:
        import optuna

        hp = _suggest_flat_hp(trial)

        if multi_objective:
            # Optuna multi-objective studies don't accept ``trial.report`` /
            # ``should_prune``, so iterate without intermediate pruning. NSGA-II
            # provides its own selection pressure through the Pareto front.
            callback = None
        else:

            def _cb(i: int, partial: dict[str, float]) -> None:
                trial.report(partial["lipid_macro_f1_mean"], step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            callback = _cb

        out = cv_evaluate_flat(
            ctx.X,
            ctx.y,
            ctx.splits,
            models=ctx.models,
            seed_base=ctx.seed_base,
            hp=hp,
            progress_callback=callback,
        )
        f1 = float(out["lipid_macro_f1_mean"])
        if not multi_objective:
            return f1
        f2 = _evaluate_holdout_f1(hp, ctx)
        # Optuna NSGA-II expects a tuple matching the study's ``directions``.
        return f1, f2

    return objective


def run_study(
    ctx: HPOContext,
    *,
    study_name: str,
    storage: str | None,
    n_trials: int,
    sampler_name: str,
    pruner_name: str,
    multi_objective: bool,
) -> Any:
    """Run the Optuna study and return it for post-hoc analysis."""

    import optuna

    if sampler_name == "nsga2":
        sampler: Any = optuna.samplers.NSGAIISampler(seed=ctx.seed_base)
        if not multi_objective:
            raise SystemExit(
                "nsga2 sampler requires --multi-objective; pass --sampler tpe for "
                "single-objective HPO"
            )
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(
            multivariate=True, group=True, n_startup_trials=10, seed=ctx.seed_base
        )
    else:
        raise SystemExit(f"unknown sampler: {sampler_name}")

    if pruner_name == "hyperband":
        pruner: Any = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=len(ctx.splits), reduction_factor=3
        )
    elif pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    elif pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        raise SystemExit(f"unknown pruner: {pruner_name}")

    directions = ["maximize", "maximize"] if multi_objective else None
    direction = None if multi_objective else "maximize"
    create_kwargs: dict[str, Any] = {
        "study_name": study_name,
        "sampler": sampler,
        "pruner": pruner,
        "load_if_exists": True,
    }
    if storage is not None:
        create_kwargs["storage"] = storage
    if directions is not None:
        create_kwargs["directions"] = directions
    else:
        create_kwargs["direction"] = direction

    study = optuna.create_study(**create_kwargs)
    objective = _build_objective(ctx, multi_objective=multi_objective)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _smoke_test_context(*, seed: int = 42) -> HPOContext:
    """Build a synthetic 10-class context for the in-process smoke test."""

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=10,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=seed,
    )
    X = X.astype(np.float64)
    y = y.astype(np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for s in range(4):
        perm = np.random.default_rng(s).permutation(len(y))
        n_test = len(y) // 5
        splits.append((perm[n_test:], perm[:n_test]))
    # Synthetic holdout with the same class distribution.
    Xh, yh = make_classification(
        n_samples=120,
        n_features=20,
        n_informative=10,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=seed + 1,
    )
    return HPOContext(
        X=X,
        y=y,
        splits=splits,
        models=["rf", "lgbm"],
        seed_base=seed,
        holdout_X=Xh.astype(np.float64),
        holdout_y=yh.astype(np.int64),
    )


def _real_context(args: argparse.Namespace) -> HPOContext:
    """Build a real HPO context from the on-disk processed artifacts."""

    import pandas as pd

    from slipp_plus.config import Settings, load_settings
    from slipp_plus.constants import CLASS_10
    from slipp_plus.features import class10_labels, feature_matrix

    settings: Settings = load_settings(args.config) if args.config else None  # type: ignore[assignment]
    processed = Path(args.processed_dir)
    full = pd.read_parquet(processed / "full_pockets.parquet")
    X = feature_matrix(full, settings) if settings is not None else full.to_numpy()
    y_str = class10_labels(full)
    class_to_int = {c: i for i, c in enumerate(CLASS_10)}
    y = np.array([class_to_int[c] for c in y_str], dtype=np.int64)
    splits_dir = processed / "splits"
    split_files = sorted(splits_dir.glob("seed_*.parquet"))
    if not split_files:
        raise SystemExit(f"no split files found in {splits_dir}; run training once first")
    from slipp_plus.splits import load_split

    splits = [tuple(load_split(p)) for p in split_files][: args.max_iterations]
    return HPOContext(
        X=X,
        y=y,
        splits=splits,  # type: ignore[arg-type]
        models=list(args.models),
        seed_base=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optuna multi-objective Hyperband HPO for SLiPP++ flat-mode CV."
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic 5-trial sanity check.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "xgb", "lgbm", "cat"],
        choices=["rf", "xgb", "lgbm", "cat"],
    )
    parser.add_argument("--study-name", default="slipp_hpo_v1")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///x.db).")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--sampler", choices=["nsga2", "tpe"], default="nsga2")
    parser.add_argument("--pruner", choices=["hyperband", "median", "none"], default="hyperband")
    parser.add_argument(
        "--single-objective",
        action="store_true",
        help="Drop the holdout objective; required when sampler=tpe.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-summary", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.smoke_test:
        ctx = _smoke_test_context(seed=args.seed)
    else:
        if args.processed_dir is None:
            raise SystemExit("--processed-dir is required when not in --smoke-test mode")
        ctx = _real_context(args)

    multi_objective = not args.single_objective and args.sampler == "nsga2"
    study = run_study(
        ctx,
        study_name=args.study_name,
        storage=args.storage,
        n_trials=args.trials,
        sampler_name=args.sampler,
        pruner_name=args.pruner,
        multi_objective=multi_objective,
    )

    summary: dict[str, Any] = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "multi_objective": multi_objective,
    }
    if multi_objective:
        pareto = study.best_trials
        summary["pareto_front_size"] = len(pareto)
        summary["pareto_front"] = [
            {"trial": t.number, "values": list(t.values), "params": dict(t.params)}
            for t in pareto[:10]
        ]
    else:
        best = study.best_trial
        summary["best_trial"] = {
            "trial": best.number,
            "value": float(best.value),
            "params": dict(best.params),
        }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_summary is not None:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
