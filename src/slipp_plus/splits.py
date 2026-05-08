"""Split generation utilities, seeded and persisted to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

from .config import Settings, normalize_split_strategy

SplitStrategy = Literal[
    "stratified_shuffle",
    "grouped",
    "grouped_mmseqs30",
    "grouped_uniprot_clustered",
    "grouped_uniprot",
]


def _factorize_sorted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int64), classes


def _factorize_preserve_order(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapping: dict[object, int] = {}
    groups: list[object] = []
    codes = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        key = value.item() if isinstance(value, np.generic) else value
        code = mapping.get(key)
        if code is None:
            code = len(groups)
            mapping[key] = code
            groups.append(key)
        codes[i] = code
    return codes, np.asarray(groups, dtype=values.dtype)


def _make_stratified_shuffle_splits(
    class_labels: np.ndarray,
    n_iterations: int,
    test_fraction: float,
    seed_base: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_iterations):
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed_base + i,
        )
        train_idx, test_idx = next(sss.split(np.zeros(len(class_labels)), class_labels))
        splits.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return splits


def _score_grouped_selection(
    size: int,
    counts: np.ndarray,
    target_size: int,
    target_class_counts: np.ndarray,
) -> float:
    size_scale = max(float(target_size), 1.0)
    class_scale = np.maximum(target_class_counts, 1.0)
    size_error = abs(size - target_size) / size_scale
    class_error = np.mean(np.abs(counts - target_class_counts) / class_scale)
    return float(size_error + class_error)


def _make_grouped_split(
    class_labels: np.ndarray,
    group_labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(class_labels) != len(group_labels):
        raise ValueError("group_labels must have the same length as class_labels")

    class_codes, classes = _factorize_sorted(class_labels)
    group_codes, groups = _factorize_preserve_order(group_labels)
    if len(groups) < 2:
        raise ValueError("grouped splitting requires at least two distinct groups")

    n_groups = len(groups)
    n_classes = len(classes)
    group_sizes = np.bincount(group_codes, minlength=n_groups).astype(np.int64)
    group_class_counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    np.add.at(group_class_counts, (group_codes, class_codes), 1)

    target_size = round(test_fraction * len(class_labels))
    total_class_counts = np.bincount(class_codes, minlength=n_classes).astype(np.int64)
    target_class_counts = total_class_counts * test_fraction

    remaining = list(np.random.default_rng(seed).permutation(n_groups))
    selected: list[int] = []
    current_size = 0
    current_counts = np.zeros(n_classes, dtype=np.int64)

    while remaining and current_size < target_size:
        best_pos = 0
        best_score = None
        for pos, group_idx in enumerate(remaining):
            candidate_size = current_size + int(group_sizes[group_idx])
            candidate_counts = current_counts + group_class_counts[group_idx]
            candidate_score = _score_grouped_selection(
                size=candidate_size,
                counts=candidate_counts,
                target_size=target_size,
                target_class_counts=target_class_counts,
            )
            if best_score is None or candidate_score < best_score:
                best_pos = pos
                best_score = candidate_score

        chosen_group = remaining.pop(best_pos)
        selected.append(chosen_group)
        current_size += int(group_sizes[chosen_group])
        current_counts += group_class_counts[chosen_group]

    selected_groups = np.array(selected, dtype=group_codes.dtype)
    test_mask = np.isin(group_codes, selected_groups)
    train_idx = np.flatnonzero(~test_mask).astype(np.int64)
    test_idx = np.flatnonzero(test_mask).astype(np.int64)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("grouped split produced an empty train or test partition")
    return train_idx, test_idx


def make_splits(
    class_labels: np.ndarray,
    n_iterations: int,
    test_fraction: float,
    seed_base: int,
    strategy: SplitStrategy = "stratified_shuffle",
    group_labels: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n_iterations`` (train_idx, test_idx) pairs for the configured split.

    Seeding: iteration ``i`` uses ``seed_base + i`` to keep each split reproducible
    on its own while the overall protocol is deterministic.
    """
    strategy = normalize_split_strategy(strategy)

    if strategy == "stratified_shuffle":
        return _make_stratified_shuffle_splits(
            class_labels=class_labels,
            n_iterations=n_iterations,
            test_fraction=test_fraction,
            seed_base=seed_base,
        )

    if strategy != "grouped":
        raise ValueError(f"unknown split strategy: {strategy}")
    if group_labels is None:
        raise ValueError("group_labels are required when strategy='grouped'")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_iterations):
        splits.append(
            _make_grouped_split(
                class_labels=class_labels,
                group_labels=group_labels,
                test_fraction=test_fraction,
                seed=seed_base + i,
            )
        )
    return splits


def persist_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> list[Path]:
    """Write split index pairs as deterministic parquet files.

    Parameters
    ----------
    splits
        Ordered ``(train_idx, test_idx)`` pairs.
    out_dir
        Destination directory for ``seed_XX.parquet`` files.

    Returns
    -------
    list[Path]
        Paths written in split order.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        path = out_dir / f"seed_{i:02d}.parquet"
        pl.DataFrame(
            {
                "index": np.concatenate([train_idx, test_idx]),
                "split": np.concatenate(
                    [
                        np.array(["train"] * len(train_idx)),
                        np.array(["test"] * len(test_idx)),
                    ]
                ),
            }
        ).write_parquet(path)
        written.append(path)
    return written


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one persisted split parquet.

    Parameters
    ----------
    path
        Path to a ``seed_XX.parquet`` file produced by ``persist_splits``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Train and test row indices as ``int64`` arrays.
    """

    df = pl.read_parquet(path)
    train_idx = (
        df.filter(pl.col("split") == "train")
        .get_column("index")
        .to_numpy()
        .astype(np.int64)
    )
    test_idx = (
        df.filter(pl.col("split") == "test")
        .get_column("index")
        .to_numpy()
        .astype(np.int64)
    )
    return train_idx, test_idx


def run_splits(
    settings: Settings,
    class_labels: np.ndarray,
    group_labels: np.ndarray | None = None,
) -> list[Path]:
    """Generate and persist splits for a loaded experiment configuration.

    Parameters
    ----------
    settings
        Experiment settings controlling split strategy, count, fraction, and
        seed.
    class_labels
        Class labels used for stratification.
    group_labels
        Optional grouping labels required by grouped split strategies.

    Returns
    -------
    list[Path]
        Persisted split parquet paths.
    """

    splits = make_splits(
        class_labels=class_labels,
        n_iterations=settings.n_iterations,
        test_fraction=settings.test_fraction,
        seed_base=settings.seed_base,
        strategy=settings.split_strategy,
        group_labels=group_labels,
    )
    return persist_splits(splits, settings.paths.processed_dir / "splits")
