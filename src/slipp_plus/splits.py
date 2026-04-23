"""25 stratified shuffle splits, seeded and persisted to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import Settings


def make_splits(
    class_labels: np.ndarray,
    n_iterations: int,
    test_fraction: float,
    seed_base: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n_iterations`` (train_idx, test_idx) pairs stratified on class.

    Seeding: iteration ``i`` uses ``seed_base + i`` to keep each split reproducible
    on its own while the overall protocol is deterministic.
    """
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


def persist_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        path = out_dir / f"seed_{i:02d}.parquet"
        df = pd.DataFrame(
            {
                "index": np.concatenate([train_idx, test_idx]),
                "split": np.concatenate(
                    [
                        np.array(["train"] * len(train_idx)),
                        np.array(["test"] * len(test_idx)),
                    ]
                ),
            }
        )
        df.to_parquet(path, index=False)
        written.append(path)
    return written


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    train_idx = df.loc[df["split"] == "train", "index"].to_numpy(dtype=np.int64)
    test_idx = df.loc[df["split"] == "test", "index"].to_numpy(dtype=np.int64)
    return train_idx, test_idx


def run_splits(settings: Settings, class_labels: np.ndarray) -> list[Path]:
    splits = make_splits(
        class_labels=class_labels,
        n_iterations=settings.n_iterations,
        test_fraction=settings.test_fraction,
        seed_base=settings.seed_base,
    )
    return persist_splits(splits, settings.paths.processed_dir / "splits")
