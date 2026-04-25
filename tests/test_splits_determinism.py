from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slipp_plus.config import Settings, load_settings
from slipp_plus.splits import make_splits


def _synthetic_labels(n: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    probs = np.array([0.50, 0.15, 0.08, 0.08, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01])
    return rng.choice(classes, size=n, p=probs)


def _synthetic_grouped_labels(
    n_groups: int = 100,
    group_size: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.array(["A", "B", "C", "D", "E"])
    groups = np.repeat(np.array([f"g{i:03d}" for i in range(n_groups)]), group_size)
    dominant = rng.choice(classes, size=n_groups, p=np.array([0.35, 0.25, 0.2, 0.12, 0.08]))
    labels: list[np.ndarray] = []
    for cls in dominant:
        remainder = rng.choice(classes, size=group_size - 1, replace=True)
        group_labels = np.concatenate([[cls], remainder])
        labels.append(rng.permutation(group_labels))
    return np.concatenate(labels), groups


def test_splits_deterministic():
    y = _synthetic_labels()
    a = make_splits(y, n_iterations=5, test_fraction=0.1, seed_base=42)
    b = make_splits(y, n_iterations=5, test_fraction=0.1, seed_base=42)
    for (at, av), (bt, bv) in zip(a, b, strict=True):
        assert np.array_equal(at, bt)
        assert np.array_equal(av, bv)


def test_splits_stratify_respects_rare_class():
    y = _synthetic_labels()
    splits = make_splits(y, n_iterations=25, test_fraction=0.1, seed_base=42)
    for _, test_idx in splits:
        classes_in_test = set(np.unique(y[test_idx]))
        # Every class must show up in every test fold (10% of even the rarest).
        assert classes_in_test == set(np.unique(y))


def test_splits_test_size_matches():
    y = _synthetic_labels(n=1000)
    splits = make_splits(y, n_iterations=3, test_fraction=0.2, seed_base=0)
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == len(y)
        assert abs(len(test_idx) / len(y) - 0.2) < 0.01


def test_grouped_splits_deterministic():
    y, groups = _synthetic_grouped_labels()
    a = make_splits(
        y,
        n_iterations=5,
        test_fraction=0.2,
        seed_base=42,
        strategy="grouped",
        group_labels=groups,
    )
    b = make_splits(
        y,
        n_iterations=5,
        test_fraction=0.2,
        seed_base=42,
        strategy="grouped",
        group_labels=groups,
    )
    for (at, av), (bt, bv) in zip(a, b, strict=True):
        assert np.array_equal(at, bt)
        assert np.array_equal(av, bv)


def test_grouped_split_alias_matches_grouped_behavior():
    y, groups = _synthetic_grouped_labels(n_groups=120, group_size=5, seed=9)
    canonical = make_splits(
        y,
        n_iterations=4,
        test_fraction=0.2,
        seed_base=17,
        strategy="grouped",
        group_labels=groups,
    )
    aliased = make_splits(
        y,
        n_iterations=4,
        test_fraction=0.2,
        seed_base=17,
        strategy="grouped_mmseqs30",
        group_labels=groups,
    )
    for (canonical_train, canonical_test), (alias_train, alias_test) in zip(
        canonical, aliased, strict=True
    ):
        assert np.array_equal(canonical_train, alias_train)
        assert np.array_equal(canonical_test, alias_test)


def test_grouped_splits_have_no_group_leakage_and_approximate_size():
    y, groups = _synthetic_grouped_labels(n_groups=120, group_size=5)
    splits = make_splits(
        y,
        n_iterations=5,
        test_fraction=0.2,
        seed_base=7,
        strategy="grouped",
        group_labels=groups,
    )
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)
        assert len(train_idx) + len(test_idx) == len(y)
        assert abs(len(test_idx) / len(y) - 0.2) <= 0.03


def test_default_stratified_path_matches_explicit_strategy():
    y = _synthetic_labels(n=1000, seed=3)
    default = make_splits(y, n_iterations=3, test_fraction=0.2, seed_base=11)
    explicit = make_splits(
        y,
        n_iterations=3,
        test_fraction=0.2,
        seed_base=11,
        strategy="stratified_shuffle",
    )
    for (default_train, default_test), (explicit_train, explicit_test) in zip(
        default, explicit, strict=True
    ):
        assert np.array_equal(default_train, explicit_train)
        assert np.array_equal(default_test, explicit_test)


def test_settings_normalize_grouped_split_aliases():
    raw = load_settings(Path("configs/day1.yaml")).model_dump(mode="python")
    raw["split_strategy"] = "grouped_uniprot_clustered"
    raw["split_group_column"] = "cluster_id"

    settings = Settings.model_validate(raw)

    assert settings.split_strategy == "grouped"
    assert settings.split_group_column == "cluster_id"


def test_settings_require_group_column_for_grouped_alias():
    raw = load_settings(Path("configs/day1.yaml")).model_dump(mode="python")
    raw["split_strategy"] = "grouped_mmseqs30"
    raw["split_group_column"] = None

    with pytest.raises(ValueError, match="split_group_column is required"):
        Settings.model_validate(raw)
