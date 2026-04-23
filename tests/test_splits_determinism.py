from __future__ import annotations

import numpy as np

from slipp_plus.splits import make_splits


def _synthetic_labels(n: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    probs = np.array([0.50, 0.15, 0.08, 0.08, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01])
    return rng.choice(classes, size=n, p=probs)


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
