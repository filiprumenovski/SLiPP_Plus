"""PROMPT.md §6.3 Step 5 Rule 1 gate parity.

Runs the full ingest against the shipped CSV and asserts row/class counts
match the paper exactly.
"""

from __future__ import annotations

import pytest

from slipp_plus.config import Settings
from slipp_plus.ingest import _read_training_csv, assert_rule_1


@pytest.mark.parametrize(
    "class_code, expected",
    [
        ("CLR", 358),
        ("MYR", 424),
        ("OLA", 329),
        ("PLM", 718),
        ("STE", 152),
        ("ADN", 414),
        ("B12", 373),
        ("BGC", 526),
        ("COA", 2020),
        ("PP", 9905),
    ],
)
def test_per_class_counts(settings: Settings, class_code: str, expected: int) -> None:
    df = _read_training_csv(settings.paths.training_csv)
    assert (df["class_10"] == class_code).sum() == expected


def test_total_count(settings: Settings) -> None:
    df = _read_training_csv(settings.paths.training_csv)
    assert len(df) == settings.validation.training_total_exact


def test_rule_1_gate_passes(settings: Settings) -> None:
    df = _read_training_csv(settings.paths.training_csv)
    counts = assert_rule_1(df, settings)
    assert sum(counts.values()) == settings.validation.training_total_exact


def test_class_binary_derivation(settings: Settings) -> None:
    df = _read_training_csv(settings.paths.training_csv)
    lipid_total = 358 + 424 + 329 + 718 + 152
    assert int(df["class_binary"].sum()) == lipid_total
