from __future__ import annotations

from pathlib import Path

from scripts.generate_ablation_matrix import load_registry, render_ablation_matrix


def test_ablation_matrix_generator_marks_current_best_and_negative_results() -> None:
    rendered = render_ablation_matrix(load_registry(Path("experiments/registry.yaml")))

    # exp-035 is the current deployable; exp-028 remains the base anchor.
    assert "| **exp-035-legacy-rescue-logistic-gate-reproducible** |" in rendered
    assert "| exp-028-compact-shape3-shell6-chem-weighted |" in rendered
    assert "NEGATIVE RESULT." in rendered
    assert "ABANDONED." in rendered
    assert "exp-011-family-plus-moe |" in rendered


def test_checked_in_ablation_matrix_matches_registry_generator() -> None:
    rendered = render_ablation_matrix(load_registry(Path("experiments/registry.yaml")))

    assert Path("reports/ablation_matrix.md").read_text(encoding="utf-8") == rendered
