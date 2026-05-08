#!/usr/bin/env python3
"""Generate the publishable ablation matrix from the experiment registry."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

HEADER = """# Ablation Matrix

Generated from `experiments/registry.yaml`. Metrics are copied from registry entries and should be refreshed whenever registry metrics change.

| experiment_id | feature_set | binary_f1 +/- std | macro_f1_10 +/- std | apo_pdb_f1 | alphafold_f1 | key_finding | superseded_by |
|---|---|---|---|---|---|---|---|
"""


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text.replace("+/-", "±").replace("+-", "±").replace("±", "±")


def _key_finding(entry: dict[str, Any]) -> str:
    notes = re.sub(r"\s+", " ", str(entry.get("notes") or "").strip())
    first_sentence = notes.split(". ", maxsplit=1)[0].strip()
    if first_sentence:
        return first_sentence + "."
    status = entry.get("status")
    if status == "abandoned":
        return "ABANDONED."
    return "-"


def _superseded_by(entries: list[dict[str, Any]]) -> dict[str, str]:
    by_source: dict[str, str] = {}
    for entry in entries:
        source = entry.get("supersedes")
        if source:
            by_source[str(source)] = str(entry["id"])
    return by_source


def render_ablation_matrix(entries: list[dict[str, Any]]) -> str:
    """Render a markdown ablation matrix from registry entries."""

    superseded_by = _superseded_by(entries)
    sorted_entries = sorted(
        entries, key=lambda item: (str(item.get("date") or ""), str(item["id"]))
    )
    lines = [HEADER]
    for entry in sorted_entries:
        metrics = entry.get("metrics") or {}
        holdouts = entry.get("holdouts") or {}
        experiment_id = str(entry["id"])
        if entry.get("is_current_best"):
            experiment_id = f"**{experiment_id}**"
        row = [
            experiment_id,
            _fmt(entry.get("feature_set")),
            _fmt(metrics.get("binary_f1")),
            _fmt(metrics.get("macro_f1_10")),
            _fmt(holdouts.get("apo_pdb_f1")),
            _fmt(holdouts.get("alphafold_f1")),
            _key_finding(entry),
            superseded_by.get(str(entry["id"]), "-"),
        ]
        lines.append("| " + " | ".join(row) + " |\n")
    return "".join(lines)


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Load registry entries from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"registry must contain a list of experiments: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/ablation_matrix.md"))
    args = parser.parse_args()

    rendered = render_ablation_matrix(load_registry(args.registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
