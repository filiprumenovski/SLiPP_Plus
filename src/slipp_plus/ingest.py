"""Day 1 ingest: curated CSV + SF2/SF3 xlsx -> validated parquets.

We short-circuit PROMPT.md §6.3 Steps 2-4 because the authors' repo ships:

* ``reference/SLiPP_2024-main/training_pockets.csv`` - the 5x-balanced
  training set (15,219 rows) with all 17 canonical descriptors and per-class
  labels already populated.
* ``data/raw/supplementary/ci5c01076_si_003.xlsx`` - Supporting File 2, the
  apo-PDB holdout with 17 descriptors pre-extracted.
* ``data/raw/supplementary/ci5c01076_si_004.xlsx`` - Supporting File 3, the
  AlphaFold holdout, same shape.

The from-scratch path (download PDBs, run fpocket) is reserved for Day 7+ and
lives in ``src/slipp_plus/download.py`` + ``pocket_extraction.py`` as stubs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Settings
from .constants import (
    AA20,
    EXTRA_VDW22,
    LIG_TO_CLASS,
    LIPID_CODES,
    SELECTED_17,
)
from .schemas import validate_holdout, validate_training


def _read_training_csv(path: Path) -> pd.DataFrame:
    """Load the authors' training_pockets.csv and produce a canonical frame.

    Returns a DataFrame containing at minimum: SELECTED_17 + EXTRA_VDW22 + AA20
    + ``class_10`` + ``class_binary`` + ``pdb_ligand``.
    """
    df = pd.read_csv(path, index_col=0)
    df["class_10"] = df["lig"].map(LIG_TO_CLASS)
    unmapped = df["class_10"].isna()
    if unmapped.any():
        bad = df.loc[unmapped, "lig"].unique().tolist()
        raise ValueError(f"unknown ligand codes in training CSV: {bad}")
    df["class_binary"] = df["class_10"].isin(LIPID_CODES).astype(int)
    df = df.rename(columns={"pdb": "pdb_ligand"})

    keep = (
        ["pdb_ligand", "class_10", "class_binary"]
        + SELECTED_17
        + EXTRA_VDW22
        + AA20
    )
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"training CSV missing columns: {missing}")
    return df[keep].copy()


def _read_holdout_xlsx(path: Path, id_col: str) -> pd.DataFrame:
    """Load SF2 or SF3. Row 0 in the xlsx is the banner; real header is row 1."""
    raw = pd.read_excel(path, header=1)
    if id_col not in raw.columns:
        raise ValueError(
            f"expected column {id_col!r} in {path.name}, got {list(raw.columns)}"
        )
    if "ligand" not in raw.columns:
        raise ValueError(f"expected 'ligand' column in {path.name}")

    missing = [c for c in SELECTED_17 if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} missing descriptor columns: {missing}")

    out = pd.DataFrame(
        {
            "structure_id": raw[id_col].astype(str),
            "ligand": raw["ligand"].astype(str),
        }
    )
    out["class_binary"] = raw["ligand"].isin(LIPID_CODES).astype(int)
    for c in SELECTED_17:
        out[c] = pd.to_numeric(raw[c], errors="coerce")
    out = out.dropna(subset=SELECTED_17).reset_index(drop=True)
    return out


def assert_rule_1(full: pd.DataFrame, settings: Settings) -> dict[str, int]:
    """Rule 1 gate: total count + per-class count exact match to paper."""
    total = len(full)
    expected_total = settings.validation.training_total_exact
    if total != expected_total:
        raise AssertionError(
            f"Rule 1 FAIL: training total {total} != expected {expected_total}"
        )

    per_class = full["class_10"].value_counts().to_dict()
    expected = settings.validation.per_class_exact
    mismatches = {
        k: (per_class.get(k, 0), expected[k])
        for k in expected
        if per_class.get(k, 0) != expected[k]
    }
    if mismatches:
        lines = "\n".join(
            f"  {k}: got {got}, expected {exp}" for k, (got, exp) in mismatches.items()
        )
        raise AssertionError(f"Rule 1 FAIL: per-class counts drifted:\n{lines}")
    return {k: per_class[k] for k in expected}


def run_ingest(settings: Settings) -> dict[str, Path]:
    """Produce processed/*.parquet and log to reports/ingest_log.md."""
    paths = settings.paths
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    full = _read_training_csv(paths.training_csv)
    counts = assert_rule_1(full, settings)

    validate_training(full, settings.feature_columns())

    apo = _read_holdout_xlsx(paths.supporting_file_2_xlsx, id_col="PDB_ID")
    af = _read_holdout_xlsx(paths.supporting_file_3_xlsx, id_col="UniProt ID code")
    validate_holdout(apo, settings.feature_columns())
    validate_holdout(af, settings.feature_columns())

    full_path = paths.processed_dir / "full_pockets.parquet"
    apo_path = paths.processed_dir / "apo_pdb_holdout.parquet"
    af_path = paths.processed_dir / "alphafold_holdout.parquet"
    full.to_parquet(full_path, index=False)
    apo.to_parquet(apo_path, index=False)
    af.to_parquet(af_path, index=False)

    log = paths.reports_dir / "ingest_log.md"
    with log.open("w") as f:
        f.write("# Ingest log\n\n")
        f.write(f"- Training CSV: `{paths.training_csv}`\n")
        f.write(f"- Total training rows: **{len(full)}** (expected "
                f"{settings.validation.training_total_exact})\n")
        f.write("- Per-class counts:\n\n")
        f.write("| class | count |\n|---|---|\n")
        for k in sorted(counts):
            f.write(f"| {k} | {counts[k]} |\n")
        f.write(f"\n- Apo-PDB holdout: **{len(apo)}** rows "
                f"(lipids={int(apo['class_binary'].sum())})\n")
        f.write(f"- AlphaFold holdout: **{len(af)}** rows "
                f"(lipids={int(af['class_binary'].sum())})\n")
        f.write(f"\n- Feature set: `{settings.feature_set}` "
                f"({len(settings.feature_columns())} columns)\n")
        f.write("\nRule 1 gate: PASS.\n")

    return {
        "full_pockets": full_path,
        "apo_pdb_holdout": apo_path,
        "alphafold_holdout": af_path,
        "ingest_log": log,
    }
