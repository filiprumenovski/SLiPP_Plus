"""Aromatic/aliphatic residue-shell feature extraction from fpocket outputs."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from Bio.PDB import PDBParser

AROMATIC_RESIDUES: frozenset[str] = frozenset({"PHE", "TYR", "TRP", "HIS"})
ALIPHATIC_RESIDUES: frozenset[str] = frozenset({"LEU", "ILE", "VAL", "ALA", "MET", "PRO", "CYS"})
SHELL_EDGES: tuple[tuple[float, float], ...] = ((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0))
POCKET_ID_PATTERN = re.compile(r"^[0-9A-Za-z]{4}_\d+$")
POCKET_FILE_PATTERN = re.compile(r"^pocket(?P<index>\d+)_(?P<kind>atm|vert)\.(?P<ext>pdb|pqr)$")
COUNT_COLUMNS: list[str] = [
    "aromatic_count_shell1",
    "aromatic_count_shell2",
    "aromatic_count_shell3",
    "aromatic_count_shell4",
    "aliphatic_count_shell1",
    "aliphatic_count_shell2",
    "aliphatic_count_shell3",
    "aliphatic_count_shell4",
]
RATIO_COLUMNS: list[str] = [
    "aromatic_aliphatic_ratio_shell1",
    "aromatic_aliphatic_ratio_shell2",
    "aromatic_aliphatic_ratio_shell3",
    "aromatic_aliphatic_ratio_shell4",
]
OUTPUT_COLUMNS: list[str] = ["pocket_id", *COUNT_COLUMNS, *RATIO_COLUMNS]
OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "pocket_id": pl.Utf8,
    **{column: pl.Int64 for column in COUNT_COLUMNS},
    **{column: pl.Float64 for column in RATIO_COLUMNS},
}


@dataclass(slots=True)
class ExtractionResult:
    rows: list[dict[str, object]]
    processed_structures: int
    extracted_pockets: int
    warnings: list[str]


def extract_features(input_dir: Path) -> pl.DataFrame:
    """Extract aromatic/aliphatic residue-shell features keyed by ``pocket_id``.

    Each pocket contributes four aromatic shell counts, four aliphatic shell
    counts, and four aromatic-to-aliphatic ratios computed from each residue's
    closest heavy-atom distance to the fpocket alpha-sphere centroid. The
    returned frame is ready for downstream joins onto the Day 1 base parquet
    via ``pocket_id``.
    """

    result = _extract_features_with_stats(input_dir)
    return _rows_to_frame(result.rows, expected_rows=result.extracted_pockets)


def write_features(input_dir: Path, output_path: Path) -> dict[str, object]:
    result = _extract_features_with_stats(input_dir)
    frame = _rows_to_frame(result.rows, expected_rows=result.extracted_pockets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    return {
        "processed_structures": result.processed_structures,
        "extracted_pockets": result.extracted_pockets,
        "warnings": len(result.warnings),
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
    }


def _symmetric_jeffreys_log_ratio(numerator: int, denominator: int) -> float:
    """Return the antisymmetric Jeffreys-smoothed log-ratio for two counts."""

    return float(math.log((numerator + 0.5) / (denominator + 0.5)))


def _jeffreys_log_ratio(numerator: int, denominator: int) -> float:
    return _symmetric_jeffreys_log_ratio(numerator, denominator)


def _extract_features_with_stats(input_dir: Path) -> ExtractionResult:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    structure_dirs = sorted(
        path for path in input_dir.iterdir() if path.is_dir() and path.name.endswith("_out")
    )
    if not structure_dirs:
        logging.info("No *_out directories found under %s", input_dir)
        return ExtractionResult(rows=[], processed_structures=0, extracted_pockets=0, warnings=[])

    worker_count = None if len(structure_dirs) > 1 else 1
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    if worker_count == 1:
        results = [_process_structure(structure_dir) for structure_dir in structure_dirs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_process_structure, structure_dirs))

    for structure_dir, structure_result in zip(structure_dirs, results, strict=True):
        logging.info(
            "Processed %s: %d pockets, %d warnings",
            structure_dir.name,
            structure_result.extracted_pockets,
            len(structure_result.warnings),
        )
        rows.extend(structure_result.rows)
        warnings.extend(structure_result.warnings)
        for message in structure_result.warnings:
            logging.warning(message)

    return ExtractionResult(
        rows=rows,
        processed_structures=len(structure_dirs),
        extracted_pockets=len(rows),
        warnings=warnings,
    )


def _process_structure(structure_dir: Path) -> ExtractionResult:
    warnings: list[str] = []
    pockets_dir = structure_dir / "pockets"
    if not pockets_dir.exists():
        warnings.append(f"{structure_dir.name}: missing pockets directory")
        return ExtractionResult(
            rows=[], processed_structures=1, extracted_pockets=0, warnings=warnings
        )
    if not pockets_dir.is_dir():
        warnings.append(f"{structure_dir.name}: pockets path is not a directory")
        return ExtractionResult(
            rows=[], processed_structures=1, extracted_pockets=0, warnings=warnings
        )

    pocket_files = sorted(pockets_dir.iterdir())
    if not pocket_files:
        warnings.append(f"{structure_dir.name}: empty pockets directory")
        return ExtractionResult(
            rows=[], processed_structures=1, extracted_pockets=0, warnings=warnings
        )

    pdb_id = structure_dir.name[:-4]
    atm_files: dict[int, Path] = {}
    vert_files: dict[int, Path] = {}

    for file_path in pocket_files:
        match = POCKET_FILE_PATTERN.match(file_path.name)
        if match is None:
            continue
        index = int(match.group("index"))
        kind = match.group("kind")
        if kind == "atm":
            atm_files[index] = file_path
        else:
            vert_files[index] = file_path

    rows: list[dict[str, object]] = []
    shared_indices = sorted(set(atm_files) & set(vert_files))

    for missing_index in sorted(set(atm_files) - set(vert_files)):
        warnings.append(f"{pdb_id}: missing vert file for pocket {missing_index}")
    for missing_index in sorted(set(vert_files) - set(atm_files)):
        warnings.append(f"{pdb_id}: missing atm file for pocket {missing_index}")

    for pocket_index in shared_indices:
        pocket_id = f"{pdb_id}_{pocket_index}"
        try:
            centroid = _compute_centroid(vert_files[pocket_index])
        except ValueError as exc:
            warnings.append(f"{pocket_id}: {exc}")
            continue
        except OSError as exc:
            warnings.append(f"{pocket_id}: failed reading vert file: {exc}")
            continue

        try:
            rows.append(_extract_pocket_row(pocket_id, atm_files[pocket_index], centroid, warnings))
        except ValueError as exc:
            warnings.append(f"{pocket_id}: {exc}")
        except OSError as exc:
            warnings.append(f"{pocket_id}: failed reading atm file: {exc}")

    return ExtractionResult(
        rows=rows, processed_structures=1, extracted_pockets=len(rows), warnings=warnings
    )


def _compute_centroid(pqr_path: Path) -> np.ndarray:
    coords: list[list[float]] = []
    with pqr_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coord = _parse_pqr_coordinates(line)
            if coord is not None:
                coords.append(coord)
    if not coords:
        raise ValueError("vert file has no alpha-sphere coordinates")
    return np.asarray(coords, dtype=float).mean(axis=0)


def _parse_pqr_coordinates(line: str) -> list[float] | None:
    fixed_fields = (line[30:38].strip(), line[38:46].strip(), line[46:54].strip())
    if all(field for field in fixed_fields):
        try:
            return [float(field) for field in fixed_fields]
        except ValueError:
            pass

    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        return [float(parts[6]), float(parts[7]), float(parts[8])]
    except ValueError:
        return None


def _extract_pocket_row(
    pocket_id: str,
    atm_path: Path,
    centroid: np.ndarray,
    warnings: list[str],
) -> dict[str, object]:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pocket_id, atm_path)
    except Exception as exc:
        raise ValueError(f"failed to parse atm pdb: {exc}") from exc

    residues = list(structure.get_residues())
    if not residues:
        raise ValueError("failed to parse atm pdb: no residues found")

    aromatic_counts = [0, 0, 0, 0]
    aliphatic_counts = [0, 0, 0, 0]

    for residue in residues:
        residue_name = residue.get_resname().strip().upper()
        residue_group = _classify_residue_name(residue_name)
        if residue_group is None:
            continue
        distance = _closest_heavy_atom_distance(residue, centroid)
        if distance is None:
            warnings.append(f"{pocket_id}: residue {residue_name} has no heavy atoms; skipped")
            continue
        shell_index = _shell_index(distance)
        if shell_index is None:
            continue
        if residue_group == "aromatic":
            aromatic_counts[shell_index] += 1
        else:
            aliphatic_counts[shell_index] += 1

    ratios = [
        _symmetric_jeffreys_log_ratio(aromatic, aliphatic)
        for aromatic, aliphatic in zip(aromatic_counts, aliphatic_counts, strict=True)
    ]
    return {
        "pocket_id": pocket_id,
        "aromatic_count_shell1": aromatic_counts[0],
        "aromatic_count_shell2": aromatic_counts[1],
        "aromatic_count_shell3": aromatic_counts[2],
        "aromatic_count_shell4": aromatic_counts[3],
        "aliphatic_count_shell1": aliphatic_counts[0],
        "aliphatic_count_shell2": aliphatic_counts[1],
        "aliphatic_count_shell3": aliphatic_counts[2],
        "aliphatic_count_shell4": aliphatic_counts[3],
        "aromatic_aliphatic_ratio_shell1": ratios[0],
        "aromatic_aliphatic_ratio_shell2": ratios[1],
        "aromatic_aliphatic_ratio_shell3": ratios[2],
        "aromatic_aliphatic_ratio_shell4": ratios[3],
    }


def _classify_residue_name(residue_name: str) -> str | None:
    if residue_name in AROMATIC_RESIDUES:
        return "aromatic"
    if residue_name in ALIPHATIC_RESIDUES:
        return "aliphatic"
    return None


def _closest_heavy_atom_distance(residue: object, centroid: np.ndarray) -> float | None:
    distances = [
        float(np.linalg.norm(atom.get_coord() - centroid))
        for atom in residue
        if atom.element.strip().upper() != "H"
    ]
    if not distances:
        return None
    return min(distances)


def _shell_index(distance: float) -> int | None:
    if distance < 0:
        raise ValueError(f"distance must be non-negative, got {distance}")
    if distance < 3.0:
        return 0
    if distance < 6.0:
        return 1
    if distance < 9.0:
        return 2
    if distance <= 12.0:
        return 3
    return None


def _rows_to_frame(rows: list[dict[str, object]], expected_rows: int) -> pl.DataFrame:
    if rows:
        frame = pl.DataFrame(rows).select(OUTPUT_COLUMNS).cast(OUTPUT_SCHEMA)
    else:
        frame = pl.DataFrame(schema=OUTPUT_SCHEMA).select(OUTPUT_COLUMNS)
    _validate_frame(frame, expected_rows=expected_rows)
    return frame


def _validate_frame(frame: pl.DataFrame, expected_rows: int) -> None:
    if frame.columns != OUTPUT_COLUMNS:
        raise ValueError(f"unexpected columns: {frame.columns}; expected {OUTPUT_COLUMNS}")
    for column, dtype in OUTPUT_SCHEMA.items():
        if frame.schema[column] != dtype:
            raise ValueError(f"column {column} has dtype {frame.schema[column]}, expected {dtype}")

    if frame.height != expected_rows:
        raise ValueError(f"row count mismatch: expected {expected_rows}, got {frame.height}")

    for row in frame.iter_rows(named=True):
        pocket_id = row["pocket_id"]
        if pocket_id is None or not POCKET_ID_PATTERN.match(str(pocket_id)):
            raise ValueError(f"invalid pocket_id: {pocket_id}")
        for column in COUNT_COLUMNS:
            value = row[column]
            if int(value) < 0:
                raise ValueError(f"{pocket_id}: negative count in {column}")
        for column in RATIO_COLUMNS:
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError(f"{pocket_id}: invalid ratio in {column}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slipp_plus.aromatic_aliphatic",
        description="Extract aromatic/aliphatic residue-shell features from fpocket outputs.",
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing *_out fpocket outputs."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path.")
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level, e.g. INFO or DEBUG."
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    try:
        summary = write_features(args.input_dir, args.output)
    except Exception as exc:
        logging.error("Extraction failed: %s", exc)
        return 1

    print(f"Processed: {summary['processed_structures']} structures")
    print(f"Pockets extracted: {summary['extracted_pockets']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Output: {summary['output_path']} ({summary['output_size_mb']:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
