"""Pocket-detector bakeoff: score P2Rank vs fpocket against known ligand sites."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from .aromatic_aliphatic import _compute_centroid
from .v49 import POCKET_HEADER

LOGGER = logging.getLogger(__name__)

HIT_THRESHOLD_A: float = 4.0
TOP_K_VALUES: tuple[int, ...] = (1, 3, 5)
TOP_K_HIT_REGISTRY: int = 10

_SCORE_FIELD = re.compile(r"^\s*Score\s*:\s*(?P<value>[-+0-9.eE]+)")

SCORE_SCHEMA: dict[str, pl.DataType] = {
    "structure_id": pl.Utf8,
    "ligand_class": pl.Utf8,
    "detector": pl.Utf8,
    "pocket_rank": pl.Int64,
    "center_x": pl.Float64,
    "center_y": pl.Float64,
    "center_z": pl.Float64,
    "score": pl.Float64,
    "dcc": pl.Float64,
    "dca": pl.Float64,
    "hit_dcc_4A": pl.Boolean,
    "hit_dca_4A": pl.Boolean,
}

PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "pocket_rank": pl.Int64,
    "center_x": pl.Float64,
    "center_y": pl.Float64,
    "center_z": pl.Float64,
    "score": pl.Float64,
}


def extract_ligand_atoms(pdb_path: Path, ligand_code: str) -> list[np.ndarray]:
    """Return heavy-atom coordinates for each ligand copy of ``ligand_code``.

    One Nx3 array per distinct (chain, resseq, icode) HETATM residue whose
    residue name matches ``ligand_code`` exactly. Hydrogens are skipped.
    Raises :class:`ValueError` if no copies are found.
    """

    wanted = ligand_code.strip().upper()
    groups: dict[tuple[str, str, str], list[list[float]]] = {}
    order: list[tuple[str, str, str]] = []

    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            if len(line) < 54:
                continue
            resname = line[17:20].strip().upper()
            if resname != wanted:
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if element == "H":
                continue
            if not element:
                atom_name = line[12:16].strip()
                if atom_name and atom_name[0] == "H":
                    continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            key = (chain, resseq, icode)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append([x, y, z])

    if not order:
        raise ValueError(f"no ligand copies found for code {ligand_code!r} in {pdb_path}")

    return [np.asarray(groups[key], dtype=float) for key in order]


def ligand_com(atoms: np.ndarray) -> np.ndarray:
    """Mean position of the provided heavy atoms."""

    if atoms.ndim != 2 or atoms.shape[1] != 3:
        raise ValueError(f"expected Nx3 coordinates, got shape {atoms.shape}")
    if atoms.shape[0] == 0:
        raise ValueError("cannot compute COM of empty atom set")
    return atoms.mean(axis=0)


def extract_fpocket_predictions(structure_out_dir: Path) -> pl.DataFrame:
    """Read fpocket pockets from ``<stem>_out/``: info-file order is the rank.

    Each pocket contributes the info-file ``Score`` field and an alpha-sphere
    centroid derived from ``pockets/pocketN_vert.pqr``. Pockets whose vert file
    is missing are skipped with a warning but keep their original rank.
    """

    if not structure_out_dir.is_dir():
        raise FileNotFoundError(f"fpocket output directory missing: {structure_out_dir}")
    info_path = _find_info_file(structure_out_dir)
    pockets_dir = structure_out_dir / "pockets"

    ordered_pockets = _parse_info_scores(info_path)
    if not ordered_pockets:
        return pl.DataFrame(schema=PREDICTION_SCHEMA)

    rows: list[dict[str, object]] = []
    for rank, (pocket_number, score) in enumerate(ordered_pockets, start=1):
        vert_path = pockets_dir / f"pocket{pocket_number}_vert.pqr"
        if not vert_path.exists():
            LOGGER.warning(
                "%s: missing vert file for pocket %d", structure_out_dir.name, pocket_number
            )
            continue
        try:
            centroid = _compute_centroid(vert_path)
        except ValueError as exc:
            LOGGER.warning(
                "%s: pocket %d centroid failed: %s", structure_out_dir.name, pocket_number, exc
            )
            continue
        rows.append(
            {
                "pocket_rank": rank,
                "center_x": float(centroid[0]),
                "center_y": float(centroid[1]),
                "center_z": float(centroid[2]),
                "score": float(score) if score is not None else float("nan"),
            }
        )

    if not rows:
        return pl.DataFrame(schema=PREDICTION_SCHEMA)
    return pl.DataFrame(rows).select(list(PREDICTION_SCHEMA)).cast(PREDICTION_SCHEMA)


def extract_p2rank_predictions(predictions_csv: Path) -> pl.DataFrame:
    """Parse a P2Rank ``*_predictions.csv`` into the standard prediction schema."""

    if not predictions_csv.exists():
        raise FileNotFoundError(f"P2Rank predictions CSV not found: {predictions_csv}")

    rows: list[dict[str, object]] = []
    with predictions_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty P2Rank predictions CSV: {predictions_csv}") from exc
        header = [cell.strip().lower() for cell in raw_header]
        required = {"rank", "score", "center_x", "center_y", "center_z"}
        missing = required - set(header)
        if missing:
            raise ValueError(
                f"{predictions_csv}: missing expected P2Rank columns {sorted(missing)}; got {header}"
            )
        idx = {name: header.index(name) for name in required}
        for raw in reader:
            if not raw or all(not cell.strip() for cell in raw):
                continue
            try:
                rank = int(float(raw[idx["rank"]].strip()))
                cx = float(raw[idx["center_x"]].strip())
                cy = float(raw[idx["center_y"]].strip())
                cz = float(raw[idx["center_z"]].strip())
                score = float(raw[idx["score"]].strip())
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{predictions_csv}: malformed row {raw}: {exc}") from exc
            rows.append(
                {
                    "pocket_rank": rank,
                    "center_x": cx,
                    "center_y": cy,
                    "center_z": cz,
                    "score": score,
                }
            )

    if not rows:
        return pl.DataFrame(schema=PREDICTION_SCHEMA)
    return (
        pl.DataFrame(rows)
        .sort("pocket_rank")
        .select(list(PREDICTION_SCHEMA))
        .cast(PREDICTION_SCHEMA)
    )


def compute_hit_metrics(
    pred_centers: np.ndarray,
    ligand_copies: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pocket DCC and DCA distances against the closest ligand copy."""

    if pred_centers.ndim != 2 or pred_centers.shape[1] != 3:
        raise ValueError(f"pred_centers must be Kx3, got shape {pred_centers.shape}")
    if not ligand_copies:
        raise ValueError("ligand_copies is empty")

    k = pred_centers.shape[0]
    if k == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    copy_coms = np.asarray([copy.mean(axis=0) for copy in ligand_copies], dtype=float)
    com_dists = np.linalg.norm(pred_centers[:, None, :] - copy_coms[None, :, :], axis=-1)
    dcc = com_dists.min(axis=1)

    dca_per_copy = np.empty((k, len(ligand_copies)), dtype=float)
    for i, copy in enumerate(ligand_copies):
        atom_dists = np.linalg.norm(pred_centers[:, None, :] - copy[None, :, :], axis=-1)
        dca_per_copy[:, i] = atom_dists.min(axis=1)
    dca = dca_per_copy.min(axis=1)

    return dcc, dca


def score_structure(
    pdb_path: Path,
    ligand_code: str,
    fpocket_dir: Path,
    p2rank_csv: Path | None,
) -> pl.DataFrame:
    """Score fpocket (and optionally P2Rank) predictions against the known ligand."""

    stem = pdb_path.stem
    ligand_copies = extract_ligand_atoms(pdb_path, ligand_code)

    frames: list[pl.DataFrame] = []
    fpocket_preds = extract_fpocket_predictions(fpocket_dir)
    frames.append(_annotate_predictions(fpocket_preds, stem, ligand_code, "fpocket", ligand_copies))

    if p2rank_csv is not None and p2rank_csv.exists():
        p2rank_preds = extract_p2rank_predictions(p2rank_csv)
        frames.append(
            _annotate_predictions(p2rank_preds, stem, ligand_code, "p2rank", ligand_copies)
        )
    elif p2rank_csv is not None:
        LOGGER.warning(
            "%s: P2Rank predictions missing at %s; skipping P2Rank rows", stem, p2rank_csv
        )

    if not frames:
        return pl.DataFrame(schema=SCORE_SCHEMA)
    combined = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]
    return combined.select(list(SCORE_SCHEMA)).cast(SCORE_SCHEMA)


def run_bakeoff(
    structure_plan: list[dict[str, object]],
    output_parquet: Path,
    workers: int = 6,
) -> dict[str, object]:
    """Score every planned structure in parallel and persist a single parquet."""

    tasks = [_normalize_plan_entry(entry) for entry in structure_plan]
    failures: list[dict[str, str]] = []
    frames: list[pl.DataFrame] = []

    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            frame_or_failure = _run_single(task)
            _collect(frame_or_failure, frames, failures)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_single, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    frame_or_failure = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    failures.append(
                        {
                            "structure_id": str(task.get("structure_id", "?")),
                            "error": f"worker crash: {exc}",
                        }
                    )
                    continue
                _collect(frame_or_failure, frames, failures)

    if frames:
        combined = pl.concat(frames, how="vertical_relaxed").cast(SCORE_SCHEMA)
    else:
        combined = pl.DataFrame(schema=SCORE_SCHEMA)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_parquet)

    structures_scored = (
        combined.select(pl.col("structure_id").n_unique()).item() if combined.height else 0
    )
    return {
        "structures": structures_scored,
        "rows": combined.height,
        "failures": failures,
        "output_path": output_parquet,
    }


def summarize(df: pl.DataFrame, out_markdown: Path | None = None) -> pl.DataFrame:
    """Aggregate hit rates per (detector, ligand_class) and per detector overall."""

    if df.height == 0:
        summary = pl.DataFrame(schema=_summary_schema())
        if out_markdown is not None:
            out_markdown.parent.mkdir(parents=True, exist_ok=True)
            out_markdown.write_text("(no rows)\n", encoding="utf-8")
        return summary

    schema = _summary_schema()
    ordered_columns = list(schema)
    per_class = _summarize_group(df, group_class=True).select(ordered_columns)
    per_all = _summarize_group(df, group_class=False).select(ordered_columns)
    summary = (
        pl.concat([per_class, per_all], how="vertical_relaxed")
        .sort(["detector", "ligand_class"])
        .cast(schema)
    )

    if out_markdown is not None:
        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text(_render_markdown_table(summary), encoding="utf-8")

    return summary


def _annotate_predictions(
    preds: pl.DataFrame,
    structure_id: str,
    ligand_class: str,
    detector: str,
    ligand_copies: list[np.ndarray],
) -> pl.DataFrame:
    if preds.height == 0:
        return pl.DataFrame(schema=SCORE_SCHEMA)
    centers = preds.select(["center_x", "center_y", "center_z"]).to_numpy()
    dcc, dca = compute_hit_metrics(centers, ligand_copies)
    return preds.with_columns(
        [
            pl.lit(structure_id).alias("structure_id"),
            pl.lit(ligand_class).alias("ligand_class"),
            pl.lit(detector).alias("detector"),
            pl.Series("dcc", dcc, dtype=pl.Float64),
            pl.Series("dca", dca, dtype=pl.Float64),
            pl.Series("hit_dcc_4A", dcc <= HIT_THRESHOLD_A, dtype=pl.Boolean),
            pl.Series("hit_dca_4A", dca <= HIT_THRESHOLD_A, dtype=pl.Boolean),
        ]
    ).select(list(SCORE_SCHEMA))


def _normalize_plan_entry(entry: dict[str, object]) -> dict[str, object]:
    pdb_path = Path(str(entry["pdb_path"]))
    p2rank_raw = entry.get("p2rank_csv")
    p2rank_csv = Path(str(p2rank_raw)) if p2rank_raw is not None else None
    return {
        "pdb_path": pdb_path,
        "ligand_class": str(entry["ligand_class"]),
        "fpocket_dir": Path(str(entry["fpocket_dir"])),
        "p2rank_csv": p2rank_csv,
        "structure_id": pdb_path.stem,
    }


def _run_single(task: dict[str, object]) -> pl.DataFrame | dict[str, str]:
    try:
        return score_structure(
            pdb_path=task["pdb_path"],
            ligand_code=str(task["ligand_class"]),
            fpocket_dir=task["fpocket_dir"],
            p2rank_csv=task["p2rank_csv"],
        )
    except Exception as exc:
        LOGGER.warning("score_structure failed for %s: %s", task.get("structure_id"), exc)
        return {
            "structure_id": str(task.get("structure_id", "?")),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _collect(
    result: pl.DataFrame | dict[str, str],
    frames: list[pl.DataFrame],
    failures: list[dict[str, str]],
) -> None:
    if isinstance(result, pl.DataFrame):
        if result.height > 0:
            frames.append(result)
    else:
        failures.append(result)


def _find_info_file(structure_out_dir: Path) -> Path:
    candidates = list(structure_out_dir.glob("*_info.txt"))
    if not candidates:
        raise FileNotFoundError(f"no *_info.txt under {structure_out_dir}")
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def _parse_info_scores(info_path: Path) -> list[tuple[int, float | None]]:
    results: list[tuple[int, float | None]] = []
    current_index: int | None = None
    current_score: float | None = None
    with info_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            header_match = POCKET_HEADER.match(line)
            if header_match is not None:
                if current_index is not None:
                    results.append((current_index, current_score))
                current_index = int(header_match.group("index"))
                current_score = None
                continue
            if current_index is None:
                continue
            score_match = _SCORE_FIELD.match(raw_line)
            if score_match is not None:
                try:
                    current_score = float(score_match.group("value"))
                except ValueError:
                    current_score = None
    if current_index is not None:
        results.append((current_index, current_score))
    return results


def _summary_schema() -> dict[str, pl.DataType]:
    columns: dict[str, pl.DataType] = {
        "detector": pl.Utf8,
        "ligand_class": pl.Utf8,
        "n_structures": pl.Int64,
    }
    for k in TOP_K_VALUES:
        columns[f"top{k}_dcc"] = pl.Float64
    for k in TOP_K_VALUES:
        columns[f"top{k}_dca"] = pl.Float64
    columns["mean_rank_first_dcc_hit"] = pl.Float64
    columns["n_no_hit"] = pl.Int64
    return columns


def _summarize_group(df: pl.DataFrame, *, group_class: bool) -> pl.DataFrame:
    structure_keys = (
        ["detector", "ligand_class", "structure_id"]
        if group_class
        else ["detector", "structure_id"]
    )
    group_keys = ["detector", "ligand_class"] if group_class else ["detector"]

    topk_expressions: list[pl.Expr] = []
    for k in TOP_K_VALUES:
        topk_expressions.append(
            ((pl.col("pocket_rank") <= k) & pl.col("hit_dcc_4A")).any().alias(f"_topk_dcc_{k}")
        )
        topk_expressions.append(
            ((pl.col("pocket_rank") <= k) & pl.col("hit_dca_4A")).any().alias(f"_topk_dca_{k}")
        )

    first_hit_expr = (
        pl.when(pl.col("hit_dcc_4A"))
        .then(pl.col("pocket_rank"))
        .otherwise(None)
        .min()
        .alias("_first_hit_rank")
    )
    any_topn_hit_expr = (
        ((pl.col("pocket_rank") <= TOP_K_HIT_REGISTRY) & pl.col("hit_dcc_4A"))
        .any()
        .alias("_any_topn_hit")
    )

    per_structure = df.group_by(structure_keys).agg(
        [*topk_expressions, first_hit_expr, any_topn_hit_expr]
    )

    agg_expressions: list[pl.Expr] = [pl.col("structure_id").n_unique().alias("n_structures")]
    for k in TOP_K_VALUES:
        agg_expressions.append(pl.col(f"_topk_dcc_{k}").mean().alias(f"top{k}_dcc"))
    for k in TOP_K_VALUES:
        agg_expressions.append(pl.col(f"_topk_dca_{k}").mean().alias(f"top{k}_dca"))
    agg_expressions.append(pl.col("_first_hit_rank").mean().alias("mean_rank_first_dcc_hit"))
    agg_expressions.append((~pl.col("_any_topn_hit")).sum().cast(pl.Int64).alias("n_no_hit"))

    aggregated = per_structure.group_by(group_keys).agg(agg_expressions)
    if not group_class:
        aggregated = aggregated.with_columns(pl.lit("ALL").alias("ligand_class"))
    return aggregated


def _render_markdown_table(summary: pl.DataFrame) -> str:
    if summary.height == 0:
        return "(no rows)\n"
    columns = list(summary.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in summary.iter_rows(named=True):
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                if value != value:
                    cells.append("NaN")
                elif column.startswith(("top", "mean_")):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(f"{value:g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_structure_plan(fpocket_root: Path, p2rank_root: Path | None) -> list[dict[str, object]]:
    """Walk ``fpocket_root/<CLASS>/*.pdb`` and pair each with fpocket + P2Rank outputs."""

    if not fpocket_root.is_dir():
        raise FileNotFoundError(f"fpocket root not found: {fpocket_root}")

    plan: list[dict[str, object]] = []
    class_dirs = sorted(path for path in fpocket_root.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        ligand_class = class_dir.name
        for pdb_path in sorted(class_dir.glob("*.pdb")):
            fpocket_dir = class_dir / f"{pdb_path.stem}_out"
            if not fpocket_dir.is_dir():
                LOGGER.warning(
                    "skipping %s: fpocket output dir missing (%s)", pdb_path, fpocket_dir
                )
                continue
            p2rank_csv: Path | None = None
            if p2rank_root is not None:
                candidate = p2rank_root / f"{pdb_path.name}_predictions.csv"
                if not candidate.exists():
                    candidate = p2rank_root / f"{pdb_path.stem}_predictions.csv"
                p2rank_csv = candidate if candidate.exists() else None
            plan.append(
                {
                    "pdb_path": pdb_path,
                    "ligand_class": ligand_class,
                    "fpocket_dir": fpocket_dir,
                    "p2rank_csv": p2rank_csv,
                }
            )
    return plan


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m slipp_plus.detector_bakeoff",
        description="Score P2Rank vs fpocket pockets against known ligand positions.",
    )
    parser.add_argument(
        "--fpocket-root",
        type=Path,
        default=Path("data/structures/source_pdbs"),
        help="Root directory containing <CLASS>/pdbXXXX.pdb + pdbXXXX_out/ siblings.",
    )
    parser.add_argument(
        "--p2rank-root",
        type=Path,
        default=Path("processed/p2rank/train_out"),
        help="Root directory containing P2Rank <stem>.pdb_predictions.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/detector_bakeoff/training_scores.parquet"),
        help="Output parquet path for per-pocket scores.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/detector_bakeoff/training_summary.parquet"),
        help="Output parquet path for per-class detector summary.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("reports/detector_bakeoff/training_summary.md"),
        help="Output markdown path for the summary table.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of worker processes for scoring.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, e.g. INFO or DEBUG.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    plan = build_structure_plan(args.fpocket_root, args.p2rank_root)
    if not plan:
        LOGGER.error("structure plan is empty under %s", args.fpocket_root)
        return 1

    LOGGER.info("bakeoff plan size: %d structures", len(plan))
    summary = run_bakeoff(plan, args.output, workers=args.workers)
    LOGGER.info(
        "scored %d structures, %d rows, %d failures -> %s",
        summary["structures"],
        summary["rows"],
        len(summary["failures"]),
        summary["output_path"],
    )
    for failure in summary["failures"]:
        LOGGER.warning("failure: %s", failure)

    scores = pl.read_parquet(args.output)
    aggregated = summarize(scores, out_markdown=args.summary_md)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.write_parquet(args.summary_output)

    print(f"Structures scored: {summary['structures']}")
    print(f"Rows written: {summary['rows']}")
    print(f"Failures: {len(summary['failures'])}")
    print(f"Scores parquet: {args.output}")
    print(f"Summary parquet: {args.summary_output}")
    print(f"Summary markdown: {args.summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
