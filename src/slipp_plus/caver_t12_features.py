"""Build persisted-output-first CAVER Tier 1-2 feature parquets."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .caver_analysis import (
    CaverPocketContext,
    cast_caver_t12_features,
    derive_caver_t12_features_by_pocket,
    safe_caver_t12_defaults,
)
from .constants import CAVER_T12_FEATURES_17


def build_training_v_caver_t12_parquet(
    base_parquet: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    reports_dir: Path = Path("reports/v_caver_t12"),
    analysis_root: Path | None = None,
) -> dict[str, object]:
    """Build a training parquet enriched with persisted CAVER T12 features.

    Parameters
    ----------
    base_parquet
        Base training parquet containing ``pdb_ligand`` and
        ``matched_pocket_number`` columns.
    manifest_path
        CSV, TSV, or parquet manifest mapping structures and pocket contexts to
        CAVER analysis directories.
    output_path
        Destination parquet path for the enriched feature table.
    reports_dir
        Directory where sanity and warning reports are written.
    analysis_root
        Optional root joined with manifest ``analysis_subdir`` values.

    Returns
    -------
    dict[str, object]
        Build summary including row count, warning count, warning messages, and
        output path.
    """

    return _build_v_caver_t12_parquet(
        base_parquet=base_parquet,
        manifest_path=manifest_path,
        output_path=output_path,
        reports_dir=reports_dir,
        key_column="pdb_ligand",
        analysis_root=analysis_root,
    )


def build_holdout_v_caver_t12_parquet(
    base_parquet: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    reports_dir: Path = Path("reports/v_caver_t12"),
    analysis_root: Path | None = None,
) -> dict[str, object]:
    """Build a holdout parquet enriched with persisted CAVER T12 features.

    Parameters
    ----------
    base_parquet
        Base holdout parquet containing ``structure_id`` and
        ``matched_pocket_number`` columns.
    manifest_path
        CSV, TSV, or parquet manifest mapping holdout structures and pocket
        contexts to CAVER analysis directories.
    output_path
        Destination parquet path for the enriched holdout table.
    reports_dir
        Directory where sanity and warning reports are written.
    analysis_root
        Optional root joined with manifest ``analysis_subdir`` values.

    Returns
    -------
    dict[str, object]
        Build summary including row count, warning count, warning messages, and
        output path.
    """

    return _build_v_caver_t12_parquet(
        base_parquet=base_parquet,
        manifest_path=manifest_path,
        output_path=output_path,
        reports_dir=reports_dir,
        key_column="structure_id",
        analysis_root=analysis_root,
    )


def _build_v_caver_t12_parquet(
    *,
    base_parquet: Path,
    manifest_path: Path,
    output_path: Path,
    reports_dir: Path,
    key_column: str,
    analysis_root: Path | None,
) -> dict[str, object]:
    base = pd.read_parquet(base_parquet)
    if key_column not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected {key_column} + matched_pocket_number columns")

    manifest = _read_manifest(manifest_path)
    manifest = _normalize_manifest_columns(manifest, key_column=key_column)

    if manifest.empty:
        raise ValueError(f"{manifest_path}: manifest is empty")

    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    manifest_by_key = {key: frame.copy() for key, frame in manifest.groupby(key_column, sort=False)}
    feature_map_by_key: dict[object, dict[int, dict[str, float]]] = {}

    for key, manifest_frame in manifest_by_key.items():
        analysis_dir = _resolve_analysis_dir(manifest_frame, analysis_root)
        context_rows = _build_pocket_contexts(manifest_frame)
        if analysis_dir is None:
            warnings.append(f"{key}: missing analysis_dir/analysis_subdir in manifest")
            feature_map_by_key[key] = {
                ctx.matched_pocket_number: safe_caver_t12_defaults() for ctx in context_rows
            }
        elif not analysis_dir.exists():
            warnings.append(f"{key}: analysis directory not found: {analysis_dir}")
            feature_map_by_key[key] = {
                ctx.matched_pocket_number: safe_caver_t12_defaults() for ctx in context_rows
            }
        else:
            feature_map_by_key[key] = derive_caver_t12_features_by_pocket(
                analysis_dir, context_rows
            )

    for _, row in base.iterrows():
        row_dict = row.to_dict()
        key = row_dict[key_column]
        matched_pocket_number = int(row_dict["matched_pocket_number"])
        feature_row = dict(row_dict)
        manifest_frame = manifest_by_key.get(key)
        if manifest_frame is None:
            warnings.append(f"{key}: missing manifest rows")
            feature_row.update(safe_caver_t12_defaults())
            rows.append(feature_row)
            continue

        feature_row.update(
            cast_caver_t12_features(
                feature_map_by_key.get(key, {}).get(
                    matched_pocket_number, safe_caver_t12_defaults()
                )
            )
        )
        rows.append(feature_row)

    enriched = pd.DataFrame(rows)
    if len(enriched) != len(base):
        raise ValueError(
            f"row drift after v_caver_t12 join: got {len(enriched)}, expected {len(base)}"
        )

    missing = [column for column in CAVER_T12_FEATURES_17 if column not in enriched.columns]
    if missing:
        raise ValueError(f"output missing v_caver_t12 columns: {missing}")

    for column in CAVER_T12_FEATURES_17:
        numeric = pd.to_numeric(enriched[column], errors="raise")
        if numeric.isna().any():
            raise ValueError(f"{column}: NaN values present after build")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column}: non-finite values present after build")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    _write_reports(enriched, warnings, reports_dir)
    return {
        "rows": len(enriched),
        "warnings": len(warnings),
        "warnings_list": warnings,
        "output_path": output_path,
    }


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"unsupported manifest format: {path}")


def _normalize_manifest_columns(manifest: pd.DataFrame, *, key_column: str) -> pd.DataFrame:
    frame = manifest.copy()
    aliases = {
        "source_id": key_column,
        "start_point_index": "starting_point_index",
        "pocket_number": "matched_pocket_number",
        "axial_length": "pocket_axial_length",
    }
    for source, target in aliases.items():
        if source in frame.columns and target not in frame.columns:
            frame = frame.rename(columns={source: target})
    required = {key_column, "matched_pocket_number", "starting_point_index", "pocket_axial_length"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    return frame


def _build_pocket_contexts(manifest_frame: pd.DataFrame) -> list[CaverPocketContext]:
    contexts: list[CaverPocketContext] = []
    for _, row in manifest_frame.iterrows():
        axis = _extract_axis(row)
        contexts.append(
            CaverPocketContext(
                matched_pocket_number=int(row["matched_pocket_number"]),
                starting_point_index=int(row["starting_point_index"]),
                pocket_axial_length=float(row["pocket_axial_length"]),
                pocket_principal_axis=axis,
            )
        )
    return contexts


def _extract_axis(row: pd.Series) -> tuple[float, float, float] | None:
    candidates = [
        ("pocket_axis_x", "pocket_axis_y", "pocket_axis_z"),
        (
            "pocket_principal_axis_x",
            "pocket_principal_axis_y",
            "pocket_principal_axis_z",
        ),
    ]
    for x_key, y_key, z_key in candidates:
        if x_key in row.index and y_key in row.index and z_key in row.index:
            return (float(row[x_key]), float(row[y_key]), float(row[z_key]))
    return None


def _resolve_analysis_dir(
    manifest_frame: pd.DataFrame,
    analysis_root: Path | None,
) -> Path | None:
    row = manifest_frame.iloc[0]
    if "analysis_dir" in manifest_frame.columns and pd.notna(row["analysis_dir"]):
        return Path(str(row["analysis_dir"]))
    if "analysis_subdir" in manifest_frame.columns and pd.notna(row["analysis_subdir"]):
        if analysis_root is None:
            return None
        return analysis_root / str(row["analysis_subdir"])
    return None


def _write_reports(enriched: pd.DataFrame, warnings: list[str], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    class_means = enriched.groupby("class_10")[CAVER_T12_FEATURES_17].mean(numeric_only=True)
    try:
        class_means_text = class_means.to_markdown() + "\n"
    except ImportError:
        class_means_text = class_means.to_csv()
    (reports_dir / "class_mean_sanity.md").write_text(class_means_text, encoding="utf-8")

    warning_counter = Counter(warning.split(":", 1)[0] for warning in warnings)
    warning_lines = ["# v_caver_t12 build warnings", ""]
    warning_lines.extend(f"- {key}: {value}" for key, value in sorted(warning_counter.items()))
    warning_lines.append("")
    warning_lines.extend(f"- {warning}" for warning in warnings)
    (reports_dir / "build_warnings.md").write_text(
        "\n".join(warning_lines) + "\n", encoding="utf-8"
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("training", "holdout"))
    parser.add_argument("--base-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/v_caver_t12"))
    parser.add_argument("--analysis-root", type=Path, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the CAVER T12 parquet builder CLI.

    Parameters
    ----------
    argv
        Optional command-line arguments. Defaults to ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.mode == "training":
        summary = build_training_v_caver_t12_parquet(
            base_parquet=args.base_parquet,
            manifest_path=args.manifest,
            output_path=args.output,
            reports_dir=args.reports_dir,
            analysis_root=args.analysis_root,
        )
    else:
        summary = build_holdout_v_caver_t12_parquet(
            base_parquet=args.base_parquet,
            manifest_path=args.manifest,
            output_path=args.output,
            reports_dir=args.reports_dir,
            analysis_root=args.analysis_root,
        )
    logging.info("v_caver_t12 build OK: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
