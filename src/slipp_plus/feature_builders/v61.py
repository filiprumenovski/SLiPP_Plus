"""Build v61 parquets from v49 artifacts by adding normalized shell features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..constants import AROMATIC_ALIPHATIC_NORMALIZED_12, FEATURE_SETS
from ..schemas import validate_holdout, validate_training


def add_normalized_shell_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target-residue totals and per-shell aromatic/aliphatic fractions."""

    out = df.copy()
    for shell in range(1, 5):
        aromatic = out[f"aromatic_count_shell{shell}"].astype(float)
        aliphatic = out[f"aliphatic_count_shell{shell}"].astype(float)
        total = aromatic + aliphatic
        out[f"target_residue_count_shell{shell}"] = total.astype(int)
        out[f"aromatic_fraction_shell{shell}"] = (aromatic / total).where(total > 0, 0.0)
        out[f"aliphatic_fraction_shell{shell}"] = (aliphatic / total).where(total > 0, 0.0)
    return out


def build_v61_from_v49(v49_dir: Path, output_dir: Path) -> dict[str, object]:
    """Materialize v61 training and holdout parquets from v49 parquets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    full = add_normalized_shell_features(pd.read_parquet(v49_dir / "full_pockets.parquet"))
    apo = add_normalized_shell_features(pd.read_parquet(v49_dir / "apo_pdb_holdout.parquet"))
    af = add_normalized_shell_features(pd.read_parquet(v49_dir / "alphafold_holdout.parquet"))

    feature_columns = FEATURE_SETS["v61"]
    validate_training(full, feature_columns)
    validate_holdout(apo, feature_columns)
    validate_holdout(af, feature_columns)

    full_path = output_dir / "full_pockets.parquet"
    apo_path = output_dir / "apo_pdb_holdout.parquet"
    af_path = output_dir / "alphafold_holdout.parquet"
    full.to_parquet(full_path, index=False)
    apo.to_parquet(apo_path, index=False)
    af.to_parquet(af_path, index=False)

    return {
        "full_rows": len(full),
        "apo_rows": len(apo),
        "af_rows": len(af),
        "added_columns": AROMATIC_ALIPHATIC_NORMALIZED_12,
        "full_output": full_path,
        "apo_output": apo_path,
        "af_output": af_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v49-dir",
        type=Path,
        default=Path("processed/v49"),
        help="Directory containing v49 full and holdout parquets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for v61 parquets.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = build_v61_from_v49(args.v49_dir, args.output_dir)
    except Exception as exc:
        print(f"v61 build failed: {exc}", file=sys.stderr)
        return 1
    print(f"full_rows: {result['full_rows']}")
    print(f"apo_rows: {result['apo_rows']}")
    print(f"af_rows: {result['af_rows']}")
    print(f"added_columns: {len(result['added_columns'])}")
    print(f"full_output: {result['full_output']}")
    print(f"apo_output: {result['apo_output']}")
    print(f"af_output: {result['af_output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
