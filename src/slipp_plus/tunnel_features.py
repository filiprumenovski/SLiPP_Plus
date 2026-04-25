"""CAVER-derived tunnel features for the ``v_tunnel`` feature set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .plm_ste_features import _axial_profile, _collect_vert_coords

TUNNEL_FEATURES_15: list[str] = [
    "tunnel_count",
    "tunnel_primary_length",
    "tunnel_primary_bottleneck_radius",
    "tunnel_primary_avg_radius",
    "tunnel_primary_curvature",
    "tunnel_primary_throughput",
    "tunnel_primary_hydrophobicity",
    "tunnel_primary_charge",
    "tunnel_primary_aromatic_fraction",
    "tunnel_max_length",
    "tunnel_total_length",
    "tunnel_min_bottleneck",
    "tunnel_branching_factor",
    "tunnel_length_over_axial",
    "tunnel_extends_beyond_pocket",
]

TUNNEL_MISSINGNESS_3: list[str] = [
    "tunnel_pocket_context_present",
    "tunnel_caver_profile_present",
    "tunnel_has_tunnel",
]

TUNNEL_FEATURES_18: list[str] = TUNNEL_FEATURES_15 + TUNNEL_MISSINGNESS_3

_INT_COLS = frozenset({
    "tunnel_count",
    "tunnel_branching_factor",
    "tunnel_extends_beyond_pocket",
    "tunnel_pocket_context_present",
    "tunnel_caver_profile_present",
    "tunnel_has_tunnel",
})

_KD = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}
_POSITIVE = frozenset({"LYS", "ARG"})
_NEGATIVE = frozenset({"ASP", "GLU"})
_AROMATIC = frozenset({"PHE", "TYR", "TRP", "HIS"})
_EPS = 1e-6
_CACHE_VERSION = 4
_CAVER_CONFIG_VERSION = 2


class MissingStartingPointColumnError(ValueError):
    """Raised when CAVER output cannot map tunnels to multiple start points."""


@dataclass(frozen=True)
class CaverSettings:
    caver_jar: Path
    probe_radius: float = 0.9
    shell_radius: float = 3.0
    shell_depth: float = 4.0
    clustering_threshold: float = 3.5
    timeout_s: int = 45
    max_structure_timeout_s: int = 300
    use_multi_start: bool = False
    java_heap: str = "768m"


@dataclass(frozen=True)
class TunnelBuildThresholds:
    max_missing_structure_frac: float = 0.10
    min_context_present_frac: float = 0.95
    min_profile_present_frac: float = 0.90


def load_caver_settings(path: Path = Path("configs/caver.yaml")) -> CaverSettings:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return CaverSettings(
        caver_jar=Path(raw.get("caver_jar", "tools/caver/caver.jar")),
        probe_radius=float(raw.get("probe_radius", 0.9)),
        shell_radius=float(raw.get("shell_radius", 3.0)),
        shell_depth=float(raw.get("shell_depth", 4.0)),
        clustering_threshold=float(raw.get("clustering_threshold", 3.5)),
        timeout_s=int(raw.get("timeout_s", 45)),
        max_structure_timeout_s=int(raw.get("max_structure_timeout_s", 300)),
        use_multi_start=bool(raw.get("use_multi_start", False)),
        java_heap=str(raw.get("java_heap", "768m")),
    )


def _safe_defaults(
    *,
    pocket_context_present: int = 0,
    caver_profile_present: int = 0,
    has_tunnel: int = 0,
) -> dict[str, float]:
    return {
        "tunnel_count": 0,
        "tunnel_primary_length": 0.0,
        "tunnel_primary_bottleneck_radius": 0.0,
        "tunnel_primary_avg_radius": 0.0,
        "tunnel_primary_curvature": 1.0,
        "tunnel_primary_throughput": 0.0,
        "tunnel_primary_hydrophobicity": 0.0,
        "tunnel_primary_charge": 0.0,
        "tunnel_primary_aromatic_fraction": 0.0,
        "tunnel_max_length": 0.0,
        "tunnel_total_length": 0.0,
        "tunnel_min_bottleneck": 0.0,
        "tunnel_branching_factor": 0,
        "tunnel_length_over_axial": 0.0,
        "tunnel_extends_beyond_pocket": 0,
        "tunnel_pocket_context_present": int(pocket_context_present),
        "tunnel_caver_profile_present": int(caver_profile_present),
        "tunnel_has_tunnel": int(has_tunnel),
    }


def _cast_features(features: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    defaults = _safe_defaults()
    for column in TUNNEL_FEATURES_18:
        value = features.get(column, defaults[column])
        if column in _INT_COLS:
            try:
                out[column] = int(value)
            except (TypeError, ValueError):
                out[column] = int(defaults[column])
        else:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = float(defaults[column])
            if not math.isfinite(parsed):
                parsed = float(defaults[column])
            out[column] = parsed
    return out


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        records: list[dict[str, str]] = []
        for row in reader:
            records.append({_normalize_header(str(k)): str(v) for k, v in row.items() if k})
    return records


def _field(row: dict[str, str], *names: str) -> str | None:
    keys = {_normalize_header(name) for name in names}
    for key, value in row.items():
        if key in keys:
            return value
    return None


def _float_field(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    value = _field(row, *names)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _int_field(row: dict[str, str], *names: str, default: int = 0) -> int:
    value = _field(row, *names)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _parse_int_value(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _tunnel_id(row: dict[str, str]) -> str:
    cluster = _field(row, "Tunnel cluster", "Cluster", "Tunnel", "Tunnel ID") or "0"
    tunnel = _field(row, "Tunnel", "Tunnel ID", "Id") or cluster
    return f"{cluster}:{tunnel}"


def _parse_tunnels_csv(path: Path) -> list[dict[str, Any]]:
    tunnels: list[dict[str, Any]] = []
    for row in _read_csv_records(path):
        starting_point_raw = _field(
            row,
            "Starting point",
            "Starting point index",
            "Starting point ID",
            "Start point",
        )
        tunnels.append(
            {
                "raw": row,
                "tunnel_id": _tunnel_id(row),
                "cluster": _field(row, "Tunnel cluster", "Cluster") or "",
                "has_starting_point": starting_point_raw is not None,
                "starting_point": _parse_int_value(starting_point_raw),
                "length": _float_field(row, "Length"),
                "bottleneck_radius": _float_field(row, "Bottleneck radius", "Bottleneck"),
                "curvature": _float_field(row, "Curvature", default=1.0),
                "throughput": _float_field(row, "Throughput"),
                "avg_radius": _float_field(row, "Avg R", "Average radius", "Average R"),
            }
        )
    return tunnels


def _tunnels_table_path(analysis_dir: Path) -> Path:
    for name in ("tunnels.csv", "tunnel_characteristics.csv"):
        candidate = analysis_dir / name
        if candidate.exists():
            return candidate
    return analysis_dir / "tunnels.csv"


def _parse_profile_points(path: Path) -> dict[str, list[tuple[float, tuple[float, float, float] | None]]]:
    if not path.exists():
        return {}
    profiles: dict[str, list[tuple[float, tuple[float, float, float] | None]]] = defaultdict(list)
    for row in _read_csv_records(path):
        tid = _tunnel_id(row)
        dist = _float_field(row, "Distance", "Distance from origin", "distance_from_origin")
        x_raw = _field(row, "X", "x")
        y_raw = _field(row, "Y", "y")
        z_raw = _field(row, "Z", "z")
        coord = None
        if x_raw is not None and y_raw is not None and z_raw is not None:
            try:
                coord = (float(x_raw), float(y_raw), float(z_raw))
            except ValueError:
                coord = None
        profiles[tid].append((dist, coord))
    for points in profiles.values():
        points.sort(key=lambda item: item[0])
    return profiles


def _parse_residue_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    if path.suffix == ".txt":
        residues: dict[str, list[str]] = defaultdict(list)
        current_cluster = ""
        cluster_re = re.compile(r"^==\s*Tunnel cluster\s+(\S+)\s*==")
        residue_re = re.compile(r"^\s*\S+\s+\S+\s+([A-Z]{3,4})\b")
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            cluster_match = cluster_re.match(line)
            if cluster_match:
                current_cluster = cluster_match.group(1)
                continue
            if not current_cluster or line.lstrip().startswith("#"):
                continue
            residue_match = residue_re.match(line)
            if residue_match:
                resname = residue_match.group(1).upper()
                if resname.startswith("HI"):
                    resname = "HIS"
                residues[f"{current_cluster}:{current_cluster}"].append(resname[:3])
                residues[current_cluster].append(resname[:3])
        return residues
    residues: dict[str, list[str]] = defaultdict(list)
    for row in _read_csv_records(path):
        tid = _tunnel_id(row)
        resname = (_field(row, "Residue", "Residue name", "AA", "resname") or "").upper()
        match = re.search(r"\b([A-Z]{3})\b", resname)
        if match:
            residues[tid].append(match.group(1))
    return residues


def _lining_features(residue_names: list[str]) -> tuple[float, float, float]:
    if not residue_names:
        return 0.0, 0.0, 0.0
    known = [name for name in residue_names if name in _KD]
    hydrophobicity = float(np.mean([_KD[name] for name in known])) if known else 0.0
    charge = float(
        sum(1 for name in residue_names if name in _POSITIVE)
        - sum(1 for name in residue_names if name in _NEGATIVE)
    )
    aromatic_fraction = float(sum(1 for name in residue_names if name in _AROMATIC)) / float(
        len(residue_names)
    )
    return hydrophobicity, charge, aromatic_fraction


def _shared_first_three_angstrom(
    a: list[tuple[float, tuple[float, float, float] | None]],
    b: list[tuple[float, tuple[float, float, float] | None]],
) -> bool:
    a3 = [(d, c) for d, c in a if d <= 3.0 and c is not None]
    b3 = [(d, c) for d, c in b if d <= 3.0 and c is not None]
    if not a3 or not b3:
        return False
    limit = min(len(a3), len(b3))
    if limit == 0:
        return False
    matches = 0
    for idx in range(limit):
        ca = np.asarray(a3[idx][1], dtype=float)
        cb = np.asarray(b3[idx][1], dtype=float)
        if float(np.linalg.norm(ca - cb)) <= 0.5:
            matches += 1
    return matches >= max(1, limit // 2)


def _branching_factor(tunnels: list[dict[str, Any]], profiles: dict[str, list[tuple[float, tuple[float, float, float] | None]]]) -> int:
    if len(tunnels) <= 1:
        return 0
    branched: set[str] = set()
    for i, left in enumerate(tunnels):
        left_id = str(left["tunnel_id"])
        for right in tunnels[i + 1 :]:
            right_id = str(right["tunnel_id"])
            if _shared_first_three_angstrom(profiles.get(left_id, []), profiles.get(right_id, [])):
                branched.add(left_id)
                branched.add(right_id)
    return len(branched)


def _features_from_tunnels(
    tunnels: list[dict[str, Any]],
    residue_map: dict[str, list[str]],
    profile_map: dict[str, list[tuple[float, tuple[float, float, float] | None]]],
    pocket_axial_length: float,
) -> dict[str, float]:
    if not tunnels:
        return _safe_defaults(pocket_context_present=1, caver_profile_present=1)

    primary = max(tunnels, key=lambda row: (float(row["throughput"]), float(row["length"])))
    hydrophobicity, charge, aromatic_fraction = _lining_features(
        residue_map.get(str(primary["tunnel_id"]), [])
    )
    primary_length = float(primary["length"])
    ratio = primary_length / (float(pocket_axial_length) + _EPS)
    if not math.isfinite(ratio):
        ratio = 0.0
    ratio = float(np.clip(ratio, 0.0, 20.0))

    bottlenecks = [float(row["bottleneck_radius"]) for row in tunnels]
    lengths = [float(row["length"]) for row in tunnels]
    return _cast_features(
        {
            "tunnel_count": len(tunnels),
            "tunnel_primary_length": primary_length,
            "tunnel_primary_bottleneck_radius": float(primary["bottleneck_radius"]),
            "tunnel_primary_avg_radius": float(primary["avg_radius"]),
            "tunnel_primary_curvature": float(primary["curvature"]),
            "tunnel_primary_throughput": float(primary["throughput"]),
            "tunnel_primary_hydrophobicity": hydrophobicity,
            "tunnel_primary_charge": charge,
            "tunnel_primary_aromatic_fraction": aromatic_fraction,
            "tunnel_max_length": max(lengths),
            "tunnel_total_length": sum(lengths),
            "tunnel_min_bottleneck": min(bottlenecks) if bottlenecks else 0.0,
            "tunnel_branching_factor": _branching_factor(tunnels, profile_map),
            "tunnel_length_over_axial": ratio,
            "tunnel_extends_beyond_pocket": int(primary_length > float(pocket_axial_length) + 3.0),
            "tunnel_pocket_context_present": 1,
            "tunnel_caver_profile_present": 1,
            "tunnel_has_tunnel": 1,
        }
    )


def _strip_waters(input_pdb: Path, output_pdb: Path) -> None:
    with input_pdb.open("r", encoding="utf-8", errors="ignore") as src, output_pdb.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.startswith("HETATM") and line[17:20].strip().upper() == "HOH":
                continue
            dst.write(line)


def _write_caver_config(
    path: Path,
    pocket_centroids: dict[int, np.ndarray],
    *,
    probe_radius: float,
    shell_radius: float,
    shell_depth: float,
    clustering_threshold: float,
) -> None:
    lines = [
        "load_tunnels no",
        "load_cluster_tree no",
        "time_sparsity 1",
        "first_frame 1",
        "last_frame 1",
        f"probe_radius {probe_radius}",
        f"shell_radius {shell_radius}",
        f"shell_depth {shell_depth}",
        "clustering average_link",
        "weighting_coefficient 1",
        f"clustering_threshold {clustering_threshold}",
        "one_tunnel_in_snapshot cheapest",
        "generate_summary yes",
        "generate_tunnel_characteristics yes",
        "generate_tunnel_profiles yes",
        "compute_tunnel_residues yes",
        "residue_contact_distance 3.0",
        "compute_bottleneck_residues yes",
        "bottleneck_contact_distance 3.0",
        "number_of_approximating_balls 12",
        "compute_errors no",
        "save_error_profiles no",
        "seed 1",
    ]
    for centroid in pocket_centroids.values():
        x, y, z = [float(v) for v in centroid]
        lines.append(f"starting_point_coordinates {x:.3f} {y:.3f} {z:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_caver(
    protein_pdb_path: Path,
    pocket_centroids: dict[int, np.ndarray],
    caver_jar: Path,
    *,
    probe_radius: float,
    shell_radius: float,
    shell_depth: float,
    clustering_threshold: float,
    timeout_s: int,
    max_structure_timeout_s: int,
    java_heap: str,
) -> tuple[Path | None, str | None, tempfile.TemporaryDirectory[str] | None]:
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    try:
        pdb_dir = tmp_path / "pdb"
        pdb_dir.mkdir()
        prepared_pdb = pdb_dir / protein_pdb_path.name
        _strip_waters(protein_pdb_path, prepared_pdb)
        output_dir = tmp_path / "caver_out"
        config_path = tmp_path / "config.txt"
        _write_caver_config(
            config_path,
            pocket_centroids,
            probe_radius=probe_radius,
            shell_radius=shell_radius,
            shell_depth=shell_depth,
            clustering_threshold=clustering_threshold,
        )
        caver_home = caver_jar.resolve().parent
        cmd = [
            "java",
            f"-Xmx{java_heap}",
            "-jar",
            str(caver_jar),
            "-home",
            str(caver_home),
            "-pdb",
            str(pdb_dir),
            "-conf",
            str(config_path),
            "-out",
            str(output_dir),
        ]
        result = subprocess.run(
            cmd,
            check=False,
            timeout=min(
                max_structure_timeout_s,
                max(timeout_s, timeout_s * max(1, len(pocket_centroids))),
            ),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tmp.cleanup()
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            return None, f"CAVER exited {result.returncode}: {'; '.join(detail[:3])}", None
        analysis = output_dir / "analysis"
        tunnels_csv = _tunnels_table_path(analysis)
        if not tunnels_csv.exists():
            tmp.cleanup()
            return None, "CAVER produced no tunnel characteristics table", None
        return analysis, None, tmp
    except subprocess.TimeoutExpired:
        tmp.cleanup()
        return None, "CAVER timeout", None
    except OSError as exc:
        tmp.cleanup()
        return None, f"CAVER invocation failed: {exc}", None


def _extract_from_analysis(
    analysis_dir: Path,
    pocket_order: list[int],
    pocket_axial_lengths: dict[int, float],
) -> dict[int, dict[str, float]]:
    tunnels = _parse_tunnels_csv(_tunnels_table_path(analysis_dir))
    if len(pocket_order) > 1 and tunnels and not any(tunnel["has_starting_point"] for tunnel in tunnels):
        raise MissingStartingPointColumnError(
            "CAVER tunnel table has no starting-point column for multi-pocket run"
        )
    profile_map = _parse_profile_points(analysis_dir / "tunnel_profiles.csv")
    residue_path = analysis_dir / "residues.csv"
    if not residue_path.exists():
        residue_path = analysis_dir / "residues.txt"
    residue_map = _parse_residue_map(residue_path)
    features: dict[int, dict[str, float]] = {}
    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tunnel in tunnels:
        by_start[int(tunnel["starting_point"])].append(tunnel)

    # CAVER versions have used both 0-based and 1-based starting point IDs.
    zero_based_hits = sum(len(by_start.get(i, [])) for i in range(len(pocket_order)))
    one_based_hits = sum(len(by_start.get(i + 1, [])) for i in range(len(pocket_order)))
    offset = 0 if zero_based_hits >= one_based_hits else 1

    for index, pocket_number in enumerate(pocket_order):
        pocket_tunnels = by_start.get(index + offset, [])
        features[pocket_number] = _features_from_tunnels(
            pocket_tunnels,
            residue_map,
            profile_map,
            pocket_axial_lengths.get(pocket_number, 0.0),
        )
    return features


def _extract_single_pocket_with_caver(
    protein_pdb: Path,
    pocket_number: int,
    centroid: np.ndarray,
    axial_length: float,
    caver_jar: Path,
    settings: dict[str, object],
) -> tuple[dict[str, float], str | None]:
    analysis, warning, tmp_ref = _run_caver(
        protein_pdb,
        {pocket_number: centroid},
        caver_jar,
        probe_radius=float(settings["probe_radius"]),
        shell_radius=float(settings["shell_radius"]),
        shell_depth=float(settings["shell_depth"]),
        clustering_threshold=float(settings["clustering_threshold"]),
        timeout_s=int(settings["timeout_s"]),
        max_structure_timeout_s=int(settings["max_structure_timeout_s"]),
        java_heap=str(settings["java_heap"]),
    )
    if analysis is None and warning and "starting point" in warning.lower():
        analysis, warning, tmp_ref = _run_caver(
            protein_pdb,
            {pocket_number: centroid},
            caver_jar,
            probe_radius=0.7,
            shell_radius=float(settings["shell_radius"]),
            shell_depth=float(settings["shell_depth"]),
            clustering_threshold=float(settings["clustering_threshold"]),
            timeout_s=int(settings["timeout_s"]),
            max_structure_timeout_s=int(settings["max_structure_timeout_s"]),
            java_heap=str(settings["java_heap"]),
        )
    if analysis is None:
        return _safe_defaults(pocket_context_present=1), warning or "CAVER failed"
    try:
        return _extract_from_analysis(analysis, [pocket_number], {pocket_number: axial_length})[
            pocket_number
        ], None
    except (OSError, ValueError) as exc:
        return _safe_defaults(pocket_context_present=1), f"CAVER output parse failed: {exc}"
    finally:
        if tmp_ref is not None:
            tmp_ref.cleanup()


def extract_pocket_tunnel_features(
    protein_pdb_path: Path,
    pocket_centroid: np.ndarray,
    pocket_axial_length: float,
    caver_jar: Path,
    *,
    probe_radius: float = 0.9,
    shell_radius: float = 3.0,
    shell_depth: float = 4.0,
    clustering_threshold: float = 3.5,
    timeout_s: int = 45,
) -> dict[str, float]:
    """Run CAVER for one pocket and return the 15 tunnel features."""

    analysis, warning, tmp_ref = _run_caver(
        protein_pdb_path,
        {0: np.asarray(pocket_centroid, dtype=float)},
        caver_jar,
        probe_radius=probe_radius,
        shell_radius=shell_radius,
        shell_depth=shell_depth,
        clustering_threshold=clustering_threshold,
        timeout_s=timeout_s,
        max_structure_timeout_s=max(timeout_s, timeout_s),
        java_heap="768m",
    )
    if analysis is None:
        if warning and "starting point" in warning.lower() and probe_radius > 0.7:
            return extract_pocket_tunnel_features(
                protein_pdb_path,
                pocket_centroid,
                pocket_axial_length,
                caver_jar,
                probe_radius=0.7,
                shell_radius=shell_radius,
                shell_depth=shell_depth,
                clustering_threshold=clustering_threshold,
                timeout_s=timeout_s,
            )
        return _safe_defaults(pocket_context_present=1)
    try:
        return _extract_from_analysis(analysis, [0], {0: pocket_axial_length})[0]
    finally:
        if tmp_ref is not None:
            tmp_ref.cleanup()


def _class_code_from_pdb_ligand(pdb_ligand: str) -> str:
    return pdb_ligand.split("/", 1)[0]


def _pdb_stem(pdb_ligand: str) -> str:
    return Path(pdb_ligand).stem


def _pocket_context(structure_dir: Path, pocket_number: int) -> tuple[np.ndarray, float]:
    vert = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
    coords = _collect_vert_coords(vert)
    centroid = coords.mean(axis=0)
    axial = _axial_profile(coords)
    axial_length = float(axial["axial_length"]) if axial is not None else 0.0
    return centroid, axial_length


def _process_structure(task: dict[str, object]) -> dict[str, object]:
    rows = list(task["rows"])  # type: ignore[arg-type]
    structure_dir = Path(str(task["structure_dir"]))
    protein_pdb = Path(str(task["protein_pdb"]))
    caver_jar = Path(str(task["caver_jar"]))
    label = str(task["label"])
    settings = task["settings"]  # type: ignore[assignment]

    warnings: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    pocket_numbers = sorted({int(row["matched_pocket_number"]) for row in rows})
    centroids: dict[int, np.ndarray] = {}
    axial_lengths: dict[int, float] = {}
    for pocket_number in pocket_numbers:
        try:
            centroid, axial_length = _pocket_context(structure_dir, pocket_number)
            centroids[pocket_number] = centroid
            axial_lengths[pocket_number] = axial_length
        except (OSError, ValueError) as exc:
            warnings.append(f"{label}/pocket{pocket_number}: pocket context failed: {exc}")

    feature_cache = {
        pocket_number: _safe_defaults(
            pocket_context_present=int(pocket_number in centroids),
        )
        for pocket_number in pocket_numbers
    }
    analysis_output_root_raw = task.get("analysis_output_root")
    persisted_analysis_dir: Path | None = None
    if centroids and protein_pdb.exists() and (
        bool(settings.get("use_multi_start", False)) or len(centroids) == 1
    ):
        analysis, warning, tmp_ref = _run_caver(
            protein_pdb,
            centroids,
            caver_jar,
            probe_radius=float(settings["probe_radius"]),
            shell_radius=float(settings["shell_radius"]),
            shell_depth=float(settings["shell_depth"]),
            clustering_threshold=float(settings["clustering_threshold"]),
            timeout_s=int(settings["timeout_s"]),
        max_structure_timeout_s=int(settings["max_structure_timeout_s"]),
        java_heap=str(settings["java_heap"]),
        )
        if analysis is None and warning and "starting point" in warning.lower():
            analysis, warning, tmp_ref = _run_caver(
                protein_pdb,
                centroids,
                caver_jar,
                probe_radius=0.7,
                shell_radius=float(settings["shell_radius"]),
                shell_depth=float(settings["shell_depth"]),
                clustering_threshold=float(settings["clustering_threshold"]),
                timeout_s=int(settings["timeout_s"]),
            max_structure_timeout_s=int(settings["max_structure_timeout_s"]),
            java_heap=str(settings["java_heap"]),
            )
        if analysis is None:
            warnings.append(f"{label}: {warning or 'CAVER failed'}")
        else:
            if analysis_output_root_raw is not None:
                analysis_output_root = Path(str(analysis_output_root_raw))
                persisted_analysis_dir = analysis_output_root / _cache_key(label)
                if persisted_analysis_dir.exists():
                    shutil.rmtree(persisted_analysis_dir)
                persisted_analysis_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(analysis, persisted_analysis_dir)
            try:
                feature_cache.update(
                    _extract_from_analysis(analysis, list(centroids.keys()), axial_lengths)
                )
            except MissingStartingPointColumnError as exc:
                warnings.append(f"{label}: {exc}; falling back to per-pocket CAVER")
                for pocket_number, centroid in centroids.items():
                    feature, pocket_warning = _extract_single_pocket_with_caver(
                        protein_pdb,
                        pocket_number,
                        centroid,
                        axial_lengths[pocket_number],
                        caver_jar,
                        settings,  # type: ignore[arg-type]
                    )
                    feature_cache[pocket_number] = feature
                    if pocket_warning is not None:
                        warnings.append(f"{label}/pocket{pocket_number}: {pocket_warning}")
            except (OSError, ValueError) as exc:
                warnings.append(f"{label}: CAVER output parse failed: {exc}")
            finally:
                if tmp_ref is not None:
                    tmp_ref.cleanup()
    elif not protein_pdb.exists():
        warnings.append(f"{label}: missing protein pdb: {protein_pdb}")
    elif centroids and protein_pdb.exists():
        for pocket_number, centroid in centroids.items():
            feature, pocket_warning = _extract_single_pocket_with_caver(
                protein_pdb,
                pocket_number,
                centroid,
                axial_lengths[pocket_number],
                caver_jar,
                settings,  # type: ignore[arg-type]
            )
            feature_cache[pocket_number] = feature
            if pocket_warning is not None:
                warnings.append(f"{label}/pocket{pocket_number}: {pocket_warning}")

    enriched: list[dict[str, object]] = []
    for row in rows:
        pocket_number = int(row["matched_pocket_number"])
        out = dict(row)
        out.update(_cast_features(feature_cache.get(pocket_number, _safe_defaults())))
        enriched.append(out)
    key_column = "pdb_ligand" if "pdb_ligand" in rows[0] else "structure_id"
    for start_idx, pocket_number in enumerate(pocket_numbers):
        manifest_row = {
            key_column: rows[0].get(key_column),
            "matched_pocket_number": pocket_number,
            "starting_point_index": start_idx,
            "pocket_axial_length": float(axial_lengths.get(pocket_number, 0.0)),
        }
        if persisted_analysis_dir is not None:
            manifest_row["analysis_dir"] = str(persisted_analysis_dir)
        manifest_rows.append(manifest_row)
    return {"rows": enriched, "warnings": warnings, "manifest_rows": manifest_rows}


def _cache_fingerprint(task: dict[str, object]) -> str:
    settings = task.get("settings", {})
    payload = {
        "cache_version": _CACHE_VERSION,
        "caver_config_version": _CAVER_CONFIG_VERSION,
        "caver_jar": str(task.get("caver_jar", "")),
        "settings": settings,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", label).strip("_")


def _process_structure_cached(task: dict[str, object]) -> dict[str, object]:
    cache_dir_raw = task.get("cache_dir")
    label = str(task["label"])
    if cache_dir_raw is None:
        return _process_structure(task)

    cache_dir = Path(str(cache_dir_raw))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(label)}.json"
    expected_fingerprint = _cache_fingerprint(task)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("cache_version") != _CACHE_VERSION:
                raise ValueError(
                    f"cache_version={cached.get('cache_version')} expected={_CACHE_VERSION}"
                )
            if cached.get("caver_config_version") != _CAVER_CONFIG_VERSION:
                raise ValueError(
                    "caver_config_version="
                    f"{cached.get('caver_config_version')} expected={_CAVER_CONFIG_VERSION}"
                )
            if cached.get("settings_fingerprint") != expected_fingerprint:
                raise ValueError("settings_fingerprint mismatch")
            return {
                "rows": cached["rows"],
                "warnings": cached.get("warnings", []),
                "cached": True,
                "label": label,
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logging.warning("%s: ignoring corrupt tunnel cache %s: %s", label, cache_path, exc)

    result = _process_structure(task)
    payload = {
        "cache_version": _CACHE_VERSION,
        "caver_config_version": _CAVER_CONFIG_VERSION,
        "settings_fingerprint": expected_fingerprint,
        "label": label,
        "rows": result["rows"],
        "warnings": result["warnings"],
    }
    tmp_path = cache_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False)
    tmp_path.replace(cache_path)
    result["cached"] = False
    result["label"] = label
    return result


def _threshold_in_unit_interval(name: str, value: float) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {parsed}")
    return parsed


def _validate_thresholds(thresholds: TunnelBuildThresholds) -> TunnelBuildThresholds:
    return TunnelBuildThresholds(
        max_missing_structure_frac=_threshold_in_unit_interval(
            "max_missing_structure_frac", thresholds.max_missing_structure_frac
        ),
        min_context_present_frac=_threshold_in_unit_interval(
            "min_context_present_frac", thresholds.min_context_present_frac
        ),
        min_profile_present_frac=_threshold_in_unit_interval(
            "min_profile_present_frac", thresholds.min_profile_present_frac
        ),
    )


def _preflight_validate_task_inputs(
    tasks: list[dict[str, object]],
    *,
    thresholds: TunnelBuildThresholds,
) -> dict[str, object]:
    if not tasks:
        raise ValueError("no tasks to process")

    missing_protein = 0
    missing_structure_dir = 0
    missing_pockets_dir = 0
    bad: list[str] = []
    for task in tasks:
        label = str(task["label"])
        protein_pdb = Path(str(task["protein_pdb"]))
        structure_dir = Path(str(task["structure_dir"]))
        pockets_dir = structure_dir / "pockets"
        has_issue = False
        if not protein_pdb.exists():
            missing_protein += 1
            has_issue = True
        if not structure_dir.exists():
            missing_structure_dir += 1
            has_issue = True
        if not pockets_dir.exists():
            missing_pockets_dir += 1
            has_issue = True
        if has_issue:
            bad.append(label)

    total = len(tasks)
    missing_any = len(bad)
    missing_frac = float(missing_any) / float(total)
    summary: dict[str, object] = {
        "structures_total": total,
        "structures_with_missing_inputs": missing_any,
        "missing_input_frac": missing_frac,
        "missing_protein_pdb": missing_protein,
        "missing_structure_dir": missing_structure_dir,
        "missing_pockets_dir": missing_pockets_dir,
        "example_labels": sorted(set(bad))[:10],
    }
    logging.info("v_tunnel preflight summary: %s", summary)
    if missing_frac > thresholds.max_missing_structure_frac:
        examples = ", ".join(summary["example_labels"])  # type: ignore[arg-type]
        raise ValueError(
            "preflight failed: missing structure inputs for "
            f"{missing_any}/{total} structures ({missing_frac:.1%}), "
            f"allowed <= {thresholds.max_missing_structure_frac:.1%}. "
            "Check --structures-root/--source-pdbs-root. "
            f"Examples: {examples}"
        )
    return summary


def _quality_metrics(enriched: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    row_count = float(len(enriched))
    if row_count <= 0:
        return {
            "row_count": 0.0,
            "context_present_frac": 0.0,
            "profile_present_frac": 0.0,
            "has_tunnel_frac": 0.0,
        }
    for column in (
        "tunnel_pocket_context_present",
        "tunnel_caver_profile_present",
        "tunnel_has_tunnel",
    ):
        if column not in enriched.columns:
            raise ValueError(f"missing expected quality column: {column}")
    metrics["row_count"] = row_count
    metrics["context_present_frac"] = float(
        pd.to_numeric(enriched["tunnel_pocket_context_present"], errors="raise").mean()
    )
    metrics["profile_present_frac"] = float(
        pd.to_numeric(enriched["tunnel_caver_profile_present"], errors="raise").mean()
    )
    metrics["has_tunnel_frac"] = float(
        pd.to_numeric(enriched["tunnel_has_tunnel"], errors="raise").mean()
    )
    return metrics


def _enforce_quality_gates(
    quality: dict[str, float],
    *,
    thresholds: TunnelBuildThresholds,
) -> None:
    context_frac = quality["context_present_frac"]
    profile_frac = quality["profile_present_frac"]
    if context_frac < thresholds.min_context_present_frac:
        raise ValueError(
            "quality gate failed: tunnel_pocket_context_present mean="
            f"{context_frac:.3f} below minimum {thresholds.min_context_present_frac:.3f}"
        )
    if profile_frac < thresholds.min_profile_present_frac:
        raise ValueError(
            "quality gate failed: tunnel_caver_profile_present mean="
            f"{profile_frac:.3f} below minimum {thresholds.min_profile_present_frac:.3f}"
        )


def _run_tasks_and_write(
    tasks: list[dict[str, object]],
    base: pd.DataFrame,
    output_path: Path,
    workers: int,
    reports_dir: Path,
    cache_dir: Path | None,
    thresholds: TunnelBuildThresholds,
    analysis_manifest_path: Path | None = None,
    analysis_output_root: Path | None = None,
) -> dict[str, object]:
    thresholds = _validate_thresholds(thresholds)
    start = time.time()
    preflight = _preflight_validate_task_inputs(tasks, thresholds=thresholds)
    if analysis_manifest_path is not None and analysis_output_root is None:
        raise ValueError("analysis_manifest_path requires analysis_output_root to be set")
    if analysis_output_root is not None:
        analysis_output_root.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        for task in tasks:
            task["cache_dir"] = str(cache_dir)
    if analysis_output_root is not None:
        for task in tasks:
            task["analysis_output_root"] = str(analysis_output_root)
    if workers <= 1 or len(tasks) == 1:
        results = [_process_structure_cached(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_structure_cached, task) for task in tasks]
            results = []
            cached_count = 0
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result.get("cached"):
                    cached_count += 1
                results.append(result)
                if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                    logging.info(
                        "v_tunnel CAVER progress: %s/%s structures (%s cached)",
                        completed,
                        len(tasks),
                        cached_count,
                    )

    all_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    all_manifest_rows: list[dict[str, object]] = []
    for result in results:
        all_rows.extend(result["rows"])  # type: ignore[arg-type]
        warnings.extend(result["warnings"])  # type: ignore[arg-type]
        all_manifest_rows.extend(result.get("manifest_rows", []))  # type: ignore[arg-type]
    for warning in warnings:
        logging.warning(warning)

    enriched = pd.DataFrame(all_rows)
    if "_row_order" in enriched.columns:
        enriched = enriched.sort_values("_row_order").reset_index(drop=True)
    if len(enriched) != len(base):
        raise ValueError(f"row drift after v_tunnel join: got {len(enriched)}, expected {len(base)}")

    for column in TUNNEL_FEATURES_18:
        if column not in enriched.columns:
            raise ValueError(f"missing tunnel feature column: {column}")
        numeric = pd.to_numeric(enriched[column], errors="raise")
        if numeric.isna().any():
            raise ValueError(f"{column}: NaN values present after build")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column}: non-finite values present after build")

    quality = _quality_metrics(enriched)
    _enforce_quality_gates(quality, thresholds=thresholds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    if analysis_manifest_path is not None:
        analysis_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_manifest_rows).to_csv(analysis_manifest_path, index=False)
    elapsed_s = time.time() - start
    summary = {
        "rows": len(enriched),
        "structures": len(tasks),
        "warnings": len(warnings),
        "warnings_list": warnings,
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
        "elapsed_s": elapsed_s,
        "cache_dir": cache_dir,
        "preflight": preflight,
        "quality": quality,
        "thresholds": thresholds,
        "analysis_output_root": analysis_output_root,
        "analysis_manifest_path": analysis_manifest_path,
    }
    _write_reports(enriched, warnings, summary, reports_dir)
    return summary


def _warning_type(warning: str) -> str:
    if ":" in warning:
        return warning.split(":", 1)[1].strip().split(":", 1)[0].strip() or "warning"
    return warning.strip().split(":", 1)[0] or "warning"


def _warning_class(warning: str) -> str:
    first = warning.split(":", 1)[0]
    if "/" in first:
        return first.split("/", 1)[0]
    return "unknown"


def _write_reports(
    enriched: pd.DataFrame,
    warnings: list[str],
    summary: dict[str, object],
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    class_means_markdown = "Class means unavailable: no class_10 column in this build."
    if "class_10" in enriched.columns:
        class_means = enriched.groupby("class_10")[TUNNEL_FEATURES_18].mean(numeric_only=True)
        class_means_markdown = _frame_to_markdown(class_means)
        (reports_dir / "class_mean_sanity.md").write_text(
            class_means_markdown + "\n", encoding="utf-8"
        )

    by_type = Counter(_warning_type(w) for w in warnings)
    by_class = Counter(_warning_class(w) for w in warnings)
    warnings_lines = ["# v_tunnel build warnings", ""]
    warnings_lines.append("## Counts by type")
    warnings_lines.extend(f"- {key}: {value}" for key, value in sorted(by_type.items()))
    warnings_lines.append("")
    warnings_lines.append("## Counts by class")
    warnings_lines.extend(f"- {key}: {value}" for key, value in sorted(by_class.items()))
    warnings_lines.append("")
    warnings_lines.append("## All warnings")
    warnings_lines.extend(f"- {warning}" for warning in warnings)
    (reports_dir / "build_warnings.md").write_text("\n".join(warnings_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# v_tunnel build summary",
        "",
        f"- rows: {summary['rows']}",
        f"- structures: {summary['structures']}",
        f"- warnings: {summary['warnings']}",
        f"- elapsed_s: {float(summary['elapsed_s']):.1f}",
        f"- output: {summary['output_path']}",
        f"- output_size_mb: {float(summary['output_size_mb']):.1f}",
        f"- cache_dir: {summary.get('cache_dir')}",
        f"- analysis_output_root: {summary.get('analysis_output_root')}",
        f"- analysis_manifest_path: {summary.get('analysis_manifest_path')}",
        "",
        "## Preflight",
        "",
    ]
    preflight = summary.get("preflight", {})
    if isinstance(preflight, dict) and preflight:
        summary_lines.extend(
            [
                f"- structures_total: {preflight.get('structures_total')}",
                f"- structures_with_missing_inputs: {preflight.get('structures_with_missing_inputs')}",
                f"- missing_input_frac: {float(preflight.get('missing_input_frac', 0.0)):.3f}",
                f"- missing_protein_pdb: {preflight.get('missing_protein_pdb')}",
                f"- missing_structure_dir: {preflight.get('missing_structure_dir')}",
                f"- missing_pockets_dir: {preflight.get('missing_pockets_dir')}",
                "",
            ]
        )
    else:
        summary_lines.extend(["- unavailable", ""])

    summary_lines.extend(
        [
            "## Quality",
            "",
        ]
    )
    quality = summary.get("quality", {})
    thresholds = summary.get("thresholds")
    if isinstance(quality, dict) and quality:
        summary_lines.extend(
            [
                f"- context_present_frac: {float(quality.get('context_present_frac', 0.0)):.3f}",
                f"- profile_present_frac: {float(quality.get('profile_present_frac', 0.0)):.3f}",
                f"- has_tunnel_frac: {float(quality.get('has_tunnel_frac', 0.0)):.3f}",
            ]
        )
    else:
        summary_lines.append("- unavailable")
    if isinstance(thresholds, TunnelBuildThresholds):
        summary_lines.extend(
            [
                f"- min_context_present_frac: {thresholds.min_context_present_frac:.3f}",
                f"- min_profile_present_frac: {thresholds.min_profile_present_frac:.3f}",
                f"- max_missing_structure_frac: {thresholds.max_missing_structure_frac:.3f}",
            ]
        )
    summary_lines.extend(
        [
            "",
        "## Class means",
        "",
        class_means_markdown,
        "",
        ]
    )
    (reports_dir / "build_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact markdown table without optional tabulate dependency."""
    out = frame.reset_index()
    columns = [str(column) for column in out.columns]
    rows = [[str(value) for value in row] for row in out.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows)) if rows else len(columns[idx])
        for idx in range(len(columns))
    ]

    def _fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [_fmt(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def build_training_v_tunnel_parquet(
    base_parquet: Path = Path("processed/v_sterol/full_pockets.parquet"),
    source_pdbs_root: Path = Path("data/structures/source_pdbs"),
    caver_jar: Path | None = None,
    output_path: Path = Path("processed/v_tunnel/full_pockets.parquet"),
    workers: int = 8,
    cache_dir: Path | None = Path("processed/v_tunnel/structure_json"),
    max_missing_structure_frac: float = 0.02,
    min_context_present_frac: float = 0.98,
    min_profile_present_frac: float = 0.95,
    analysis_output_root: Path | None = None,
    analysis_manifest_path: Path | None = None,
) -> dict[str, object]:
    settings = load_caver_settings()
    if caver_jar is not None:
        settings = CaverSettings(
            caver_jar=caver_jar,
            probe_radius=settings.probe_radius,
            shell_radius=settings.shell_radius,
            shell_depth=settings.shell_depth,
            clustering_threshold=settings.clustering_threshold,
            timeout_s=settings.timeout_s,
            max_structure_timeout_s=settings.max_structure_timeout_s,
            use_multi_start=settings.use_multi_start,
            java_heap=settings.java_heap,
        )
    if not settings.caver_jar.exists():
        raise FileNotFoundError(f"CAVER jar not found: {settings.caver_jar}")
    base = pd.read_parquet(base_parquet)
    if "pdb_ligand" not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected pdb_ligand + matched_pocket_number columns")

    tasks: list[dict[str, object]] = []
    for pdb_ligand, frame in base.groupby("pdb_ligand", sort=False):
        class_code = _class_code_from_pdb_ligand(str(pdb_ligand))
        stem = _pdb_stem(str(pdb_ligand))
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(source_pdbs_root / class_code / f"{stem}_out"),
                "protein_pdb": str(source_pdbs_root / str(pdb_ligand)),
                "caver_jar": str(settings.caver_jar),
                "label": str(pdb_ligand),
                "settings": {
                    "probe_radius": settings.probe_radius,
                    "shell_radius": settings.shell_radius,
                    "shell_depth": settings.shell_depth,
                    "clustering_threshold": settings.clustering_threshold,
                    "timeout_s": settings.timeout_s,
                    "max_structure_timeout_s": settings.max_structure_timeout_s,
                    "use_multi_start": settings.use_multi_start,
                    "java_heap": settings.java_heap,
                },
            }
        )
    return _run_tasks_and_write(
        tasks,
        base,
        output_path,
        workers,
        Path("reports/v_tunnel"),
        cache_dir,
        TunnelBuildThresholds(
            max_missing_structure_frac=max_missing_structure_frac,
            min_context_present_frac=min_context_present_frac,
            min_profile_present_frac=min_profile_present_frac,
        ),
        analysis_manifest_path=analysis_manifest_path,
        analysis_output_root=analysis_output_root,
    )


def build_holdout_v_tunnel_parquet(
    base_parquet: Path,
    structures_root: Path,
    caver_jar: Path | None,
    output_path: Path,
    workers: int = 8,
    cache_dir: Path | None = None,
    max_missing_structure_frac: float = 0.10,
    min_context_present_frac: float = 0.90,
    min_profile_present_frac: float = 0.80,
    analysis_output_root: Path | None = None,
    analysis_manifest_path: Path | None = None,
) -> dict[str, object]:
    settings = load_caver_settings()
    if caver_jar is not None:
        settings = CaverSettings(
            caver_jar=caver_jar,
            probe_radius=settings.probe_radius,
            shell_radius=settings.shell_radius,
            shell_depth=settings.shell_depth,
            clustering_threshold=settings.clustering_threshold,
            timeout_s=settings.timeout_s,
            max_structure_timeout_s=settings.max_structure_timeout_s,
            use_multi_start=settings.use_multi_start,
            java_heap=settings.java_heap,
        )
    if not settings.caver_jar.exists():
        raise FileNotFoundError(f"CAVER jar not found: {settings.caver_jar}")
    base = pd.read_parquet(base_parquet)
    if "structure_id" not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected structure_id + matched_pocket_number columns")

    tasks: list[dict[str, object]] = []
    for structure_id, frame in base.groupby("structure_id", sort=False):
        stem = str(structure_id)
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(structures_root / f"{stem}_out"),
                "protein_pdb": str(structures_root / f"{stem}.pdb"),
                "caver_jar": str(settings.caver_jar),
                "label": stem,
                "settings": {
                    "probe_radius": settings.probe_radius,
                    "shell_radius": settings.shell_radius,
                    "shell_depth": settings.shell_depth,
                    "clustering_threshold": settings.clustering_threshold,
                    "timeout_s": settings.timeout_s,
                    "max_structure_timeout_s": settings.max_structure_timeout_s,
                    "use_multi_start": settings.use_multi_start,
                    "java_heap": settings.java_heap,
                },
            }
        )
    if cache_dir is None:
        cache_dir = output_path.parent / "structure_json" / output_path.stem
    return _run_tasks_and_write(
        tasks,
        base,
        output_path,
        workers,
        Path("reports/v_tunnel"),
        cache_dir,
        TunnelBuildThresholds(
            max_missing_structure_frac=max_missing_structure_frac,
            min_context_present_frac=min_context_present_frac,
            min_profile_present_frac=min_profile_present_frac,
        ),
        analysis_manifest_path=analysis_manifest_path,
        analysis_output_root=analysis_output_root,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser("training", help="Build processed/v_tunnel/full_pockets.parquet.")
    training.add_argument("--base-parquet", type=Path, default=Path("processed/v_sterol/full_pockets.parquet"))
    training.add_argument("--source-pdbs-root", type=Path, default=Path("data/structures/source_pdbs"))
    training.add_argument("--caver-jar", type=Path, default=None)
    training.add_argument("--output", type=Path, default=Path("processed/v_tunnel/full_pockets.parquet"))
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--cache-dir", type=Path, default=Path("processed/v_tunnel/structure_json"))
    training.add_argument("--no-cache", action="store_true", help="Disable per-structure JSON cache/resume.")
    training.add_argument("--max-missing-structure-frac", type=float, default=0.02)
    training.add_argument("--min-context-present-frac", type=float, default=0.98)
    training.add_argument("--min-profile-present-frac", type=float, default=0.95)
    training.add_argument("--analysis-output-root", type=Path, default=None)
    training.add_argument("--analysis-manifest", type=Path, default=None)

    holdout = sub.add_parser("holdout", help="Build a v_tunnel holdout parquet.")
    holdout.add_argument("--base-parquet", type=Path, required=True)
    holdout.add_argument("--structures-root", type=Path, required=True)
    holdout.add_argument("--caver-jar", type=Path, default=None)
    holdout.add_argument("--output", type=Path, required=True)
    holdout.add_argument("--workers", type=int, default=8)
    holdout.add_argument("--cache-dir", type=Path, default=None)
    holdout.add_argument("--no-cache", action="store_true", help="Disable per-structure JSON cache/resume.")
    holdout.add_argument("--max-missing-structure-frac", type=float, default=0.10)
    holdout.add_argument("--min-context-present-frac", type=float, default=0.90)
    holdout.add_argument("--min-profile-present-frac", type=float, default=0.80)
    holdout.add_argument("--analysis-output-root", type=Path, default=None)
    holdout.add_argument("--analysis-manifest", type=Path, default=None)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    try:
        if args.command == "training":
            summary = build_training_v_tunnel_parquet(
                base_parquet=args.base_parquet,
                source_pdbs_root=args.source_pdbs_root,
                caver_jar=args.caver_jar,
                output_path=args.output,
                workers=args.workers,
                cache_dir=None if args.no_cache else args.cache_dir,
                max_missing_structure_frac=args.max_missing_structure_frac,
                min_context_present_frac=args.min_context_present_frac,
                min_profile_present_frac=args.min_profile_present_frac,
                analysis_output_root=args.analysis_output_root,
                analysis_manifest_path=args.analysis_manifest,
            )
        elif args.command == "holdout":
            summary = build_holdout_v_tunnel_parquet(
                base_parquet=args.base_parquet,
                structures_root=args.structures_root,
                caver_jar=args.caver_jar,
                output_path=args.output,
                workers=args.workers,
                cache_dir=None if args.no_cache else args.cache_dir,
                max_missing_structure_frac=args.max_missing_structure_frac,
                min_context_present_frac=args.min_context_present_frac,
                min_profile_present_frac=args.min_profile_present_frac,
                analysis_output_root=args.analysis_output_root,
                analysis_manifest_path=args.analysis_manifest,
            )
        else:
            raise ValueError(f"unknown command {args.command}")
    except Exception as exc:
        logging.error("v_tunnel build failed: %s", exc)
        return 1

    print(f"Rows: {summary['rows']}")
    print(f"Structures: {summary['structures']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Output: {summary['output_path']} ({summary['output_size_mb']:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


from .constants import TUNNEL_FEATURES_15 as _REGISTRY_TUNNEL_FEATURES_15  # noqa: E402
from .constants import TUNNEL_FEATURES_18 as _REGISTRY_TUNNEL_FEATURES_18  # noqa: E402
from .constants import TUNNEL_MISSINGNESS_3 as _REGISTRY_TUNNEL_MISSINGNESS_3  # noqa: E402

assert TUNNEL_FEATURES_15 == _REGISTRY_TUNNEL_FEATURES_15, (
    "tunnel_features.TUNNEL_FEATURES_15 drifted from constants.TUNNEL_FEATURES_15"
)
assert TUNNEL_MISSINGNESS_3 == _REGISTRY_TUNNEL_MISSINGNESS_3, (
    "tunnel_features.TUNNEL_MISSINGNESS_3 drifted from constants.TUNNEL_MISSINGNESS_3"
)
assert TUNNEL_FEATURES_18 == _REGISTRY_TUNNEL_FEATURES_18, (
    "tunnel_features.TUNNEL_FEATURES_18 drifted from constants.TUNNEL_FEATURES_18"
)
