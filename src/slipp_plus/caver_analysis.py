"""Persisted-output-first CAVER parsing and Tier 1-2 feature derivation."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import CAVER_T12_FEATURES_17

_EPS = 1e-6
_HIGH_CURVATURE_THRESHOLD = 0.15
_NEUTRAL_ALIGNMENT_ANGLE_DEG = 90.0


@dataclass(frozen=True)
class CaverProfilePoint:
    distance: float
    radius: float
    coord: tuple[float, float, float] | None


@dataclass(frozen=True)
class CaverTunnel:
    tunnel_id: str
    cluster: str
    starting_point_index: int
    length: float
    throughput: float
    bottleneck_radius: float
    avg_radius: float
    curvature: float
    profile_points: tuple[CaverProfilePoint, ...]
    lining_residues: tuple[str, ...]


@dataclass(frozen=True)
class CaverPocketContext:
    matched_pocket_number: int
    starting_point_index: int
    pocket_axial_length: float
    pocket_principal_axis: tuple[float, float, float] | None = None


def safe_caver_t12_defaults() -> dict[str, float]:
    """Return finite default values for every Tier 1-2 CAVER feature.

    Returns
    -------
    dict[str, float]
        Feature map keyed by ``CAVER_T12_FEATURES_17``. Defaults encode "no
        tunnel observed" while preserving finite numeric values for schema
        validation and model scoring.
    """

    return {
        "caver_tunnel_count": 0,
        "caver_primary_length": 0.0,
        "caver_total_length": 0.0,
        "caver_primary_bottleneck_radius": 0.0,
        "caver_median_bottleneck_radius": 0.0,
        "caver_primary_mean_radius": 0.0,
        "caver_primary_radius_std": 0.0,
        "caver_primary_radius_min": 0.0,
        "caver_primary_radius_max": 0.0,
        "caver_primary_radius_skewness": 0.0,
        "caver_primary_straightness": 1.0,
        "caver_primary_mean_curvature": 0.0,
        "caver_primary_max_curvature": 0.0,
        "caver_primary_high_curvature_count": 0,
        "caver_primary_alignment_angle_deg": _NEUTRAL_ALIGNMENT_ANGLE_DEG,
        "caver_primary_bottleneck_count": 0,
        "caver_primary_length_over_axial": 0.0,
    }


def cast_caver_t12_features(features: dict[str, float | int]) -> dict[str, float]:
    """Coerce a partial CAVER feature map into the canonical numeric schema.

    Parameters
    ----------
    features
        Raw or partially-derived CAVER feature values.

    Returns
    -------
    dict[str, float]
        Complete feature map in ``CAVER_T12_FEATURES_17`` order with invalid,
        missing, or non-finite values replaced by safe defaults.
    """

    defaults = safe_caver_t12_defaults()
    out: dict[str, float] = {}
    int_columns = {
        "caver_tunnel_count",
        "caver_primary_high_curvature_count",
        "caver_primary_bottleneck_count",
    }
    for column in CAVER_T12_FEATURES_17:
        value = features.get(column, defaults[column])
        if column in int_columns:
            try:
                out[column] = int(value)
            except (TypeError, ValueError):
                out[column] = int(defaults[column])
            continue
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
            records.append({_normalize_header(str(key)): str(value) for key, value in row.items() if key})
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


def _tunnel_id(row: dict[str, str]) -> str:
    cluster = _field(row, "Tunnel cluster", "Cluster", "Tunnel", "Tunnel ID") or "0"
    tunnel = _field(row, "Tunnel", "Tunnel ID", "Id") or cluster
    return f"{cluster}:{tunnel}"


def _tunnels_table_path(analysis_dir: Path) -> Path:
    for name in ("tunnels.csv", "tunnel_characteristics.csv"):
        candidate = analysis_dir / name
        if candidate.exists():
            return candidate
    return analysis_dir / "tunnels.csv"


def _parse_residue_map(path: Path) -> dict[str, tuple[str, ...]]:
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
                residues[current_cluster].append(resname[:3])
                residues[f"{current_cluster}:{current_cluster}"].append(resname[:3])
        return {key: tuple(value) for key, value in residues.items()}
    residues = defaultdict(list)
    for row in _read_csv_records(path):
        tid = _tunnel_id(row)
        resname = (_field(row, "Residue", "Residue name", "AA", "resname") or "").upper()
        match = re.search(r"\b([A-Z]{3})\b", resname)
        if match:
            residues[tid].append(match.group(1))
    return {key: tuple(value) for key, value in residues.items()}


def _parse_profile_map(path: Path) -> dict[str, tuple[CaverProfilePoint, ...]]:
    if not path.exists():
        return {}
    profiles: dict[str, list[CaverProfilePoint]] = defaultdict(list)
    for row in _read_csv_records(path):
        tid = _tunnel_id(row)
        distance = _float_field(row, "Distance", "Distance from origin", "distance_from_origin")
        radius = _float_field(row, "R", "Radius", default=0.0)
        x_raw = _field(row, "X", "x")
        y_raw = _field(row, "Y", "y")
        z_raw = _field(row, "Z", "z")
        coord = None
        if x_raw is not None and y_raw is not None and z_raw is not None:
            try:
                coord = (float(x_raw), float(y_raw), float(z_raw))
            except ValueError:
                coord = None
        profiles[tid].append(CaverProfilePoint(distance=distance, radius=radius, coord=coord))
    for key in profiles:
        profiles[key].sort(key=lambda point: point.distance)
    return {key: tuple(value) for key, value in profiles.items()}


def parse_caver_tunnels(analysis_dir: Path) -> tuple[CaverTunnel, ...]:
    """Parse persisted CAVER output tables from one analysis directory.

    Parameters
    ----------
    analysis_dir
        Directory containing CAVER tunnel characteristics plus optional profile
        and residue-lining tables.

    Returns
    -------
    tuple[CaverTunnel, ...]
        Parsed tunnels with profile points and lining residues attached where
        available.

    Raises
    ------
    ValueError
        If CAVER output contains ambiguous tunnel identifiers.
    """

    profile_map = _parse_profile_map(analysis_dir / "tunnel_profiles.csv")
    residue_path = analysis_dir / "residues.csv"
    if not residue_path.exists():
        residue_path = analysis_dir / "residues.txt"
    residue_map = _parse_residue_map(residue_path)
    tunnels: list[CaverTunnel] = []
    for row in _read_csv_records(_tunnels_table_path(analysis_dir)):
        tunnel_id = _tunnel_id(row)
        tunnels.append(
            CaverTunnel(
                tunnel_id=tunnel_id,
                cluster=_field(row, "Tunnel cluster", "Cluster") or "",
                starting_point_index=_int_field(
                    row,
                    "Starting point",
                    "Starting point index",
                    "Starting point ID",
                    "Start point",
                    default=0,
                ),
                length=_float_field(row, "Length"),
                throughput=_float_field(row, "Throughput"),
                bottleneck_radius=_float_field(row, "Bottleneck radius", "Bottleneck"),
                avg_radius=_float_field(row, "Avg R", "Average radius", "Average R"),
                curvature=_float_field(row, "Curvature", default=1.0),
                profile_points=profile_map.get(tunnel_id, tuple()),
                lining_residues=residue_map.get(tunnel_id, tuple()),
            )
        )
    tunnel_ids = [tunnel.tunnel_id for tunnel in tunnels]
    if len(set(tunnel_ids)) != len(tunnel_ids):
        duplicates = sorted({tunnel_id for tunnel_id in tunnel_ids if tunnel_ids.count(tunnel_id) > 1})
        raise ValueError(
            f"ambiguous tunnel identifiers in {analysis_dir}: {duplicates[:5]}"
        )
    return tuple(tunnels)


def select_primary_tunnel(tunnels: Iterable[CaverTunnel]) -> CaverTunnel | None:
    """Select the highest-priority tunnel for pocket-level descriptors.

    Parameters
    ----------
    tunnels
        Candidate CAVER tunnels associated with a pocket start point.

    Returns
    -------
    CaverTunnel | None
        Tunnel with maximal throughput, then length, then average radius; or
        ``None`` when no tunnels are available.
    """

    tunnel_list = list(tunnels)
    if not tunnel_list:
        return None
    return max(tunnel_list, key=lambda tunnel: (tunnel.throughput, tunnel.length, tunnel.avg_radius))


def derive_caver_t12_features(
    tunnels: Iterable[CaverTunnel],
    pocket_axial_length: float,
    pocket_principal_axis: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Derive Tier 1-2 CAVER features for one matched fpocket pocket.

    Parameters
    ----------
    tunnels
        CAVER tunnels mapped to the pocket's start point.
    pocket_axial_length
        Axial length of the matched fpocket pocket, used for length
        normalization.
    pocket_principal_axis
        Optional principal axis vector used to measure tunnel alignment.

    Returns
    -------
    dict[str, float]
        Complete canonical CAVER feature map.
    """

    tunnel_list = list(tunnels)
    if not tunnel_list:
        return safe_caver_t12_defaults()

    primary = select_primary_tunnel(tunnel_list)
    if primary is None:
        return safe_caver_t12_defaults()

    radii = np.asarray([point.radius for point in primary.profile_points], dtype=float)
    lengths = np.asarray([tunnel.length for tunnel in tunnel_list], dtype=float)
    bottlenecks = np.asarray([tunnel.bottleneck_radius for tunnel in tunnel_list], dtype=float)
    curvature_values = _curvature_series(primary.profile_points)
    out = {
        "caver_tunnel_count": len(tunnel_list),
        "caver_primary_length": float(primary.length),
        "caver_total_length": float(np.sum(lengths)) if lengths.size else 0.0,
        "caver_primary_bottleneck_radius": float(primary.bottleneck_radius),
        "caver_median_bottleneck_radius": float(np.median(bottlenecks)) if bottlenecks.size else 0.0,
        "caver_primary_mean_radius": float(np.mean(radii)) if radii.size else float(primary.avg_radius),
        "caver_primary_radius_std": float(np.std(radii)) if radii.size else 0.0,
        "caver_primary_radius_min": float(np.min(radii)) if radii.size else float(primary.bottleneck_radius),
        "caver_primary_radius_max": float(np.max(radii)) if radii.size else float(primary.avg_radius),
        "caver_primary_radius_skewness": _radius_skewness(radii),
        "caver_primary_straightness": _straightness(primary.profile_points, primary.length),
        "caver_primary_mean_curvature": float(np.mean(curvature_values)) if curvature_values.size else 0.0,
        "caver_primary_max_curvature": float(np.max(curvature_values)) if curvature_values.size else 0.0,
        "caver_primary_high_curvature_count": int(np.sum(curvature_values >= _HIGH_CURVATURE_THRESHOLD)) if curvature_values.size else 0,
        "caver_primary_alignment_angle_deg": _alignment_angle_deg(primary.profile_points, pocket_principal_axis),
        "caver_primary_bottleneck_count": _bottleneck_count(radii),
        "caver_primary_length_over_axial": _length_over_axial(primary.length, pocket_axial_length),
    }
    return cast_caver_t12_features(out)


def group_tunnels_by_starting_point(tunnels: Iterable[CaverTunnel]) -> dict[int, tuple[CaverTunnel, ...]]:
    """Group parsed tunnels by CAVER starting-point index.

    Parameters
    ----------
    tunnels
        Parsed CAVER tunnels.

    Returns
    -------
    dict[int, tuple[CaverTunnel, ...]]
        Mapping from starting-point index to all tunnels emitted from that
        point.
    """

    grouped: dict[int, list[CaverTunnel]] = defaultdict(list)
    for tunnel in tunnels:
        grouped[tunnel.starting_point_index].append(tunnel)
    return {key: tuple(value) for key, value in grouped.items()}


def derive_caver_t12_features_by_pocket(
    analysis_dir: Path,
    pocket_contexts: Iterable[CaverPocketContext],
) -> dict[int, dict[str, float]]:
    """Derive CAVER features for all matched pockets in one analysis directory.

    Parameters
    ----------
    analysis_dir
        Persisted CAVER output directory for a structure.
    pocket_contexts
        Matched fpocket pocket metadata used to align CAVER starting points
        back to pocket numbers.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from ``matched_pocket_number`` to canonical CAVER feature map.
    """

    context_list = list(pocket_contexts)
    tunnels = parse_caver_tunnels(analysis_dir)
    grouped = group_tunnels_by_starting_point(tunnels)
    offset = _starting_point_offset(grouped, context_list)
    return {
        context.matched_pocket_number: derive_caver_t12_features(
            grouped.get(context.starting_point_index + offset, tuple()),
            pocket_axial_length=context.pocket_axial_length,
            pocket_principal_axis=context.pocket_principal_axis,
        )
        for context in context_list
    }


def _starting_point_offset(
    grouped: dict[int, tuple[CaverTunnel, ...]],
    pocket_contexts: Iterable[CaverPocketContext],
) -> int:
    context_list = list(pocket_contexts)
    zero_based_hits = sum(len(grouped.get(context.starting_point_index, tuple())) for context in context_list)
    one_based_hits = sum(
        len(grouped.get(context.starting_point_index + 1, tuple())) for context in context_list
    )
    return 0 if zero_based_hits >= one_based_hits else 1


def _straightness(profile_points: tuple[CaverProfilePoint, ...], path_length: float) -> float:
    coords = [point.coord for point in profile_points if point.coord is not None]
    if len(coords) < 2 or path_length <= _EPS:
        return 1.0
    start = np.asarray(coords[0], dtype=float)
    end = np.asarray(coords[-1], dtype=float)
    return float(np.clip(np.linalg.norm(end - start) / max(path_length, _EPS), 0.0, 1.0))


def _curvature_series(profile_points: tuple[CaverProfilePoint, ...]) -> np.ndarray:
    coords = [point.coord for point in profile_points if point.coord is not None]
    if len(coords) < 3:
        return np.asarray([], dtype=float)
    values: list[float] = []
    coord_array = [np.asarray(coord, dtype=float) for coord in coords]
    for idx in range(1, len(coord_array) - 1):
        first = coord_array[idx] - coord_array[idx - 1]
        second = coord_array[idx + 1] - coord_array[idx]
        denom = np.linalg.norm(first) * np.linalg.norm(second) * np.linalg.norm(first + second)
        if denom <= _EPS:
            values.append(0.0)
            continue
        cross = np.linalg.norm(np.cross(first, second))
        values.append(float((2.0 * cross) / denom))
    return np.asarray(values, dtype=float)


def _alignment_angle_deg(
    profile_points: tuple[CaverProfilePoint, ...],
    pocket_principal_axis: tuple[float, float, float] | None,
) -> float:
    if pocket_principal_axis is None:
        return _NEUTRAL_ALIGNMENT_ANGLE_DEG
    coords = [point.coord for point in profile_points if point.coord is not None]
    if len(coords) < 2:
        return _NEUTRAL_ALIGNMENT_ANGLE_DEG
    axis = np.asarray(pocket_principal_axis, dtype=float)
    tunnel_axis = np.asarray(coords[-1], dtype=float) - np.asarray(coords[0], dtype=float)
    axis_norm = np.linalg.norm(axis)
    tunnel_norm = np.linalg.norm(tunnel_axis)
    if axis_norm <= _EPS or tunnel_norm <= _EPS:
        return _NEUTRAL_ALIGNMENT_ANGLE_DEG
    cosine = float(np.dot(axis, tunnel_axis) / (axis_norm * tunnel_norm))
    cosine = float(np.clip(abs(cosine), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _bottleneck_count(radii: np.ndarray) -> int:
    if radii.size < 3:
        return 0
    count = 0
    for idx in range(1, radii.size - 1):
        center = radii[idx]
        if center < radii[idx - 1] and center <= radii[idx + 1]:
            count += 1
    return count


def _length_over_axial(length: float, pocket_axial_length: float) -> float:
    axial_length = float(pocket_axial_length)
    if abs(axial_length) <= _EPS:
        return 0.0
    ratio = float(length) / axial_length
    if not math.isfinite(ratio):
        return 0.0
    return float(np.clip(ratio, 0.0, 20.0))


def _radius_skewness(radii: np.ndarray) -> float:
    if radii.size < 3:
        return 0.0
    centered = radii - float(np.mean(radii))
    std = float(np.std(radii))
    if std <= _EPS:
        return 0.0
    return float(np.mean((centered / std) ** 3))
