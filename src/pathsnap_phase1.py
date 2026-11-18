"""Phase 1 data pipeline for PathSnap.

The raw route is a sequence of traversed arc ids, not a per-observation label.
This module therefore aligns each GPS observation to the closest *route arc* in
metres.  That assumption is isolated in ``align_reference_arcs`` and every
alignment distance is retained for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
import math
from typing import Iterable

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray


def _read_whitespace(path: Path, names: list[str], dtype: dict | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(trip_dir: str | Path, partition: str | None = None) -> Trip:
    """Load one four-file trip and preserve arc id as the original row index."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    partition = partition or trip_dir.parent.name
    needed = {suffix: trip_dir / f"{trip_id}.{suffix}" for suffix in ("track", "nodes", "arcs", "route")}
    missing = [str(path) for path in needed.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{trip_id}: missing required files: {missing}")
    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=_read_whitespace(needed["track"], ["lon", "lat", "timestamp"]),
        nodes=_read_whitespace(needed["nodes"], ["lon", "lat"]),
        arcs=_read_whitespace(needed["arcs"], ["from_node", "to_node"], {"from_node": "int64", "to_node": "int64"}),
        route=_read_whitespace(needed["route"], ["arc_id"], {"arc_id": "int64"})["arc_id"].to_numpy(),
    )


def discover_trips(data_root: str | Path, partitions: Iterable[str] = ("train", "student_test"), trip_ids: set[str] | None = None) -> list[Trip]:
    root = Path(data_root)
    trips: list[Trip] = []
    for partition in partitions:
        directory = root / partition
        if not directory.is_dir():
            raise FileNotFoundError(f"Partition directory not found: {directory}")
        trips.extend(load_trip(item, partition) for item in sorted(directory.iterdir()) if item.is_dir() and (trip_ids is None or item.name in trip_ids))
    return trips


def validate_trip(trip: Trip) -> dict:
    """Return validation facts and warnings without mutating raw inputs."""
    errors: list[str] = []
    warnings: list[str] = []
    for name, frame, columns in (("track", trip.track, 3), ("nodes", trip.nodes, 2), ("arcs", trip.arcs, 2)):
        if frame.shape[1] != columns:
            errors.append(f"{name} has {frame.shape[1]} columns; expected {columns}")
        if frame.isna().any().any():
            errors.append(f"{name} contains missing values")
    if trip.route.size == 0:
        errors.append("route is empty")
    if not trip.track["lon"].between(-180, 180).all() or not trip.track["lat"].between(-90, 90).all():
        errors.append("track has invalid longitude/latitude")
    if not trip.nodes["lon"].between(-180, 180).all() or not trip.nodes["lat"].between(-90, 90).all():
        errors.append("nodes has invalid longitude/latitude")
    if not ((trip.arcs >= 0).all().all() and (trip.arcs < len(trip.nodes)).all().all()):
        errors.append("arc endpoint refers to a non-existent node")
    invalid_route = int(((trip.route < 0) | (trip.route >= len(trip.arcs))).sum())
    if invalid_route:
        errors.append(f"route has {invalid_route} arc ids outside [0, {len(trip.arcs) - 1}]")
    dt = trip.track["timestamp"].diff().iloc[1:]
    if (dt <= 0).any():
        errors.append("timestamps are not strictly increasing")
    if (dt > 10).any():
        warnings.append(f"{int((dt > 10).sum())} timestamp gap(s) exceed 10 seconds")
    if len(trip.route) == len(trip.track):
        warnings.append("route and track have equal lengths; this still does not prove row alignment")
    return {"trip_id": trip.trip_id, "partition": trip.partition, "ok": not errors, "errors": errors, "warnings": warnings}


class LocalProjection:
    """Local equirectangular projection in metres, appropriate for one city-sized trip."""
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        self.cos_lat0 = math.cos(math.radians(lat0))

    @classmethod
    def from_trip(cls, trip: Trip) -> "LocalProjection":
        return cls(float(trip.nodes.lon.mean()), float(trip.nodes.lat.mean()))

    def project(self, lon: np.ndarray | pd.Series, lat: np.ndarray | pd.Series) -> np.ndarray:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = EARTH_RADIUS_M * np.radians(lon - self.lon0) * self.cos_lat0
        y = EARTH_RADIUS_M * np.radians(lat - self.lat0)
        return np.column_stack((x, y))


@dataclass
class RoadNetwork:
    trip: Trip
    projection: LocalProjection
    start_xy: np.ndarray
    end_xy: np.ndarray
    length_m: np.ndarray
    bearing_deg: np.ndarray
    grid: dict[tuple[int, int], list[int]]
    cell_size_m: float

    @classmethod
    def from_trip(cls, trip: Trip, cell_size_m: float = 100.0) -> "RoadNetwork":
        projection = LocalProjection.from_trip(trip)
        node_xy = projection.project(trip.nodes.lon, trip.nodes.lat)
        endpoints = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
        start_xy, end_xy = node_xy[endpoints[:, 0]], node_xy[endpoints[:, 1]]
        delta = end_xy - start_xy
        length_m = np.linalg.norm(delta, axis=1)
        bearing_deg = (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360) % 360
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for arc_id, (start, end) in enumerate(zip(start_xy, end_xy)):
            min_cell = np.floor(np.minimum(start, end) / cell_size_m).astype(int)
            max_cell = np.floor(np.maximum(start, end) / cell_size_m).astype(int)
            for x in range(min_cell[0], max_cell[0] + 1):
                for y in range(min_cell[1], max_cell[1] + 1):
                    grid[(x, y)].append(arc_id)
        return cls(trip, projection, start_xy, end_xy, length_m, bearing_deg, dict(grid), cell_size_m)

    def candidate_arc_ids(self, point_xy: np.ndarray, radius_m: float = 80.0, max_candidates: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve nearby arcs with a uniform-grid spatial index, sorted by segment distance."""
        cell = np.floor(point_xy / self.cell_size_m).astype(int)
        span = int(math.ceil(radius_m / self.cell_size_m))
        ids: set[int] = set()
        for x in range(cell[0] - span, cell[0] + span + 1):
            for y in range(cell[1] - span, cell[1] + span + 1):
                ids.update(self.grid.get((x, y), ()))
        if not ids:
            return np.array([], dtype=np.int64), np.array([], dtype=float), np.array([], dtype=float)
        arc_ids = np.fromiter(ids, dtype=np.int64)
        distances, fractions = segment_distances(point_xy, self.start_xy[arc_ids], self.end_xy[arc_ids])
        keep = np.argsort(distances)[:max_candidates]
        return arc_ids[keep], distances[keep], fractions[keep]


def segment_distances(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = ends - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(np.einsum("ij,ij->i", point - starts, vectors), denom, out=np.zeros_like(denom), where=denom > 0)
    fractions = np.clip(fractions, 0.0, 1.0)
    projected = starts + fractions[:, None] * vectors
    return np.linalg.norm(point - projected, axis=1), fractions


def _movement_features(track_xy: np.ndarray, timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step = np.zeros(len(track_xy))
    heading = np.full(len(track_xy), np.nan)
    if len(track_xy) > 1:
        delta = np.diff(track_xy, axis=0)
        step[1:] = np.linalg.norm(delta, axis=1)
        heading[1:] = (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360) % 360
    dt = np.diff(timestamp, prepend=np.nan)
    speed = np.divide(step, dt, out=np.full(len(step), np.nan), where=np.isfinite(dt) & (dt > 0))
    return step, dt, speed, heading


def align_reference_arcs(network: RoadNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Align observations to closest arc from the reference-route sequence only.

    This is an explicit modelling assumption because the provided files contain
    no per-observation alignment. It must be replaced if organisers provide one.
    """
    allowed = np.unique(network.trip.route)
    starts, ends = network.start_xy[allowed], network.end_xy[allowed]
    points = network.projection.project(network.trip.track.lon, network.trip.track.lat)
    labels = np.empty(len(points), dtype=np.int64)
    distances = np.empty(len(points), dtype=float)
    # Vectorised over route arcs and looped over points to control memory use.
    for i, point in enumerate(points):
        d, _ = segment_distances(point, starts, ends)
        best = int(np.argmin(d))
        labels[i], distances[i] = allowed[best], d[best]
    return labels, distances


def extract_candidate_features(network: RoadNetwork, radius_m: float = 80.0, max_candidates: int = 8) -> tuple[pd.DataFrame, dict]:
    """Build candidate-level features and binary labels for one labelled trip."""
    track = network.trip.track
    points = network.projection.project(track.lon, track.lat)
    step, dt, speed, heading = _movement_features(points, track.timestamp.to_numpy(dtype=float))
    reference_arc, alignment_distance = align_reference_arcs(network)
    rows: list[dict] = []
    previous_ids: np.ndarray | None = None
    for obs_id, point in enumerate(points):
        arc_ids, distances, fractions = network.candidate_arc_ids(point, radius_m, max_candidates)
        previous_to_nodes = set()
        if previous_ids is not None and len(previous_ids):
            previous_to_nodes = set(network.trip.arcs.iloc[previous_ids].to_node.tolist())
        for arc_id, distance, fraction in zip(arc_ids, distances, fractions):
            road_heading = network.bearing_deg[arc_id]
            hdiff = np.nan if not np.isfinite(heading[obs_id]) else abs((road_heading - heading[obs_id] + 180) % 360 - 180)
            rows.append({
                "trip_id": network.trip.trip_id, "partition": network.trip.partition, "observation_id": obs_id,
                "timestamp": track.timestamp.iat[obs_id], "arc_id": int(arc_id), "label": int(arc_id == reference_arc[obs_id]),
                "distance_to_arc_m": float(distance), "projection_fraction": float(fraction), "arc_length_m": float(network.length_m[arc_id]),
                "arc_bearing_deg": float(road_heading), "gps_heading_deg": float(heading[obs_id]), "heading_difference_deg": float(hdiff),
                "gps_step_m": float(step[obs_id]), "time_delta_s": float(dt[obs_id]), "speed_mps": float(speed[obs_id]),
                "connects_from_previous_candidates": int(network.trip.arcs.from_node.iat[arc_id] in previous_to_nodes),
                "reference_arc_id": int(reference_arc[obs_id]), "reference_alignment_distance_m": float(alignment_distance[obs_id]),
            })
        previous_ids = arc_ids
    candidates = pd.DataFrame(rows)
    observations_with_candidates = candidates.observation_id.nunique() if not candidates.empty else 0
    labels_found = int(candidates.label.sum()) if not candidates.empty else 0
    diagnostics = {
        "trip_id": network.trip.trip_id, "observations": len(track), "observations_with_candidates": observations_with_candidates,
        "candidate_rows": len(candidates), "labelled_observations": labels_found,
        "candidate_miss_rate": 1 - labels_found / len(track),
        "reference_alignment_distance_median": float(np.median(alignment_distance)),
        "reference_alignment_distance_max": float(np.max(alignment_distance)),
    }
    return candidates, diagnostics


def trip_eda_row(trip: Trip, validation: dict) -> dict:
    network = RoadNetwork.from_trip(trip)
    degrees = np.bincount(trip.arcs.to_numpy().ravel(), minlength=len(trip.nodes))
    dt = trip.track.timestamp.diff().iloc[1:]
    components = weak_component_count(len(trip.nodes), trip.arcs.to_numpy(dtype=np.int64))
    return {
        "trip_id": trip.trip_id, "partition": trip.partition, "valid": validation["ok"],
        "track_rows": len(trip.track), "node_rows": len(trip.nodes), "arc_rows": len(trip.arcs), "route_rows": len(trip.route),
        "missing_values": int(trip.track.isna().sum().sum() + trip.nodes.isna().sum().sum() + trip.arcs.isna().sum().sum()),
        "non_increasing_timestamps": int((dt <= 0).sum()), "gaps_over_10_seconds": int((dt > 10).sum()),
        "median_dt_s": float(dt.median()), "max_dt_s": float(dt.max()),
        "track_lon_min": float(trip.track.lon.min()), "track_lon_max": float(trip.track.lon.max()),
        "track_lat_min": float(trip.track.lat.min()), "track_lat_max": float(trip.track.lat.max()),
        "isolated_nodes": int((degrees == 0).sum()), "weak_components": components,
        "zero_length_arcs": int((network.length_m == 0).sum()),
        "validation_errors": " | ".join(validation["errors"]), "validation_warnings": " | ".join(validation["warnings"]),
    }


def weak_component_count(n_nodes: int, arcs: np.ndarray) -> int:
    """Count weak components without an extra graph dependency."""
    parent = np.arange(n_nodes)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for left, right in arcs:
        left_root, right_root = find(int(left)), find(int(right))
        if left_root != right_root:
            parent[left_root] = right_root
    return len({find(i) for i in range(n_nodes)})
