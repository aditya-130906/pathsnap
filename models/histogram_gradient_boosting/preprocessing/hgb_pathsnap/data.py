"""Strict, documented readers for the four PathSnap trip file types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PARTITIONS = ("train", "student_test")
EXTENSIONS = ("track", "nodes", "arcs", "route")


@dataclass(frozen=True)
class Trip:
    trip_id: str
    partition: str
    path: Path
    # .track: longitude, latitude, time (seconds)
    track: np.ndarray
    # .nodes: longitude, latitude; node ID is zero-based row index
    nodes: np.ndarray
    # .arcs: from_node_id, to_node_id; arc ID is zero-based row index
    arcs: np.ndarray
    # .route: chronological reference arc IDs (NOT per-GPS-point labels)
    route: np.ndarray | None = None


def iter_trip_directories(dataset_root: Path, partition: str):
    """Yield trip directories in numeric order, keeping their 8-digit IDs."""
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition {partition!r}; expected {PARTITIONS}")
    directory = Path(dataset_root) / partition
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing partition directory: {directory}")
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_dir():
            if not path.name.isdigit():
                raise ValueError(f"Unexpected nonnumeric trip directory: {path}")
            yield path


def _read_table(path: Path, columns: int, dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    # ndmin=2 preserves the number of columns even for a single-row file.
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns; found {values.shape[1]}")
    if not len(values):
        raise ValueError(f"{path}: file must not be empty")
    return values


def load_trip(
    dataset_root: Path,
    partition: str,
    trip_id: str,
    *,
    include_reference: bool = False,
) -> Trip:
    """Load one trip. Reference routes are excluded from inference by default."""
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition {partition!r}")
    trip_id = str(trip_id)
    if not trip_id.isdigit():
        raise ValueError("trip_id must consist of digits")
    trip_dir = Path(dataset_root) / partition / trip_id
    track = _read_table(trip_dir / f"{trip_id}.track", 3, np.float64)
    nodes = _read_table(trip_dir / f"{trip_id}.nodes", 2, np.float64)
    arcs = _read_table(trip_dir / f"{trip_id}.arcs", 2, np.int64)
    route = None
    if include_reference:
        route = _read_table(trip_dir / f"{trip_id}.route", 1, np.int64)[:, 0]

    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"{trip_dir}: GPS or node coordinates contain NaN/Inf")
    if not np.all((track[:, 0] >= -180) & (track[:, 0] <= 180)):
        raise ValueError(f"{trip_dir}: invalid GPS longitude")
    if not np.all((track[:, 1] >= -90) & (track[:, 1] <= 90)):
        raise ValueError(f"{trip_dir}: invalid GPS latitude")
    if not np.all((nodes[:, 0] >= -180) & (nodes[:, 0] <= 180)):
        raise ValueError(f"{trip_dir}: invalid node longitude")
    if not np.all((nodes[:, 1] >= -90) & (nodes[:, 1] <= 90)):
        raise ValueError(f"{trip_dir}: invalid node latitude")
    if np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError(f"{trip_dir}: timestamps must strictly increase")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"{trip_dir}: arc endpoints refer to nonexistent nodes")
    if route is not None and (np.any(route < 0) or np.any(route >= len(arcs))):
        raise ValueError(f"{trip_dir}: reference route contains nonexistent arc IDs")

    return Trip(trip_id, partition, trip_dir, track, nodes, arcs, route)
