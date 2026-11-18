"""Approximate observation labels from ordered reference routes, separate from features.

The route supplies arc IDs in travel order, not GPS point labels. Each GPS point
is provisionally assigned its closest route arc. This geometric proxy ignores
route chronology and can be wrong at crossings, parallel roads, and repeated
arcs. Use the distance and candidate coverage diagnostics before model fitting.
"""

from __future__ import annotations

import numpy as np

from .data import Trip
from .features import _segment_projection, project_local_meters


def align_reference(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    if trip.route is None:
        raise ValueError("Reference route was not loaded")
    origin = np.median(trip.track[:, :2], axis=0)
    points = project_local_meters(trip.track[:, :2], origin)
    nodes = project_local_meters(trip.nodes, origin)
    route_arcs = np.unique(trip.route)
    starts = nodes[trip.arcs[route_arcs, 0]]
    ends = nodes[trip.arcs[route_arcs, 1]]
    labels = np.empty(len(points), dtype=np.int64)
    distances = np.empty(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        d, _ = _segment_projection(point, starts, ends)
        nearest = int(np.argmin(d))
        labels[index] = route_arcs[nearest]
        distances[index] = d[nearest]
    return labels, distances


def label_candidates(records: list[dict], trip: Trip, *, max_alignment_m: float = 30.0) -> tuple[list[dict], dict]:
    """Attach proxy labels and diagnostics; never pass these columns to a model."""
    if max_alignment_m <= 0:
        raise ValueError("max_alignment_m must be positive")
    labels, distances = align_reference(trip)
    covered = set()
    eligible = set()
    for row in records:
        index = row["observation_index"]
        valid = bool(distances[index] <= max_alignment_m)
        row["label_valid"] = int(valid)
        row["label"] = int(valid and row["candidate_arc_id"] == labels[index])
        if valid:
            eligible.add(index)
            if row["label"]:
                covered.add(index)
    diagnostics = {
        "partition": trip.partition,
        "trip_id": trip.trip_id,
        "observations": len(trip.track),
        "candidate_rows": len(records),
        "eligible_observations": len(eligible),
        "covered_eligible_observations": len(covered),
        "candidate_miss_rate": 1 - len(covered) / len(eligible) if eligible else None,
        "alignment_distance_median": float(np.median(distances)),
        "alignment_distance_max": float(np.max(distances)),
        "alignment_method": "nearest_route_arc_geometry_proxy",
    }
    return records, diagnostics
