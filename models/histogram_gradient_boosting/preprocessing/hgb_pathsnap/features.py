"""Road candidate generation and GPS/road feature extraction.

IMPORTANT: This is an *inference-compatible* feature pipeline. It deliberately
does not open .route files or invent observation-level target labels.

Usage:
    python -m hgb_pathsnap.features --dataset ".../Student folder" \
        --partition train --trip 00000001 --output evidence/features_00000001.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .data import PARTITIONS, Trip, load_trip


EARTH_RADIUS_M = 6_371_008.8
FIELDS = (
    "trip_id", "observation_index", "timestamp_s", "candidate_arc_id",
    "candidate_rank", "candidate_count", "distance_to_road_m",
    "distance_to_start_m", "distance_to_end_m", "segment_length_m",
    "projected_fraction", "heading_difference_deg", "gps_heading_deg",
    "road_bearing_deg", "gps_step_distance_m", "gps_speed_mps",
    "timestamp_delta_s", "has_reliable_heading", "large_time_gap",
    "speed_was_clipped", "outside_search_radius",
    "connects_from_previous_candidates",
)


def project_local_meters(lonlat: np.ndarray, origin_lonlat: np.ndarray) -> np.ndarray:
    """Local tangent-plane approximation in metres, centred on each trip.

    Appropriate as a fast candidate-search approximation for city-scale trips;
    no single UTM zone/latitude scale is imposed across all geographical trips.
    For unusually wide trips, replace this with a proper local geodesic projection.
    """
    lonlat = np.asarray(lonlat, dtype=np.float64)
    reference = np.asarray(origin_lonlat, dtype=np.float64)
    diffs = np.deg2rad(lonlat - reference)
    return EARTH_RADIUS_M * diffs * np.array([np.cos(np.deg2rad(reference[1])), 1.0])


def _bearing_deg(vectors: np.ndarray) -> np.ndarray:
    """Compass bearing, clockwise from north, for (east, north) vectors."""
    return (np.degrees(np.arctan2(vectors[..., 0], vectors[..., 1])) + 360.0) % 360.0


def _angle_difference_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs((first - second + 180.0) % 360.0 - 180.0)


def _segment_projection(point: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """Find exact distance to each supplied finite segment, not its infinite line."""
    vectors = ends - starts
    length_sq = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.clip(
        np.einsum("ij,ij->i", point - starts, vectors) / np.maximum(length_sq, 1e-12),
        0.0, 1.0,
    )
    snapped_points = starts + fractions[:, None] * vectors
    distance = np.linalg.norm(point - snapped_points, axis=1)
    return distance, fractions


def generate_candidate_features(
    trip: Trip,
    *,
    top_k: int = 8,
    radius_m: float = 150.0,
    max_observations: int | None = None,
    batch_size: int = 512,
) -> list[dict]:
    """Generate ranked candidate-road features without using reference routes.

    A KD-tree of arc endpoints and midpoints provides a fast *shortlist*; exact
    point-to-segment distance then ranks the shortlisted arcs. Long segments
    whose sampled points are far away can be missed; report/validate candidate
    recall before using the outputs for training or final evaluation.
    """
    if trip.route is not None:
        raise ValueError("Pass an inference-only Trip (include_reference=False) to avoid leakage")
    if top_k < 1 or radius_m <= 0 or batch_size < 1:
        raise ValueError("top_k, radius_m, and batch_size must be positive")
    if max_observations is not None and max_observations < 1:
        raise ValueError("max_observations must be positive")

    observations = trip.track
    if max_observations is not None:
        observations = observations[:max_observations]

    # Use one coordinate system per trip rather than mixing raw angular units.
    origin = np.median(trip.track[:, :2], axis=0)
    xy = project_local_meters(observations[:, :2], origin)
    nodes_xy = project_local_meters(trip.nodes, origin)
    starts = nodes_xy[trip.arcs[:, 0]]
    ends = nodes_xy[trip.arcs[:, 1]]
    arc_vectors = ends - starts
    lengths = np.linalg.norm(arc_vectors, axis=1)
    road_bearings = _bearing_deg(arc_vectors)

    # Spatial index uses samples only to shortlist arcs; distances are exact
    # point-to-segment distances within the local plane after shortlisting.
    index_points = np.stack((starts, (starts + ends) / 2.0, ends), axis=1).reshape(-1, 2)
    arc_lookup = np.repeat(np.arange(len(trip.arcs), dtype=np.int64), 3)
    tree = cKDTree(index_points)
    probes = min(len(index_points), max(96, 16 * top_k))

    changes = np.zeros_like(xy)
    changes[1:] = np.diff(xy, axis=0)
    step_distances = np.linalg.norm(changes, axis=1)
    dt = np.zeros(len(observations))
    dt[1:] = np.diff(observations[:, 2])
    unbounded_speed = np.divide(
        step_distances, dt, out=np.zeros_like(dt), where=dt > 0
    )
    # Store finite numeric features, preserving an explicit outlier indicator.
    speeds = np.minimum(unbounded_speed, 80.0)
    heading = _bearing_deg(changes)
    reliable_heading = (step_distances >= 3.0) & (dt > 0) & (dt <= 60.0)
    gap_flag = dt > 60.0

    records: list[dict] = []
    previous_to_nodes: set[int] = set()
    for batch_start in range(0, len(observations), batch_size):
        batch_end = min(batch_start + batch_size, len(observations))
        _, sampled_indices = tree.query(xy[batch_start:batch_end], k=probes, workers=1)
        sampled_indices = np.asarray(sampled_indices).reshape(batch_end - batch_start, -1)

        for local_index, sample_row in enumerate(sampled_indices):
            index = batch_start + local_index
            point = xy[index]
            candidate_ids = np.unique(arc_lookup[sample_row])
            distances, fractions = _segment_projection(
                point, starts[candidate_ids], ends[candidate_ids]
            )
            nearest_order = np.lexsort((candidate_ids, distances))
            in_range = nearest_order[distances[nearest_order] <= radius_m]
            # Robust fallback: always produce at least one candidate when all
            # nearby roads fall outside the radius (e.g. severe GPS noise).
            nearest = in_range[:top_k] if len(in_range) else nearest_order[:1]
            count = len(nearest)
            for rank, candidate_offset in enumerate(nearest, start=1):
                arc_id = int(candidate_ids[candidate_offset])
                d = float(distances[candidate_offset])
                records.append({
                    "trip_id": trip.trip_id,
                    "observation_index": index,
                    "timestamp_s": float(observations[index, 2]),
                    "candidate_arc_id": arc_id,
                    "candidate_rank": rank,
                    "candidate_count": count,
                    "distance_to_road_m": round(d, 4),
                    "distance_to_start_m": round(float(np.linalg.norm(point - starts[arc_id])), 4),
                    "distance_to_end_m": round(float(np.linalg.norm(point - ends[arc_id])), 4),
                    "segment_length_m": round(float(lengths[arc_id]), 4),
                    "projected_fraction": round(float(fractions[candidate_offset]), 6),
                    "heading_difference_deg": round(float(_angle_difference_deg(heading[index], road_bearings[arc_id])), 4)
                    if reliable_heading[index] else 0.0,
                    "gps_heading_deg": round(float(heading[index]), 4) if reliable_heading[index] else 0.0,
                    "road_bearing_deg": round(float(road_bearings[arc_id]), 4),
                    "gps_step_distance_m": round(float(step_distances[index]), 4),
                    "gps_speed_mps": round(float(speeds[index]), 4),
                    "timestamp_delta_s": float(dt[index]),
                    "has_reliable_heading": int(reliable_heading[index]),
                    "large_time_gap": int(gap_flag[index]),
                    "speed_was_clipped": int(unbounded_speed[index] > 80.0),
                    "outside_search_radius": int(d > radius_m),
                    "connects_from_previous_candidates": int(trip.arcs[arc_id, 0] in previous_to_nodes),
                })
            previous_to_nodes = set(int(node) for node in trip.arcs[candidate_ids[nearest], 1])
    return records


def save_features(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--trip", required=True, help="Eight-digit trip ID, e.g. 00000001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--radius-m", type=float, default=150.0)
    parser.add_argument("--max-observations", type=int, default=None)
    arguments = parser.parse_args()
    trip = load_trip(arguments.dataset, arguments.partition, arguments.trip, include_reference=False)
    records = generate_candidate_features(
        trip,
        top_k=arguments.top_k,
        radius_m=arguments.radius_m,
        max_observations=arguments.max_observations,
    )
    save_features(records, arguments.output)
    n_observations = len({row["observation_index"] for row in records})
    print(
        f"Generated {len(records)} candidate feature rows for {n_observations} GPS observations; "
        f"{sum(row['outside_search_radius'] for row in records)} outside-radius fallback rows"
    )
    print(f"Saved: {arguments.output}")
    print("NOTE: These are unlabeled candidate features; .route cannot be joined by row index.")


if __name__ == "__main__":
    main()
