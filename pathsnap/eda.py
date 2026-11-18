"""Reproducible PathSnap raw-data inventory and diagnostic statistics.

Usage:
    python -m pathsnap.eda --dataset ".../Student folder" --output evidence
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import EXTENSIONS, PARTITIONS, Trip, iter_trip_directories, load_trip


def weak_component_count(n_nodes: int, arcs: np.ndarray) -> int:
    """Weakly connected components (edge direction ignored), union-find in numpy/python."""
    parent = np.arange(n_nodes)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in arcs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return int(len({find(x) for x in range(n_nodes)}))


def summarise_trip(trip: Trip) -> dict:
    """Describe data quality without treating .route as GPS-point labels."""
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    time_steps = np.diff(track[:, 2])
    # A weakly connected component ignores direction for the EDA graph census.
    components = weak_component_count(len(nodes), arcs)
    incident = np.bincount(arcs.ravel(), minlength=len(nodes))
    route_breaks = 0
    if trip.route is not None and len(trip.route) > 1:
        selected_arcs = arcs[trip.route]
        route_breaks = int(np.count_nonzero(selected_arcs[:-1, 1] != selected_arcs[1:, 0]))
    result = {
        "partition": trip.partition,
        "trip_id": trip.trip_id,
        "gps_observations": len(track),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "reference_arc_ids": len(trip.route) if trip.route is not None else None,
        "timestamp_steps_1s": int(np.count_nonzero(time_steps == 1)),
        "irregular_timestamp_steps": int(np.count_nonzero(time_steps != 1)),
        "gaps_over_60s": int(np.count_nonzero(time_steps > 60)),
        "gaps_over_600s": int(np.count_nonzero(time_steps > 600)),
        "largest_time_gap_s": float(time_steps.max()) if len(time_steps) else 0.0,
        "consecutive_duplicate_gps": int(np.count_nonzero(np.all(np.diff(track[:, :2], axis=0) == 0, axis=1))),
        "longitude_min": float(track[:, 0].min()),
        "longitude_max": float(track[:, 0].max()),
        "latitude_min": float(track[:, 1].min()),
        "latitude_max": float(track[:, 1].max()),
        "weak_road_components": int(components),
        "isolated_road_nodes": int(np.count_nonzero(incident == 0)),
        "reference_arc_endpoint_breaks": route_breaks,
    }
    for extension in EXTENSIONS:
        result[f"{extension}_bytes"] = (trip.path / f"{trip.trip_id}.{extension}").stat().st_size
    return result


def inspect_dataset(dataset_root: Path, output_dir: Path) -> tuple[list[dict], dict]:
    """Inspect every available trip and save verifiable CSV/JSON evidence."""
    rows = []
    for partition in PARTITIONS:
        for path in iter_trip_directories(dataset_root, partition):
            trip = load_trip(dataset_root, partition, path.name, include_reference=True)
            row = summarise_trip(trip)
            rows.append(row)
            print(
                f"[{partition}] {path.name}: {row['gps_observations']} GPS, "
                f"{row['road_arcs']} arcs, {row['weak_road_components']} components",
                flush=True,
            )
    if not rows:
        raise RuntimeError("No trips found")

    groups = {}
    for partition in PARTITIONS:
        subset = [row for row in rows if row["partition"] == partition]
        groups[partition] = {
            "trips": len(subset),
            "gps_observations": sum(row["gps_observations"] for row in subset),
            "road_nodes": sum(row["road_nodes"] for row in subset),
            "road_arcs": sum(row["road_arcs"] for row in subset),
            "reference_arc_ids": sum(row["reference_arc_ids"] for row in subset),
            "irregular_timestamp_steps": sum(row["irregular_timestamp_steps"] for row in subset),
            "gaps_over_60s": sum(row["gaps_over_60s"] for row in subset),
            "max_timestamp_gap_s": max((row["largest_time_gap_s"] for row in subset), default=0),
            "reference_arc_endpoint_breaks": sum(row["reference_arc_endpoint_breaks"] for row in subset),
        }
    report = {
        "partitions": groups,
        "notes": [
            "Arc IDs and node IDs are zero-based row indices, local to each trip.",
            "Reference .route is a chronological arc sequence, NOT labels aligned one-to-one with .track.",
            "Weak road components ignore edge direction; reference arc endpoint breaks are diagnostic only.",
            "Evaluator-testing trips are not present in the supplied student dataset.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trip_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing train/ and student_test/")
    parser.add_argument("--output", type=Path, default=Path("evidence"))
    arguments = parser.parse_args()
    _, summary = inspect_dataset(arguments.dataset, arguments.output)
    print(json.dumps(summary["partitions"], indent=2))
    print(f"Saved {arguments.output / 'trip_stats.csv'} and dataset_summary.json")


if __name__ == "__main__":
    main()
