"""Build reproducible candidate and label CSVs for the train and student_test partitions.

Usage (repository root):
    python -m pathsnap.build --dataset "data/Student folder" --output outputs/features
    python -m pathsnap.build --dataset "data/Student folder" --output outputs/features_smoke --smoke 4

Outputs (all untracked by Git)
------------------------------
<partition>_candidates.csv          ID columns + FEATURE_COLUMNS, one row per (observation, candidate arc).
                                    Built WITHOUT .route; contains no label information.
<partition>_observation_labels.csv  approximate per-observation reference arc from labels.align_route
                                    (trip_id, observation_index, reference_arc_id, route_position,
                                    alignment_distance_m, label_valid). Training/evaluation only.
label_quality.csv                   per-trip alignment quality and candidate miss rate.
build_config.json                   settings, feature list, trips, row counts, timings.

Only the `train` and `student_test` partitions are read; evaluator trips are never part of this build.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .candidates import CandidateConfig, FEATURE_COLUMNS, ID_COLUMNS, generate_candidates
from .data import PARTITIONS, find_dataset_root, list_trip_ids, load_trip
from .labels import LabelConfig, align_route, alignment_quality

LABEL_COLUMNS = ["trip_id", "observation_index", "reference_arc_id", "route_position",
                 "alignment_distance_m", "label_valid"]
INT_FEATURES = {"candidate_rank", "candidate_count", "outside_radius", "heading_reliable", "large_time_gap",
                "speed_clipped", "prev_candidate_link", "next_candidate_link", "from_node_in_degree",
                "to_node_out_degree", "has_reverse_arc"}


def process_trip(args) -> tuple[dict, dict, dict]:
    dataset_root, partition, trip_id, cand_cfg, label_cfg = args
    t0 = time.time()
    # candidates: route deliberately NOT loaded
    trip = load_trip(dataset_root, partition, trip_id, include_reference=False)
    cand = generate_candidates(trip, cand_cfg)
    # labels: separate load with the route
    trip_ref = load_trip(dataset_root, partition, trip_id, include_reference=True)
    labels = align_route(trip_ref, label_cfg)
    quality = alignment_quality(trip_ref, labels, cand["observation_index"], cand["candidate_arc_id"])
    quality.update(partition=partition, candidate_rows=int(len(cand["observation_index"])),
                   seconds=round(time.time() - t0, 2))
    return cand, labels, quality


def _write_csv(path: Path, columns: list[str], blocks: list[dict], int_cols: set[str]) -> int:
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for block in blocks:
            cols = []
            for c in columns:
                v = block[c]
                if c == "trip_id":
                    cols.append(v.tolist())
                elif c in int_cols or np.issubdtype(np.asarray(v).dtype, np.integer):
                    cols.append(np.asarray(v).astype(np.int64).tolist())
                else:
                    cols.append(np.round(np.asarray(v, dtype=np.float64), 6).tolist())
            writer.writerows(zip(*cols))
            rows += len(cols[0])
    return rows


def build(dataset: Path, output: Path, partitions=PARTITIONS, trips_per_partition: int | None = None,
          trip_ids: list[str] | None = None, jobs: int = 1, cand_cfg=CandidateConfig(), label_cfg=LabelConfig()):
    dataset = find_dataset_root(Path(dataset))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary = {"dataset_root": str(dataset), "candidate_config": cand_cfg.to_dict(),
               "label_config": label_cfg.to_dict(), "id_columns": ID_COLUMNS, "feature_columns": FEATURE_COLUMNS,
               "label_columns": LABEL_COLUMNS, "partitions": {}}
    quality_rows = []
    for partition in partitions:
        if partition not in PARTITIONS:
            raise ValueError(f"only {PARTITIONS} may be built; got {partition!r}")
        ids = list_trip_ids(dataset, partition)
        if trip_ids:
            ids = [t for t in ids if t in set(trip_ids)]
        if trips_per_partition:
            ids = ids[:trips_per_partition]
        tasks = [(dataset, partition, t, cand_cfg, label_cfg) for t in ids]
        t0 = time.time()
        if jobs > 1:
            with Pool(jobs) as pool:
                results = pool.map(process_trip, tasks)
        else:
            results = [process_trip(task) for task in tasks]
        cand_rows = _write_csv(output / f"{partition}_candidates.csv", ID_COLUMNS + FEATURE_COLUMNS,
                               [r[0] for r in results], INT_FEATURES)
        label_rows = _write_csv(output / f"{partition}_observation_labels.csv", LABEL_COLUMNS,
                                [r[1] for r in results], {"label_valid"})
        q = [r[2] for r in results]
        quality_rows.extend(q)
        valid = sum(x["valid_observations"] for x in q)
        missed = sum(x["candidate_miss_rate"] * x["valid_observations"] for x in q if x["valid_observations"])
        summary["partitions"][partition] = {
            "trips": ids, "n_trips": len(ids), "candidate_rows": cand_rows, "observations": label_rows,
            "valid_observations": valid, "label_valid_share": valid / max(label_rows, 1),
            "candidate_miss_rate": missed / max(valid, 1), "seconds": round(time.time() - t0, 1)}
        print(f"[{partition}] trips={len(ids)} observations={label_rows} candidates={cand_rows} "
              f"valid={valid / max(label_rows, 1):.3f} miss_rate={missed / max(valid, 1):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    with (output / "label_quality.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(quality_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(quality_rows)
    summary["seconds_total"] = round(time.time() - started, 1)
    summary["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (output / "build_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="folder containing (or above) train/ and student_test/")
    p.add_argument("--output", type=Path, default=Path("outputs") / "features")
    p.add_argument("--partitions", nargs="*", default=list(PARTITIONS))
    p.add_argument("--smoke", type=int, default=None, help="only the first N trips of each partition")
    p.add_argument("--trips", nargs="*", default=None, help="exact zero-padded trip IDs")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--radius-m", type=float, default=CandidateConfig.radius_m)
    p.add_argument("--max-candidates", type=int, default=CandidateConfig.max_candidates)
    a = p.parse_args()
    cfg = CandidateConfig(radius_m=a.radius_m, max_candidates=a.max_candidates)
    build(a.dataset, a.output, a.partitions, a.smoke, a.trips, a.jobs, cfg)


if __name__ == "__main__":
    main()
