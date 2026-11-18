"""Run Phase 1 inventory, validation, EDA and candidate-feature extraction."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap_phase1 import RoadNetwork, extract_candidate_features, load_trip, trip_eda_row, validate_trip


def _process_trip(trip_dir: str, partition: str, make_features: bool, radius_m: float, max_candidates: int):
    """Process one complete trip in a separate process when requested."""
    trip = load_trip(trip_dir, partition)
    validation = validate_trip(trip)
    eda = trip_eda_row(trip, validation)
    if not make_features:
        return validation, eda, None, None
    features, diagnostics = extract_candidate_features(RoadNetwork.from_trip(trip), radius_m, max_candidates)
    return validation, eda, features, diagnostics


def _process_spec(spec):
    return _process_trip(*spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data") / "Student folder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase1")
    parser.add_argument("--partitions", nargs="*", default=["train", "student_test"])
    parser.add_argument("--trip-ids", nargs="*", help="Optional exact trip IDs for a quick subset run.")
    parser.add_argument("--feature-trips", nargs="*", help="Trip IDs to featurise; omit to featurise every loaded trip.")
    parser.add_argument("--radius-m", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1, help="Independent trips to process concurrently; use 3 on a typical laptop.")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = set(args.feature_trips) if args.feature_trips else None
    requested_ids = set(args.trip_ids) if args.trip_ids else None
    specs = []
    for partition in args.partitions:
        directory = args.data_root / partition
        if not directory.is_dir():
            raise FileNotFoundError(f"Partition directory not found: {directory}")
        for trip_dir in sorted(item for item in directory.iterdir() if item.is_dir()):
            if requested_ids is None or trip_dir.name in requested_ids:
                specs.append((str(trip_dir), partition, selected_ids is None or trip_dir.name in selected_ids, args.radius_m, args.max_candidates))
    if not specs:
        raise ValueError("No trips matched the requested partitions and IDs")
    if args.workers == 1:
        results = [_process_trip(*spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_process_spec, specs))
    validations = [result[0] for result in results]
    rows = [result[1] for result in results]
    import pandas as pd
    eda = pd.DataFrame(rows)
    eda.to_csv(args.output_dir / "trip_inventory_and_eda.csv", index=False)
    with open(args.output_dir / "validation.json", "w", encoding="utf-8") as stream:
        json.dump(validations, stream, indent=2)
    all_features = [result[2] for result in results if result[2] is not None]
    diagnostics = [result[3] for result in results if result[3] is not None]
    if not all_features:
        raise ValueError(f"None of --feature-trips found: {args.feature_trips}")
    pd.concat(all_features, ignore_index=True).to_csv(args.output_dir / "candidate_features.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "candidate_diagnostics.csv", index=False)
    print(f"trips={len(specs)} valid={sum(item['ok'] for item in validations)} invalid={sum(not item['ok'] for item in validations)} workers={args.workers}")
    print(f"inventory={args.output_dir / 'trip_inventory_and_eda.csv'}")
    for item in diagnostics:
        print("feature_trip={trip_id} observations={observations} candidate_rows={candidate_rows} "
              "labels={labelled_observations} miss_rate={candidate_miss_rate:.4f} "
              "alignment_median={reference_alignment_distance_median:.2f} alignment_max={reference_alignment_distance_max:.2f}".format(**item))


if __name__ == "__main__":
    main()
