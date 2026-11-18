"""Run Phase 1 inventory, validation, EDA and candidate-feature extraction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap_phase1 import RoadNetwork, discover_trips, extract_candidate_features, trip_eda_row, validate_trip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data") / "Student folder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase1")
    parser.add_argument("--partitions", nargs="*", default=["train", "student_test"])
    parser.add_argument("--trip-ids", nargs="*", help="Optional exact trip IDs for a quick subset run.")
    parser.add_argument("--feature-trips", nargs="*", help="Trip IDs to featurise; omit to featurise every loaded trip.")
    parser.add_argument("--radius-m", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trips = discover_trips(args.data_root, args.partitions, set(args.trip_ids) if args.trip_ids else None)
    validations = [validate_trip(trip) for trip in trips]
    rows = [trip_eda_row(trip, check) for trip, check in zip(trips, validations)]
    import pandas as pd
    eda = pd.DataFrame(rows)
    eda.to_csv(args.output_dir / "trip_inventory_and_eda.csv", index=False)
    with open(args.output_dir / "validation.json", "w", encoding="utf-8") as stream:
        json.dump(validations, stream, indent=2)
    selected = trips if args.feature_trips is None else [trip for trip in trips if trip.trip_id in set(args.feature_trips)]
    if not selected:
        raise ValueError(f"None of --feature-trips found: {args.feature_trips}")
    all_features, diagnostics = [], []
    for trip in selected:
        features, info = extract_candidate_features(RoadNetwork.from_trip(trip), args.radius_m, args.max_candidates)
        all_features.append(features)
        diagnostics.append(info)
    pd.concat(all_features, ignore_index=True).to_csv(args.output_dir / "candidate_features.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "candidate_diagnostics.csv", index=False)
    print(f"trips={len(trips)} valid={sum(item['ok'] for item in validations)} invalid={sum(not item['ok'] for item in validations)}")
    print(f"inventory={args.output_dir / 'trip_inventory_and_eda.csv'}")
    for item in diagnostics:
        print("feature_trip={trip_id} observations={observations} candidate_rows={candidate_rows} "
              "labels={labelled_observations} miss_rate={candidate_miss_rate:.4f} "
              "alignment_median={reference_alignment_distance_median:.2f} alignment_max={reference_alignment_distance_max:.2f}".format(**item))


if __name__ == "__main__":
    main()
