"""Build a shared candidate CSV for one permitted partition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hgb_pathsnap.data import PARTITIONS, iter_trip_directories, load_trip
from hgb_pathsnap.features import FIELDS, generate_candidate_features
from hgb_pathsnap.labels import label_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trip", help="Optional single trip ID for smoke testing")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--radius-m", type=float, default=150.0)
    parser.add_argument("--max-alignment-m", type=float, default=30.0)
    parser.add_argument("--unlabeled", action="store_true", help="Do not read reference routes")
    args = parser.parse_args()
    trip_ids = [args.trip] if args.trip else [p.name for p in iter_trip_directories(args.dataset, args.partition)]
    if not trip_ids:
        raise ValueError("No trips found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    fieldnames = list(FIELDS) + ([] if args.unlabeled else ["label_valid", "label"])
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for trip_id in trip_ids:
            trip = load_trip(args.dataset, args.partition, trip_id, include_reference=False)
            records = generate_candidate_features(trip, top_k=args.top_k, radius_m=args.radius_m)
            if not args.unlabeled:
                reference_trip = load_trip(args.dataset, args.partition, trip_id, include_reference=True)
                records, diagnostic = label_candidates(records, reference_trip, max_alignment_m=args.max_alignment_m)
                diagnostics.append(diagnostic)
            writer.writerows(records)
            print(f"{args.partition}/{trip_id}: {len(trip.track)} observations, {len(records)} candidates", flush=True)
    if diagnostics:
        path = args.output.with_suffix(".diagnostics.json")
        path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
