"""Score every candidate row with a saved HGB model, retaining road arc IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def score(model_path: Path, candidates_path: Path, output_path: Path) -> int:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["feature_names"]
    identifiers = ["trip_id", "observation_index", "candidate_arc_id"]
    if not hasattr(model, "predict_proba") or len(set(features)) != len(features):
        raise ValueError("Invalid model bundle")
    frame = pd.read_csv(candidates_path, usecols=identifiers + features,
                        dtype={"trip_id": str, "observation_index": "int32",
                               "candidate_arc_id": "int32", **{name: "float32" for name in features}})
    if frame.empty:
        raise ValueError("Candidate CSV is empty")
    X = frame[features].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Candidate CSV contains non-finite model inputs")
    probabilities = model.predict_proba(X)[:, 1]
    result = frame[identifiers].copy()
    result["probability"] = probabilities
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Scored {score(args.model, args.candidates, args.output)} candidates")


if __name__ == "__main__":
    main()
