"""Compare five Phase 1 candidate classifiers on generated feature CSVs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from candidate_models import DistanceBaseline, GaussianNB, KNN, LogisticGD, Preprocessor, RandomStumpForest, fit_and_score

FEATURES = ["distance_to_arc_m", "projection_fraction", "arc_length_m", "arc_bearing_deg", "heading_difference_deg", "gps_step_m", "time_delta_s", "speed_mps", "connects_from_previous_candidates"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", type=Path, required=True); p.add_argument("--student-test", type=Path, required=True); p.add_argument("--output", type=Path, default=Path("outputs") / "model_comparison.csv"); args = p.parse_args()
    train, test = pd.read_csv(args.train), pd.read_csv(args.student_test)
    if train.label.sum() == 0 or test.label.sum() == 0: raise ValueError("No positive candidate labels; inspect candidate generation or alignment.")
    prep = Preprocessor().fit(train[FEATURES].to_numpy(float)); x_train, x_test = prep.transform(train[FEATURES].to_numpy(float)), prep.transform(test[FEATURES].to_numpy(float))
    y_train, y_test = train.label.to_numpy(int), test.label.to_numpy(int)
    groups = (test.trip_id.astype(str) + ":" + test.observation_id.astype(str)).to_numpy()
    models = [DistanceBaseline(), GaussianNB(), LogisticGD(), KNN(), RandomStumpForest()]
    results = [fit_and_score(m, x_train, y_train, x_test, y_test, groups) for m in models]
    args.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(results).to_csv(args.output, index=False)
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__": main()
