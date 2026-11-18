# Random Forest: candidate classifier (team dataset)

Trains on the team's shared `candidate_features.csv` produced by `scripts/run_phase1.py` (80 m radius, at most 8 candidates, labels = closest reference-route arc).

## Run (from the repo root)
```
python scripts/run_phase1.py --data-root "data/Student folder" --output-dir outputs/full
python models/random_forest/train_rf.py --features outputs/full/candidate_features.csv --diagnostics outputs/full/candidate_diagnostics.csv --data-root "data/Student folder" --output-dir outputs/random_forest
```
About 3 minutes for the features and about 6 minutes for the RF (5 CV folds + final fit). Add `--base-features-only` to use only the 8 team schema columns.

## Features
- **Team schema (8):** distance_to_arc_m, projection_fraction, arc_length_m, heading_difference_deg, gps_step_m, time_delta_s, speed_mps, connects_from_previous_candidates.
- **Derived per observation (5)**, computed only from those columns: candidate_rank, n_candidates, distance_minus_min, heading_minus_min, heading_missing.
- **Excluded:** `reference_arc_id` and `reference_alignment_distance_m` are derived from `.route`, so using them would leak the label. Also excluded: identifiers, timestamp, and absolute bearings (arc_bearing_deg, gps_heading_deg), which describe a direction on one particular map and don't generalise to other trips.
- Missing heading on the first point (NaN) → 90° plus a `heading_missing` flag. NaN or infinite speeds → 0.

## Protocol
- Training trips 00000033 and 00000054 are excluded: their median GPS-to-route alignment distance is above 50 m (122 m and 139 m), so their labels are unreliable.
- 5-fold GroupKFold by trip_id, seed 42. Student-test trips are used only for evaluation.
- Top-1 accuracy is measured over **all** 19,160 student-test observations; points with no correct candidate count as wrong.
- Isotonic calibration is fitted on out-of-fold training probabilities only.

## Hyperparameters (`config.json`)
150 trees, max_depth 20, min_samples_leaf 10, max_features sqrt, max_samples 0.25, class_weight balanced_subsample, Gini criterion.

## Results (student-test, 10 trips)
| Metric | Value |
|---|---|
| CV macro-F1 | 0.863 ± 0.023 |
| CV train macro-F1 | 0.873 |
| Student-test macro-F1 | 0.866 |
| Top-1 accuracy | **0.809** (nearest-road baseline 0.564) |
| Candidate accuracy | 0.927 |
| Precision / Recall | 0.673 / 0.918 |
| Log loss (raw → calibrated) | 0.176 → 0.129 |
| Route continuity | 0.901 |
| Candidate miss rate | 0.0014 |
| Train / inference time | 62 s / 0.71 s |

Outputs in `outputs/random_forest/`: summary.json, summary_row.csv (team comparison format), cv_folds.csv, student_test_predictions.csv (trip_id, observation_id, arc_id, candidate_rank, prob, prob_calibrated; input for the HMM), random_forest.joblib, isotonic_calibrator.joblib, plots/.
