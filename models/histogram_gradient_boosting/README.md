# Histogram Gradient Boosting (HGB)

- **Author:** Aayaan Kamil, SRN PES2UG24CS017
- **Algorithm:** `sklearn.ensemble.HistGradientBoostingClassifier` (binary: is this candidate arc the reference arc?)
- **Results:** `outputs/histogram_gradient_boosting/`

> **Pipeline note.** This HGB run was trained on candidate CSVs built by the HGB-run preprocessing in
> `preprocessing/hgb_pathsnap/` (top-8 nearest arcs within 150 m, 18 features, nearest-route-arc proxy labels,
> observations > 30 m from the route dropped). It is a **third** candidate/label pipeline, different from the
> root `pathsnap/` package and the Phase 1 pipeline (see the root README). Its numbers are **not directly
> comparable** with models trained on those pipelines. The shared pipelines were not modified; the preprocessing
> is vendored under the distinct package name `hgb_pathsnap` so it cannot clash with `pathsnap/`.

## Features (18, float32, no scaler or imputer)

`candidate_rank, candidate_count, distance_to_road_m, distance_to_start_m, distance_to_end_m, segment_length_m,
projected_fraction, heading_difference_deg, gps_heading_deg, road_bearing_deg, gps_step_distance_m, gps_speed_mps,
timestamp_delta_s, has_reliable_heading, large_time_gap, speed_was_clipped, outside_search_radius,
connects_from_previous_candidates`

Excluded: `trip_id, observation_index, timestamp_s, candidate_arc_id, label_valid, label, reference_arc_id,
reference_alignment_distance_m` (identifiers and label-derived fields). Ordered list: `feature_schema.json`.

## Training methodology

- Only `label_valid == 1` candidates are used. Labels are **geometric proxies**: `.route` has no per-GPS-point
  alignment, so each observation is labelled with the nearest route arc; observations > 30 m from any route arc are
  excluded (25,380 train and 142 student-test observations). Crossings and parallel roads can make labels wrong.
- 80 train trips, 5-fold `GroupKFold` by `trip_id`; final model refit on all 80 trips; the 10 `student_test` trips
  are held-out evaluation only. No evaluator trips were used.
- 183,037 eligible train observations / 1,325,791 candidate rows; 19,018 / 135,930 for student test.
- Early stopping disabled (no internal row-level split).

## Hyperparameters (`config.json`)

`learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, max_bins=255,
early_stopping=False, random_state=42` (loss `log_loss`). Environment: scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6.

## Actual results

| Metric | CV (out of fold) | Student test |
| --- | ---: | ---: |
| Candidate macro F1 | 0.8633 (fold mean 0.8634 ± 0.0275) | 0.8798 |
| Observation Top-1 accuracy* | 0.7759 | 0.8076 |
| Positive precision | 0.7753 | 0.7976 |
| Positive recall | 0.7522 | 0.7884 |
| Candidate miss rate | 0.4824% | 0.2208% |
| Log loss | 0.1499 | 0.1312 |

Final fit: 10.81 s. Student-test inference: 0.26 s. Per-fold fit/inference times: `outputs/histogram_gradient_boosting/cv_folds.csv`.
Student-test confusion matrix (rows = actual 0/1, columns = predicted 0/1): `[[113157, 3797], [4015, 14961]]`.

\*Top-1 is over proxy-label-eligible observations only (observations > 30 m from the route are excluded), unlike
the Random Forest figure which counts all student-test observations. These are proxy-label metrics, not
independently verified route accuracy; the 95% Top-1 target was **not** reached. `smoke` runs were a small
subset test and are not committed.

## Files

| Path | Content |
| --- | --- |
| `hgb_model.joblib` | dict bundle: `model` (HistGradientBoostingClassifier), `feature_names`, `config`, `label_method` |
| `config.json`, `feature_schema.json` | configuration with source-CSV SHA-256 hashes; ordered feature schema |
| `train_hgb.py`, `predict_hgb.py` | training/evaluation and inference scripts |
| `preprocessing/` | feature/label builder used for this run (`build_features.py`, `run_eda.py`, `hgb_pathsnap/`) |
| `outputs/histogram_gradient_boosting/` | `metrics.json`, `summary_row.csv`, `cv_folds.csv`, `training.log`, `plots/`, probability CSVs, `diagnostics/` |

## Reproduce training

```powershell
pip install -r models/histogram_gradient_boosting/requirements.txt scipy
python models/histogram_gradient_boosting/preprocessing/build_features.py --dataset "data/Student folder" --partition train --output work/full_train.csv
python models/histogram_gradient_boosting/preprocessing/build_features.py --dataset "data/Student folder" --partition student_test --output work/full_student_test.csv
python models/histogram_gradient_boosting/train_hgb.py --train work/full_train.csv --student-test work/full_student_test.csv
# add --smoke for a quick check (10 train trips, 2 test trips, 2 folds, 10 iterations)
```

Check `python models/histogram_gradient_boosting/preprocessing/build_features.py --help` for options. The raw dataset and
the large candidate CSVs are not committed; the hashes of the CSVs used in the original run are in `config.json`.
(`config.json` records the original local paths of those CSVs.) `train_hgb.py` writes to
`outputs/histogram_gradient_boosting/` by default.

## Load the trained model

```python
import joblib
bundle = joblib.load("models/histogram_gradient_boosting/hgb_model.joblib")
features = bundle["feature_names"]            # ordered; must match feature_schema.json
p = bundle["model"].predict_proba(candidate_frame[features].to_numpy(dtype="float32"))[:, 1]
```

Score a candidate CSV (keeps trip, observation and arc IDs; ignores labels):

```powershell
python models/histogram_gradient_boosting/predict_hgb.py --model models/histogram_gradient_boosting/hgb_model.joblib --candidates PATH_TO_CANDIDATES.csv --output predictions.csv
```

`student_test_all_candidate_probabilities.csv` scores all 137,056 student-test candidate rows (for the HMM decoder);
`student_test_candidate_probabilities.csv` only the proxy-label-eligible rows. Neither contains labels. The model
was pickled with scikit-learn 1.9.1; load it with a compatible version.
