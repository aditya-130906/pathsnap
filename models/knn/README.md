# K-Nearest Neighbours: Akshat Shaurya (PES2UG24CS047)

Instance-based candidate classifier on the team's Phase 1 candidate tables. For each (GPS observation, candidate arc), it finds the *k* most similar labelled training candidates in scaled feature space and returns the distance-weighted share that were true arcs. The highest-probability candidate per observation is the classifier-only prediction; the probabilities are the emission scores for the HMM/Viterbi stage.

## Reproduce

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --output-dir outputs/train_all --radius-m 80 --max-candidates 8
python scripts/run_phase1.py --data-root "data/Student folder" --partitions student_test --output-dir outputs/student_test_all --radius-m 80 --max-candidates 8
python models/knn/train_knn.py `
  --train-features outputs/train_all/candidate_features.csv `
  --train-diagnostics outputs/train_all/candidate_diagnostics.csv `
  --student-test-features outputs/student_test_all/candidate_features.csv `
  --student-test-diagnostics outputs/student_test_all/candidate_diagnostics.csv `
  --data-root "data/Student folder" --output-dir outputs/knn
```
Runtime: about 13 minutes on 2 CPU cores (k sweep about 4 min, 5 CV folds about 5 min, test inference about 1.5 min). Add `--skip-k-sweep` to save about 4 minutes.

## Features (15)

- **Phase 1 schema (11, the same inputs as Gaussian NB):** `distance_to_arc_m`, `projection_fraction`, `arc_length_m`, `candidate_rank`, `relative_distance_to_nearest`, `heading_difference_deg`, `gps_step_m`, `time_delta_s`, `speed_mps`, `large_gap`, `connects_from_previous_candidates`.
- **Derived per observation (4),** computed only from those columns: `n_candidates`, `distance_minus_min`, `heading_minus_min`, `heading_missing`.
- **Excluded:**
  - Route-derived columns (`reference_arc_id`, `reference_route_position`, `reference_alignment_distance_m`), which would leak the label.
  - Identifiers and timestamps.
  - Absolute bearings (`arc_bearing_deg`, `gps_heading_deg`), which describe a direction on one particular map and don't generalise across trips.
- **Missing values:** missing heading on the first point becomes 90° plus a `heading_missing` flag. Remaining non-finite values are filled by median imputation.

## Model

- **Why KNN:** it's non-parametric and makes no assumption about the shape of the decision boundary. It tests whether true arcs form local neighbourhoods in feature space (small distance, aligned heading, low rank). Distance-weighted votes give a candidate probability directly.
- **Preprocessing (a scikit-learn `Pipeline`, fitted on each fold's training trips only):**
  - median imputation;
  - `arcsinh` on 7 heavy-tailed features (distances, lengths, step, time gap, speed), so a few extreme values don't dominate the Euclidean distance;
  - `StandardScaler` on all features.
  
  **Scaling is essential for KNN**, because every feature contributes to the distance on equal terms.
- **Classifier:** `KNeighborsClassifier(n_neighbors=50, weights="distance", metric="euclidean")`.
- **Objective:** none to optimise. KNN is a lazy learner: "training" only stores the data, and prediction is a weighted majority vote (no loss curve exists).
- **Class imbalance (15.3% positives):** no reweighting. The ranking used for Top-1 is unaffected by the class prior.
- **Computational cost:** prediction is O(N_train) per query. To keep it tractable, each fit stores a random sample of **200,000** training candidates (seed 42), and CV validation is scored on **60,000** sampled rows per fold. The final model is evaluated on all 126,881 student-test candidates.

## Choosing k

The k sweep on fold 1 (`k_sweep.csv`, `plots/k_sweep.png`):

| k | Validation macro-F1 | Validation log loss |
| --- | --- | --- |
| 5 | 0.8480 | 0.492 |
| 15 | 0.8548 | 0.295 |
| 50 | 0.8557 | 0.219 |
| 150 | 0.8562 | 0.192 |

Small k gives noisy, overconfident probabilities (high log loss). Macro-F1 plateaus from k ≈ 50. k = 50 was chosen as the balance between accuracy, probability quality and inference cost; k = 150 is about 3× slower to predict for +0.0005 macro-F1.

## Protocol

- Training trips `00000033` and `00000054` are excluded, because their median GPS-to-route alignment distance is above 50 m (the same rule as Random Forest). That leaves 78 training trips and 1,183,524 candidates.
- 5-fold GroupKFold by `trip_id`, seed 42. Student-test trips are used only for evaluation.
- Top-1 accuracy is measured over **all** 19,160 student-test observations; points without a correct candidate count as wrong.
- Isotonic calibration is fitted on out-of-fold training probabilities only.

## Results (student_test, 10 trips)

| Metric | Value |
| --- | --- |
| CV Macro F1 (mean ± std) | 0.8691 ± 0.0138 |
| Student-test Macro F1 | 0.8820 |
| **Top-1 accuracy** | **0.8078**, vs. 0.5554 nearest-road baseline |
| Candidate accuracy | 0.9406 |
| Precision / Recall (true-arc class, threshold 0.5) | 0.8148 / 0.7835 |
| Log loss, raw → calibrated | 0.1692 → 0.1503 |
| Brier score, raw → calibrated | 0.0438 → 0.0436 |
| Route continuity (classifier only) | 0.8984 |
| Candidate miss rate | 0.0038 |
| Training / inference time | 0.47 s / 83.4 s |

Per-fold validation:

| Fold | Macro F1 | Precision | Recall | Log loss |
| --- | --- | --- | --- | --- |
| 1 | 0.8588 | 0.7701 | 0.7545 | 0.206 |
| 2 | 0.8502 | 0.7626 | 0.7388 | 0.206 |
| 3 | 0.8816 | 0.8330 | 0.7644 | 0.170 |
| 4 | 0.8776 | 0.8179 | 0.7601 | 0.167 |
| 5 | 0.8771 | 0.8142 | 0.7666 | 0.166 |

## Analysis

- **Train macro-F1 is 1.000, and this is a property of distance weighting, not evidence of overfitting.** With `weights="distance"`, a training point is its own nearest neighbour at distance 0 and receives infinite weight, so training-set predictions are trivially perfect. Generalisation should be judged from validation and test: CV 0.869 ± 0.014 and test 0.882 are consistent, and the five folds are stable (std 0.014).
- **Balanced errors.** Precision (0.81) and recall (0.78) are similar, unlike Random Forest (balanced class weights → high recall) and Gaussian NB (high recall, 0.52 precision).
- **Probability quality.** Log loss 0.169 is far better than Gaussian NB (0.510), because the vote averages 50 neighbours. Isotonic calibration improves it further to 0.150.
- **Main limitation: inference cost.** 83 s for 126,881 candidates, against well under 3 s for the tree models, because every query is compared with the stored training sample. Storing all 1.18 M rows would be about 6× slower. KNN is the slowest model to deploy in the API.
- **Other limitations:** the Euclidean distance treats all scaled features as equally important (no feature selection), and performance depends on the training subsample (fixed seed for reproducibility).

## Files in `outputs/knn/`

`summary.json`, `summary_row.csv` (team comparison format), `cv_folds.csv`, `k_sweep.csv`, `knn.joblib` (pipeline and feature list), `isotonic_calibrator.joblib`, `student_test_predictions.csv` (trip_id, observation_id, arc_id, candidate_rank, prob, prob_calibrated, label; input for the HMM), and `plots/` (confusion_matrix, cv_folds, calibration, k_sweep).
