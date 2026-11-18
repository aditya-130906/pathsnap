# PathSnap: Distributed Map-Matching and Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon.

PathSnap matches noisy GPS observations to the road network and reconstructs the route a vehicle actually drove. A classifier scores the candidate road arcs near each GPS point, and an HMM with Viterbi decoding turns those scores into a connected route. The system runs on two laptops: a FastAPI model server and an evaluator UI.

## Team

| Member | SRN | Model | Other responsibilities |
| --- | --- | --- | --- |
| Akshansh Agrawal | PES2UG24CS045 | Logistic Regression | Metrics, HMM and Viterbi decoder |
| Aayaan Kamil | PES2UG24CS017 | Gradient Boosting | EDA, Folium maps |
| Akshat Shaurya | PES2UG24CS047 | Random Forest | Labels, candidates, features, Bayesian tuning |
| Aditya Rao | PES2UG24CS034 | Gaussian Naive Bayes | FastAPI backend, UI, deployment |

A fifth model (Decision Tree) is planned to meet the five-model minimum; owner to be confirmed.

## Dataset

Each trip folder holds four headerless, whitespace-separated files:

| File | Columns | Meaning |
| --- | --- | --- |
| `.track` | longitude, latitude, timestamp (s) | GPS observations, nominally 1 Hz |
| `.nodes` | longitude, latitude | Road nodes; row index = node ID |
| `.arcs` | from_node_id, to_node_id | Directed road segments; row index = arc ID |
| `.route` | arc_id | Chronological reference arcs driven. Not one label per GPS point |

Partitions: 80 `train` trips (fitting and cross-validation), 10 `student_test` trips (model comparison), 10 evaluator trips (final demo only, never used in development). The supplied data is not committed; place it under `data/Student folder/`.

## Approach

1. **Validation and cleaning:** check shapes, types and ID ranges; sort by time; remove invalid coordinates and duplicate timestamps; flag large time gaps and impossible jumps.
2. **Projection:** each trip is projected to metres with a local projection around its centroid, so all distances are physical. No Euclidean distance on raw degrees.
3. **Reference labels:** `.route` is aligned to the GPS sequence to give each observation a reference arc. The alignment distance is kept so label quality can be inspected, and unreliable labels are excluded. The route is used only for labels and scoring, never as a model feature.
4. **Candidate generation:** for each GPS point, the road arcs within a search radius, nearest first, capped per point. Built without the route so it runs on unseen trips. Candidate miss rate is recorded.
5. **Features:** per (GPS point, candidate arc): distance and distance rank, heading difference to the arc bearing, movement (time gap, step, speed), and arc geometry.
6. **Candidate classifiers:** binary label (is this the reference arc?). The highest-probability candidate per observation is the classifier-only prediction.
7. **Bayesian optimisation:** Optuna (TPE) on the selected model with group-aware cross-validation.
8. **HMM and Viterbi:** classifier probabilities as emissions, road-network connectivity as transitions, to produce a connected route. Compared against classifier-only predictions.
9. **Deployment:** FastAPI on Laptop 1 (model, decoder, road network, server-side scoring); UI on Laptop 2 (trip selection and route display).

## Models

| Model | Family | Owner | Why it is in the comparison |
| --- | --- | --- | --- |
| Logistic Regression | Linear | Akshansh | Calibrated probabilities for the HMM; interpretable, fast baseline |
| Gaussian Naive Bayes | Probabilistic | Aditya | Fastest probabilistic baseline; tests the feature-independence assumption |
| Random Forest | Ensemble (bagging) | Akshat | Nonlinear feature interactions with lower variance than one tree |
| Gradient Boosting | Ensemble (boosting) | Aayaan | Usually strongest on tabular features; optimises log loss |
| Decision Tree | Tree | TBC | Explainable nonlinear rules; shows the overfitting that ensembles fix |

All models use the same candidates, features, labels, folds and random seed (42), so the comparison is fair.

## Validation and metrics

- 5-fold `GroupKFold` with `trip_id` as the group on the 80 training trips; the 10 student-test trips for comparison.
- Preprocessing (for example scalers) is fitted on training trips only.
- Reported per model: CV Macro F1, student-test Macro F1, precision, recall, candidate accuracy, observation-level Top-1 accuracy, candidate miss rate, log loss, route continuity, training time and inference time.
- Primary selection metric: CV Macro F1, then student-test Top-1 accuracy and log loss.

## Repository layout

```text
src/
  pathsnap_phase1.py      loading, validation, projection, road network, label alignment, candidate features
  candidate_models.py     Phase 1 smoke-test classifiers and evaluation helpers
scripts/
  run_phase1.py           builds EDA inventory and candidate feature CSVs
  train_models.py         Phase 1 smoke-test model comparison
tests/
  test_phase1.py          pipeline tests
outputs/                  smoke-test CSVs (full runs are not committed)
requirements.txt
```

## How to run

```powershell
pip install -r requirements.txt
python scripts/run_phase1.py --data-root "data/Student folder" --output-dir outputs/phase1
python -m pytest tests/test_phase1.py
```

`run_phase1.py` validates every loaded trip, writes an EDA inventory, and extracts candidate-level features (`--radius-m`, default 80; `--max-candidates`, default 8; `--feature-trips` to limit the run).

Phase 1 smoke comparison:

```powershell
python scripts/train_models.py --train outputs/smoke_train/candidate_features.csv --student-test outputs/smoke_student_test/candidate_features.csv --output outputs/smoke_model_comparison.csv
```

The smoke comparison in `outputs/smoke_model_comparison.csv` uses one training trip and one student-test trip only, so its numbers are not model-selection evidence. Full results for the five models above are added as each model is trained on all 80 training trips.

## Integrity

- Evaluator trips are never used for training, feature selection, tuning or thresholds.
- No hardcoded predictions; ground truth stays server-side and is never sent to the model or UI.
- Every reported number is produced by code in this repository.
- AI assistance was used and is disclosed as required by the event; every member can explain the code and results.
