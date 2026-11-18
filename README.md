# PathSnap: Distributed Map-Matching and Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon.

PathSnap matches noisy GPS observations to the road network and reconstructs the route a vehicle actually drove. A classifier scores the candidate road arcs near each GPS point, and an HMM with Viterbi decoding turns those scores into a connected route. The system runs on two laptops: a FastAPI model server and an evaluator UI.

## Team

| Member | SRN | Model | Other responsibilities |
| --- | --- | --- | --- |
| Akshansh Agrawal | PES2UG24CS045 | Logistic Regression | Metrics, HMM and Viterbi decoder |
| Aayaan Kamil | PES2UG24CS017 | HistGradientBoostingClassifier | EDA, Folium maps |
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
| HistGradientBoostingClassifier | Ensemble (boosting) | Aayaan | Histogram-binned gradient boosting: much faster than GradientBoostingClassifier on ~1.4 M candidate rows; optimises log loss |
| Decision Tree | Tree | TBC | Explainable nonlinear rules; shows the overfitting that ensembles fix |

All models use the same candidates, features, labels, folds and random seed (42), so the comparison is fair.

## Shared preprocessing package (`pathsnap/`)

One canonical package, numpy-only, so every model trains on identical candidates, labels and features.

| Module | Role |
| --- | --- |
| `pathsnap/data.py` | Strict readers for `.track`, `.nodes`, `.arcs` and optional `.route`; zero-padded trip IDs kept as strings; only `train` and `student_test` are readable |
| `pathsnap/geo.py` | Per-trip local projection to metres (origin = median GPS position) and exact point-to-segment geometry |
| `pathsnap/candidates.py` | Inference-compatible candidates and 24 features. **Never reads `.route`** (raises if a route is attached) |
| `pathsnap/labels.py` | Separate, documented observation-label approximation from `.route` (training and scoring only) |
| `pathsnap/build.py` | Writes reproducible CSVs for both partitions |
| `pathsnap/eda.py` | Dataset inventory and diagnostics |
| `pathsnap/metrics.py` | Candidate and observation-level metrics |

**Candidates:** arcs within 50 m of each GPS point (grid index over points sampled every 10 m along every arc, so long arcs are not missed), exact distance, nearest 12 kept. An observation with nothing inside 50 m keeps its single nearest arc (`outside_radius = 1`).

**Features (model inputs, `FEATURE_COLUMNS`):** distance (raw, log, rank, count, minus nearest, ratio to nearest, outside-radius flag); road length and projection position, distance to arc ends; heading difference, its cosine and a reliability flag; step, speed (capped 80 m/s), time delta, large-gap and speed-clipped flags; connectivity (link to the previous/next observation's candidates, from-node in-degree, to-node out-degree, reverse arc exists). `trip_id`, `observation_index`, `timestamp_s` and the trip-local `candidate_arc_id` are identifiers, never features.

**Labels (`pathsnap/labels.py`):** `.route` is an ordered arc sequence, not one label per observation. Each observation is aligned to a route position by monotonic dynamic programming (time order kept, revisits handled, distance cost clipped at 100 m, heading term to separate the two directions of a two-way road). Observations whose aligned arc is more than 50 m away get `label_valid = 0` and are excluded from fitting and scoring. Labels, alignment distances and validity flags live in a separate CSV and are never model inputs.

**Build outputs** (`outputs/features/`, untracked): `train_candidates.csv`, `student_test_candidates.csv`, `train_observation_labels.csv`, `student_test_observation_labels.csv`, `label_quality.csv`, `build_config.json`.

Full build on all 90 trips (measured locally, 12 processes, about 1 minute):

| Partition | Trips | Observations | Candidate rows | Valid labels | Candidate miss rate |
| --- | --- | --- | --- | --- | --- |
| train | 80 | 208,417 | 1,361,579 | 88.9% | 0.61% |
| student_test | 10 | 19,160 | 147,062 | 99.9% | 0.08% |

Median alignment distance per trip is about 4.6 m (train) and 4.8 m (student_test). Most invalid train labels come from two trips whose GPS largely lies away from the reference route: `00000054` (11% valid, 19,313 observations) and `00000033` (41% valid).

**Label limitations:** the alignment is an approximation (arc boundaries near intersections can shift by an observation or two), reference routes are not guaranteed error-free (the routes contain 14 arc endpoint breaks in total), and invalid stretches are excluded rather than corrected.

## Validation and metrics

- 5-fold `GroupKFold` with `trip_id` as the group on the 80 training trips; the 10 student-test trips for comparison.
- Preprocessing (for example scalers) is fitted on training trips only.
- Reported per model: CV Macro F1, student-test Macro F1, precision, recall, candidate accuracy, observation-level Top-1 accuracy, candidate miss rate, log loss, route continuity, training time and inference time.
- Primary selection metric: CV Macro F1, then student-test Top-1 accuracy and log loss.

## Repository layout

```text
pathsnap/                 canonical shared preprocessing package (see above)
models/
  logistic_regression/    train_logreg.py + config.json (Akshansh)
  random_forest/          train_rf.py + config.json (Akshat; uses the Phase 1 features)
notebooks/
  LogisticRegression_Colab.ipynb   Colab: clone, upload dataset, build features, train and evaluate LR
tests/
  test_pathsnap.py        package + LR training-loop tests (real-data test when PATHSNAP_DATA is set)
  test_phase1.py          Phase 1 pipeline tests
src/, scripts/            Phase 1 data-processing pipeline (run_phase1.py); used by models/random_forest
outputs/                  committed historical results; new feature/model outputs are git-ignored
requirements.txt
```

## How to run

### Shared features (any machine with numpy)

```powershell
pip install -r requirements.txt
python -m pathsnap.build --dataset "data/Student folder" --output outputs/features
python -m pathsnap.build --dataset "data/Student folder" --output outputs/features_smoke --smoke 4   # quick check
```

### Logistic Regression

```powershell
python models/logistic_regression/train_logreg.py --features outputs/features --dataset "data/Student folder" --output outputs/logistic_regression
```

Pipeline per fold: signed `log1p` on heavy-tailed features, median imputer, `StandardScaler`, `LogisticRegression` (L2, C = 1, lbfgs), all fitted on the fold's training trips only. Class imbalance (about 14% positive candidates) is handled with balanced sample weights from each fold's training labels. 5-fold `GroupKFold` by `trip_id` on the 80 train trips, refit on all of them, then one evaluation on the 10 student_test trips. Outputs: `summary.json`, `summary_row.csv`, `cv_folds.csv`, `coefficients.csv`, `config_used.json`, `logistic_regression.joblib` and `student_test_predictions.csv` (per-candidate probabilities for the HMM).

In Colab, open `notebooks/LogisticRegression_Colab.ipynb`, keep `SMOKE = True` for the first run, then set it to `False`. The notebook asks you to upload `Student folder.zip`; Google Drive is not needed. Full run: about 3–6 min to build features and 3–8 min to train on a standard Colab CPU, peak memory about 2 GB.

### Tests

```powershell
python -m pytest tests/test_pathsnap.py
$env:PATHSNAP_DATA = "data/Student folder"; python -m pytest tests/test_pathsnap.py   # adds the real-data test
```

### Phase 1 data-processing pipeline (`scripts/run_phase1.py`, used by the Random Forest)

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --trip-ids 00000001 --output-dir outputs/preprocessing_smoke --feature-trips 00000001 --radius-m 80 --max-candidates 8
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --output-dir outputs/train_all --radius-m 80 --max-candidates 8
python scripts/run_phase1.py --data-root "data/Student folder" --partitions student_test --output-dir outputs/student_test_all --radius-m 80 --max-candidates 8
python -m pytest tests/test_phase1.py
```

It loads and validates the four files, projects to local metres, builds directed road segments with a uniform-grid index, keeps up to 8 candidates within 80 m, derives training labels by dynamic-programming alignment to the ordered route (point-to-segment distance plus heading mismatch, forward progress with repeated arcs allowed, a wider forward window at long GPS gaps), and exports spatial, directional, temporal and road-connectivity features. `.route` is used only for labels, never as an inference feature.

**Two pipelines exist.** `pathsnap/` (50 m, 12 candidates, separate label CSV) and Phase 1 (80 m, 8 candidates, labels in the feature table) produce different candidate sets and labels, so model results from the two are not directly comparable. The team should pick one for the final model comparison.

The smoke comparison in `outputs/smoke_model_comparison.csv` used one training trip and one student-test trip only; it is not model-selection evidence.

## Integrity

- Evaluator trips are never used for training, feature selection, tuning or thresholds.
- No hardcoded predictions; ground truth stays server-side and is never sent to the model or UI.
- Every reported number is produced by code in this repository.
- AI assistance was used and is disclosed as required by the event; every member can explain the code and results.
