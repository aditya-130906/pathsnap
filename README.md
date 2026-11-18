# PathSnap: Distributed Map-Matching and Route Reconstruction

**PES University · UE24CS352A Machine Learning · Hackathon**

PathSnap reconstructs the route a vehicle actually drove from noisy GPS observations. For every GPS point, a trained classifier scores the nearby road arcs. A Hidden Markov Model with Viterbi decoding then turns those scores into a single connected route. The system is deployed across two machines: a FastAPI model server and an evaluator-facing UI.

```text
GPS track ──► projection to metres ──► candidate arcs ──► features ──► classifier ──► HMM / Viterbi ──► route
                                          (no .route)                  P(true arc)    connectivity
```

## Final technical report

**[Open the team's final report: A-G03_PathSnap_Report.pdf](https://github.com/aditya-130906/pathsnap/blob/main/A-G03_PathSnap_Report.pdf)** (Team A-G03 final submission; a copy is at `outputs/report/A-G03_PathSnap_Report.pdf`).

> `outputs/report/PathSnap_Previous_System_Report.pdf` is an older report on the earlier system, kept for reference only.

---

## Final system (deployed)

**Pipeline:** HGB candidate scoring (Aayaan's model, 18 features, top-8 arcs within 150 m) → HMM/Viterbi route
reconstruction with Optuna-tuned parameters → FastAPI backend (Laptop 1) → Folium evaluator UI (Laptop 2).

**Accuracy progression** (student_test, held out; Top-1 over the same 19,018 label-eligible observations,
142 of 19,160 observations have no usable proxy label):

| Stage | Top-1 | Route continuity | Route F1 |
| --- | --- | --- | --- |
| Original HGB | 80.76% | 0.890 | 0.685 |
| Optuna-tuned HGB | 81.10% | | |
| Tuned HGB + HMM v1 | 87.71% | 0.967 | 0.956 |
| Original HGB + HMM v1 | 88.78% | 0.975 | 0.962 |
| **Original HGB + HMM v3 (deployed)** | **89.30%** | **0.975** | **0.969** |

Secondary metric (direction-corrected labels, 814 labels changed, see below): HGB 83.47%,
**HGB + HMM v3 90.83%**.

- **The ~95% target was not reached.** Best verified held-out Top-1: **89.30%** (original labels),
  90.83% with direction-corrected labels.
- **HMM v3** came from error analysis: ~60% of v1 errors picked the opposite-direction twin of the correct road. v3 allows
  only small backward movement along an arc, penalises U-turns onto the twin, adds a heading-agreement emission term, and
  is tuned by Optuna on **out-of-fold** train emissions (`pathsnap_app/oof_hgb.py`, 5-fold GroupKFold) instead of
  over-confident in-sample ones (20 train trips, 40 trials; config
  `outputs/hmm_viterbi/tuning_v3_oof/best_hmm_config.json`).
- **Label noise:** about half of the remaining direction errors occur where the reference route contains both directions
  of a road, so the nearest-arc proxy label is arbitrary there. The secondary metric replaces a label that points
  against a reliable GPS heading (>90°) by its reverse twin when the twin is on the route; the primary metric is unchanged.
- Remaining error sources: proxy-label ambiguity at arc boundaries and on two-way roads, off-route GPS stretches,
  parallel roads, and slow/stopped observations without a reliable heading (21% error rate there vs 11% elsewhere).
- HGB Bayesian optimisation: Optuna TPESampler(seed=42), 12 trials, GroupKFold(3) by trip_id over 40 random train trips; CV Top-1 0.7060 →
  0.7238. Best: learning_rate=0.0235, max_iter=100, max_leaf_nodes=25, max_depth=None, min_samples_leaf=54, l2_regularization=0.0618.
  Tuned model: `models/histogram_gradient_boosting/hgb_model_tuned.joblib`; history in `outputs/histogram_gradient_boosting/optuna/`.
- HMM v1 tuning (in-sample emissions): Optuna TPE on 14 train trips, 30 trials; config `outputs/hmm_viterbi/tuning/best_hmm_config.json`.
- Five-model comparison (with pipeline and denominator notes): `outputs/comparison/model_comparison.md`.
- Earlier system report (previous version, not the final submission): `outputs/report/PathSnap_Previous_System_Report.pdf`. Route maps: `outputs/hmm_viterbi/maps/`.
- Two-laptop setup, firewall and demo script: [DEPLOYMENT.md](DEPLOYMENT.md).

### Run the deployed system

```powershell
# Laptop 1 (backend)
python -m pip install -r requirements-api.txt
$env:PATHSNAP_TRIPS_DIR = "C:\pathsnap_trips"          # folder with <trip_id>/<trip_id>.track/.nodes/.arcs[/.route]
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Laptop 2 (UI)
python -m pip install -r requirements-ui.txt
$env:PATHSNAP_API_URL = "http://<LAPTOP-1-IP>:8000"
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8501     # open http://localhost:8501
```

### Reproduce the final results

```powershell
python models/histogram_gradient_boosting/preprocessing/build_features.py --dataset "data/Student folder/Student folder" --partition train --output work/full_train.csv
python models/histogram_gradient_boosting/preprocessing/build_features.py --dataset "data/Student folder/Student folder" --partition student_test --output work/full_student_test.csv
python models/histogram_gradient_boosting/optimize_hgb.py --train work/full_train.csv --student-test work/full_student_test.csv --trials 12 --cv-trips 40 --folds 3
python -m pathsnap_app.tune_hmm --features work/full_train.csv --dataset "data/Student folder/Student folder" --trials 30 --n-trips 14 --max-obs 2500 --output outputs/hmm_viterbi/tuning
python -m pathsnap_app.evaluate_hmm --features work/full_student_test.csv --dataset "data/Student folder/Student folder" --partition student_test --config outputs/hmm_viterbi/tuning/best_hmm_config.json --output outputs/hmm_viterbi/student_test_original_hgb_viterbi
python tools/compare_models.py; python tools/make_route_maps.py --dataset "data/Student folder" --trips 00000077 00000091 00000009; python tools/build_report.py
```

The feature CSVs rebuilt this way match the SHA-256 hashes recorded in `models/histogram_gradient_boosting/config.json`.
HGB inference runs through scikit-learn, or through `pathsnap_app/hgb_numpy.py` (the same fitted trees evaluated in numpy,
identical to scikit-learn's probabilities to 1e-16 on all 137,056 student_test candidates) where scikit-learn cannot load.

---

## Team

| Member | SRN | Model | Other responsibilities |
| --- | --- | --- | --- |
| Akshansh Agrawal | PES2UG24CS045 | Logistic Regression | Metrics, HMM and Viterbi decoder |
| Aayaan Kamil | PES2UG24CS017 | HistGradientBoostingClassifier | EDA, Folium maps |
| Akshat Shaurya | PES2UG24CS047 | Random Forest, K-Nearest Neighbours | Labels, candidates, features, Bayesian tuning |
| Aditya Rao | PES2UG24CS034 | Gaussian Naive Bayes | FastAPI backend, UI, deployment |

---

## Dataset

Each trip folder contains four headerless, whitespace-separated files:

| File | Columns | Meaning |
| --- | --- | --- |
| `.track` | longitude, latitude, timestamp (s) | GPS observations, nominally 1 Hz |
| `.nodes` | longitude, latitude | Road nodes; row index = node ID |
| `.arcs` | from_node_id, to_node_id | Directed road segments; row index = arc ID |
| `.route` | arc_id | Chronological reference arcs driven; **not** one label per GPS point |

| Partition | Trips | Use |
| --- | --- | --- |
| `train` | 80 | Model fitting, cross-validation, hyperparameter optimisation |
| `student_test` | 10 | Development-stage model comparison, decoder tuning |
| Evaluator | 10 | Final demonstration only; never used during development |

The supplied data is not committed. Place it under `data/Student folder/`.

---

## Method

| Stage | Description |
| --- | --- |
| 1. Validation | Shape, type and ID-range checks; time ordering; invalid coordinates and duplicate timestamps removed; large gaps and impossible jumps flagged |
| 2. Projection | Each trip is projected to metres around its own origin. Trips span several continents, so raw-degree distances are never used |
| 3. Reference labels | `.route` is aligned to the GPS sequence by monotonic dynamic programming. Alignment distance is retained for auditing; unreliable labels are excluded. The route is used for supervision and scoring only, never as a feature |
| 4. Candidates | Road arcs within a search radius of each GPS point, nearest first, capped per point. Built without `.route`, so identical at inference. Candidate miss rate is recorded |
| 5. Features | Per (GPS point, candidate arc): distance and rank, heading agreement, motion (time gap, step, speed), arc geometry and network connectivity |
| 6. Classification | Binary target: *is this candidate the reference arc?* The highest-probability candidate per observation is the classifier-only prediction |
| 7. Optimisation | Bayesian hyperparameter search (Optuna, TPE) on the selected model with trip-grouped cross-validation |
| 8. Decoding | HMM: classifier probabilities as emissions, road connectivity as transitions; Viterbi returns the most likely connected route. Compared against classifier-only predictions |
| 9. Deployment | Laptop 1: FastAPI (model, decoder, road network, server-side scoring). Laptop 2: UI (trip selection, route and metric display) |

---

## Models

| Model | Family | Owner | Role in the comparison |
| --- | --- | --- | --- |
| Logistic Regression | Linear | Akshansh | Well-calibrated probabilities for the HMM; interpretable, fast baseline |
| Gaussian Naive Bayes | Probabilistic | Aditya | Fastest probabilistic baseline; tests the feature-independence assumption |
| Random Forest | Ensemble (bagging) | Akshat | Captures nonlinear feature interactions with lower variance than a single tree |
| HistGradientBoostingClassifier | Ensemble (boosting) | Aayaan | Histogram-binned boosting; scales to ~1.4 M candidate rows and directly optimises log loss |
| K-Nearest Neighbours | Instance-based | Akshat | Non-parametric: classifies a candidate by its most similar labelled candidates. Tests whether true arcs form local neighbourhoods in feature space; distance-weighted votes yield candidate probabilities. Requires feature scaling |

All models share the same candidates, features, labels, folds and random seed (42), so the comparison is fair.

---

## Shared preprocessing package (`pathsnap/`)

The canonical, numpy-only package. Every model trains on identical candidates, labels and features.

| Module | Responsibility |
| --- | --- |
| `pathsnap/data.py` | Strict readers for `.track`, `.nodes`, `.arcs` and optional `.route`; zero-padded trip IDs kept as strings; only `train` and `student_test` are readable |
| `pathsnap/geo.py` | Per-trip local projection (origin = median GPS position) and exact point-to-segment geometry |
| `pathsnap/candidates.py` | Inference-compatible candidates and 24 features. **Never reads `.route`** (raises if one is attached) |
| `pathsnap/labels.py` | Separate, documented observation-label alignment from `.route` (training and scoring only) |
| `pathsnap/build.py` | Reproducible CSV export for both partitions |
| `pathsnap/eda.py` | Dataset inventory and diagnostics |
| `pathsnap/metrics.py` | Candidate- and observation-level metrics |

**Candidates.** Arcs within 50 m of each GPS point, found via a grid index over points sampled every 10 m along every arc (so long arcs are not missed). Exact distances are computed and the nearest 12 kept. An observation with nothing inside 50 m keeps its single nearest arc (`outside_radius = 1`).

**Features (`FEATURE_COLUMNS`).**

| Group | Features |
| --- | --- |
| Distance | raw, log, rank, candidate count, minus nearest, ratio to nearest, outside-radius flag |
| Arc geometry | arc length, projection position, distance to arc endpoints |
| Heading | heading difference, its cosine, heading-reliability flag |
| Motion | step, speed (capped at 80 m/s), time delta, large-gap flag, speed-clipped flag |
| Connectivity | link to previous/next observation's candidates, from-node in-degree, to-node out-degree, reverse-arc exists |

`trip_id`, `observation_index`, `timestamp_s` and the trip-local `candidate_arc_id` are identifiers, never features.

**Labels (`pathsnap/labels.py`).** Each observation is aligned to a *route position* by monotonic dynamic programming:
- time order is preserved and revisited arcs are handled;
- the distance cost is clipped at 100 m;
- a heading term separates the two directions of a two-way road.

Observations whose aligned arc lies more than 50 m away get `label_valid = 0` and are excluded from fitting and scoring. Labels, alignment distances and validity flags are stored in a separate CSV and are never model inputs.

**Build outputs** (`outputs/features/`, untracked): `train_candidates.csv`, `student_test_candidates.csv`, `train_observation_labels.csv`, `student_test_observation_labels.csv`, `label_quality.csv`, `build_config.json`.

**Full build, all 90 trips** (12 processes, about 1 minute):

| Partition | Trips | Observations | Candidate rows | Valid labels | Candidate miss rate |
| --- | --- | --- | --- | --- | --- |
| train | 80 | 208,417 | 1,361,579 | 88.9% | 0.61% |
| student_test | 10 | 19,160 | 147,062 | 99.9% | 0.08% |

Median alignment distance per trip is about 4.6 m (train) and 4.8 m (student_test). Most invalid training labels come from two trips whose GPS lies largely away from the reference route: `00000054` (11% valid, 19,313 observations) and `00000033` (41% valid).

**Label limitations.**
- The alignment is an approximation: arc boundaries near intersections can shift by an observation or two.
- Reference routes are not guaranteed error-free; they contain 14 arc-endpoint breaks in total.
- Invalid stretches are excluded rather than corrected.

---

## Validation protocol and metrics

- **Cross-validation:** 5-fold `GroupKFold` on the 80 training trips, grouped by `trip_id`, so no trip contributes to both training and validation.
- **Comparison set:** the 10 `student_test` trips, evaluated once per model after refitting on all training trips.
- **Leakage control:** all learned preprocessing (scalers, imputers) is fitted on training folds only.
- **Reported per model:** CV Macro F1, student-test Macro F1, precision, recall, candidate accuracy, observation-level Top-1 accuracy, candidate miss rate, log loss, route continuity, training time, inference time.
- **Selection criterion:** CV Macro F1 first, then student-test Top-1 accuracy and log loss.

*Candidate accuracy* scores every (point, candidate) row, most of which are easy negatives. *Top-1 accuracy* asks whether the single best-scoring candidate per GPS point is the reference arc, which is the actual map-matching objective.

---

## Results to date

> ⚠️ **Not yet comparable.** These runs used different preprocessing pipelines, so their candidate sets and labels differ. Final selection requires re-running every model on the single pipeline the team adopts (see [Pipelines](#pipelines)).

| Model | Pipeline | CV Macro F1 | Student-test Macro F1 | Student-test Top-1 |
| --- | --- | --- | --- | --- |
| Random Forest | Phase 1 (80 m, 8 cand.) | 0.8631 ± 0.023 | 0.8658 | 0.8097 |
| HistGradientBoosting | HGB-run (150 m, 8 cand.) | 0.8633 | 0.8798 | 0.8076* |
| Gaussian Naive Bayes | Phase 1 (80 m, 8 cand.) | 0.7610 | 0.7851 | 0.7839 |
| Logistic Regression | `pathsnap/` (50 m, 12 cand.) | see `models/logistic_regression/README.md` | | |
| K-Nearest Neighbours | — | pending | pending | pending |

\* HGB Top-1 is computed over proxy-label-eligible observations only.

### Random Forest (Phase 1 pipeline)

`RandomForestClassifier`: 150 trees, `max_depth` 20, `min_samples_leaf` 10, `max_features` √p, `max_samples` 0.25, `class_weight="balanced_subsample"`, Gini criterion.
- **Data:** trained on 78 trips (`00000033` and `00000054` excluded as unreliable), 1,340,613 candidate rows.
- **Features:** the 8 Phase 1 schema features plus 5 per-observation relative features (rank, candidate count, distance and heading relative to the best candidate, missing-heading flag).
- **Leakage control:** the label-derived columns `reference_arc_id` and `reference_alignment_distance_m` are explicitly excluded.

| Metric | Value |
| --- | --- |
| CV Macro F1 (train folds) | 0.8631 ± 0.0228 (0.8726) |
| Student-test Macro F1 | 0.8658 |
| Student-test Top-1 accuracy | **0.8097**, vs. 0.5632 nearest-road baseline |
| Candidate accuracy | 0.9265 |
| Precision / Recall | 0.6712 / 0.9180 |
| Log loss, raw → isotonic-calibrated | 0.1759 → 0.1289 |
| Route continuity (classifier only) | 0.9009 |
| Candidate miss rate | 0.0014 |
| Training / inference time | 132 s / 2.4 s |

Train, CV and test Macro F1 lie within 0.01 of each other: no evidence of overfitting. The isotonic calibrator is fitted on out-of-fold training probabilities only. Artifacts are in `outputs/random_forest/`, including `student_test_predictions.csv` with per-candidate probabilities for the HMM.

### Gaussian Naive Bayes (Phase 1 pipeline)

Five-fold group-aware CV on 80 training trips. Model bundle, folds, summary and candidate probabilities are in `outputs/gaussian_nb/`. To be re-run through `pathsnap/` if that becomes the canonical builder.

### HistGradientBoostingClassifier (HGB-run pipeline)

Built with its own candidate and label builder (150 m, top 8 candidates, 18 features), vendored under `models/histogram_gradient_boosting/preprocessing/`. Model, code and README: `models/histogram_gradient_boosting/`. Results: `outputs/histogram_gradient_boosting/`.

`outputs/smoke_model_comparison.csv` comes from a one-trip smoke test and is **not** model-selection evidence.

---

## Pipelines

Three candidate and label builders currently exist:

| Pipeline | Radius / cap | Labels stored | Used by |
| --- | --- | --- | --- |
| `pathsnap/` (canonical) | 50 m / 12 | Separate CSV, validity flag | Logistic Regression |
| Phase 1 (`src/`, `scripts/`) | 80 m / 8 | Inside the feature table | Random Forest, Gaussian NB |
| HGB-run (vendored) | 150 m / 8 | Own builder | HistGradientBoosting |

**Action:** adopt one pipeline (recommended: `pathsnap/`) and re-run all five models on it before the final comparison, Bayesian optimisation and HMM integration.

---

## Repository layout

```text
pathsnap/                          canonical shared preprocessing package
models/
  logistic_regression/             train_logreg.py, config.json                  (Akshansh)
  random_forest/                   train_rf.py, config.json, README.md           (Akshat)
  knn/                             planned                                       (Akshat)
  histogram_gradient_boosting/     train_hgb.py, predict_hgb.py, preprocessing/  (Aayaan)
src/, scripts/                     Phase 1 pipeline (run_phase1.py)
tests/
  test_pathsnap.py                 package and LR training-loop tests
  test_phase1.py                   Phase 1 pipeline tests
outputs/                           committed results; new feature/model outputs are git-ignored
requirements.txt
```

---

## How to run

### 1. Shared features (`pathsnap/`)

```powershell
pip install -r requirements.txt
python -m pathsnap.build --dataset "data/Student folder" --output outputs/features
python -m pathsnap.build --dataset "data/Student folder" --output outputs/features_smoke --smoke 4   # quick check
```

### 2. Logistic Regression

```powershell
python models/logistic_regression/train_logreg.py --features outputs/features --dataset "data/Student folder" --output outputs/logistic_regression
```

The pipeline runs inside each fold, fitted on that fold's training trips only:
- signed `log1p` on heavy-tailed features;
- median imputer, then `StandardScaler`;
- `LogisticRegression` (L2, C = 1, lbfgs).

Class imbalance (about 14% positives) is handled with balanced sample weights. Use `--backend numpy` where scikit-learn is unavailable; it fits the same objective. A full run takes about 3–6 minutes for features and 3–8 minutes for training, with peak memory around 2 GB.

### 3. Random Forest (Phase 1 pipeline)

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --output-dir outputs/full
python models/random_forest/train_rf.py --features outputs/full/candidate_features.csv --diagnostics outputs/full/candidate_diagnostics.csv --data-root "data/Student folder" --output-dir outputs/random_forest
```

Feature build takes about 3 minutes; training takes about 6–10 minutes (5 folds plus the final fit). Add `--base-features-only` to restrict the model to the 8 schema features.

### 4. Phase 1 pipeline: individual partitions and tests

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --trip-ids 00000001 --output-dir outputs/preprocessing_smoke --feature-trips 00000001
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --output-dir outputs/train_all
python scripts/run_phase1.py --data-root "data/Student folder" --partitions student_test --output-dir outputs/student_test_all
python -m pytest tests/test_phase1.py
```

### 5. Tests

```powershell
python -m pytest tests/test_pathsnap.py
$env:PATHSNAP_DATA = "data/Student folder"; python -m pytest tests/test_pathsnap.py   # adds the real-data test
```

> `outputs/full/` holds a ~340 MB feature CSV and must not be committed (GitHub's limit is 100 MB). Keep `outputs/full/` and `data/` in `.gitignore`.

---

## Academic integrity

- Evaluator trips are never used for training, feature selection, tuning or thresholds.
- No hardcoded predictions; ground truth stays server-side and is never exposed to the model or UI.
- Every reported number is produced by code in this repository.
- AI assistance was used and is disclosed as required by the event. Every member can explain the code and results.
