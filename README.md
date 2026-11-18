# PathSnap - Data Processing Pipeline

This repository contains the shared data-processing stage for PathSnap. It loads the supplied trip files, validates them, constructs a directed road network, derives supervised labels from the ordered reference route, generates nearby road candidates, and produces the common candidate-level feature table for model experiments.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Place the supplied dataset at `data/Student folder/`. Raw coursework data is not stored in this repository.

## Run a smoke test

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --trip-ids 00000001 --output-dir outputs/preprocessing_smoke --feature-trips 00000001 --radius-m 80 --max-candidates 8
```

## Generate shared features

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --partitions train --output-dir outputs/train_all --radius-m 80 --max-candidates 8
python scripts/run_phase1.py --data-root "data/Student folder" --partitions student_test --output-dir outputs/student_test_all --radius-m 80 --max-candidates 8
```

## Pipeline

1. Load and validate `.track`, `.nodes`, `.arcs`, and `.route`.
2. Project longitude and latitude into local metre coordinates.
3. Build directed road segments and a uniform-grid spatial index.
4. Generate up to eight candidates within an 80-metre radius for each GPS observation.
5. Derive training labels by dynamic-programming alignment to the ordered reference-route sequence. The alignment combines point-to-segment distance and heading mismatch, permits forward progress and repeated route arcs, and widens its forward window at long GPS gaps.
6. Export candidate-level spatial, directional, temporal, and road-connectivity features.

`.route` is used only to derive permitted training and evaluation labels. It is not an inference feature.

## Team model folders

Each model must read the same candidate tables created by this pipeline. The completed Random Forest work is under `models/random_forest/`. Gaussian Naive Bayes is under `models/gaussian_nb/`.

## Gaussian Naive Bayes full run

The committed Gaussian NB result uses 80 training trips, five-fold group-aware cross-validation, and 10 student-test trips. The actual summary, fold results, trained model bundle, and candidate probabilities are in `outputs/gaussian_nb/`. Generated full candidate tables are excluded because they are reproducible from the supplied raw data and too large for the repository.
