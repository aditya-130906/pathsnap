# PathSnap Phase 1

This folder is separate from the original uploads. It contains a repeatable Phase 1 pipeline for the supplied `.track`, `.nodes`, `.arcs`, and `.route` files.

## Alignment assumption

The materials specify that `.route` is a chronological sequence of traversed arc IDs and explicitly say not to assume row-wise correspondence with `.track`. They do not provide an observation-to-arc alignment table. For Phase 1, `align_reference_arcs` assigns each GPS observation the closest arc from its own reference route in a local metre-based projection. The CSV retains `reference_alignment_distance_m`, so this inferred label quality is inspectable. Replace only that function if organisers supply a different alignment rule.

## Run

From this folder, using a Python environment with NumPy and pandas:

```powershell
python scripts/run_phase1.py --data-root "data/Student folder" --output-dir outputs/phase1
python -m pytest tests/test_phase1.py
```

The runner validates all loaded trips, writes an EDA inventory, and extracts candidate-level features for every loaded trip unless `--feature-trips` narrows the run.

## First evaluation: model selection and training

After generating separate candidate feature CSVs for training and student-test trips, run:

```powershell
python scripts/train_models.py --train outputs/smoke_train/candidate_features.csv --student-test outputs/smoke_student_test/candidate_features.csv --output outputs/smoke_model_comparison.csv
```

It compares five distinct candidate classifiers: a distance baseline, Gaussian Naive Bayes, logistic regression trained by gradient descent, KNN, and a random-stump ensemble. The smoke comparison uses one training trip and one student-test trip only; run the data builder over all 80 training trips before making a final selection.
