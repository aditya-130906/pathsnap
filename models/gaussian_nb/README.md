# Gaussian Naive Bayes

This model scores each candidate road segment independently using the shared PathSnap candidate table. It estimates a Gaussian distribution for each numeric feature in the matching and non-matching classes, then applies Bayes' rule to produce an emission probability for HMM/Viterbi.

The model uses only inference-time features. Route-derived columns, identifiers, timestamps, and labels are excluded.

## Run

```powershell
python models/gaussian_nb/train_gnb.py `
  --train-features outputs/train_all/candidate_features.csv `
  --train-diagnostics outputs/train_all/candidate_diagnostics.csv `
  --student-test-features outputs/student_test_all/candidate_features.csv `
  --student-test-diagnostics outputs/student_test_all/candidate_diagnostics.csv `
  --output-dir outputs/gaussian_nb
```

The script performs five-fold group-aware cross-validation by `trip_id`, fits the final model on all training trips, evaluates on student-test trips, and writes a model bundle, HMM-ready candidate probabilities, fold results, and a comparison-row CSV.
