# Logistic Regression — Akshansh Agrawal (PES2UG24CS045)

Candidate classifier on the shared `pathsnap` features: for each (GPS observation, candidate arc) it predicts the
probability that the arc is the observation's reference arc. The highest-probability candidate per observation is
the classifier-only prediction; the probabilities are the emission scores for the later HMM/Viterbi stage.

## Reproduce

```powershell
python -m pathsnap.build --dataset "data/Student folder" --output outputs/features
python models/logistic_regression/train_logreg.py --features outputs/features --dataset "data/Student folder" --output outputs/logistic_regression
python models/logistic_regression/train_logreg.py --features outputs/features --dataset "data/Student folder" --output outputs/logistic_regression/ablation_balanced_weights --class-balance balanced_sample_weight
python models/logistic_regression/make_plots.py --results outputs/logistic_regression --features outputs/features --dataset "data/Student folder" --map-trip 00000052
```

`--backend auto` (default) uses scikit-learn when it can be imported and the numpy implementation otherwise.

## Model

- **Pipeline (fitted inside each fold on that fold's training trips only):** signed `log1p` on heavy-tailed features → median imputer → standard scaler → L2-regularised logistic regression (C = 1.0).
- **Objective:** C · Σ logloss + ½‖w‖² (binary cross-entropy with L2 penalty, intercept unpenalised).
- **Inputs:** the 24 `pathsnap.candidates.FEATURE_COLUMNS`. Trip/observation/arc IDs, labels, alignment distances and validity flags are never inputs.
- **Training rows:** observations with a reliable approximate label (`label_valid = 1`); 1,313,713 candidate rows from 80 train trips.
- **Validation:** GroupKFold(5) by trip_id on train; single student_test evaluation.
- **Class imbalance:** positives are 14% of candidate rows. Unweighted vs balanced sample weights were compared on the same CV folds and **unweighted was selected by CV Macro F1** (0.8595 vs 0.8133; CV log loss 0.1690 vs 0.2633). Balanced weights inflated positive probabilities (precision 0.557 vs 0.803) without improving Top-1.
- **Implementation used for these results: `numpy`.** The training laptop's Windows Application Control policy blocks scikit-learn's compiled files, so `numpy_logreg.py` implements the same pipeline and the same convex objective with Newton's method (exact Hessian, line search). The final fit converged in 9 iterations; `tests/test_pathsnap.py` checks that the gradient of the objective is ≈ 0 at the solution, i.e. the same optimum scikit-learn's lbfgs reaches.

## Results

| Metric | Value |
| --- | --- |
| CV Macro F1 (mean ± std, 5 folds) | 0.8595 ± 0.0226 |
| CV train Macro F1 (overfitting check) | 0.8594 |
| CV Top-1 accuracy | 0.7749 |
| student_test Macro F1 | 0.8749 |
| student_test Top-1 accuracy | 0.8006 |
| Top-1 when the reference arc is among candidates | 0.8012 |
| Candidate miss rate | 0.0008 |
| Precision / recall (positive class, threshold 0.5) | 0.8035 / 0.7608 |
| Candidate accuracy | 0.9447 |
| Log loss | 0.1490 |
| Route continuity (classifier-only, before HMM) | 0.8797 |
| Training time (full refit) | 4.4 s |
| Inference time (19,133 student_test observations) | 0.059 s |

Per-fold validation scores:

| Fold | Val trips | Val Macro F1 | Train Macro F1 | Val Top-1 | Val log loss |
| --- | --- | --- | --- | --- | --- |
| 0 | 15 | 0.8578 | 0.8598 | 0.7817 | 0.1844 |
| 1 | 17 | 0.8855 | 0.8524 | 0.8221 | 0.1400 |
| 2 | 16 | 0.8794 | 0.8543 | 0.8152 | 0.1424 |
| 3 | 16 | 0.8373 | 0.8635 | 0.7287 | 0.1660 |
| 4 | 16 | 0.8376 | 0.8673 | 0.7267 | 0.2122 |

**Fit:** train and validation Macro F1 are nearly equal in every fold, so there is no sign of overfitting; the gap to tree ensembles is consistent with mild underfitting from the linear decision boundary.

**Coefficients** (`plots/coefficients.png`): the strongest evidence against an arc is being farther than the nearest candidate (`distance_minus_min`), a large heading difference and a low distance rank; links to the previous/next observation's candidates and heading agreement push towards the true arc. Several distance features are correlated, so individual coefficients should not be over-interpreted.

**Top-1 vs Macro F1:** Macro F1 scores every candidate row at threshold 0.5; Top-1 picks one arc per observation, which is what map matching needs. Observations whose reference arc is missing from the candidates count as wrong.

## Limitations

- Labels are an approximation from route alignment (see `pathsnap/labels.py`); 11% of train observations are excluded as unreliable, mostly trips `00000054` and `00000033`.
- A linear model cannot represent feature interactions; the engineered relative features (rank, distance minus nearest, cosine heading) partly compensate.
- Results come from the shared `pathsnap` features (50 m radius, 12 candidates); models trained on the Phase 1 features (80 m, 8 candidates) are not directly comparable.

## Files in `outputs/logistic_regression/`

`summary.json`, `summary_row.csv` (team comparison row), `cv_folds.csv`, `coefficients.csv`, `config_used.json`,
`logistic_regression_params.json` (portable model parameters for the API), `logistic_regression.joblib`,
`convergence_history.csv`, `student_test_predictions.csv` (per-candidate probabilities for the HMM),
`ablation_balanced_weights/`, and `plots/` (CV folds, confusion matrix, coefficients, calibration, convergence,
class-weight ablation, Folium map of student_test trip 00000052).
