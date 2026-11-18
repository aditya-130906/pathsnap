| Model | Pipeline | CV Macro F1 | Test Macro F1 | Test Top-1 | Precision | Recall | Miss rate | Log loss | Train s | Inference s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | pathsnap | 0.8595 | 0.8749 | 0.8006 | 0.8035 | 0.7608 | 0.0008 | 0.1490 | 4.4 | 0.06 |
| Random Forest | phase1 | 0.8630 | 0.8664 | 0.8087 | 0.6728 | 0.9178 | 0.0014 | 0.1761 | 61.8 | 0.71 |
| Gaussian Naive Bayes | phase1 | 0.7610 | 0.7851 | 0.7839 | 0.5215 | 0.8903 | 0.0038 | 0.5096 | 0.9 | 0.07 |
| K-Nearest Neighbours | phase1 | 0.8691 | 0.8820 | 0.8078 | 0.8148 | 0.7835 | 0.0038 | 0.1692 | 0.5 | 83.41 |
| Histogram Gradient Boosting (selected) | hgb | 0.8633 | 0.8798 | 0.8076 | 0.7976 | 0.7884 | 0.0022 | 0.1312 | 10.8 | 0.26 |

- **phase1**: Phase 1 (scripts/run_phase1.py): <=8 candidates within 80 m. Random Forest was trained before the Phase 1 label update (nearest-route-arc labels, trips 33/54 excluded); Gaussian NB and KNN after it (DP route-alignment labels)
- **pathsnap**: pathsnap package: <=12 candidates within 50 m, DP alignment labels (valid <= 50 m)
- **hgb**: hgb_pathsnap: top-8 within 150 m, nearest-route-arc proxy labels (eligible <= 30 m)
- Top-1 denominator, Logistic Regression: 19,133 observations with valid labels
- Top-1 denominator, Random Forest: all 19,160 observations
- Top-1 denominator, Gaussian Naive Bayes: all 19,160 observations
- Top-1 denominator, K-Nearest Neighbours: all 19,160 observations
- Top-1 denominator, Histogram Gradient Boosting (selected): 19,018 eligible observations
