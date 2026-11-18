"""Train and evaluate Gaussian Naive Bayes on PathSnap candidate features."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
EPS = 1e-9

# Every column below is available when `.route` is absent at inference time.
FEATURES = [
    "distance_to_arc_m", "projection_fraction", "arc_length_m", "candidate_rank",
    "relative_distance_to_nearest", "heading_difference_deg", "gps_step_m",
    "time_delta_s", "speed_mps", "large_gap", "connects_from_previous_candidates",
]


class NumericPreprocessor:
    def fit(self, x: np.ndarray) -> "NumericPreprocessor":
        self.median_ = np.nanmedian(x, axis=0)
        filled = np.where(np.isfinite(x), x, self.median_)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = np.maximum(filled.std(axis=0), 1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(x), x, self.median_)
        return (filled - self.mean_) / self.scale_


class GaussianNaiveBayes:
    def __init__(self, variance_smoothing: float):
        self.variance_smoothing = variance_smoothing

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        if len(np.unique(y)) != 2:
            raise ValueError("Gaussian Naive Bayes needs both positive and negative candidate labels")
        self.prior_ = np.clip(y.mean(), EPS, 1 - EPS)
        self.mean_0, self.mean_1 = x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)
        self.var_0 = x[y == 0].var(axis=0) + self.variance_smoothing
        self.var_1 = x[y == 1].var(axis=0) + self.variance_smoothing
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        log_0 = np.log(1 - self.prior_) - 0.5 * np.sum(np.log(self.var_0) + (x - self.mean_0) ** 2 / self.var_0, axis=1)
        log_1 = np.log(self.prior_) - 0.5 * np.sum(np.log(self.var_1) + (x - self.mean_1) ** 2 / self.var_1, axis=1)
        positive = 1 / (1 + np.exp(-np.clip(log_1 - log_0, -40, 40)))
        return np.column_stack((1 - positive, positive))


def metrics(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> dict:
    probability = np.clip(probability, EPS, 1 - EPS)
    predicted = (probability >= 0.5).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    positive_f1 = 2 * precision * recall / max(precision + recall, EPS)
    negative_precision = tn / max(tn + fn, 1)
    negative_recall = tn / max(tn + fp, 1)
    negative_f1 = 2 * negative_precision * negative_recall / max(negative_precision + negative_recall, EPS)
    top_1 = []
    for observation in np.unique(groups):
        rows = np.flatnonzero(groups == observation)
        top_1.append(int(y[rows[np.argmax(probability[rows])]] == 1))
    return {
        "candidate_accuracy": float(np.mean(predicted == y)),
        "precision": precision,
        "recall": recall,
        "candidate_f1": positive_f1,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "log_loss": float(-(y * np.log(probability) + (1 - y) * np.log(1 - probability)).mean()),
        "top1_accuracy": float(np.mean(top_1)),
        "observations_with_candidates": len(top_1),
    }


def group_folds(trips: np.ndarray, n_folds: int) -> list[np.ndarray]:
    unique_trips = np.array(sorted(np.unique(trips)))
    if len(unique_trips) < n_folds:
        raise ValueError(f"Need at least {n_folds} training trips; found {len(unique_trips)}")
    return [fold for fold in np.array_split(unique_trips, n_folds) if len(fold)]


def prepare(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate feature table is missing: {sorted(missing)}")
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=int)
    groups = (frame["trip_id"].astype(str) + ":" + frame["observation_id"].astype(str)).to_numpy()
    return x, y, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-diagnostics", type=Path, required=True)
    parser.add_argument("--student-test-features", type=Path, required=True)
    parser.add_argument("--student-test-diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "gaussian_nb")
    parser.add_argument("--cv-folds", type=int, default=CONFIG["cv_folds"], help="Use 1 only for a smoke run without cross-validation.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_features, dtype={"trip_id": str})
    student = pd.read_csv(args.student_test_features, dtype={"trip_id": str})
    train_diag = pd.read_csv(args.train_diagnostics, dtype={"trip_id": str})
    student_diag = pd.read_csv(args.student_test_diagnostics, dtype={"trip_id": str})
    x_train, y_train, _ = prepare(train)
    x_student, y_student, student_groups = prepare(student)
    if not y_train.sum() or not y_student.sum():
        raise ValueError("No positive labels found; investigate candidate generation and route alignment")

    folds = []
    if args.cv_folds > 1:
        for fold_id, validation_trips in enumerate(group_folds(train["trip_id"].to_numpy(), args.cv_folds), start=1):
            validation_mask = train["trip_id"].isin(validation_trips).to_numpy()
            prep = NumericPreprocessor().fit(x_train[~validation_mask])
            model = GaussianNaiveBayes(CONFIG["variance_smoothing"]).fit(prep.transform(x_train[~validation_mask]), y_train[~validation_mask])
            valid_probability = model.predict_proba(prep.transform(x_train[validation_mask]))[:, 1]
            valid_groups = (train.loc[validation_mask, "trip_id"].astype(str) + ":" + train.loc[validation_mask, "observation_id"].astype(str)).to_numpy()
            row = metrics(y_train[validation_mask], valid_probability, valid_groups)
            row.update(fold=fold_id, validation_trips="|".join(validation_trips), validation_candidates=int(validation_mask.sum()))
            folds.append(row)
    fold_frame = pd.DataFrame(folds)
    fold_frame.to_csv(args.output_dir / "cv_folds.csv", index=False)

    started = time.perf_counter()
    preprocessor = NumericPreprocessor().fit(x_train)
    model = GaussianNaiveBayes(CONFIG["variance_smoothing"]).fit(preprocessor.transform(x_train), y_train)
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    student_probability = model.predict_proba(preprocessor.transform(x_student))[:, 1]
    inference_time = time.perf_counter() - started
    result = metrics(y_student, student_probability, student_groups)
    candidate_miss_rate = 1 - float(student_diag["labelled_observations"].sum()) / float(student_diag["observations"].sum())
    summary = {
        "model_name": "GaussianNaiveBayes",
        "algorithm_family": "probabilistic generative classifier",
        "hyperparameters": {"variance_smoothing": CONFIG["variance_smoothing"]},
        "features": FEATURES,
        "cv_macro_f1": None if fold_frame.empty else float(fold_frame["macro_f1"].mean()),
        "cv_macro_f1_std": None if fold_frame.empty else float(fold_frame["macro_f1"].std(ddof=0)),
        "student_test_macro_f1": result["macro_f1"],
        "student_test_top1_accuracy": result["top1_accuracy"],
        "candidate_miss_rate": candidate_miss_rate,
        "precision": result["precision"], "recall": result["recall"],
        "candidate_accuracy": result["candidate_accuracy"], "log_loss": result["log_loss"],
        "training_time_s": training_time, "inference_time_s": inference_time,
        "n_train_trips": int(train["trip_id"].nunique()), "n_train_candidates": len(train),
        "n_student_test_trips": int(student["trip_id"].nunique()), "n_student_test_candidates": len(student),
        "n_student_test_observations": int(student_diag["observations"].sum()),
        "validation": "not run: smoke mode" if args.cv_folds == 1 else f"{args.cv_folds}-fold group-aware CV by trip_id", "random_seed": CONFIG["random_seed"],
        "candidate_radius_m": CONFIG["candidate_radius_m"], "max_candidates": CONFIG["max_candidates"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    row = {key: summary[key] for key in ["model_name", "algorithm_family", "cv_macro_f1", "student_test_macro_f1", "student_test_top1_accuracy", "candidate_miss_rate", "precision", "recall", "log_loss", "training_time_s", "inference_time_s"]}
    row["hyperparameters"] = json.dumps(summary["hyperparameters"])
    pd.DataFrame([row]).to_csv(args.output_dir / "summary_row.csv", index=False)
    with open(args.output_dir / "gaussian_nb.pkl", "wb") as stream:
        pickle.dump({"model": model, "preprocessor": preprocessor, "features": FEATURES, "config": CONFIG}, stream)
    predictions = student[["trip_id", "observation_id", "arc_id", "label"]].copy()
    predictions["candidate_probability"] = student_probability
    predictions["candidate_rank_by_probability"] = predictions.groupby(["trip_id", "observation_id"])["candidate_probability"].rank(method="first", ascending=False).astype(int)
    predictions.to_csv(args.output_dir / "student_test_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
