"""Train Aayaan's local HistGradientBoostingClassifier on shared candidate CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import GroupKFold

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR / "preprocessing"))
from hgb_pathsnap.features import FIELDS

# Explicit allowlist: IDs, timestamps, reference labels and diagnostics cannot
# silently enter the model when a new CSV column appears.
FEATURES = (
    "candidate_rank", "candidate_count", "distance_to_road_m",
    "distance_to_start_m", "distance_to_end_m", "segment_length_m",
    "projected_fraction", "heading_difference_deg", "gps_heading_deg",
    "road_bearing_deg", "gps_step_distance_m", "gps_speed_mps",
    "timestamp_delta_s", "has_reliable_heading", "large_time_gap",
    "speed_was_clipped", "outside_search_radius",
    "connects_from_previous_candidates",
)
META = ("trip_id", "observation_index", "candidate_arc_id", "label_valid", "label")
DEFAULT_CONFIG = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = list(FEATURES) + list(META)
    header = pd.read_csv(path, nrows=0).columns
    missing = set(columns) - set(header)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not set(FEATURES).issubset(FIELDS):
        raise ValueError("Model feature list contains a field outside the inference feature schema")
    dtypes = {name: "float32" for name in FEATURES}
    dtypes.update(trip_id="category", observation_index="int32",
                  candidate_arc_id="int32", label_valid="uint8", label="uint8")
    frame = pd.read_csv(path, usecols=columns, dtype=dtypes)
    if frame.empty or not np.isfinite(frame[list(FEATURES)].to_numpy()).all():
        raise ValueError(f"{path}: empty or non-finite model features")
    if not frame["label_valid"].isin([0, 1]).all() or not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be binary")
    if ((frame["label"] == 1) & (frame["label_valid"] == 0)).any():
        raise ValueError(f"{path}: positive label on an ineligible observation")
    if frame.duplicated(["trip_id", "observation_index", "candidate_arc_id"]).any():
        raise ValueError(f"{path}: duplicate candidate per observation")
    if frame.groupby(["trip_id", "observation_index"], observed=True)["label"].sum().gt(1).any():
        raise ValueError(f"{path}: multiple positive candidates for one observation")
    if frame.groupby(["trip_id", "observation_index"], observed=True)["label_valid"].nunique().gt(1).any():
        raise ValueError(f"{path}: inconsistent label eligibility within an observation")
    if not frame["trip_id"].astype(str).str.fullmatch(r"\d{8}").all():
        raise ValueError(f"{path}: trip IDs lost their eight-digit format")
    return frame


def selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame.label_valid == 1].reset_index(drop=True)
    if selected.empty or selected.label.nunique() != 2:
        raise ValueError("Eligible candidates need both binary classes")
    return selected


def counts(frame: pd.DataFrame) -> dict:
    return {
        "trips": int(frame.trip_id.nunique()),
        "observations": int(frame.groupby(["trip_id", "observation_index"], observed=True).ngroups),
        "candidate_rows": int(len(frame)),
        "positive_candidates": int(frame.label.sum()),
    }


def metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    truth = frame.label.to_numpy(dtype=np.uint8)
    predicted = (probabilities >= 0.5).astype(np.uint8)
    scored = frame[["trip_id", "observation_index", "label"]].copy()
    scored["probability"] = probabilities
    top_indices = scored.groupby(["trip_id", "observation_index"], observed=True)["probability"].idxmax()
    top1 = float(scored.loc[top_indices, "label"].mean())
    observations = int(len(top_indices))
    positives = int(truth.sum())
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(truth, predicted, zero_division=0)),
        "positive_recall": float(recall_score(truth, predicted, zero_division=0)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "observation_top1_accuracy": top1,
        "candidate_miss_rate": float(1 - positives / observations),
        "eligible_observations": observations,
        "eligible_candidate_rows": int(len(frame)),
        "positive_candidates": positives,
    }


def save_plots(output: Path, truth: np.ndarray, probabilities: np.ndarray,
               fold_rows: list[dict], student_metrics: dict) -> None:
    cm = confusion_matrix(truth, probabilities >= 0.5, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["other candidate", "reference proxy"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Student-test candidate confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "student_confusion_matrix.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(truth, probabilities)
    ap = average_precision_score(truth, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"Average precision = {ap:.3f}")
    ax.axhline(float(truth.mean()), color="gray", linestyle="--", label="Class prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Student-test candidate precision-recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "student_precision_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"CV {row['fold']}" for row in fold_rows] + ["Student test"],
           [row["macro_f1"] for row in fold_rows] + [student_metrics["macro_f1"]])
    ax.set(ylabel="Candidate macro F1", ylim=(0, 1), title="Trip-grouped folds and held-out student test")
    fig.tight_layout()
    fig.savefig(output / "macro_f1_comparison.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "training.log", encoding="utf-8")])
    log = logging.getLogger("hgb")
    config = dict(DEFAULT_CONFIG)
    folds = 5
    log.info("Reading shared candidate CSVs")
    train_all = read_candidates(args.train)
    student_all = read_candidates(args.student_test)
    if set(train_all.trip_id.astype(str).unique()) & set(student_all.trip_id.astype(str).unique()):
        raise ValueError("Train and student-test trip IDs overlap")
    if args.smoke:
        train_ids = list(train_all.trip_id.cat.categories)[:10]
        test_ids = list(student_all.trip_id.cat.categories)[:2]
        train_all = train_all.loc[train_all.trip_id.isin(train_ids)].copy()
        student_all = student_all.loc[student_all.trip_id.isin(test_ids)].copy()
        config["max_iter"] = 10
        folds = 2
    elif train_all.trip_id.nunique() != 80 or student_all.trip_id.nunique() != 10:
        raise ValueError("Full run requires exactly 80 train and 10 student-test trips")
    train = selected_rows(train_all)
    student = selected_rows(student_all)
    train_counts, student_counts = counts(train), counts(student)
    log.info("Eligible train: %s", train_counts)
    log.info("Eligible student-test: %s", student_counts)
    X = train.loc[:, FEATURES].to_numpy(dtype=np.float32, copy=False)
    y = train.label.to_numpy(dtype=np.uint8)
    groups = train.trip_id.astype(str).to_numpy()
    X_student = student.loc[:, FEATURES].to_numpy(dtype=np.float32, copy=False)
    y_student = student.label.to_numpy(dtype=np.uint8)
    fold_rows = []
    oof = np.empty(len(train), dtype=np.float64)
    for fold, (fit_idx, valid_idx) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups), start=1):
        if set(groups[fit_idx]) & set(groups[valid_idx]):
            raise AssertionError("Trip leakage across a cross-validation fold")
        model = HistGradientBoostingClassifier(**config)
        start = time.perf_counter()
        model.fit(X[fit_idx], y[fit_idx])
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        probability = model.predict_proba(X[valid_idx])[:, 1]
        inference_s = time.perf_counter() - start
        oof[valid_idx] = probability
        result = {"fold": fold, "fit_trips": int(len(set(groups[fit_idx]))),
                  "validation_trips": int(len(set(groups[valid_idx]))),
                  "training_seconds": fit_s, "inference_seconds": inference_s}
        result.update(metrics(train.iloc[valid_idx], probability))
        fold_rows.append(result)
        log.info("Fold %s: %s", fold, result)
    cv_metrics = metrics(train, oof)
    cv_metrics["mean_fold_macro_f1"] = float(np.mean([row["macro_f1"] for row in fold_rows]))
    cv_metrics["std_fold_macro_f1"] = float(np.std([row["macro_f1"] for row in fold_rows], ddof=1))
    log.info("Out-of-fold CV: %s", cv_metrics)

    final_model = HistGradientBoostingClassifier(**config)
    start = time.perf_counter()
    final_model.fit(X, y)
    final_training_s = time.perf_counter() - start
    start = time.perf_counter()
    student_probability = final_model.predict_proba(X_student)[:, 1]
    student_inference_s = time.perf_counter() - start
    student_metrics = metrics(student, student_probability)
    student_metrics.update(training_seconds=final_training_s, inference_seconds=student_inference_s)
    log.info("Student test: %s", student_metrics)

    bundle = {"model": final_model, "feature_names": list(FEATURES),
              "config": config, "label_method": "nearest_route_arc_geometry_proxy"}
    model_path = output / "hgb_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    loaded = joblib.load(model_path)
    check = loaded["model"].predict_proba(X_student[:100])[:, 1]
    if not np.allclose(check, student_probability[:100]):
        raise AssertionError("Reloaded model probabilities differ")

    schema = {"model_input_features": list(FEATURES), "dtypes": {name: "float32" for name in FEATURES},
              "excluded_fields": ["trip_id", "observation_index", "timestamp_s", "candidate_arc_id",
                                  "label_valid", "label", "reference_arc_id", "reference_alignment_distance_m"],
              "preprocessing": "No fitted transformer; validated finite float32 features passed directly to HGB"}
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    provenance = {"train_csv": str(args.train.resolve()), "train_sha256": sha256(args.train),
                  "student_test_csv": str(args.student_test.resolve()),
                  "student_test_sha256": sha256(args.student_test), "sklearn_version": sklearn.__version__,
                  "numpy_version": np.__version__, "pandas_version": pd.__version__,
                  "smoke": args.smoke, "cv_folds": folds, "random_state": 42,
                  "label_method": "nearest_route_arc_geometry_proxy", "config": config,
                  "train_before_label_filter": counts(train_all),
                  "train_used": train_counts,
                  "student_test_before_label_filter": counts(student_all),
                  "student_test_used": student_counts}
    (output / "configuration.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    results = {"cross_validation_oof": cv_metrics, "folds": fold_rows,
               "student_test": student_metrics,
               "student_confusion_matrix": confusion_matrix(y_student, student_probability >= 0.5, labels=[0, 1]).tolist()}
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output / "cv_folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    predictions = student[["trip_id", "observation_index", "candidate_arc_id"]].copy()
    predictions["probability"] = student_probability
    predictions.to_csv(output / "student_test_candidate_probabilities.csv", index=False)
    save_plots(output, y_student, student_probability, fold_rows, student_metrics)
    log.info("Saved and reloaded model at %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--student-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "histogram_gradient_boosting")
    parser.add_argument("--smoke", action="store_true", help="Use 10 train trips, 2 test trips, 2 folds and 10 iterations")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
