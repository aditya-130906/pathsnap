"""Random Forest candidate classifier on the team's shared candidate_features.csv.

Usage (from the repo root):
  python models/random_forest/train_rf.py --features outputs/full/candidate_features.csv \
      --diagnostics outputs/full/candidate_diagnostics.csv --data-root "data/Student folder" \
      --output-dir outputs/random_forest

The feature CSV is produced by scripts/run_phase1.py (both partitions in one file, split by the
`partition` column). Columns derived from the reference route (reference_arc_id,
reference_alignment_distance_m) are NEVER used as features: they would leak the label.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
SEED = CONFIG["random_seed"]

# Team schema columns used as model inputs (all available at inference without .route)
BASE_FEATURES = ["distance_to_arc_m", "projection_fraction", "arc_length_m", "heading_difference_deg",
                 "gps_step_m", "time_delta_s", "speed_mps", "connects_from_previous_candidates"]
# Per-observation relative features computed from the same columns (no label information)
DERIVED_FEATURES = ["candidate_rank", "n_candidates", "distance_minus_min", "heading_minus_min", "heading_missing"]
# Never used: label-derived (leakage) or identifiers / absolute directions
EXCLUDED = ["label", "reference_arc_id", "reference_alignment_distance_m", "trip_id", "partition", "observation_id",
            "timestamp", "arc_id", "arc_bearing_deg", "gps_heading_deg"]


def add_derived(df):
    g = df.groupby(["trip_id", "observation_id"])
    df["candidate_rank"] = g["distance_to_arc_m"].rank(method="first") - 1
    df["n_candidates"] = g["arc_id"].transform("size")
    df["distance_minus_min"] = df["distance_to_arc_m"] - g["distance_to_arc_m"].transform("min")
    df["heading_missing"] = df["heading_difference_deg"].isna().astype(int)
    df["heading_difference_deg"] = df["heading_difference_deg"].fillna(90.0)  # undefined heading -> neutral
    df["heading_minus_min"] = df["heading_difference_deg"] - df.groupby(
        ["trip_id", "observation_id"])["heading_difference_deg"].transform("min")
    for col in ["speed_mps", "time_delta_s", "gps_step_m"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def make_model():
    return RandomForestClassifier(**CONFIG["hyperparameters"], random_state=SEED, n_jobs=-1)


def top1(df, prob, n_obs_total, arcs_by_trip=None):
    """Observation-level Top-1 accuracy over ALL GPS observations (no candidates / no true candidate = wrong)."""
    d = df[["trip_id", "observation_id", "arc_id", "label"]].copy()
    d["prob"] = prob
    best = d.loc[d.groupby(["trip_id", "observation_id"])["prob"].idxmax()]
    acc = best["label"].sum() / n_obs_total
    cont = np.nan
    if arcs_by_trip is not None:
        ok = tot = 0
        for t, g in best.groupby("trip_id"):
            a = g.sort_values("observation_id")["arc_id"].to_numpy()
            arcs = arcs_by_trip[t]
            good = (a[:-1] == a[1:]) | (arcs[a[:-1], 1] == arcs[a[1:], 0])
            ok += good.sum(); tot += len(good)
        cont = ok / max(tot, 1)
    return float(acc), float(cont)


def cls_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                macro_f1=f1_score(y, pred, average="macro"), log_loss=log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7)),
                brier=brier_score_loss(y, prob)), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--diagnostics", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs") / "random_forest")
    ap.add_argument("--base-features-only", action="store_true", help="use only the 8 team schema features")
    args = ap.parse_args()
    out = args.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)

    df = add_derived(pd.read_csv(args.features, dtype={"trip_id": str}))
    diag = pd.read_csv(args.diagnostics, dtype={"trip_id": str})
    features = BASE_FEATURES if args.base_features_only else BASE_FEATURES + DERIVED_FEATURES

    # trips whose GPS lies far from the reference route have unreliable labels -> excluded from training
    unreliable = diag.loc[diag.reference_alignment_distance_median > CONFIG["unreliable_median_alignment_m"], "trip_id"]
    tr = df[(df.partition == "train") & ~df.trip_id.isin(unreliable)].reset_index(drop=True)
    te = df[df.partition == "student_test"].reset_index(drop=True)
    te_trips = te.trip_id.unique()
    n_obs_test = int(diag.loc[diag.trip_id.isin(te_trips), "observations"].sum())
    arcs_by_trip = {t: np.loadtxt(args.data_root / "student_test" / t / f"{t}.arcs", dtype=np.int64, ndmin=2)
                    for t in te_trips}
    X, y, groups = tr[features].to_numpy(np.float32), tr["label"].to_numpy(), tr["trip_id"].to_numpy()
    Xte, yte = te[features].to_numpy(np.float32), te["label"].to_numpy()
    test_diag = diag[diag.trip_id.isin(te_trips)]
    info = dict(excluded_unreliable_train_trips=sorted(unreliable.tolist()), n_train_trips=int(tr.trip_id.nunique()),
                n_train_candidates=int(len(tr)), n_test_trips=len(te_trips), n_test_candidates=int(len(te)),
                n_test_observations=n_obs_test,
                candidate_miss_rate=float(1 - test_diag.labelled_observations.sum() / n_obs_test))
    print(json.dumps(info, indent=1))

    gkf = GroupKFold(n_splits=CONFIG["cv_folds"])
    oof = np.zeros(len(y))
    folds = []
    for k, (a, b) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        m = make_model().fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
        tr_f1 = f1_score(y[a][::5], (m.predict_proba(X[a][::5])[:, 1] >= 0.5).astype(int), average="macro")
        val = cls_metrics(y[b], oof[b])[0]
        folds.append(dict(fold=k + 1, train_macro_f1=tr_f1, val_macro_f1=val["macro_f1"],
                          val_log_loss=val["log_loss"], seconds=time.time() - t0))
        print(f"fold {k+1}: val macro-F1 {val['macro_f1']:.4f} | train macro-F1 {tr_f1:.4f} | {time.time()-t0:.0f}s")
    folds = pd.DataFrame(folds)
    folds.to_csv(out / "cv_folds.csv", index=False)

    t0 = time.time()
    model = make_model().fit(X, y)
    train_time = time.time() - t0
    t0 = time.time()
    prob = model.predict_proba(Xte)[:, 1]
    infer_time = time.time() - t0
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof, y)  # train OOF only
    prob_cal = iso.predict(prob)

    raw, pred = cls_metrics(yte, prob)
    cal, _ = cls_metrics(yte, prob_cal)
    t1, cont = top1(te, prob, n_obs_test, arcs_by_trip)
    b1, bcont = top1(te, -te["distance_to_arc_m"].to_numpy(), n_obs_test, arcs_by_trip)

    summary = dict(model_name="RandomForest", algorithm_family="Ensemble (bagged decision trees)",
                   hyperparameters=CONFIG["hyperparameters"], features=features,
                   cv_macro_f1=float(folds.val_macro_f1.mean()), cv_macro_f1_std=float(folds.val_macro_f1.std()),
                   cv_train_macro_f1=float(folds.train_macro_f1.mean()),
                   student_test_macro_f1=raw["macro_f1"], student_test_top1_accuracy=t1,
                   candidate_accuracy=raw["accuracy"], precision=raw["precision"], recall=raw["recall"],
                   log_loss=raw["log_loss"], brier=raw["brier"], calibrated_log_loss=cal["log_loss"],
                   calibrated_brier=cal["brier"], route_continuity=cont,
                   nearest_road_baseline_top1=b1, nearest_road_baseline_continuity=bcont,
                   training_time_s=train_time, inference_time_s=infer_time, **info,
                   validation=f"{CONFIG['cv_folds']}-fold GroupKFold (group = trip_id)", random_seed=SEED,
                   software=dict(python=platform.python_version(), sklearn=sklearn.__version__))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    keys = ["model_name", "algorithm_family", "cv_macro_f1", "student_test_macro_f1", "student_test_top1_accuracy",
            "candidate_miss_rate", "precision", "recall", "log_loss", "training_time_s", "inference_time_s"]
    row = {k: summary[k] for k in keys}
    row["hyperparameters"] = json.dumps(CONFIG["hyperparameters"])
    pd.DataFrame([row]).to_csv(out / "summary_row.csv", index=False)

    joblib.dump(dict(model=model, features=features), out / "random_forest.joblib", compress=3)
    joblib.dump(iso, out / "isotonic_calibrator.joblib")
    exp = te[["trip_id", "observation_id", "arc_id", "candidate_rank", "distance_to_arc_m"]].copy()
    exp["prob"], exp["prob_calibrated"], exp["label"] = prob, prob_cal, yte
    exp.sort_values(["trip_id", "observation_id", "candidate_rank"]).to_csv(out / "student_test_predictions.csv", index=False)

    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["other candidate", "true arc"]).plot(
        cmap="Blues", values_format="d")
    plt.title("Random Forest – student-test confusion matrix")
    plt.savefig(out / "plots" / "confusion_matrix.png", dpi=130, bbox_inches="tight"); plt.close()
    x = np.arange(len(folds))
    plt.figure(figsize=(6, 4))
    plt.bar(x - .2, folds.train_macro_f1, .4, label="train"); plt.bar(x + .2, folds.val_macro_f1, .4, label="validation")
    plt.xticks(x, [f"fold {i}" for i in folds.fold]); plt.ylim(0.5, 1); plt.legend()
    plt.title("Random Forest – GroupKFold macro-F1")
    plt.savefig(out / "plots" / "cv_folds.png", dpi=130, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(5, 5))
    for p, lbl in [(prob, "raw RF"), (prob_cal, "isotonic-calibrated")]:
        fp, mp = calibration_curve(yte, p, n_bins=15, strategy="quantile")
        plt.plot(mp, fp, "o-", label=lbl)
    plt.plot([0, 1], [0, 1], "k--", label="perfect"); plt.legend(); plt.title("Reliability diagram (student test)")
    plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
    plt.savefig(out / "plots" / "calibration.png", dpi=130, bbox_inches="tight"); plt.close()
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    imp.plot.barh(figsize=(6, 5), title="Random Forest – feature importance")
    plt.savefig(out / "plots" / "feature_importance.png", dpi=130, bbox_inches="tight"); plt.close()

    print("\n===== RESULTS (student-test) =====")
    for k in ["cv_macro_f1", "cv_macro_f1_std", "cv_train_macro_f1", "student_test_macro_f1",
              "student_test_top1_accuracy", "candidate_accuracy", "precision", "recall", "log_loss",
              "calibrated_log_loss", "route_continuity", "candidate_miss_rate", "nearest_road_baseline_top1",
              "training_time_s", "inference_time_s"]:
        print(f"{k:30s} {summary[k]:.4f}")


if __name__ == "__main__":
    main()
