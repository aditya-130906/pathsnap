"""K-Nearest Neighbours candidate classifier on the team's Phase 1 candidate tables.

Usage (from the repo root):
  python models/knn/train_knn.py \
    --train-features outputs/train_all/candidate_features.csv \
    --train-diagnostics outputs/train_all/candidate_diagnostics.csv \
    --student-test-features outputs/student_test_all/candidate_features.csv \
    --student-test-diagnostics outputs/student_test_all/candidate_diagnostics.csv \
    --data-root "data/Student folder" --output-dir outputs/knn

Route-derived columns (reference_arc_id, reference_route_position, reference_alignment_distance_m),
identifiers, timestamps and absolute bearings are never used as features.
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
SEED = CONFIG["random_seed"]

# Phase 1 schema columns (same 11 inputs as the Gaussian NB model) - all available without .route
BASE_FEATURES = ["distance_to_arc_m", "projection_fraction", "arc_length_m", "candidate_rank",
                 "relative_distance_to_nearest", "heading_difference_deg", "gps_step_m", "time_delta_s",
                 "speed_mps", "large_gap", "connects_from_previous_candidates"]
# Per-observation relative features computed from those columns (no label information)
DERIVED_FEATURES = ["n_candidates", "distance_minus_min", "heading_minus_min", "heading_missing"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES
# Heavy-tailed features get an arcsinh (log-like) transform so a few extreme values do not dominate Euclidean distance
LOG_FEATURES = ["distance_to_arc_m", "arc_length_m", "relative_distance_to_nearest", "gps_step_m",
                "time_delta_s", "speed_mps", "distance_minus_min"]


def add_derived(df):
    key = ["trip_id", "observation_id"]
    g = df.groupby(key)
    df["n_candidates"] = g["arc_id"].transform("size")
    df["distance_minus_min"] = df["distance_to_arc_m"] - g["distance_to_arc_m"].transform("min")
    df["heading_missing"] = df["heading_difference_deg"].isna().astype(int)
    df["heading_difference_deg"] = df["heading_difference_deg"].fillna(90.0)  # undefined heading -> neutral 90 deg
    df["heading_minus_min"] = df["heading_difference_deg"] - df.groupby(key)["heading_difference_deg"].transform("min")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def make_model():
    log_idx = [FEATURES.index(f) for f in LOG_FEATURES]
    other_idx = [i for i in range(len(FEATURES)) if i not in log_idx]
    pre = ColumnTransformer([
        ("log", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("asinh", FunctionTransformer(np.arcsinh)),  # log-like, defined for all values, picklable
                          ("scale", StandardScaler())]), log_idx),
        ("lin", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), other_idx),
    ])
    return Pipeline([("pre", pre), ("knn", KNeighborsClassifier(**CONFIG["hyperparameters"], n_jobs=-1))])


def subsample(n, cap, rng):
    return np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))


def top1(df, prob, n_obs_total, arcs_by_trip=None):
    """Observation-level Top-1 over ALL GPS observations (no candidates / no true candidate = wrong)."""
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
    ap.add_argument("--train-features", type=Path, required=True)
    ap.add_argument("--train-diagnostics", type=Path, required=True)
    ap.add_argument("--student-test-features", type=Path, required=True)
    ap.add_argument("--student-test-diagnostics", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs") / "knn")
    ap.add_argument("--skip-k-sweep", action="store_true")
    args = ap.parse_args()
    out = args.output_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cap_fit, cap_val = CONFIG["max_fit_rows"], CONFIG["max_val_rows"]

    tr_diag = pd.read_csv(args.train_diagnostics, dtype={"trip_id": str})
    te_diag = pd.read_csv(args.student_test_diagnostics, dtype={"trip_id": str})
    tr = add_derived(pd.read_csv(args.train_features, dtype={"trip_id": str}))
    te = add_derived(pd.read_csv(args.student_test_features, dtype={"trip_id": str}))

    unreliable = tr_diag.loc[tr_diag.reference_alignment_distance_median > CONFIG["unreliable_median_alignment_m"],
                             "trip_id"].tolist()
    tr = tr[~tr.trip_id.isin(unreliable)].reset_index(drop=True)
    X, y, groups = tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy(), tr["trip_id"].to_numpy()
    Xte, yte = te[FEATURES].to_numpy(np.float32), te["label"].to_numpy()
    te_trips = te.trip_id.unique()
    n_obs_test = int(te_diag.loc[te_diag.trip_id.isin(te_trips), "observations"].sum())
    arcs_by_trip = {t: np.loadtxt(args.data_root / "student_test" / t / f"{t}.arcs", dtype=np.int64, ndmin=2)
                    for t in te_trips}
    info = dict(excluded_unreliable_train_trips=sorted(unreliable), n_train_trips=int(tr.trip_id.nunique()),
                n_train_candidates=int(len(tr)), n_test_trips=len(te_trips), n_test_candidates=int(len(te)),
                n_test_observations=n_obs_test,
                candidate_miss_rate=float(1 - te_diag.labelled_observations.sum() / n_obs_test),
                train_positive_rate=float(y.mean()), max_fit_rows=cap_fit, max_val_rows=cap_val)
    print(json.dumps(info, indent=1))

    gkf = GroupKFold(n_splits=CONFIG["cv_folds"])
    splits = list(gkf.split(X, y, groups))

    # ---- k sweep on fold 1 (bias-variance of the neighbourhood size) ----
    if not args.skip_k_sweep:
        a, b = splits[0]
        fa, vb = a[subsample(len(a), cap_fit, rng)], b[subsample(len(b), cap_val, rng)]
        sweep = []
        for k in CONFIG["k_sweep"]:
            m = make_model().set_params(knn__n_neighbors=k).fit(X[fa], y[fa])
            pv = m.predict_proba(X[vb])[:, 1]
            pt = m.predict_proba(X[fa[:20000]])[:, 1]
            sweep.append(dict(k=k, val_macro_f1=cls_metrics(y[vb], pv)[0]["macro_f1"],
                              train_macro_f1=f1_score(y[fa[:20000]], (pt >= .5).astype(int), average="macro"),
                              val_log_loss=log_loss(y[vb], np.clip(pv, 1e-7, 1 - 1e-7))))
            print(f"k={k:4d}  val macro-F1 {sweep[-1]['val_macro_f1']:.4f}  train macro-F1 {sweep[-1]['train_macro_f1']:.4f}")
        sw = pd.DataFrame(sweep)
        sw.to_csv(out / "k_sweep.csv", index=False)
        plt.figure(figsize=(6, 4))
        plt.plot(sw.k, sw.train_macro_f1, "o-", label="train (fold 1)")
        plt.plot(sw.k, sw.val_macro_f1, "o-", label="validation (fold 1)")
        plt.xscale("log"); plt.xlabel("k (neighbours)"); plt.ylabel("macro-F1"); plt.legend()
        plt.title("KNN – effect of k (small k overfits, large k oversmooths)")
        plt.savefig(out / "plots" / "k_sweep.png", dpi=130, bbox_inches="tight"); plt.close()

    # ---- 5-fold GroupKFold CV ----
    oof_idx, oof_prob, folds = [], [], []
    for k, (a, b) in enumerate(splits):
        t0 = time.time()
        fa, vb = a[subsample(len(a), cap_fit, rng)], b[subsample(len(b), cap_val, rng)]
        m = make_model().fit(X[fa], y[fa])
        pv = m.predict_proba(X[vb])[:, 1]
        pt = m.predict_proba(X[fa[:20000]])[:, 1]
        val = cls_metrics(y[vb], pv)[0]
        tr_f1 = f1_score(y[fa[:20000]], (pt >= .5).astype(int), average="macro")
        oof_idx.append(vb); oof_prob.append(pv)
        folds.append(dict(fold=k + 1, n_val_trips=len(np.unique(groups[b])), train_macro_f1=tr_f1,
                          val_macro_f1=val["macro_f1"], val_precision=val["precision"], val_recall=val["recall"],
                          val_log_loss=val["log_loss"], seconds=time.time() - t0))
        print(f"fold {k+1}: val macro-F1 {val['macro_f1']:.4f} | train macro-F1 {tr_f1:.4f} | {time.time()-t0:.0f}s")
    folds = pd.DataFrame(folds)
    folds.to_csv(out / "cv_folds.csv", index=False)
    oof_idx, oof_prob = np.concatenate(oof_idx), np.concatenate(oof_prob)

    # ---- final fit on (a capped sample of) all permitted training trips ----
    fit = subsample(len(y), cap_fit, rng)
    t0 = time.time()
    model = make_model().fit(X[fit], y[fit])
    train_time = time.time() - t0
    t0 = time.time()
    prob = model.predict_proba(Xte)[:, 1]
    infer_time = time.time() - t0
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_prob, y[oof_idx])
    prob_cal = iso.predict(prob)

    raw, pred = cls_metrics(yte, prob)
    cal, _ = cls_metrics(yte, prob_cal)
    t1, cont = top1(te, prob, n_obs_test, arcs_by_trip)
    t1c, _ = top1(te, prob_cal, n_obs_test, arcs_by_trip)
    b1, _ = top1(te, -te["distance_to_arc_m"].to_numpy(), n_obs_test, arcs_by_trip)

    summary = dict(model_name="KNN", algorithm_family="Instance-based (k-nearest neighbours)",
                   hyperparameters=CONFIG["hyperparameters"], features=FEATURES, log_features=LOG_FEATURES,
                   cv_macro_f1=float(folds.val_macro_f1.mean()), cv_macro_f1_std=float(folds.val_macro_f1.std()),
                   cv_train_macro_f1=float(folds.train_macro_f1.mean()),
                   student_test_macro_f1=raw["macro_f1"], student_test_top1_accuracy=t1,
                   calibrated_top1_accuracy=t1c, candidate_accuracy=raw["accuracy"], precision=raw["precision"],
                   recall=raw["recall"], log_loss=raw["log_loss"], brier=raw["brier"],
                   calibrated_log_loss=cal["log_loss"], calibrated_brier=cal["brier"], route_continuity=cont,
                   nearest_road_baseline_top1=b1, training_time_s=train_time, inference_time_s=infer_time,
                   **info, validation=f"{CONFIG['cv_folds']}-fold GroupKFold (group = trip_id)", random_seed=SEED,
                   software=dict(python=platform.python_version(), sklearn=sklearn.__version__))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    keys = ["model_name", "algorithm_family", "cv_macro_f1", "student_test_macro_f1", "student_test_top1_accuracy",
            "candidate_miss_rate", "precision", "recall", "log_loss", "training_time_s", "inference_time_s"]
    row = {k: summary[k] for k in keys}
    row["hyperparameters"] = json.dumps(CONFIG["hyperparameters"])
    pd.DataFrame([row]).to_csv(out / "summary_row.csv", index=False)

    joblib.dump(dict(model=model, features=FEATURES), out / "knn.joblib", compress=3)
    joblib.dump(iso, out / "isotonic_calibrator.joblib")
    exp = te[["trip_id", "observation_id", "arc_id", "candidate_rank", "distance_to_arc_m"]].copy()
    exp["prob"], exp["prob_calibrated"], exp["label"] = prob, prob_cal, yte
    exp.sort_values(["trip_id", "observation_id", "candidate_rank"]).to_csv(out / "student_test_predictions.csv", index=False)

    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["other candidate", "true arc"]).plot(
        cmap="Blues", values_format="d")
    plt.title("KNN – student-test confusion matrix")
    plt.savefig(out / "plots" / "confusion_matrix.png", dpi=130, bbox_inches="tight"); plt.close()
    x = np.arange(len(folds))
    plt.figure(figsize=(6, 4))
    plt.bar(x - .2, folds.train_macro_f1, .4, label="train"); plt.bar(x + .2, folds.val_macro_f1, .4, label="validation")
    plt.xticks(x, [f"fold {i}" for i in folds.fold]); plt.ylim(0.5, 1); plt.legend()
    plt.title("KNN – GroupKFold macro-F1")
    plt.savefig(out / "plots" / "cv_folds.png", dpi=130, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(5, 5))
    for p, lbl in [(prob, "raw KNN"), (prob_cal, "isotonic-calibrated")]:
        fp, mp = calibration_curve(yte, p, n_bins=15, strategy="quantile")
        plt.plot(mp, fp, "o-", label=lbl)
    plt.plot([0, 1], [0, 1], "k--", label="perfect"); plt.legend(); plt.title("Reliability diagram (student test)")
    plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
    plt.savefig(out / "plots" / "calibration.png", dpi=130, bbox_inches="tight"); plt.close()

    print("\n===== RESULTS (student-test) =====")
    for k in ["cv_macro_f1", "cv_macro_f1_std", "cv_train_macro_f1", "student_test_macro_f1",
              "student_test_top1_accuracy", "calibrated_top1_accuracy", "candidate_accuracy", "precision", "recall",
              "log_loss", "calibrated_log_loss", "route_continuity", "candidate_miss_rate",
              "nearest_road_baseline_top1", "training_time_s", "inference_time_s"]:
        print(f"{k:30s} {summary[k]:.4f}")


if __name__ == "__main__":
    main()
