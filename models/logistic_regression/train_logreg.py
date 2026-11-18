"""Logistic Regression candidate classifier (Akshansh Agrawal).

Usage (repository root, after `python -m pathsnap.build ...`):
    python models/logistic_regression/train_logreg.py --features outputs/features \
        --dataset "data/Student folder" --output outputs/logistic_regression

Procedure
---------
1. Load the shared candidate CSVs (pathsnap.build) and attach the approximate reference arc from the separate
   observation-label CSV. Model inputs are exactly pathsnap.candidates.FEATURE_COLUMNS; IDs, labels, alignment
   distances and validity flags are never inputs.
2. Rows of observations with label_valid = 0 (aligned arc > 50 m away) are excluded from fitting and scoring.
3. 5-fold GroupKFold by trip_id on the 80 train trips. Inside every fold the pipeline
   (log1p of heavy-tailed features -> median imputer -> StandardScaler -> LogisticRegression) is fitted on the
   fold's training trips only. Class imbalance (about 1 positive per 7-8 candidates) is handled by balanced
   sample weights computed from each fold's training labels.
4. Refit on all 80 train trips, then ONE evaluation on the 10 student_test trips.
5. Save fold metrics, student-test metrics, coefficients, config, the fitted pipeline and per-candidate
   student-test probabilities (input for the later HMM) to the output directory (untracked by Git).

Note on probabilities: balanced weighting shifts probabilities upwards for the positive class, which raises
log loss but not Top-1 accuracy; the HMM stage should use these as relative emission scores or recalibrate.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathsnap.candidates import FEATURE_COLUMNS  # noqa: E402
from pathsnap.metrics import candidate_metrics, observation_top1, route_continuity  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = json.loads((HERE / "config.json").read_text())


# ----------------------------------------------------------------------------- data
def load_partition(features_dir: Path, partition: str) -> dict:
    """Candidate rows with features (X), reference label (y) and validity, as numpy arrays."""
    features_dir = Path(features_dir)
    cand_path = features_dir / f"{partition}_candidates.csv"
    label_path = features_dir / f"{partition}_observation_labels.csv"
    try:
        import pandas as pd
        cand = pd.read_csv(cand_path, dtype={"trip_id": str})
        lab = pd.read_csv(label_path, dtype={"trip_id": str})
        trip = cand["trip_id"].to_numpy(str)
        obs = cand["observation_index"].to_numpy(np.int64)
        arc = cand["candidate_arc_id"].to_numpy(np.int64)
        X = cand[FEATURE_COLUMNS].to_numpy(np.float64)
        l_trip = lab["trip_id"].to_numpy(str); l_obs = lab["observation_index"].to_numpy(np.int64)
        l_ref = lab["reference_arc_id"].to_numpy(np.int64); l_valid = lab["label_valid"].to_numpy(np.int64)
    except ImportError:  # dependency-light fallback (used by tests on machines without pandas)
        with cand_path.open(encoding="utf-8") as f:
            rd = csv.reader(f); hdr = next(rd); rows = list(rd)
        ix = {h: i for i, h in enumerate(hdr)}
        trip = np.array([r[ix["trip_id"]] for r in rows])
        obs = np.array([int(r[ix["observation_index"]]) for r in rows], np.int64)
        arc = np.array([int(r[ix["candidate_arc_id"]]) for r in rows], np.int64)
        X = np.array([[float(r[ix[c]]) for c in FEATURE_COLUMNS] for r in rows], np.float64)
        with label_path.open(encoding="utf-8") as f:
            lr = list(csv.DictReader(f))
        l_trip = np.array([r["trip_id"] for r in lr]); l_obs = np.array([int(r["observation_index"]) for r in lr])
        l_ref = np.array([int(r["reference_arc_id"]) for r in lr]); l_valid = np.array([int(r["label_valid"]) for r in lr])

    trips = np.unique(np.concatenate([trip, l_trip]))
    tcode = {t: i for i, t in enumerate(trips)}
    stride = int(max(obs.max(initial=0), l_obs.max(initial=0)) + 1)
    key = np.array([tcode[t] for t in trip], np.int64) * stride + obs
    lkey = np.array([tcode[t] for t in l_trip], np.int64) * stride + l_obs
    order = np.argsort(lkey)
    pos = order[np.searchsorted(lkey[order], key)]
    assert np.array_equal(lkey[pos], key), "every candidate row must have an observation label row"
    return {"trip_id": trip, "observation_index": obs, "candidate_arc_id": arc, "X": X,
            "y": (arc == l_ref[pos]).astype(np.int64), "valid": l_valid[pos].astype(bool), "obs_key": key,
            "obs_valid_keys": lkey[l_valid.astype(bool)], "obs_trip": l_trip, "obs_valid": l_valid.astype(bool)}


def subset(d: dict, mask: np.ndarray) -> dict:
    """Rows selected by mask; n_valid_obs counts every validly labelled observation of the selected trips."""
    meta = ("obs_valid_keys", "obs_trip", "obs_valid", "n_valid_obs")
    out = {k: v[mask] for k, v in d.items() if k not in meta}
    out.update({k: d[k] for k in ("obs_trip", "obs_valid", "obs_valid_keys")})
    trips = np.unique(d["trip_id"][mask])
    out["n_valid_obs"] = int((np.isin(d["obs_trip"], trips) & d["obs_valid"]).sum())
    return out


# ----------------------------------------------------------------------------- model
def signed_log1p(X, columns=()):
    X = np.array(X, dtype=np.float64, copy=True)
    for c in columns:
        X[:, c] = np.sign(X[:, c]) * np.log1p(np.abs(X[:, c]))
    return X


def make_model(cfg: dict):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    cols = [FEATURE_COLUMNS.index(c) for c in cfg["log1p_features"]]
    return Pipeline([
        ("log1p", FunctionTransformer(functools.partial(signed_log1p, columns=cols), validate=False)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(random_state=cfg["random_seed"], **cfg["hyperparameters"])),
    ])


def make_numpy_model(cfg: dict):
    """Same pipeline and objective in numpy (see numpy_logreg.py); used where sklearn cannot load."""
    sys.path.insert(0, str(HERE))
    from numpy_logreg import NumpyLogisticRegression
    hp = cfg["hyperparameters"]
    return NumpyLogisticRegression(C=hp["C"], max_iter=100, tol=1e-6,
                                   log1p_columns=[FEATURE_COLUMNS.index(c) for c in cfg["log1p_features"]])


def resolve_backend(name: str):
    if name == "numpy":
        return "numpy", make_numpy_model
    try:
        import sklearn  # noqa: F401
        return "sklearn", make_model
    except ImportError:
        if name == "sklearn":
            raise
        return "numpy", make_numpy_model


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """n_samples / (n_classes * count(class)), the same rule as sklearn's class_weight='balanced'."""
    counts = np.bincount(y, minlength=2).astype(np.float64)
    return (len(y) / (2.0 * np.maximum(counts, 1)))[y]


def fit(model, X, y, cfg):
    if cfg["class_balance"] == "balanced_sample_weight":
        model.fit(X, y, logreg__sample_weight=balanced_weights(y))
    else:
        model.fit(X, y)
    return model


def group_kfold(groups: np.ndarray, n_splits: int):
    try:
        from sklearn.model_selection import GroupKFold
        yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)
    except ImportError:  # same greedy rule as sklearn: largest groups first, into the lightest fold
        uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        fold_of = np.empty(len(uniq), int); load = np.zeros(n_splits)
        for g in np.argsort(-counts, kind="stable"):
            f = int(np.argmin(load)); fold_of[g] = f; load[f] += counts[g]
        for f in range(n_splits):
            test = fold_of[inv] == f
            yield np.flatnonzero(~test), np.flatnonzero(test)


def evaluate(d: dict, prob: np.ndarray, threshold: float) -> dict:
    m = candidate_metrics(d["y"], prob, threshold)
    m.update(observation_top1(d["obs_key"], d["y"], prob, d["n_valid_obs"]))
    return m


# ----------------------------------------------------------------------------- run
def run(features_dir: Path, output_dir: Path, cfg: dict = DEFAULT_CONFIG, dataset: Path | None = None,
        model_factory=make_model) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    train_all = load_partition(features_dir, "train")
    test_all = load_partition(features_dir, "student_test")
    train = subset(train_all, train_all["valid"])
    test = subset(test_all, test_all["valid"])
    assert not set(train["trip_id"]) & set(test["trip_id"]), "train and student_test trips must not overlap"
    print(f"loaded in {time.time() - t_load:.0f}s | train rows={len(train['y'])} trips={len(set(train['trip_id']))} "
          f"| student_test rows={len(test['y'])} trips={len(set(test['trip_id']))} | positive rate={train['y'].mean():.3f}")

    folds = []
    for k, (a, b) in enumerate(group_kfold(train["trip_id"], cfg["cv_folds"])):
        ma = np.zeros(len(train["y"]), bool); ma[a] = True
        tr, va = subset(train, ma), subset(train, ~ma)
        assert not set(tr["trip_id"]) & set(va["trip_id"])
        model = model_factory(cfg)
        t0 = time.time(); fit(model, tr["X"], tr["y"], cfg); fit_s = time.time() - t0
        t0 = time.time(); p_va = model.predict_proba(va["X"])[:, 1]; pred_s = time.time() - t0
        mv = evaluate(va, p_va, cfg["threshold"])
        mt = evaluate(tr, model.predict_proba(tr["X"])[:, 1], cfg["threshold"])
        folds.append({"fold": k, "train_trips": len(set(tr["trip_id"])), "val_trips": len(set(va["trip_id"])),
                      "fit_seconds": fit_s, "predict_seconds": pred_s,
                      "n_iter": int(np.max(getattr(model[-1] if hasattr(model, "steps") else model, "n_iter_", -1))),
                      **{f"val_{x}": v for x, v in mv.items()}, **{f"train_{x}": mt[x] for x in ("macro_f1", "top1_accuracy", "log_loss")}})
        print(f"fold {k}: val macro_f1={mv['macro_f1']:.4f} top1={mv['top1_accuracy']:.4f} log_loss={mv['log_loss']:.4f} "
              f"| train macro_f1={mt['macro_f1']:.4f} ({fit_s:.1f}s)", flush=True)

    model = model_factory(cfg)
    t0 = time.time(); fit(model, train["X"], train["y"], cfg); train_s = time.time() - t0
    t0 = time.time(); p_te = model.predict_proba(test["X"])[:, 1]; infer_s = time.time() - t0
    mte = evaluate(test, p_te, cfg["threshold"])

    # classifier-only route continuity on student_test (needs arc endpoints from the raw .arcs files)
    continuity = None
    if dataset is not None:
        from pathsnap.data import find_dataset_root, load_trip
        root = find_dataset_root(Path(dataset)); vals = []
        for t in np.unique(test["trip_id"]):
            m = test["trip_id"] == t
            arcs = load_trip(root, "student_test", t).arcs
            o, a, p = test["observation_index"][m], test["candidate_arc_id"][m], p_te[m]
            order = np.lexsort((-p, o)); first = np.r_[True, o[order][1:] != o[order][:-1]]
            best = order[first]
            vals.append(route_continuity(o[best], a[best], arcs[a[best], 0], arcs[a[best], 1]))
        continuity = float(np.mean(vals))

    cvm = {k: float(np.mean([f[f"val_{k}"] for f in folds])) for k in ("macro_f1", "top1_accuracy", "log_loss", "precision", "recall")}
    cvs = float(np.std([f["val_macro_f1"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0
    summary = {
        "model_name": cfg["model_name"], "algorithm_family": cfg["algorithm_family"], "owner": cfg["owner"],
        "hyperparameters": cfg["hyperparameters"], "class_balance": cfg["class_balance"], "random_seed": cfg["random_seed"],
        "validation": f"GroupKFold({cfg['cv_folds']}) by trip_id on train; single student_test evaluation",
        "cv_macro_f1": cvm["macro_f1"], "cv_macro_f1_std": cvs, "cv_top1_accuracy": cvm["top1_accuracy"],
        "cv_log_loss": cvm["log_loss"], "cv_precision": cvm["precision"], "cv_recall": cvm["recall"],
        "cv_train_macro_f1": float(np.mean([f["train_macro_f1"] for f in folds])),
        "student_test_macro_f1": mte["macro_f1"], "student_test_top1_accuracy": mte["top1_accuracy"],
        "student_test_top1_given_reference_in_candidates": mte["top1_given_reference_in_candidates"],
        "candidate_miss_rate": mte["candidate_miss_rate"], "precision": mte["precision"], "recall": mte["recall"],
        "f1_positive": mte["f1"], "candidate_accuracy": mte["candidate_accuracy"], "log_loss": mte["log_loss"],
        "route_continuity": continuity, "training_time_s": train_s, "inference_time_s": infer_s,
        "n_train_trips": len(set(train["trip_id"])), "n_train_candidates": int(len(train["y"])),
        "n_student_test_observations_evaluated": mte["observations_evaluated"], "feature_columns": FEATURE_COLUMNS,
        "python": platform.python_version(),
    }
    summary["backend"] = cfg.get("backend", "sklearn")
    summary["numpy"] = np.__version__
    final_est = model[-1] if hasattr(model, "steps") else model
    summary["final_fit_iterations"] = int(np.max(getattr(final_est, "n_iter_", -1)))
    summary["final_fit_converged"] = bool(getattr(final_est, "converged_", True))
    try:
        import sklearn; summary["sklearn"] = sklearn.__version__
    except ImportError:
        summary["sklearn"] = None

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (output_dir / "config_used.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _write_rows(output_dir / "cv_folds.csv", folds)
    flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in summary.items() if k != "feature_columns"}
    _write_rows(output_dir / "summary_row.csv", [flat])
    with (output_dir / "student_test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["trip_id", "observation_index", "candidate_arc_id", "probability"])
        w.writerows(zip(test["trip_id"].tolist(), test["observation_index"].tolist(),
                        test["candidate_arc_id"].tolist(), np.round(p_te, 6).tolist()))
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_"):
        _write_rows(output_dir / "coefficients.csv",
                    [{"feature": c, "coefficient": float(v)} for c, v in zip(FEATURE_COLUMNS, np.ravel(est.coef_))])
    if hasattr(model, "save_json"):          # numpy backend: portable parameters + convergence history
        model.save_json(output_dir / "logistic_regression_params.json", FEATURE_COLUMNS)
        _write_rows(output_dir / "convergence_history.csv", model.history_)
    try:
        import joblib; joblib.dump(model, output_dir / "logistic_regression.joblib")
    except ImportError:
        pass
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_columns"}, indent=1, default=float))
    return summary


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "logistic_regression")
    p.add_argument("--dataset", type=Path, default=None, help="raw dataset folder, for route continuity only")
    p.add_argument("--folds", type=int, default=None, help="override CV folds (e.g. 2 for a smoke run)")
    p.add_argument("--C", type=float, default=None)
    p.add_argument("--class-balance", choices=["balanced_sample_weight", "none"], default=None)
    p.add_argument("--backend", choices=["auto", "sklearn", "numpy"], default="auto",
                   help="auto = sklearn if it can be imported, otherwise the equivalent numpy implementation")
    a = p.parse_args()
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if a.folds: cfg["cv_folds"] = a.folds
    if a.C is not None: cfg["hyperparameters"]["C"] = a.C
    if a.class_balance: cfg["class_balance"] = a.class_balance
    backend, factory = resolve_backend(a.backend)
    cfg["backend"] = backend
    print("backend:", backend)
    run(a.features, a.output, cfg, a.dataset, model_factory=factory)


if __name__ == "__main__":
    main()
