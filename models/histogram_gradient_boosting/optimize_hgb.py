"""Bayesian hyperparameter optimisation of the HGB model with Optuna (TPE), group-aware CV on TRAIN trips only.

Usage (repository root):
    python models/histogram_gradient_boosting/optimize_hgb.py --train work/full_train.csv \
        --student-test work/full_student_test.csv --trials 12 --cv-trips 40 --folds 3

* Same 18 features (feature_schema.json), same candidates and proxy labels as the original HGB run.
* Only label_valid == 1 rows are used (as in train_hgb.py).
* Objective: out-of-fold observation-level Top-1 accuracy (argmax probability per observation, eligible
  observations only), GroupKFold by trip_id over a fixed random subset of `cv_trips` train trips (for speed).
  Macro F1 and log loss are recorded for every trial too.
* Trial 0 is the original configuration, so tuned vs untuned are compared under identical CV.
* The best configuration is refitted on ALL 80 train trips and evaluated ONCE on the student_test trips.
Outputs: outputs/histogram_gradient_boosting/optuna/ (trials, best params, CV comparison, tuned metrics) and
models/histogram_gradient_boosting/hgb_model_tuned.joblib (same bundle format as hgb_model.joblib).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = json.loads((HERE / "feature_schema.json").read_text())["model_input_features"]
BASE = {k: v for k, v in json.loads((HERE / "config.json").read_text())["config"].items()}


def load(path):
    df = pd.read_csv(path, dtype={"trip_id": str})
    return df[df.label_valid == 1].reset_index(drop=True)


def top1(df, prob):
    d = df[["trip_id", "observation_index", "label"]].assign(p=prob)
    best = d.loc[d.groupby(["trip_id", "observation_index"]).p.idxmax()]
    return float(best.label.mean()), int(len(best))


def make(params, seed):
    p = {**BASE, **params}
    p.update(early_stopping=False, random_state=seed)
    return HistGradientBoostingClassifier(**p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--student-test", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--cv-trips", type=int, default=40)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "histogram_gradient_boosting" / "optuna")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    train = load(a.train)
    rng = np.random.default_rng(a.seed)
    cv_trips = sorted(rng.choice(sorted(train.trip_id.unique()), a.cv_trips, replace=False))
    cv = train[train.trip_id.isin(cv_trips)].reset_index(drop=True)
    X, y, g = cv[FEATURES].to_numpy(np.float32), cv.label.to_numpy(), cv.trip_id.to_numpy()
    splits = list(GroupKFold(a.folds).split(X, y, g))
    print(f"CV: {len(cv_trips)} train trips, {len(cv):,} rows, {a.folds} folds", flush=True)

    def cv_score(params):
        oof = np.zeros(len(y))
        for tr, va in splits:
            m = make(params, a.seed).fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
        t1, _ = top1(cv, oof)
        return t1, f1_score(y, oof >= .5, average="macro"), log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7))

    def objective(trial):
        depth = trial.suggest_categorical("max_depth_or_none", ["none", "set"])
        params = {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                  "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
                  "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
                  "max_depth": None if depth == "none" else trial.suggest_int("max_depth", 4, 16),
                  "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                  "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)}
        t0 = time.time(); t1, f1, ll = cv_score(params)
        trial.set_user_attr("macro_f1", f1); trial.set_user_attr("log_loss", ll)
        trial.set_user_attr("seconds", time.time() - t0); trial.set_user_attr("params", params)
        print(f"trial {trial.number}: top1={t1:.4f} macroF1={f1:.4f} logloss={ll:.4f} ({time.time()-t0:.0f}s) {params}", flush=True)
        return t1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed), study_name="hgb")
    study.enqueue_trial({"max_depth_or_none": "none", "learning_rate": BASE["learning_rate"], "max_iter": BASE["max_iter"],
                         "max_leaf_nodes": BASE["max_leaf_nodes"], "min_samples_leaf": BASE["min_samples_leaf"],
                         "l2_regularization": BASE["l2_regularization"]})
    t_start = time.time()
    study.optimize(objective, n_trials=a.trials)
    opt_seconds = time.time() - t_start
    trials = study.trials_dataframe()
    trials.to_csv(a.output / "optuna_trials.csv", index=False)
    base_t = study.trials[0]; best_t = study.best_trial
    best_params = best_t.user_attrs["params"]

    # refit best on ALL train trips, evaluate once on student_test (and the original config for reference)
    test = load(a.student_test)
    Xa, ya = train[FEATURES].to_numpy(np.float32), train.label.to_numpy()
    Xt, yt = test[FEATURES].to_numpy(np.float32), test.label.to_numpy()
    res = {}
    for name, params in [("original", base_t.user_attrs["params"]), ("tuned", best_params)]:
        t0 = time.time(); m = make(params, a.seed).fit(Xa, ya); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(Xt)[:, 1]; inf_s = time.time() - t0
        t1, n = top1(test, p)
        res[name] = {"params": params, "student_test_top1_accuracy": t1, "student_test_eligible_observations": n,
                     "student_test_macro_f1": float(f1_score(yt, p >= .5, average="macro")),
                     "student_test_log_loss": float(log_loss(yt, np.clip(p, 1e-7, 1 - 1e-7))),
                     "fit_seconds": fit_s, "inference_seconds": inf_s}
        if name == "tuned":
            joblib.dump({"model": m, "feature_names": FEATURES, "config": {**BASE, **params},
                         "label_method": "nearest_route_arc_geometry_proxy", "tuning": "optuna_tpe"},
                        HERE / "hgb_model_tuned.joblib")
        print(name, {k: v for k, v in res[name].items() if k != "params"}, flush=True)
    summary = {"sampler": f"Optuna TPESampler(seed={a.seed})", "trials": a.trials, "objective": "OOF observation Top-1 (eligible obs)",
               "cv": f"GroupKFold({a.folds}) by trip_id over {len(cv_trips)} random train trips", "cv_trips": cv_trips,
               "cv_rows": int(len(cv)), "optimisation_seconds": opt_seconds,
               "original_cv": {"top1": base_t.value, "macro_f1": base_t.user_attrs["macro_f1"], "log_loss": base_t.user_attrs["log_loss"]},
               "tuned_cv": {"top1": best_t.value, "macro_f1": best_t.user_attrs["macro_f1"], "log_loss": best_t.user_attrs["log_loss"]},
               "best_params": best_params, "best_trial": best_t.number, "student_test": res,
               "sklearn": sklearn.__version__}
    (a.output / "optuna_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (a.output / "best_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("original_cv", "tuned_cv", "best_params")}, indent=1, default=str))


if __name__ == "__main__":
    main()
