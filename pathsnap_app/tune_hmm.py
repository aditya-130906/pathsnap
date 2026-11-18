"""Bayesian optimisation (Optuna TPE) of the HMM/Viterbi parameters on TRAIN trips only.

Usage:
    python -m pathsnap_app.tune_hmm --features work/full_train.csv --dataset "data/Student folder" \
        --trials 40 --output outputs/hmm_viterbi/tuning

Objective: post-Viterbi observation Top-1 accuracy over label-eligible observations of a fixed set of train trips.
Emissions come from the final HGB model, which was fitted on all train trips, so emission probabilities on these
trips are in-sample (more confident than on unseen trips). The student_test trips are never used here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "histogram_gradient_boosting" / "preprocessing"))
sys.path.insert(0, str(ROOT))

from hgb_pathsnap.data import load_trip  # noqa: E402
from pathsnap_app.hgb_numpy import load_hgb  # noqa: E402
from pathsnap_app.hmm import HMMConfig, RoadGraph, viterbi  # noqa: E402

MODEL = ROOT / "models" / "histogram_gradient_boosting" / "hgb_model.joblib"


def prepare(features: Path, dataset: Path, n_trips: int, max_obs: int, seed: int):
    df = pd.read_csv(features, dtype={"trip_id": str})
    model, names, _ = load_hgb(MODEL)
    sizes = df.groupby("trip_id").observation_index.nunique()
    rng = np.random.default_rng(seed)
    pool = sorted(sizes[sizes <= max_obs].index)
    trips = sorted(rng.choice(pool, size=min(n_trips, len(pool)), replace=False))
    df = df[df.trip_id.isin(trips)].copy()
    df["prob"] = model.predict_proba(df[names].to_numpy(np.float32))[:, 1]
    data = []
    for tid, d in df.groupby("trip_id"):
        trip = load_trip(dataset, "train", tid, include_reference=False)
        g = RoadGraph(trip.nodes, trip.arcs)
        d = d.sort_values(["observation_index", "candidate_rank"])
        obs = d.observation_index.to_numpy()
        st = np.r_[0, np.flatnonzero(obs[1:] != obs[:-1]) + 1, len(obs)]
        oids = obs[st[:-1]]
        a, f, p = d.candidate_arc_id.to_numpy(), d.projected_fraction.to_numpy(), d.prob.to_numpy()
        cands = [dict(arc=a[s:e], frac=f[s:e], prob=p[s:e]) for s, e in zip(st[:-1], st[1:])]
        truth = d[d.label == 1].set_index("observation_index").candidate_arc_id.reindex(oids).to_numpy()
        elig = d.groupby("observation_index").label_valid.max().reindex(oids).fillna(0).to_numpy().astype(bool)
        data.append((tid, g, cands, g.project(trip.track[oids, :2]), trip.track[oids, 2], truth, elig))
    return trips, data


def score(data, cfg: HMMConfig):
    ok = tot = 0
    for _, g, cands, gps, times, truth, elig in data:
        path = viterbi(cands, gps, times, g, cfg)
        pred = np.array([c["arc"][k] for c, k in zip(cands, path)])
        ok += int(((pred == truth) & elig).sum()); tot += int(elig.sum())
    return ok / max(tot, 1)


def main():
    global MODEL
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--n-trips", type=int, default=16)
    p.add_argument("--max-obs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=Path, default=MODEL, help="HGB joblib bundle")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    MODEL = a.model
    a.output.mkdir(parents=True, exist_ok=True)
    trips, data = prepare(a.features, a.dataset, a.n_trips, a.max_obs, a.seed)
    print("tuning trips:", trips, flush=True)
    base = HMMConfig()
    t0 = time.time(); base_score = score(data, base)
    print(f"default config: {base_score:.4f} ({time.time() - t0:.0f}s)", flush=True)

    def objective(trial):
        cfg = HMMConfig(w_emit=trial.suggest_float("w_emit", 0.2, 4.0, log=True),
                        beta_m=trial.suggest_float("beta_m", 1.0, 60.0, log=True),
                        beta_per_s=trial.suggest_float("beta_per_s", 0.0, 10.0),
                        unreachable_logp=trial.suggest_float("unreachable_logp", -80.0, -5.0))
        return score(data, cfg)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed),
                                study_name="hmm_viterbi")
    study.enqueue_trial({"w_emit": base.w_emit, "beta_m": base.beta_m, "beta_per_s": base.beta_per_s,
                         "unreachable_logp": base.unreachable_logp})
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.value:.4f} best {s.best_value:.4f}", flush=True)])
    best = HMMConfig(**{**base.to_dict(), **study.best_params})
    study.trials_dataframe().to_csv(a.output / "optuna_trials.csv", index=False)
    (a.output / "best_hmm_config.json").write_text(json.dumps(best.to_dict(), indent=2) + "\n")
    (a.output / "tuning_summary.json").write_text(json.dumps({
        "sampler": "Optuna TPESampler(seed=%d)" % a.seed, "trials": a.trials, "tuning_trips": trips,
        "tuning_observations": int(sum(len(d[2]) for d in data)),
        "eligible_tuning_observations": int(sum(d[6].sum() for d in data)),
        "objective": "post-Viterbi Top-1 over label-eligible observations of the tuning train trips",
        "default_score": base_score, "best_score": study.best_value, "best_params": study.best_params,
        "seconds": time.time() - t0,
        "note": "emissions are in-sample HGB probabilities (final model fitted on all train trips)"}, indent=2) + "\n")
    print("best", study.best_value, study.best_params)


if __name__ == "__main__":
    main()
