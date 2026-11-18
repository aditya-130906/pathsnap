"""Inference engine used by the FastAPI backend: trip files -> HGB candidates/features -> HGB -> Viterbi -> route.

Training logic never runs here. Candidate generation and the 18 features are produced by the exact functions
the HGB model was trained with (models/histogram_gradient_boosting/preprocessing/hgb_pathsnap, top-8 within 150 m),
and the feature matrix is built in the saved order from feature_schema.json / the model bundle.

The reference route (.route), if present, is read ONLY for server-side scoring after prediction. It is never
passed to candidate generation, features, HGB or Viterbi, and its geometry is never returned to the UI.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "histogram_gradient_boosting" / "preprocessing"))
sys.path.insert(0, str(ROOT))

from hgb_pathsnap.data import Trip  # noqa: E402
from hgb_pathsnap.features import generate_candidate_features  # noqa: E402
from hgb_pathsnap.labels import label_candidates  # noqa: E402
from pathsnap_app.hgb_numpy import load_hgb  # noqa: E402
from pathsnap_app.hmm import HMMConfig, RoadGraph, continuity, reconstruct_route, route_overlap, viterbi  # noqa: E402

_TUNED = ROOT / "models" / "histogram_gradient_boosting" / "hgb_model_tuned.joblib"
_ORIGINAL = ROOT / "models" / "histogram_gradient_boosting" / "hgb_model.joblib"
# Deployed: original HGB + HMM tuned for it. Selected on TRAIN trips (HMM tuning objective 0.911 vs 0.899 for the
# Optuna-tuned HGB + its HMM); student_test was not used to choose. Override with PATHSNAP_MODEL / PATHSNAP_HMM_CONFIG.
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", _ORIGINAL))
SCHEMA_PATH = ROOT / "models" / "histogram_gradient_boosting" / "feature_schema.json"
HMM_CONFIG_PATH = Path(os.environ.get("PATHSNAP_HMM_CONFIG", ROOT / "outputs" / "hmm_viterbi" / "tuning" / "best_hmm_config.json"))
CANDIDATE_TOP_K, CANDIDATE_RADIUS_M = 8, 150.0          # must match the HGB training run
TRIP_ID_CHARS = set("0123456789")


class TripNotFound(Exception):
    pass


def _read(path: Path, cols: int, dtype):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.shape[1] != cols or not len(arr):
        raise ValueError(f"{path.name}: expected {cols} columns and at least one row")
    return arr


def discover_trips(trips_dir: Path) -> dict[str, Path]:
    """Trip folders (named by zero-padded digits) under trips_dir, at any depth, that hold .track/.nodes/.arcs."""
    found = {}
    for p in sorted(Path(trips_dir).rglob("*.track")):
        tid = p.stem
        if set(tid) <= TRIP_ID_CHARS and p.parent.name == tid and (p.parent / f"{tid}.nodes").exists() \
                and (p.parent / f"{tid}.arcs").exists():
            found.setdefault(tid, p.parent)
    return found


def load_trip_folder(folder: Path, tid: str) -> Trip:
    track = _read(folder / f"{tid}.track", 3, np.float64)
    nodes = _read(folder / f"{tid}.nodes", 2, np.float64)
    arcs = _read(folder / f"{tid}.arcs", 2, np.int64)
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("GPS or node coordinates contain NaN/Inf")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("arcs reference nonexistent nodes")
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]          # strictly increasing time, as in training data
    return Trip(tid, "inference", folder, track[keep], nodes, arcs, None)


class Engine:
    def __init__(self, trips_dir: str | Path, model_path=MODEL_PATH, hmm_config_path=HMM_CONFIG_PATH):
        self.trips_dir = Path(trips_dir)
        self.model_path = Path(model_path)
        self.model, names, self.backend = load_hgb(model_path)
        schema = json.loads(Path(SCHEMA_PATH).read_text())["model_input_features"]
        if names is not None and list(names) != list(schema):
            raise RuntimeError("model feature order differs from feature_schema.json")
        self.features = list(schema)
        p = Path(hmm_config_path)
        self.hmm_config = HMMConfig(**json.loads(p.read_text())) if p.exists() else HMMConfig()
        self.hmm_config_source = str(p.relative_to(ROOT)) if p.exists() else "defaults"
        self.refresh()

    def refresh(self):
        self.trips = discover_trips(self.trips_dir)

    def info(self):
        return {"model": "HistGradientBoostingClassifier", "model_file": self.model_path.name, "inference_backend": self.backend,
                "n_features": len(self.features), "feature_order": self.features,
                "candidates": {"top_k": CANDIDATE_TOP_K, "radius_m": CANDIDATE_RADIUS_M},
                "hmm_config": self.hmm_config.to_dict(), "hmm_config_source": self.hmm_config_source,
                "trips_dir": str(self.trips_dir), "n_trips": len(self.trips)}

    def predict(self, trip_id: str) -> dict:
        if not trip_id or not set(trip_id) <= TRIP_ID_CHARS:
            raise ValueError("trip_id must be digits, e.g. 00000052")
        if trip_id not in self.trips:
            self.refresh()
            if trip_id not in self.trips:
                raise TripNotFound(trip_id)
        folder = self.trips[trip_id]
        t = {}
        t0 = time.perf_counter()
        trip = load_trip_folder(folder, trip_id)
        records = generate_candidate_features(trip, top_k=CANDIDATE_TOP_K, radius_m=CANDIDATE_RADIUS_M)
        df = pd.DataFrame.from_records(records)
        t["candidates_features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        X = df[self.features].to_numpy(np.float32)
        if not np.isfinite(X).all():
            raise ValueError("non-finite feature values")
        df["prob"] = self.model.predict_proba(X)[:, 1]
        t["hgb_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = RoadGraph(trip.nodes, trip.arcs)
        df = df.sort_values(["observation_index", "candidate_rank"]).reset_index(drop=True)
        obs = df.observation_index.to_numpy()
        st = np.r_[0, np.flatnonzero(obs[1:] != obs[:-1]) + 1, len(obs)]
        oids = obs[st[:-1]]
        a, f, p = df.candidate_arc_id.to_numpy(), df.projected_fraction.to_numpy(), df.prob.to_numpy()
        cands = [dict(arc=a[s:e], frac=f[s:e], prob=p[s:e]) for s, e in zip(st[:-1], st[1:])]
        path = viterbi(cands, g.project(trip.track[oids, :2]), trip.track[oids, 2], g, self.hmm_config)
        vit = np.array([c["arc"][k] for c, k in zip(cands, path)])
        hgb = np.array([c["arc"][int(np.argmax(c["prob"]))] for c in cands])
        route = reconstruct_route(g, list(vit))
        t["viterbi_s"] = time.perf_counter() - t0

        def poly(arc_ids):
            return [[[float(trip.nodes[i, 1]), float(trip.nodes[i, 0])] for i in trip.arcs[x]] for x in arc_ids]

        hgb_route = [x for i, x in enumerate(hgb) if i == 0 or x != hgb[i - 1]]
        out = {
            "trip_id": trip_id,
            "n_observations": int(len(trip.track)), "n_candidates": int(len(df)),
            "gps": [[float(r[1]), float(r[0]), float(r[2])] for r in trip.track],
            "hgb_only_route": poly(hgb_route),
            "reconstructed_route": poly(route),
            "reconstructed_arc_ids": [int(x) for x in route],
            "matched_arc_per_observation": [int(x) for x in vit],
            "route_continuity": {"hgb_only": continuity(g, hgb), "hgb_viterbi": continuity(g, vit)},
            "timings_s": {k: round(v, 3) for k, v in t.items()},
            "evaluation": None,
        }
        route_file = folder / f"{trip_id}.route"
        if route_file.exists():                              # server-side scoring only
            ref = _read(route_file, 1, np.int64)[:, 0]
            ref_trip = Trip(trip_id, "inference", folder, trip.track, trip.nodes, trip.arcs, ref)
            labelled, _ = label_candidates([dict(r) for r in records], ref_trip, max_alignment_m=30.0)
            lab = pd.DataFrame.from_records(labelled)
            elig = lab.groupby("observation_index").label_valid.max().reindex(oids).fillna(0).to_numpy().astype(bool)
            truth = lab[lab.label == 1].set_index("observation_index").candidate_arc_id.reindex(oids).to_numpy()
            n_el = int(elig.sum())
            ev = {"eligible_observations": n_el, "ineligible_observations": int(len(oids) - n_el),
                  "hgb_top1_accuracy": float(((hgb == truth) & elig).sum() / max(n_el, 1)),
                  "hgb_viterbi_top1_accuracy": float(((vit == truth) & elig).sum() / max(n_el, 1)),
                  "denominator": "observations within 30 m of the reference route (proxy labels)"}
            ev.update({f"hgb_only_{k}": v for k, v in route_overlap(hgb_route, ref, g.length).items()})
            ev.update({f"hgb_viterbi_{k}": v for k, v in route_overlap(route, ref, g.length).items()})
            out["evaluation"] = ev
        return out


def default_trips_dir() -> Path:
    return Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "evaluation_trips"))
