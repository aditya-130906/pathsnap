"""Tests for the shared pathsnap package and the Logistic Regression training loop.

Synthetic tests always run. Real-data tests run when PATHSNAP_DATA points at the dataset folder
(the folder containing, or above, train/ and student_test/).
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap.candidates import FEATURE_COLUMNS, CandidateConfig, generate_candidates  # noqa: E402
from pathsnap.data import Trip  # noqa: E402
from pathsnap.geo import point_segment_distance, project_local_meters  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import candidate_metrics, observation_top1, route_continuity  # noqa: E402

LEAKY = {"label", "reference_arc_id", "alignment_distance_m", "label_valid", "route_position", "candidate_arc_id",
         "trip_id", "observation_index"}


def _synthetic_trip(with_route=False):
    """A straight east-west two-way road (2 long arcs) plus a parallel road 40 m north, GPS 5 m off the main road."""
    lat0, lon0 = 50.0, 8.0
    m_per_deg_lat = 111_195.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat0))
    def ll(x, y): return [lon0 + x / m_per_deg_lon, lat0 + y / m_per_deg_lat]
    nodes = np.array([ll(0, 0), ll(1000, 0), ll(0, 40), ll(1000, 40)])
    arcs = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])
    xs = np.arange(100, 900, 10.0)
    track = np.array([ll(x, 5.0) + [i] for i, x in enumerate(xs)])
    route = np.array([0]) if with_route else None
    return Trip("00000042", "train", Path("."), track, nodes, arcs, route)


def test_segment_distance():
    d, f = point_segment_distance(np.array([3.0]), np.array([4.0]), 0.0, 0.0, 10.0, 0.0)
    assert np.allclose(d, 4.0) and np.allclose(f, 0.3)
    d, f = point_segment_distance(np.array([-3.0]), np.array([4.0]), 0.0, 0.0, 10.0, 0.0)
    assert np.allclose(d, 5.0) and np.allclose(f, 0.0)


def test_projection_metres():
    xy = project_local_meters(np.array([[8.001, 50.0]]), np.array([8.0, 50.0]))
    assert 71 < xy[0, 0] < 72 and abs(xy[0, 1]) < 1e-9


def test_candidates_refuse_route_and_have_no_leaky_features():
    with pytest.raises(ValueError):
        generate_candidates(_synthetic_trip(with_route=True))
    assert not LEAKY & set(FEATURE_COLUMNS)


def test_candidates_long_arcs_found_ranked_and_finite():
    c = generate_candidates(_synthetic_trip(), CandidateConfig(radius_m=50, max_candidates=12))
    n_obs = 80
    assert set(c["observation_index"]) == set(range(n_obs))
    first = c["candidate_rank"] == 1
    # nearest candidates are the main road (arcs 0/1, 5 m away), never the parallel road (35 m away)
    assert set(c["candidate_arc_id"][first]) <= {0, 1}
    assert np.allclose(c["distance_m"][first], 5.0, atol=0.05)
    # all 4 arcs (1 km long, sampled) are within 50 m of every point
    assert np.all(c["candidate_count"] == 4)
    X = np.column_stack([np.asarray(c[k], dtype=float) for k in FEATURE_COLUMNS])
    assert np.isfinite(X).all()
    # heading: GPS moves east, so arc 0 (west->east) agrees and arc 1 (east->west) is opposite
    mid = (c["observation_index"] == 40)
    h = dict(zip(c["candidate_arc_id"][mid], c["heading_difference_deg"][mid]))
    assert h[0] < 1 and h[1] > 179
    assert np.all(c["has_reverse_arc"] == 1)
    assert c["trip_id"][0] == "00000042"


def test_alignment_is_monotonic_and_handles_revisits():
    # out-and-back along the same road: route = arc 0 then arc 1 (reverse); GPS goes east then west
    t = _synthetic_trip(with_route=True)
    lat0, lon0 = 50.0, 8.0
    m_lat = 111_195.0; m_lon = m_lat * np.cos(np.radians(lat0))
    xs = np.r_[np.arange(100, 900, 20.0), np.arange(900, 100, -20.0)]
    track = np.array([[lon0 + x / m_lon, lat0 + 5 / m_lat, i] for i, x in enumerate(xs)])
    trip = Trip("00000042", "train", Path("."), track, t.nodes, t.arcs, np.array([0, 1]))
    lab = align_route(trip)
    assert np.all(np.diff(lab["route_position"]) >= 0)
    assert np.all(lab["reference_arc_id"][:35] == 0) and np.all(lab["reference_arc_id"][-35:] == 1)
    assert np.all(lab["label_valid"] == 1)


def test_metrics_known_values():
    y = np.array([1, 0, 0, 1, 0, 0])
    p = np.array([.9, .2, .1, .3, .6, .1])
    m = candidate_metrics(y, p)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 3
    assert np.isclose(m["precision"], 0.5) and np.isclose(m["recall"], 0.5)
    obs = np.array([0, 0, 0, 1, 1, 1])
    t = observation_top1(obs, y, p, n_valid_observations=3)   # 3rd valid observation had no candidates
    assert np.isclose(t["top1_accuracy"], 1 / 3)               # obs 0 right, obs 1 wrong (0.6 beats 0.3)
    assert np.isclose(t["candidate_miss_rate"], 1 / 3)
    assert np.isclose(route_continuity(np.array([0, 1, 2]), np.array([5, 5, 7]), np.array([1, 1, 2]),
                                       np.array([2, 2, 3])), 1.0)


def _load_trainer():
    spec = importlib.util.spec_from_file_location("train_logreg", ROOT / "models/logistic_regression/train_logreg.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class NumpyLogReg:
    """Tiny stand-in with the same fit/predict_proba interface, so the loop is testable without sklearn."""
    def fit(self, X, y, logreg__sample_weight=None):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w = np.ones(len(y)) if logreg__sample_weight is None else logreg__sample_weight
        self.coef_ = np.zeros((1, Z.shape[1]))
        for _ in range(300):
            p = 1 / (1 + np.exp(-Z @ self.coef_[0]))
            self.coef_[0] -= 0.5 * Z.T @ (w * (p - y)) / w.sum()
        return self
    def predict_proba(self, X):
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        p = 1 / (1 + np.exp(-Z @ self.coef_[0]))
        return np.c_[1 - p, p]


def test_group_kfold_fallback_is_disjoint():
    mod = _load_trainer()
    groups = np.repeat(np.array([f"{i:08d}" for i in range(7)]), [5, 3, 8, 2, 6, 4, 1])
    seen = []
    for a, b in mod.group_kfold(groups, 3):
        assert not set(groups[a]) & set(groups[b]); seen.extend(b)
    assert sorted(seen) == list(range(len(groups)))


DATA = os.environ.get("PATHSNAP_DATA")


@pytest.mark.skipif(not DATA, reason="set PATHSNAP_DATA to run real-data tests")
def test_real_data_build_and_training_loop(tmp_path):
    from pathsnap.build import build
    summary = build(Path(DATA), tmp_path / "features", trips_per_partition=3, jobs=1)
    for part in ("train", "student_test"):
        s = summary["partitions"][part]
        assert s["n_trips"] == 3 and s["candidate_rows"] > s["observations"] > 0
        assert all(len(t) == 8 for t in s["trips"])
    mod = _load_trainer()
    cfg = dict(mod.DEFAULT_CONFIG, cv_folds=3)
    res = mod.run(tmp_path / "features", tmp_path / "lr", cfg, dataset=Path(DATA), model_factory=lambda c: NumpyLogReg())
    assert 0 < res["student_test_top1_accuracy"] <= 1 and 0 <= res["candidate_miss_rate"] < 0.05
    assert (tmp_path / "lr" / "cv_folds.csv").exists() and (tmp_path / "lr" / "student_test_predictions.csv").exists()


def _numpy_lr():
    sys.path.insert(0, str(ROOT / "models" / "logistic_regression"))
    from numpy_logreg import NumpyLogisticRegression
    return NumpyLogisticRegression


def test_numpy_logreg_reaches_optimum_of_sklearn_objective():
    """At the solution the gradient of C*sum(s*logloss) + 0.5*||w||^2 (intercept unpenalised) is ~0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 4)); X[:, 3] = np.abs(X[:, 3]) * 50
    true_w = np.array([1.5, -2.0, 0.5, 0.0])
    y = (rng.random(5000) < 1 / (1 + np.exp(-(X[:, :3] @ true_w[:3] - 1.0)))).astype(int)
    s = np.where(y == 1, 2.0, 0.7)
    m = _numpy_lr()(C=0.5, log1p_columns=[3]).fit(X, y, logreg__sample_weight=s)
    assert m.converged_
    Z = np.c_[m._transform(X), np.ones(len(y))]
    beta = np.r_[m.coef_[0], m.intercept_]
    p = 1 / (1 + np.exp(-Z @ beta))
    grad = 0.5 * Z.T @ (s * (p - y)) + np.r_[beta[:-1], 0.0]
    assert np.max(np.abs(grad)) < 1e-3
    # objective decreases monotonically over Newton iterations
    obj = [h["objective"] for h in m.history_]
    assert all(b <= a + 1e-9 for a, b in zip(obj, obj[1:]))
    # recovers the sign of the true effects (standardised scale)
    assert m.coef_[0][0] > 0 and m.coef_[0][1] < 0


def test_numpy_logreg_json_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3)); y = (X[:, 0] + rng.normal(size=500) > 0).astype(int)
    m = _numpy_lr()(C=1.0, log1p_columns=[2]).fit(X, y)
    m.save_json(tmp_path / "p.json", ["a", "b", "c"])
    import json as _j
    m2 = _numpy_lr().from_dict(_j.loads((tmp_path / "p.json").read_text()))
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
