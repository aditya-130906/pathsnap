"""Tests for the deployment pipeline: numpy HGB inference, HMM/Viterbi, engine and FastAPI backend.

Real-data tests run when PATHSNAP_DATA points at the dataset folder containing student_test/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap_app.hgb_numpy import NumpyHGB, load_hgb  # noqa: E402
from pathsnap_app.hmm import HMMConfig, RoadGraph, reconstruct_route, viterbi  # noqa: E402

MODEL = ROOT / "models" / "histogram_gradient_boosting" / "hgb_model.joblib"


def test_numpy_hgb_matches_sklearn_exactly():
    sk = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6)); X[rng.random(X.shape) < 0.05] = np.nan
    y = ((np.nan_to_num(X[:, 0]) + X[:, 1] ** 2 > 0.5)).astype(int)
    m = sk.HistGradientBoostingClassifier(max_iter=40, random_state=0).fit(X, y)
    assert np.allclose(NumpyHGB.from_sklearn_like(m).predict_proba(X), m.predict_proba(X), atol=1e-12)


def test_saved_model_loads_with_schema_order():
    import json
    model, names, backend = load_hgb(MODEL, prefer_sklearn=False)
    schema = json.loads((MODEL.parent / "feature_schema.json").read_text())["model_input_features"]
    assert backend == "numpy" and names == schema and model.n_features_in_ == 18
    p = model.predict_proba(np.zeros((5, 18), np.float32))[:, 1]
    assert np.all((p > 0) & (p < 1))


def _line_graph():
    """Nodes on a line 100 m apart, eastbound arcs 0..3 plus one isolated parallel arc 4 (10 m north)."""
    lat0 = 50.0; m_lon = 111_195 * np.cos(np.radians(lat0))
    xs = [0, 100, 200, 300, 400]
    nodes = [[8 + x / m_lon, lat0] for x in xs] + [[8 + 150 / m_lon, lat0 + 10 / 111_195], [8 + 250 / m_lon, lat0 + 10 / 111_195]]
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]])
    return RoadGraph(np.array(nodes), arcs, origin_lonlat=np.array([8.0, lat0]))


def test_viterbi_follows_connected_road_over_noisy_emission():
    g = _line_graph()
    xs = np.arange(20, 400, 20.0)
    gps = np.c_[xs, np.full(len(xs), 4.0)]
    cands = []
    for i, x in enumerate(xs):
        arc = int(x // 100)
        prob = [0.7, 0.3]
        if 150 < x < 250:                          # classifier prefers the isolated parallel arc here
            prob = [0.4, 0.6]
        cands.append(dict(arc=np.array([arc, 4]), frac=np.array([(x % 100) / 100, 0.5]), prob=np.array(prob)))
    path = viterbi(cands, gps, np.arange(len(xs), dtype=float), g, HMMConfig())
    chosen = [c["arc"][k] for c, k in zip(cands, path)]
    assert 4 not in chosen                          # greedy argmax would pick arc 4; Viterbi keeps the road
    assert reconstruct_route(g, chosen) == [0, 1, 2, 3]


def test_viterbi_neutral_across_large_gap():
    g = _line_graph()
    cands = [dict(arc=np.array([0, 4]), frac=np.array([.5, .5]), prob=np.array([.9, .1])),
             dict(arc=np.array([0, 4]), frac=np.array([.5, .5]), prob=np.array([.1, .9]))]
    path = viterbi(cands, np.array([[50, 0], [200, 10]]), np.array([0.0, 500.0]), g, HMMConfig(gap_s=60))
    assert list(path) == [0, 1]                    # across a 500 s gap each step follows its own emission


DATA = os.environ.get("PATHSNAP_DATA")


@pytest.mark.skipif(not DATA, reason="set PATHSNAP_DATA to run real-data tests")
def test_api_end_to_end_on_real_trips(monkeypatch):
    from fastapi.testclient import TestClient
    tdir = next(p for p in Path(DATA).rglob("student_test") if p.is_dir())
    monkeypatch.setenv("PATHSNAP_TRIPS_DIR", str(tdir))
    import importlib
    import api.main as api_main
    importlib.reload(api_main)
    c = TestClient(api_main.app)
    assert c.get("/health").json()["n_features"] == 18
    trips = c.get("/trips").json()["trips"]
    assert len(trips) == 10
    for tid in ("00000052", "00000091"):
        r = c.post("/predict", json={"trip_id": tid}).json()
        assert r["reconstructed_route"] and len(r["gps"]) == r["n_observations"]
        ev = r["evaluation"]
        assert ev["hgb_viterbi_top1_accuracy"] >= ev["hgb_top1_accuracy"]
        assert "reference" not in " ".join(r.keys())      # reference geometry never leaves the server
    assert c.post("/predict", json={"trip_id": "99999999"}).status_code == 404
    assert c.post("/predict", json={"trip_id": "../x"}).status_code == 422
    assert c.get("/results/00000052").status_code == 200
