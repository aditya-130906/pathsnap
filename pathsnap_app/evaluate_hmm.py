"""Evaluate HGB-only vs HGB + HMM/Viterbi on a labelled HGB candidate CSV.

Usage:
    python -m pathsnap_app.evaluate_hmm --features work/full_student_test.csv --dataset "data/Student folder" \
        --partition student_test --output outputs/hmm_viterbi/student_test

Primary metric: observation-level Top-1 accuracy over label-eligible observations (label_valid == 1 in the HGB
proxy-label CSV: GPS fix within 30 m of a reference-route arc). Denominator = number of eligible observations;
an eligible observation whose true arc is not among its candidates counts as wrong. The same candidate set,
labels and denominator are used for HGB-only and HGB + Viterbi, so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "histogram_gradient_boosting" / "preprocessing"))
sys.path.insert(0, str(ROOT))

from hgb_pathsnap.data import load_trip  # noqa: E402
from pathsnap_app.hgb_numpy import load_hgb  # noqa: E402
from pathsnap_app.hmm import HMMConfig, RoadGraph, continuity, reconstruct_route, route_overlap, viterbi  # noqa: E402

MODEL = ROOT / "models" / "histogram_gradient_boosting" / "hgb_model.joblib"


def decode_trip(df: pd.DataFrame, trip, cfg: HMMConfig):
    g = RoadGraph(trip.nodes, trip.arcs)
    df = df.sort_values(["observation_index", "candidate_rank"])
    obs = df.observation_index.to_numpy()
    starts = np.r_[0, np.flatnonzero(obs[1:] != obs[:-1]) + 1, len(obs)]
    oids = obs[starts[:-1]]
    arc, frac, prob = df.candidate_arc_id.to_numpy(), df.projected_fraction.to_numpy(), df.prob.to_numpy()
    hd, hr = df.heading_difference_deg.to_numpy(), df.has_reliable_heading.to_numpy()
    cands = [dict(arc=arc[s:e], frac=frac[s:e], prob=prob[s:e], hdiff=hd[s:e], hrel=hr[s:e]) for s, e in zip(starts[:-1], starts[1:])]
    gps = g.project(trip.track[oids, :2]); times = trip.track[oids, 2]
    path = viterbi(cands, gps, times, g, cfg)
    vit = np.array([c["arc"][k] for c, k in zip(cands, path)])
    hgb = np.array([c["arc"][int(np.argmax(c["prob"]))] for c in cands])
    return g, oids, hgb, vit


def evaluate(features: Path, dataset: Path, partition: str, cfg: HMMConfig, trips=None, model=None, verbose=True):
    df = pd.read_csv(features, dtype={"trip_id": str})
    if trips is not None:
        df = df[df.trip_id.isin(trips)]
    if model is None:
        model, names, _ = load_hgb(MODEL)
    else:
        model, names = model
    df["prob"] = model.predict_proba(df[names].to_numpy(np.float32))[:, 1]
    rows, per_trip = [], []
    t0 = time.time()
    for tid, d in df.groupby("trip_id", sort=True):
        trip = load_trip(dataset, partition, tid, include_reference=True)
        g, oids, hgb, vit = decode_trip(d, trip, cfg)
        lab = d[d.label == 1].set_index("observation_index").candidate_arc_id
        elig = d.groupby("observation_index").label_valid.max().reindex(oids).fillna(0).to_numpy().astype(bool)
        truth = lab.reindex(oids).to_numpy()
        hgb_ok = (hgb == truth) & elig; vit_ok = (vit == truth) & elig
        # SECONDARY metric: direction-corrected proxy labels. When the reference route contains both directions of a
        # road, the nearest-arc proxy picks between two equally near arcs arbitrarily. If the GPS heading is reliable
        # and the label arc points against the motion (>90 deg) while its reverse twin is on the route, use the twin.
        twin_of = {(int(u), int(v)): k for k, (u, v) in enumerate(trip.arcs)}
        route_set = set(int(x) for x in trip.route)
        lab_h = d[d.label == 1].set_index("observation_index").heading_difference_deg.reindex(oids).to_numpy()
        rel = d.groupby("observation_index").has_reliable_heading.max().reindex(oids).to_numpy() > 0
        truth_dc = truth.copy(); n_fix = 0
        for i in np.flatnonzero(elig & rel & ~np.isnan(truth)):
            u, v = trip.arcs[int(truth[i])]
            tw = twin_of.get((int(v), int(u)))
            if tw is not None and tw in route_set and lab_h[i] > 90:
                truth_dc[i] = tw; n_fix += 1
        hgb_dc = (hgb == truth_dc) & elig; vit_dc = (vit == truth_dc) & elig
        r_h = route_overlap(reconstruct_route(g, list(hgb)), trip.route, g.length)
        r_v = route_overlap(reconstruct_route(g, list(vit)), trip.route, g.length)
        per_trip.append(dict(trip_id=tid, observations=len(oids), eligible=int(elig.sum()),
                             hgb_top1=hgb_ok.sum() / max(elig.sum(), 1), viterbi_top1=vit_ok.sum() / max(elig.sum(), 1),
                             hgb_continuity=continuity(g, hgb), viterbi_continuity=continuity(g, vit),
                             hgb_route_f1=r_h["route_f1"], viterbi_route_f1=r_v["route_f1"]))
        rows.append((len(oids), elig.sum(), hgb_ok.sum(), vit_ok.sum(), hgb_dc.sum(), vit_dc.sum(), n_fix))
        if verbose:
            p = per_trip[-1]
            print(f"{tid}: eligible {p['eligible']}/{p['observations']} HGB {p['hgb_top1']:.4f} -> Viterbi {p['viterbi_top1']:.4f} "
                  f"| continuity {p['hgb_continuity']:.3f} -> {p['viterbi_continuity']:.3f}", flush=True)
    n, e, h, v, hdc, vdc, nfix = (int(sum(x)) for x in zip(*rows))
    pt = pd.DataFrame(per_trip)
    summary = {"partition": partition, "trips": len(pt), "observations": n, "eligible_observations": e,
               "ineligible_observations": n - e,
               "hgb_top1_accuracy": h / e, "viterbi_top1_accuracy": v / e,
               "hgb_correct": h, "viterbi_correct": v,
               "secondary_direction_corrected_labels": {
                   "labels_changed": nfix, "hgb_top1_accuracy": hdc / e, "viterbi_top1_accuracy": vdc / e,
                   "rule": "label arc against reliable GPS heading (>90 deg) and its reverse twin on the route -> twin"},
               "hgb_route_continuity": float(pt.hgb_continuity.mean()), "viterbi_route_continuity": float(pt.viterbi_continuity.mean()),
               "hgb_route_f1_mean": float(pt.hgb_route_f1.mean()), "viterbi_route_f1_mean": float(pt.viterbi_route_f1.mean()),
               "decode_seconds": time.time() - t0, "hmm_config": cfg.to_dict(),
               "denominator": "label-eligible observations (HGB proxy labels: GPS within 30 m of a reference-route arc)"}
    return summary, pt


def main():
    global MODEL
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--partition", choices=["train", "student_test"], required=True)
    p.add_argument("--config", type=Path, default=None, help="JSON with HMMConfig fields (e.g. tuned)")
    p.add_argument("--model", type=Path, default=MODEL, help="HGB joblib bundle")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    MODEL = a.model
    cfg = HMMConfig(**json.loads(a.config.read_text())) if a.config else HMMConfig()
    summary, pt = evaluate(a.features, a.dataset, a.partition, cfg)
    print(json.dumps(summary, indent=1))
    if a.output:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        pt.to_csv(a.output / "per_trip.csv", index=False)


if __name__ == "__main__":
    main()
