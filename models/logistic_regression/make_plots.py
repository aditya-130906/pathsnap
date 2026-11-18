"""Report plots and a Folium map for the Logistic Regression results (generated from saved outputs only).

Usage (repository root):
    python models/logistic_regression/make_plots.py --results outputs/logistic_regression \
        --features outputs/features --dataset "data/Student folder" --map-trip 00000052

Writes to <results>/plots/: cv_folds.png, confusion_matrix.png, coefficients.png, calibration.png,
convergence.png, class_weight_ablation.png and student_test_<trip>_map.html.
Every number plotted is read from files written by train_logreg.py; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "outputs" / "logistic_regression")
    p.add_argument("--features", type=Path, default=ROOT / "outputs" / "features")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--map-trip", default="00000052")
    a = p.parse_args()
    R = a.results; out = R / "plots"; out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((R / "summary.json").read_text())

    # 1. CV folds: train vs validation Macro F1 (overfitting check) + Top-1
    folds = read_csv(R / "cv_folds.csv")
    k = [int(f["fold"]) for f in folds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k, [float(f["train_macro_f1"]) for f in folds], "o-", label="train Macro F1")
    ax.plot(k, [float(f["val_macro_f1"]) for f in folds], "o-", label="validation Macro F1")
    ax.plot(k, [float(f["val_top1_accuracy"]) for f in folds], "s--", label="validation Top-1")
    ax.set_xticks(k); ax.set_xlabel("GroupKFold fold (by trip)"); ax.set_ylim(0.6, 1.0); ax.legend(); ax.grid(alpha=.3)
    ax.set_title(f"Logistic Regression CV: Macro F1 {summary['cv_macro_f1']:.3f} ± {summary['cv_macro_f1_std']:.3f}")
    fig.tight_layout(); fig.savefig(out / "cv_folds.png", dpi=130); plt.close(fig)

    # student_test predictions joined with labels
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_logreg import load_partition
    test = load_partition(a.features, "student_test")
    pred = read_csv(R / "student_test_predictions.csv")
    prob_by_key = {(r["trip_id"], int(r["observation_index"]), int(r["candidate_arc_id"])): float(r["probability"]) for r in pred}
    m = test["valid"]
    keys = list(zip(test["trip_id"][m], test["observation_index"][m], test["candidate_arc_id"][m]))
    prob = np.array([prob_by_key[k_] for k_ in keys]); y = test["y"][m]

    # 2. confusion matrix (candidate level, threshold 0.5)
    pr = (prob >= 0.5).astype(int)
    cm = np.array([[((pr == 0) & (y == 0)).sum(), ((pr == 1) & (y == 0)).sum()],
                   [((pr == 0) & (y == 1)).sum(), ((pr == 1) & (y == 1)).sum()]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred: wrong arc", "pred: true arc"]); ax.set_yticks([0, 1], ["wrong arc", "true arc"])
    ax.set_title("student_test confusion matrix (candidates)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=130); plt.close(fig)

    # 3. coefficients (standardised features)
    coef = read_csv(R / "coefficients.csv")
    coef.sort(key=lambda r: float(r["coefficient"]))
    fig, ax = plt.subplots(figsize=(7, 7))
    vals = [float(r["coefficient"]) for r in coef]
    ax.barh([r["feature"] for r in coef], vals, color=["#c0392b" if v < 0 else "#2471a3" for v in vals])
    ax.axvline(0, color="k", lw=.8); ax.set_title("Standardised coefficients (log-odds per 1 SD)")
    fig.tight_layout(); fig.savefig(out / "coefficients.png", dpi=130); plt.close(fig)

    # 4. calibration (reliability) on student_test
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(prob, bins) - 1, 0, 9)
    mp = [prob[idx == b].mean() for b in range(10) if (idx == b).any()]
    fy = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect"); ax.plot(mp, fy, "o-", label="Logistic Regression")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed share of true arcs"); ax.legend()
    ax.set_title(f"Calibration on student_test (log loss {summary['log_loss']:.3f})")
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)

    # 5. Newton convergence of the final fit (numpy backend only)
    if (R / "convergence_history.csv").exists():
        h = read_csv(R / "convergence_history.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy([int(r["iteration"]) for r in h], [float(r["grad_max"]) for r in h], "o-")
        ax.set_xlabel("Newton iteration"); ax.set_ylabel("max |gradient|"); ax.grid(alpha=.3)
        ax.set_title("Final fit convergence (objective is convex)")
        fig.tight_layout(); fig.savefig(out / "convergence.png", dpi=130); plt.close(fig)

    # 6. class-weight ablation (both runs on the same folds)
    abl = R / "ablation_balanced_weights" / "summary.json"
    if abl.exists():
        b = json.loads(abl.read_text())
        names = ["CV Macro F1", "test Macro F1", "test Top-1", "test log loss"]
        keys_ = ["cv_macro_f1", "student_test_macro_f1", "student_test_top1_accuracy", "log_loss"]
        x = np.arange(len(names)); w = .38
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w / 2, [summary[k_] for k_ in keys_], w, label="unweighted (selected)")
        ax.bar(x + w / 2, [b[k_] for k_ in keys_], w, label="balanced sample weights")
        ax.set_xticks(x, names); ax.legend(); ax.set_title("Class-weighting choice (selected by CV Macro F1)")
        fig.tight_layout(); fig.savefig(out / "class_weight_ablation.png", dpi=130); plt.close(fig)

    # 7. Folium map: GPS, nearby road network, reference route, predicted top-1 arcs
    import folium
    from pathsnap.data import find_dataset_root, load_trip
    trip = load_trip(find_dataset_root(a.dataset), "student_test", a.map_trip, include_reference=True)
    nodes, arcs = trip.nodes, trip.arcs
    rows = [r for r in pred if r["trip_id"] == a.map_trip]
    best = {}
    for r in rows:
        o = int(r["observation_index"]); pr_ = float(r["probability"])
        if o not in best or pr_ > best[o][1]:
            best[o] = (int(r["candidate_arc_id"]), pr_)
    cand_arcs = {int(r["candidate_arc_id"]) for r in rows}
    fmap = folium.Map(location=[float(np.mean(trip.track[:, 1])), float(np.mean(trip.track[:, 0]))], zoom_start=15,
                      tiles="OpenStreetMap")
    def layer(ids, name, color, weight, show=True):
        fg = folium.FeatureGroup(name=name, show=show)
        for aid in ids:
            f, t = arcs[aid]
            folium.PolyLine([(nodes[f, 1], nodes[f, 0]), (nodes[t, 1], nodes[t, 0])], color=color, weight=weight,
                            opacity=.85).add_to(fg)
        fg.add_to(fmap)
    layer(sorted(cand_arcs), "Road network (candidate arcs)", "#9e9e9e", 2)
    layer(trip.route, "Reference route", "#1b9e77", 7)
    layer(sorted({v[0] for v in best.values()}), "Predicted route (LR top-1)", "#d95f02", 3)
    fg = folium.FeatureGroup(name="GPS observations")
    for lon, lat, t in trip.track:
        folium.CircleMarker((lat, lon), radius=2, color="#1f78b4", fill=True, opacity=.8).add_to(fg)
    fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;'
              'border:1px solid #999;font-size:13px">'
              f'<b>student_test {a.map_trip} — Logistic Regression</b><br>'
              '<span style="color:#1f78b4">&#9679;</span> GPS observations<br>'
              '<span style="color:#1b9e77">&#9644;</span> Reference route<br>'
              '<span style="color:#d95f02">&#9644;</span> Predicted (top-1 per observation)<br>'
              '<span style="color:#9e9e9e">&#9644;</span> Candidate road arcs</div>')
    fmap.get_root().html.add_child(folium.Element(legend))
    fmap.save(str(out / f"student_test_{a.map_trip}_map.html"))
    print("plots written to", out)


if __name__ == "__main__":
    main()
