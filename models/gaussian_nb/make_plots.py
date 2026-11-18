"""Report plots and a Folium map for the Gaussian Naive Bayes results (from saved outputs only, no retraining).

Usage (repository root):
    python models/gaussian_nb/make_plots.py --results outputs/gaussian_nb --dataset "data/Student folder" --map-trip 00000052

Reads cv_folds.csv, summary.json, student_test_predictions.csv (which carries the evaluation label) and the
fitted model bundle gaussian_nb.pkl. Writes to <results>/plots/: cv_folds.png, confusion_matrix.png,
calibration.png, probability_histogram.png, class_separation.png and student_test_<trip>_map.html.
Needs numpy, matplotlib and folium only (pandas is not required).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class _Bundle:
    """Stand-in for the classes pickled from train_gnb.py; only their fitted attributes are needed."""


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("GaussianNaiveBayes", "NumericPreprocessor"):
            return _Bundle
        return super().find_class(module, name)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "outputs" / "gaussian_nb")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--map-trip", default="00000052")
    a = p.parse_args()
    R = a.results; out = R / "plots"; out.mkdir(parents=True, exist_ok=True)
    s = json.loads((R / "summary.json").read_text())

    # 1. CV folds (validation only: train-fold scores were not logged by train_gnb.py)
    folds = read_csv(R / "cv_folds.csv")
    k = [int(f["fold"]) for f in folds]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(k, [float(f["macro_f1"]) for f in folds], "o-", label="validation Macro F1")
    ax[0].plot(k, [float(f["top1_accuracy"]) for f in folds], "s--", label="validation Top-1")
    ax[0].set_xticks(k); ax[0].set_xlabel("fold (contiguous trip blocks)"); ax[0].set_ylim(0.5, 1.0)
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"Gaussian NB CV: Macro F1 {s['cv_macro_f1']:.3f} ± {s['cv_macro_f1_std']:.3f}")
    ax[1].bar(k, [float(f["log_loss"]) for f in folds], color="#c0392b")
    ax[1].set_xticks(k); ax[1].set_xlabel("fold"); ax[1].set_title("validation log loss per fold")
    fig.tight_layout(); fig.savefig(out / "cv_folds.png", dpi=130); plt.close(fig)

    pred = read_csv(R / "student_test_predictions.csv")
    y = np.array([int(r["label"]) for r in pred]); prob = np.array([float(r["candidate_probability"]) for r in pred])

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

    # 3. calibration (reliability)
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(prob, bins) - 1, 0, 9)
    have = [b for b in range(10) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect")
    ax.plot([prob[idx == b].mean() for b in have], [y[idx == b].mean() for b in have], "o-", label="Gaussian NB")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed share of true arcs"); ax.legend()
    ax.set_title(f"Calibration on student_test (log loss {s['log_loss']:.3f})")
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)

    # 4. probability histogram by true class (NB pushes probabilities to 0/1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(prob[y == 0], bins=50, alpha=.6, label="wrong arc", density=True)
    ax.hist(prob[y == 1], bins=50, alpha=.6, label="true arc", density=True)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Predicted probabilities by true class (student_test)")
    fig.tight_layout(); fig.savefig(out / "probability_histogram.png", dpi=130); plt.close(fig)

    # 5. class separation per feature from the fitted Gaussians (standardised feature space)
    with open(R / "gaussian_nb.pkl", "rb") as f:
        bundle = _Unpickler(f).load()
    m = bundle["model"]; feats = bundle["features"]
    sep = (m.mean_1 - m.mean_0) / np.sqrt((m.var_0 + m.var_1) / 2)
    order = np.argsort(sep)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([feats[i] for i in order], sep[order], color=["#c0392b" if v < 0 else "#2471a3" for v in sep[order]])
    ax.axvline(0, color="k", lw=.8)
    ax.set_title("Class separation per feature\n(true-arc mean − wrong-arc mean) / pooled SD")
    fig.tight_layout(); fig.savefig(out / "class_separation.png", dpi=130); plt.close(fig)
    with open(out / "class_separation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["feature", "mean_true_arc", "mean_wrong_arc", "var_true_arc",
                                                           "var_wrong_arc", "separation"])
        for i, n in enumerate(feats):
            w.writerow([n, m.mean_1[i], m.mean_0[i], m.var_1[i], m.var_0[i], sep[i]])

    # 6. Folium map: GPS, candidate road arcs, reference route, predicted top-1 arcs
    import folium
    from pathsnap.data import find_dataset_root, load_trip
    trip = load_trip(find_dataset_root(a.dataset), "student_test", a.map_trip, include_reference=True)
    nodes, arcs = trip.nodes, trip.arcs
    rows = [r for r in pred if r["trip_id"] == a.map_trip]
    top = {int(r["arc_id"]) for r in rows if r["candidate_rank_by_probability"] == "1"}
    cand = {int(r["arc_id"]) for r in rows}
    fmap = folium.Map(location=[float(np.mean(trip.track[:, 1])), float(np.mean(trip.track[:, 0]))], zoom_start=15,
                      tiles="OpenStreetMap")
    def layer(ids, name, color, weight):
        fg = folium.FeatureGroup(name=name)
        for aid in ids:
            f_, t_ = arcs[aid]
            folium.PolyLine([(nodes[f_, 1], nodes[f_, 0]), (nodes[t_, 1], nodes[t_, 0])], color=color, weight=weight,
                            opacity=.85).add_to(fg)
        fg.add_to(fmap)
    layer(sorted(cand), "Road network (candidate arcs)", "#9e9e9e", 2)
    layer(trip.route, "Reference route", "#1b9e77", 7)
    layer(sorted(top), "Predicted route (GNB top-1)", "#7570b3", 3)
    fg = folium.FeatureGroup(name="GPS observations")
    for lon, lat, _t in trip.track:
        folium.CircleMarker((lat, lon), radius=2, color="#1f78b4", fill=True, opacity=.8).add_to(fg)
    fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;'
        'border:1px solid #999;font-size:13px">'
        f'<b>student_test {a.map_trip} — Gaussian Naive Bayes</b><br>'
        '<span style="color:#1f78b4">&#9679;</span> GPS observations<br>'
        '<span style="color:#1b9e77">&#9644;</span> Reference route<br>'
        '<span style="color:#7570b3">&#9644;</span> Predicted (top-1 per observation)<br>'
        '<span style="color:#9e9e9e">&#9644;</span> Candidate road arcs</div>'))
    fmap.save(str(out / f"student_test_{a.map_trip}_map.html"))

    # numbers used by the README (written to file so nothing is retyped)
    obs_rows = {}
    for r in pred:
        obs_rows.setdefault((r["trip_id"], r["observation_id"]), []).append(r)
    stats = {"confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
             "share_probabilities_below_0.05_or_above_0.95": float(np.mean((prob < .05) | (prob > .95))),
             "student_test_observations_with_candidates": len(obs_rows),
             "cv_macro_f1_std_sample_ddof1": float(np.std([float(f["macro_f1"]) for f in folds], ddof=1))}
    (out / "plot_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=1)); print("plots written to", out)


if __name__ == "__main__":
    main()
