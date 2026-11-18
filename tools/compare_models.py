"""Five-model comparison table + plot, read from each model's committed summary files.

Usage: python tools/compare_models.py   -> outputs/comparison/model_comparison.{csv,md,png}

The five models were trained by different team members on three candidate/label pipelines, so the table records
the pipeline and the Top-1 denominator of every row. Only rows that share a pipeline are strictly comparable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"

PIPELINES = {
    "phase1": ("Phase 1 (scripts/run_phase1.py): <=8 candidates within 80 m. Random Forest was trained before the "
               "Phase 1 label update (nearest-route-arc labels, trips 33/54 excluded); Gaussian NB and KNN after it "
               "(DP route-alignment labels)"),
    "pathsnap": "pathsnap package: <=12 candidates within 50 m, DP alignment labels (valid <= 50 m)",
    "hgb": "hgb_pathsnap: top-8 within 150 m, nearest-route-arc proxy labels (eligible <= 30 m)",
}
MODELS = [
    # name, owner, summary_row path, pipeline, top-1 denominator
    ("Logistic Regression", "Akshansh", "logistic_regression/summary_row.csv", "pathsnap", "19,133 observations with valid labels"),
    ("Random Forest", "Akshat", "random_forest/summary_row.csv", "phase1", "all 19,160 observations"),
    ("Gaussian Naive Bayes", "Aditya", "gaussian_nb/summary_row.csv", "phase1", "all 19,160 observations"),
    ("K-Nearest Neighbours", "-", "knn/summary_row.csv", "phase1", "all 19,160 observations"),
    ("Histogram Gradient Boosting (selected)", "Aayaan", "histogram_gradient_boosting/summary_row.csv", "hgb",
     "19,018 eligible observations"),
]
COLS = ["cv_macro_f1", "student_test_macro_f1", "student_test_top1_accuracy", "precision", "recall",
        "candidate_miss_rate", "log_loss", "training_time_s", "inference_time_s"]


def main():
    rows = []
    for name, owner, path, pipe, denom in MODELS:
        with open(O / path, encoding="utf-8") as f:
            r = next(csv.DictReader(f))
        rows.append({"model": name, "owner": owner, "pipeline": pipe, "top1_denominator": denom,
                     **{c: float(r[c]) for c in COLS}})
    out = O / "comparison"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    md = ["| Model | Pipeline | CV Macro F1 | Test Macro F1 | Test Top-1 | Precision | Recall | Miss rate | Log loss | Train s | Inference s |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['model']} | {r['pipeline']} | {r['cv_macro_f1']:.4f} | {r['student_test_macro_f1']:.4f} | "
                  f"{r['student_test_top1_accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                  f"{r['candidate_miss_rate']:.4f} | {r['log_loss']:.4f} | {r['training_time_s']:.1f} | {r['inference_time_s']:.2f} |")
    md.append("")
    md += [f"- **{k}**: {v}" for k, v in PIPELINES.items()]
    md += [f"- Top-1 denominator, {r['model']}: {r['top1_denominator']}" for r in rows]
    (out / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    names = [r["model"].replace(" (selected)", "*") for r in rows]
    colors = {"phase1": "#4e79a7", "pathsnap": "#59a14f", "hgb": "#e15759"}
    c = [colors[r["pipeline"]] for r in rows]
    for i, (key, title) in enumerate([("cv_macro_f1", "CV Macro F1"), ("student_test_top1_accuracy", "student_test Top-1"),
                                      ("log_loss", "student_test log loss (lower is better)")]):
        ax[i].barh(names, [r[key] for r in rows], color=c); ax[i].set_title(title); ax[i].invert_yaxis()
        for y_, r in enumerate(rows):
            ax[i].text(r[key], y_, f" {r[key]:.3f}", va="center", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    fig.legend(handles, list(colors), loc="lower center", ncol=3, title="candidate/label pipeline")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out / "model_comparison.png", dpi=130)
    print((out / "model_comparison.md").read_text())


if __name__ == "__main__":
    main()
