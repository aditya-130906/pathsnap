"""Route visualisations for the report: raw GPS vs HGB-only vs HGB + Viterbi vs reference route.

Usage: python tools/make_route_maps.py --dataset "data/Student folder" --trips 00000077 00000091 00000009
Writes outputs/hmm_viterbi/maps/<trip>.html (Folium, layer control) and <trip>.png (static, for the PDF).
The reference route is drawn here for offline analysis only; the evaluator UI never receives it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap_app.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--trips", nargs="+", required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "hmm_viterbi" / "maps")
    a = ap.parse_args()
    tdir = next(p for p in a.dataset.rglob("student_test") if p.is_dir())
    eng = Engine(tdir)
    a.output.mkdir(parents=True, exist_ok=True)
    for tid in a.trips:
        r = eng.predict(tid)
        folder = eng.trips[tid]
        nodes = np.loadtxt(folder / f"{tid}.nodes", ndmin=2); arcs = np.loadtxt(folder / f"{tid}.arcs", ndmin=2, dtype=int)
        ref = np.loadtxt(folder / f"{tid}.route", ndmin=1, dtype=int)
        ref_poly = [[[nodes[i, 1], nodes[i, 0]] for i in arcs[x]] for x in ref]
        gps = np.array(r["gps"]); ev = r["evaluation"]
        m = folium.Map(tiles="OpenStreetMap"); m.fit_bounds([[gps[:, 0].min(), gps[:, 1].min()], [gps[:, 0].max(), gps[:, 1].max()]])
        layers = [("Reference route (analysis only)", ref_poly, "#555555", 8, .45),
                  ("HGB-only matched arcs", r["hgb_only_route"], "#e66101", 3, .9),
                  ("HGB + Viterbi reconstructed route", r["reconstructed_route"], "#1b9e77", 4, .95)]
        for name, polys, col, w, op in layers:
            fg = folium.FeatureGroup(name=name)
            for s in polys:
                folium.PolyLine(s, color=col, weight=w, opacity=op).add_to(fg)
            fg.add_to(m)
        fg = folium.FeatureGroup(name="Raw GPS")
        for la, lo, _ in gps[:: max(1, len(gps) // 3000)]:
            folium.CircleMarker((la, lo), radius=2, color="#1f78b4", fill=True, weight=1).add_to(fg)
        fg.add_to(m); folium.LayerControl(collapsed=False).add_to(m)
        m.get_root().html.add_child(folium.Element(
            f"<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:#fff;padding:8px;border:1px solid #999;font:13px sans-serif'>"
            f"<b>student_test {tid}</b><br>Top-1: HGB {ev['hgb_top1_accuracy']:.1%} → HGB+Viterbi {ev['hgb_viterbi_top1_accuracy']:.1%}<br>"
            "<span style='color:#1f78b4'>&#9679;</span> raw GPS <span style='color:#e66101'>&#9644;</span> HGB-only "
            "<span style='color:#1b9e77'>&#9644;</span> HGB+Viterbi <span style='color:#555'>&#9644;</span> reference</div>"))
        m.save(str(a.output / f"{tid}.html"))
        fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for k, (title, polys, col) in enumerate([("HGB only", r["hgb_only_route"], "#e66101"),
                                                 ("HGB + HMM/Viterbi", r["reconstructed_route"], "#1b9e77")]):
            for s in ref_poly:
                ax[k].plot([p[1] for p in s], [p[0] for p in s], color="#999999", lw=5, alpha=.5)
            for s in polys:
                ax[k].plot([p[1] for p in s], [p[0] for p in s], color=col, lw=1.6)
            ax[k].scatter(gps[:, 1], gps[:, 0], s=2, color="#1f78b4", alpha=.5)
            acc = ev["hgb_top1_accuracy"] if k == 0 else ev["hgb_viterbi_top1_accuracy"]
            f1 = ev["hgb_only_route_f1"] if k == 0 else ev["hgb_viterbi_route_f1"]
            ax[k].set_title(f"{title}: Top-1 {acc:.1%}, route F1 {f1:.3f}")
            ax[k].set_xlabel("longitude"); ax[k].set_aspect(1 / np.cos(np.radians(gps[:, 0].mean())))
        ax[0].set_ylabel("latitude")
        fig.suptitle(f"student_test {tid}: grey = reference route, blue = raw GPS")
        fig.tight_layout(); fig.savefig(a.output / f"{tid}.png", dpi=120); plt.close(fig)
        print(tid, {k: round(v, 4) for k, v in ev.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
