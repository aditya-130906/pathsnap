"""PathSnap evaluator UI (Laptop 2). Talks to the FastAPI backend on Laptop 1 over the LAN and draws Folium maps.

Start (repository root on Laptop 2):
    $env:PATHSNAP_API_URL = "http://<LAPTOP1-IP>:8000"
    uvicorn ui.app:app --host 0.0.0.0 --port 8501
Then open http://localhost:8501 in the browser on Laptop 2. The API URL can also be changed on the page.
Needs only: fastapi, uvicorn, requests, folium (no model, no dataset, no ML code on this laptop).
"""

from __future__ import annotations

import html
import os

import folium
import requests
from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse

app = FastAPI(title="PathSnap UI")
DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "")
TIMEOUT = float(os.environ.get("PATHSNAP_API_TIMEOUT", "300"))

STYLE = """
<style>
 :root{--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--muted:#5b6475;--line:#d9dde5;--ok:#1b7f4b;--bad:#b42318;--accent:#2457c5}
 *{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}
 header{padding:14px 20px;background:#1d2433;color:#fff} header h1{margin:0;font-size:20px} header p{margin:4px 0 0;color:#c7cdd8;font-size:13px}
 main{max-width:1200px;margin:0 auto;padding:16px} .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
 label{font-size:13px;color:var(--muted);display:block;margin-bottom:4px} input,select{font-size:15px;padding:7px;border:1px solid var(--line);border-radius:6px;min-width:260px}
 button{font-size:15px;padding:8px 16px;border:0;border-radius:6px;background:var(--accent);color:#fff;cursor:pointer} .row{display:flex;gap:12px;flex-wrap:wrap;align-items:end}
 .ok{color:var(--ok);font-weight:600} .bad{color:var(--bad);font-weight:600} table{border-collapse:collapse;width:100%;font-size:14px}
 td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left} th{color:var(--muted);font-weight:600}
 .map iframe{width:100%;height:620px;border:1px solid var(--line);border-radius:8px} .muted{color:var(--muted);font-size:13px}
</style>"""


def page(body: str) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                        f"<title>PathSnap Evaluator</title>{STYLE}</head><body><header><h1>PathSnap — Map Matching Evaluator</h1>"
                        f"<p>GPS trajectory → HGB candidate scoring → HMM/Viterbi route reconstruction</p></header><main>{body}</main></body></html>")


def api_form(api: str, status: str = "") -> str:
    return (f"<div class='card'><form method='get' action='/' class='row'><div><label>Backend URL (Laptop 1)</label>"
            f"<input name='api' value='{html.escape(api)}' placeholder='http://192.168.x.x:8000'></div>"
            f"<button>Connect</button></form><p>{status}</p></div>")


@app.get("/", response_class=HTMLResponse)
def home(api: str = Query(DEFAULT_API)):
    api = api.rstrip("/")
    if not api:
        return page(api_form(api, "<span class='muted'>Enter the backend address shown on Laptop 1.</span>"))
    try:
        h = requests.get(f"{api}/health", timeout=10).json()
        trips = requests.get(f"{api}/trips", timeout=20).json()["trips"]
    except Exception as e:
        return page(api_form(api, f"<span class='bad'>Cannot reach backend: {html.escape(str(e))}</span>"))
    status = (f"<span class='ok'>Connected</span> · model {h['model']} ({h['n_features']} features, "
              f"{h['inference_backend']} inference) · {len(trips)} trips available")
    opts = "".join(f"<option>{t}</option>" for t in trips)
    form = (f"<div class='card'><form method='post' action='/run' class='row'><input type='hidden' name='api' value='{html.escape(api)}'>"
            f"<div><label>Trip</label><select name='trip_id'>{opts}</select></div><button>Run map matching</button></form>"
            f"<p class='muted'>Processing takes a few seconds to about a minute depending on trip length.</p></div>")
    return page(api_form(api, status) + form)


def build_map(res: dict) -> str:
    gps = res["gps"]
    lats = [p[0] for p in gps]; lons = [p[1] for p in gps]
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    fg_gps = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(gps) // 4000)
    for la, lo, _t in gps[::step]:
        folium.CircleMarker((la, lo), radius=2, color="#1f78b4", fill=True, fill_opacity=.8, weight=1).add_to(fg_gps)
    fg_hgb = folium.FeatureGroup(name="HGB-only matched arcs", show=False)
    for seg in res["hgb_only_route"]:
        folium.PolyLine(seg, color="#e66101", weight=3, opacity=.8).add_to(fg_hgb)
    fg_route = folium.FeatureGroup(name="Reconstructed route (HGB + Viterbi)")
    for seg in res["reconstructed_route"]:
        folium.PolyLine(seg, color="#1b9e77", weight=6, opacity=.85).add_to(fg_route)
    for fg in (fg_route, fg_hgb, fg_gps):
        fg.add_to(m)
    folium.Marker((lats[0], lons[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((lats[-1], lons[-1]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:#fff;padding:8px 10px;border:1px solid #999;"
        "font:13px system-ui'><b>Legend</b><br><span style='color:#1f78b4'>&#9679;</span> Raw GPS"
        "<br><span style='color:#1b9e77'>&#9644;</span> Reconstructed route (HGB+Viterbi)"
        "<br><span style='color:#e66101'>&#9644;</span> HGB-only arcs (toggle)</div>"))
    return m._repr_html_()


@app.post("/run", response_class=HTMLResponse)
def run(api: str = Form(...), trip_id: str = Form(...)):
    api = api.rstrip("/")
    try:
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=TIMEOUT)
    except Exception as e:
        return page(api_form(api, f"<span class='bad'>Request failed: {html.escape(str(e))}</span>"))
    if r.status_code != 200:
        return page(api_form(api, f"<span class='bad'>Backend error {r.status_code}: {html.escape(r.text[:300])}</span>"))
    res = r.json()
    ev = res.get("evaluation")
    rows = [("Trip", res["trip_id"]), ("GPS observations", f"{res['n_observations']:,}"),
            ("Candidate arcs scored", f"{res['n_candidates']:,}"),
            ("Route continuity: HGB-only → HGB+Viterbi",
             f"{res['route_continuity']['hgb_only']:.3f} → {res['route_continuity']['hgb_viterbi']:.3f}"),
            ("Processing time (s)", ", ".join(f"{k} {v}" for k, v in res["timings_s"].items()))]
    if ev:
        rows += [("Top-1 accuracy: HGB-only", f"{ev['hgb_top1_accuracy']:.2%}"),
                 ("Top-1 accuracy: HGB + Viterbi", f"<b>{ev['hgb_viterbi_top1_accuracy']:.2%}</b>"),
                 ("Scored observations", f"{ev['eligible_observations']:,} ({ev['ineligible_observations']} without a usable label)"),
                 ("Route F1 (length-weighted): HGB-only → HGB+Viterbi",
                  f"{ev['hgb_only_route_f1']:.3f} → {ev['hgb_viterbi_route_f1']:.3f}")]
    else:
        rows.append(("Accuracy", "<span class='muted'>no reference route on the server for this trip</span>"))
    table = "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"
    body = (api_form(api, f"<span class='ok'>Processed trip {trip_id} successfully</span> · <a href='/?api={html.escape(api)}'>choose another trip</a>")
            + f"<div class='card'>{table}</div><div class='card map'>{build_map(res)}</div>")
    return page(body)
