"""PathSnap FastAPI backend (Laptop 1).

Start (repository root):
    $env:PATHSNAP_TRIPS_DIR = "C:\\path\\to\\evaluation_trips"     # folder containing trip folders (e.g. 00000052/)
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health              status, model and HMM configuration
    GET  /trips               available trip IDs (folders found under PATHSNAP_TRIPS_DIR)
    POST /predict             body {"trip_id": "00000052"} -> full prediction + server-side evaluation
    GET  /results/{trip_id}   last prediction for that trip (cached in memory)
    POST /reload              rescan the trips folder
Every request is logged to this console (required for the evaluation).
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap_app.engine import Engine, TripNotFound, default_trips_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API", version="1.0")
origins = os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

engine = Engine(default_trips_dir())
results: dict[str, dict] = {}
log.info("engine ready: %s", {k: v for k, v in engine.info().items() if k != "feature_order"})


class PredictRequest(BaseModel):
    trip_id: str = Field(..., min_length=1, max_length=16, pattern=r"^\d+$", examples=["00000052"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %s (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.perf_counter() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", **engine.info()}


@app.get("/trips")
def trips():
    engine.refresh()
    return {"trips": sorted(engine.trips), "count": len(engine.trips)}


@app.post("/reload")
def reload_trips():
    engine.refresh()
    return {"count": len(engine.trips)}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        out = engine.predict(req.trip_id)
    except TripNotFound:
        raise HTTPException(404, f"trip {req.trip_id} not found in {engine.trips_dir}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    results[req.trip_id] = out
    ev = out["evaluation"] or {}
    log.info("PREDICT trip=%s obs=%d candidates=%d HGB_top1=%s HGB+Viterbi_top1=%s timings=%s", req.trip_id,
             out["n_observations"], out["n_candidates"], f"{ev.get('hgb_top1_accuracy', float('nan')):.4f}",
             f"{ev.get('hgb_viterbi_top1_accuracy', float('nan')):.4f}", out["timings_s"])
    return out


@app.get("/results/{trip_id}")
def result(trip_id: str):
    if trip_id not in results:
        raise HTTPException(404, "no result yet; POST /predict first")
    return results[trip_id]
