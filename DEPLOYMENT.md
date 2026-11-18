# PathSnap two-laptop deployment

**Status:** the backend and UI were tested on one laptop (both servers on the same machine, talking over HTTP).
The physical two-laptop setup has **not** been tested yet; follow these steps on the day.

| Laptop | Runs | Needs |
| --- | --- | --- |
| **Laptop 1 (backend)** | FastAPI + HGB model + HMM/Viterbi + road networks + evaluation trips | this repository, `requirements-api.txt`, the trip folders |
| **Laptop 2 (UI)** | evaluator web UI (Folium maps) | this repository (only `ui/` is used), `requirements-ui.txt` |

Laptop 2 needs **no model, no dataset and no ML code**. It only calls Laptop 1 over HTTP.

## 1. Laptop 1: backend

```powershell
git clone https://github.com/aditya-130906/pathsnap.git
cd pathsnap
python -m pip install -r requirements-api.txt
```

Put the evaluation trips in one folder. Each trip is a sub-folder named by its ID that holds its `.track`, `.nodes` and
`.arcs` (and `.route`, which is used **only** for server-side scoring):

```text
C:\pathsnap_trips\00000052\00000052.track   .nodes   .arcs   .route
C:\pathsnap_trips\00000091\...
```

Start the backend (listens on all network interfaces):

```powershell
$env:PATHSNAP_TRIPS_DIR = "C:\pathsnap_trips"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Keep this window visible: every request from Laptop 2 is logged here (`POST /predict from 192.168.x.y -> 200`, plus a
`PREDICT trip=...` line with the accuracies). The evaluators will ask to see this log.

Find Laptop 1's LAN IP address:

```powershell
ipconfig      # "IPv4 Address" of the Wi-Fi adapter, e.g. 192.168.1.23
```

Check locally: open `http://localhost:8000/health` and `http://localhost:8000/trips` in a browser on Laptop 1.

Optional settings: `PATHSNAP_MODEL` (path to a different HGB joblib bundle), `PATHSNAP_CORS_ORIGINS`
(comma-separated, default `*`).

## 2. Laptop 2: evaluator UI

```powershell
git clone https://github.com/aditya-130906/pathsnap.git
cd pathsnap
python -m pip install -r requirements-ui.txt
$env:PATHSNAP_API_URL = "http://192.168.1.23:8000"      # Laptop 1's IP from ipconfig
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8501
```

Open `http://localhost:8501` on Laptop 2. The backend URL can also be typed into the page and changed at any time.

First check from Laptop 2's browser: `http://192.168.1.23:8000/health` must return JSON.

## 3. Network and firewall checklist

1. Both laptops on the **same Wi-Fi** (a phone hotspot works well; campus Wi-Fi often isolates clients).
2. On Laptop 1, allow Python through Windows Defender Firewall for **Private** networks when Windows asks, or run once
   in an administrator PowerShell:
   `New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
3. Mark the Wi-Fi as a **Private** network on Laptop 1 (Settings → Network → Wi-Fi → network profile).
4. If `/health` works on Laptop 1 but not from Laptop 2: firewall, wrong IP, different network, or client isolation.
5. `ping 192.168.1.23` from Laptop 2 should answer (ICMP may be blocked even when HTTP works).

## 4. Five-minute demo script

1. Laptop 1: show the backend window running (`Application startup complete`) and `/health` in a browser.
2. Laptop 2: open `http://localhost:8501`, enter Laptop 1's URL, press **Connect** → "Connected" and the trip list.
3. Pick any trip, press **Run map matching**.
4. Point to Laptop 1's console: the `POST /predict from <Laptop 2 IP>` line and the `PREDICT trip=...` line.
5. On Laptop 2: show the metrics (Top-1 HGB-only vs HGB+Viterbi, route F1, continuity, timings) and the map:
   blue dots = raw GPS, green line = reconstructed route, toggle "HGB-only matched arcs" (orange) to show the
   difference Viterbi makes. Use the layer control and zoom.
6. Repeat with a second trip to show it is not hardcoded.
7. If asked: ground truth stays on Laptop 1 (the UI receives only metrics); evaluator trips were never used in
   training or tuning.

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| UI says "Cannot reach backend" | check IP/port, firewall (step 3), same network |
| `/trips` is empty | `PATHSNAP_TRIPS_DIR` must point to the folder that contains the trip sub-folders; call `POST /reload` or restart |
| 404 for a trip | the trip folder/files are missing or misnamed (`<id>/<id>.track` etc.) |
| Slow first request | long trips (several thousand GPS points) take tens of seconds; the UI waits up to 300 s (`PATHSNAP_API_TIMEOUT`) |
| scikit-learn cannot load on Laptop 1 | the backend falls back automatically to the verified numpy HGB evaluator (`inference_backend: numpy` in `/health`) |
