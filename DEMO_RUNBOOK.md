# SkyGuard-AI — Verified Demo Runbook

> Every command in this file was run successfully during wiring verification
> (2026-11-20, branch `skyguard-mamba`). If a step fails, stop and fix that
> step — do not skip ahead.
>
> **All data in this demo is simulated.** The seeded history comes from the
> repo's own `test_weather.csv` (flat synthetic values) and the live frames
> below are hand-sent test frames, not real sensor readings. Say so on stage.

## 0. One-time setup (first run on a machine only)

```bash
cd ~/workspace/skyguard-ai/upstream

# Python deps — system Python is externally managed, so use a venv.
# (This venv already exists at ~/workspace/skyguard-ai/venv.)
python3 -m venv ../venv
../venv/bin/pip install -r requirements.txt

# Frontend deps
npm install
```

## 1. Reset to a clean demo state

```bash
cd ~/workspace/skyguard-ai/upstream
# Stop any earlier backend/frontend first (see §7), then:
rm -f backend/data/skyguard.db backend/data/skyguard.db-wal backend/data/skyguard.db-shm
```

The backend recreates the SQLite database and seeds the admin account on
first startup. Deleting the DB is what clears incidents, telemetry and
stations from a previous rehearsal.

## 2. Start the backend (port 8000)

```bash
cd ~/workspace/skyguard-ai/upstream
DEMO_MODE=true WEATHER_POLL_ENABLED=false ../venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- `DEMO_MODE=true` — local demo mode: ephemeral JWT secret, local SQLite,
  and the admin account is seeded from the documented demo login (see §4).
  Never use demo mode for a real deployment.
- `WEATHER_POLL_ENABLED=false` — disables the Open-Meteo poller so the demo
  needs no internet and no live weather feed. The ESP32 ingest path used
  below does not depend on it.
- Leave this terminal running. Verify in a second terminal:

```bash
curl -s http://127.0.0.1:8000/api/v1/health
# {"status":"ONLINE","service":"SkyGuard-AI Backend","version":"2.1.0",
#  "database":"SQLite (WAL Mode)","demo_mode":true, ...}
```

## 3. Seed stations and load demo history

```bash
cd ~/workspace/skyguard-ai/upstream
DEMO_MODE=true ../venv/bin/python scripts/demo_checklist.py
```

Expected output (exit code 0):

```
Seeded 2 demo stations.
Loaded telemetry for 1 stations from test_weather.csv.
All 48 rows replayed successfully.   (x4: initial replay + 3 rehearsals)
Demo checklist completed successfully.
```

This seeds the two prototype stations and loads the deterministic history:

| Station | Name | Coordinates |
| --- | --- | --- |
| AWS-01 | Udupi | 13.34° N, 74.74° E |
| AWS-02 | Dakshina Kannada | 12.87° N, 74.88° E |

It also validates a database backup/restore round-trip. The script is
idempotent — replays after the first show `DUPLICATE`, which is expected.

> **Baseline quirk — read before the live demo.** `test_weather.csv` is
> 15,000 rows of one flat value (25.0 °C / 60 % / 1010.0 hPa), so after this
> step AWS-01's "normal" is exactly 25.0 °C. A live frame at realistic
> coastal values (e.g. 28.5 °C) is scored as a +3.5 °C drift
> (`LOCALIZED_ANOMALY_UNCONFIRMED`, fusion ≈ 0.95) and raises an incident
> before you inject anything. **For the live "normal" frames in §5, use
> 25.0 °C / 60 % / 1010.0 hPa** — those score `NORMAL` (fusion ≈ 0.047).
> AWS-02 has no CSV history, so it accepts realistic values as normal.
>
> **Jitter the values.** The pipeline flags a stuck sensor when the last
> four observations are exactly identical on temperature, humidity and
> pressure (`SENSOR_FLATLINE`). Never send the same exact values four times
> in a row — vary each frame by ±0.1–0.3 °C / ±1 % / ±0.2 hPa (see §5c).
> Verified: identical 25.0 °C frames were NORMAL at first, then flipped to
> `SENSOR_FLATLINE` once enough identical frames had accumulated; jittered
> frames stay NORMAL.

## 4. Start the frontend (port 5173)

```bash
cd ~/workspace/skyguard-ai/upstream
npm run dev
```

Open **http://localhost:5173/** in the browser. (Use `localhost`, not
`127.0.0.1` — in this VM Vite binds the IPv6 localhost and `127.0.0.1:5173`
does not answer.)

Wiring, as verified — no `.env` file needed:

- `src/utils/apiClient.js` uses `VITE_API_BASE_URL` when set, otherwise
  same-origin `/api/v1`.
- `vite.config.js` proxies `/api` to `http://127.0.0.1:8000`, so the dev
  frontend reaches the backend with no extra config.
- Backend CORS (`backend/app/config.py`, `ALLOWED_ORIGINS`) defaults to
  `http://localhost:5173`; a preflight `OPTIONS` from that origin returns
  `access-control-allow-origin: http://localhost:5173` with credentials.
- `npm run build` also passes (built in ~4 s) if you want a production
  bundle instead of the dev server.

### Demo login — where it is documented, not the password

- The login screen's **FOR SIH EVALUATORS** box, with a
  **Fill evaluator credentials** button:
  `src/components/auth/LoginScreen.jsx`, lines 102–116.
- Documented in `README.md`, "Login" paragraph (under the install block).
- The seed password is set in `backend/app/config.py`
  (`DEFAULT_ADMIN_PASSWORD`, `DEMO_MODE` default); an existing database is
  never re-seeded, so if login fails on an old DB, wipe it (§1) and restart.

Log in as **Central Admin** using that box. The dashboard (Command Center)
should show both stations; the Fleet Map shows them at the coordinates above.

## 5. Live telemetry — the exact frames to send

Get a token first (the password is the one shown in the login box in §4;
it is typed interactively and never written to a file):

```bash
read -s -p "Admin password (see login box, §4): " PW; echo
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/admin/login \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"admin\",\"password\":\"$PW\"}" | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
```

### 5a. Normal frames (both stations, ~1 minute before the spike)

```bash
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# AWS-01 must match the flat CSV baseline (25.0 °C, see §3 quirk);
# AWS-02 has no CSV history, so realistic coastal values stay NORMAL there.
curl -s -X POST http://127.0.0.1:8000/api/v1/telemetry/esp32/ingest \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"seq\": 1, \"device_id\": \"esp32-aws01-edge\", \"station_id\": \"AWS-01\", \"timestamp\": \"$TS\",
       \"sensors\": {\"temperature\": {\"value\": 25.0, \"unit\": \"°C\", \"wmo_flag\": 0},
                    \"humidity\": {\"value\": 60.0, \"unit\": \"%\", \"wmo_flag\": 0},
                    \"pressure\": {\"value\": 1010.0, \"unit\": \"hPa\", \"wmo_flag\": 0},
                    \"battery_v\": {\"value\": 12.6, \"unit\": \"V\"}},
       \"derived\": {}, \"edge_ai\": {\"classification\": \"NOMINAL\", \"anomaly_score\": 0.02}}"
echo
curl -s -X POST http://127.0.0.1:8000/api/v1/telemetry/esp32/ingest \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"seq\": 1, \"device_id\": \"esp32-aws02-edge\", \"station_id\": \"AWS-02\", \"timestamp\": \"$TS\",
       \"sensors\": {\"temperature\": {\"value\": 29.0, \"unit\": \"°C\", \"wmo_flag\": 0},
                    \"humidity\": {\"value\": 71.0, \"unit\": \"%\", \"wmo_flag\": 0},
                    \"pressure\": {\"value\": 1009.4, \"unit\": \"hPa\", \"wmo_flag\": 0},
                    \"battery_v\": {\"value\": 12.6, \"unit\": \"V\"}},
       \"derived\": {}, \"edge_ai\": {\"classification\": \"NOMINAL\", \"anomaly_score\": 0.02}}"
echo
```

Expected: HTTP 200, `"status": "ACCEPTED"`,
`cloud_ml.status: "NORMAL"`, fusion ≈ 0.047 for both stations.

> The ingest payload **must include a `timestamp`** (ISO 8601, within 5
> minutes of now) — without it the pipeline returns
> HTTP 422 `SOURCE_TIMESTAMP_REQUIRED`. The endpoint docstring in
> `backend/app/api/v1/telemetry.py` does not show this field.

Check the dashboard data landed:

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/stations/fleet/live | python3 -m json.tool | head -30
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/v1/stations/AWS-01/telemetry/stats
# {"success": true, "station_id": "AWS-01", "total_records": 15005+, ...}
```

### 5b. Fault — temperature spike on AWS-01

Send the same frame with temperature **45.0 °C** (fresh timestamp):

```bash
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
curl -s -X POST http://127.0.0.1:8000/api/v1/telemetry/esp32/ingest \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"seq\": 99, \"device_id\": \"esp32-aws01-edge\", \"station_id\": \"AWS-01\", \"timestamp\": \"$TS\",
       \"sensors\": {\"temperature\": {\"value\": 45.0, \"unit\": \"°C\", \"wmo_flag\": 0},
                    \"humidity\": {\"value\": 72.0, \"unit\": \"%\", \"wmo_flag\": 0},
                    \"pressure\": {\"value\": 1009.5, \"unit\": \"hPa\", \"wmo_flag\": 0},
                    \"battery_v\": {\"value\": 12.6, \"unit\": \"V\"}},
       \"derived\": {}, \"edge_ai\": {\"classification\": \"NOMINAL\", \"anomaly_score\": 0.02}}"
```

Expected: HTTP 200, `cloud_ml.status: "LOCALIZED_ANOMALY_UNCONFIRMED"`,
`root_cause: "THERMAL_SPIKE"`, fusion ≈ 0.68. An incident is raised
immediately — verify:

```bash
curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8000/api/v1/incidents?station_id=AWS-01"
# count 1: {"id": "INC-...", "variable": "core:THERMAL_SPIKE" | "temp:THERMAL_SPIKE",
#           "severity": "high", "status": "open", "fault_risk": ~0.68, ...}
```

The incident carries evidence: `reason_codes`
(`THERMAL_SPIKE`, `TEMPORAL_ANOMALY`, `PEERS_INSUFFICIENT`), a plain-language
`explanation`, `recommended_actions`, and the full assessment in
`evidence_data` (fetch one incident at
`GET /api/v1/incidents/{incident_id}`).

> **Do not demo fault injection from the Fault Lab view for this.**
> `POST /stations/{id}/faults/inject` (the Fault Lab button) only records an
> `active_fault` row; it is applied by `weather_service.reevaluate()`, which
> builds rows from cached Open-Meteo readings — with the poller off
> (`WEATHER_POLL_ENABLED=false`) it changes nothing, and the ESP32 ingest
> path never consults it (verified: a normal frame ingested while a SPIKE
> fault was "active" still scored NORMAL and raised no incident). Inject the
> fault **in the values you send through ingest**, as in §5b. Fixing the
> inject path is an open item for the coordinator, not part of this runbook.

### 5c. Recovery — three fresh normal frames auto-resolve the incident

Send three AWS-01 frames at ~25 °C / ~60 % / ~1010 hPa with **current**
timestamps (each a second apart — timestamps must be newer than the spike),
**jittering the values slightly each time** (identical values four in a
row trip the `SENSOR_FLATLINE` rule — see §3):

```bash
# Frame 1: 25.1 °C / 60 % / 1010.1 hPa
# Frame 2: 24.9 °C / 61 % / 1009.9 hPa
# Frame 3: 25.2 °C / 60 % / 1010.0 hPa
# (Same ingest call as §5a/§5b, one fresh timestamp per frame, seq 50/51/52.)
```

After the third, the open incident flips to `resolved` automatically
(`GET /api/v1/incidents?station_id=AWS-01` shows `"status": "resolved"`).
Acknowledging first from the Incidents view also works:
`POST /api/v1/incidents/{id}/adjudicate {"action": "acknowledge"}` → 200.

## 6. The 2–3 minute demo flow (what to click, in order)

1. **(0:00) Login** — open http://localhost:5173, Central Admin tab, click
   **Fill evaluator credentials**, log in.
2. **(0:20) Command Center + Fleet Map** — both stations live: AWS-01 Udupi
   and AWS-02 Dakshina Kannada, status NORMAL, latest readings shown.
   (Data behind this view: `GET /stations/fleet/live`.)
3. **(0:50) ESP32 Live Monitor** — show the latest AWS-01 frame.
4. **(1:10) Inject the spike** — run the §5b curl in a terminal (or the
   ESP32 sensor simulator, `ESP32/sensor_simulator.html`, pointed at
   `http://127.0.0.1:8000/api/v1/telemetry/esp32/ingest` — its default
   target is a hardcoded LAN address, change it first). Refresh the
   **Incidents** view: one open incident, severity high, root cause
   THERMAL_SPIKE.
5. **(1:50) Open the incident** — read the explanation and recommended
   actions out loud; click Acknowledge.
6. **(2:20) Recovery** — run the three §5c normal frames, refresh Incidents:
   the incident shows `resolved`. Close on the map: both stations NORMAL.

## 6b. What the audience sees when a fault is injected (decision + action)

Every surface below shows the same thing in plain words: **the decision,
what it means, and what the operator should do next** — the decision
banner sits at the top of the view, above the raw scores and evidence.

- **Fault Lab (the injection itself).** Click **Inject Fault on AWS-02**:
  the Fault Lab first records the fault, then sends the faulted readings
  through the pipeline (the inject endpoint alone only records a fault
  row — the readings are what the model scores). Within a few seconds an
  **INJECTION RESULT** panel appears: `DECISION: LOCAL SENSOR FAULT` (or
  `LOCAL SENSOR FAULT — UNCONFIRMED`), the one-line explanation, and
  **What to do next** with the recommended action and a **Review Incident**
  button. If the faulted readings score within normal bounds, the panel
  says so honestly ("No incident was raised…") instead of inventing one.
  A **Send 3 Normal Frames to Recover** button sends the §5c recovery
  frames in one click.
- **Incident Triage (sidebar).** The sidebar's **Incident Triage** nav item
  shows a red badge with the open-incident count, and the **Fleet Status**
  panel underneath it lists stations needing attention and each station's
  model status (IF ACTIVE/PENDING, Mamba ACTIVE/CANDIDATE/NOT REGISTERED)
  — all updated by the app's 5-second fleet sync, no reload.
- **Incidents table.** Each open row shows the decision headline and the
  suggested action, plus a button for that action — e.g. **Acknowledge**
  (or **Confirm Genuine** / **Flag Defect** when the reasoning suggests
  it). Clicking it calls `POST /incidents/{id}/adjudicate` and the row
  updates immediately. An acknowledged incident shows **Close Incident**.
- **Incident detail (click Review).** The top of the modal is a decision
  banner: the decision in plain words, the likely cause, the scores that
  matter (Hybrid with its IF/Mamba contributions, IF score vs threshold,
  Mamba score vs threshold and its largest prediction error), and
  **WHAT TO DO NEXT** from the backend's recommended action. Below it,
  the Model Prediction panel shows the same Mamba and hybrid numbers.
  The footer buttons are **Acknowledge**, **Confirm Genuine**,
  **Flag Defect**, and **Close Incident** — all wired to
  `POST /incidents/{id}/adjudicate` (`ACKNOWLEDGE` / `GENUINE` /
  `REJECT` / `ACCEPT`); the button the reasoning suggests is highlighted.
  Press **Acknowledge** live: the incident moves to `acknowledged` and
  the Close Incident button appears; press it and the incident closes.
- **Station HUD and ESP32 Live Monitor.** The same decision banner appears
  above the gauges/sensors whenever the station's latest assessment is
  not NORMAL, with a **Review open incident** button that jumps to the
  triage queue. (Fixed 2026-11-20: the live monitor was hardcoded to
  AWS-01, so an operator for any other station saw an empty monitor and
  no decision at all. It now follows the logged-in station — the admin
  sees AWS-01, an AWS-02 operator sees AWS-02.)
- **Command Center.** Each station row shows the decision in plain words,
  the recommended action, and the hybrid score under its status badge,
  plus a **Triage Incident** button when that station has an open incident.

**What to say on stage:** read the banner out loud — "The model decided
this is a local sensor fault, not weather, because only this station
shows the departure while its neighbours do not. The recommended action
is to inspect the sensor." Then press the suggested button and show the
status change. All demo data is simulated — say so.

## Mamba scoring (ML add-on — run after the main demo, §1–§6)

This turns on the ML layer for **AWS-02 (Dakshina Kannada)** so a live
frame carries a real Mamba score and a hybrid score. It is an add-on
because it changes AWS-02's baseline: after the backfill, AWS-02's
"normal" is the district's real late-monsoon values (≈24.6 °C,
≈91 %RH), not the 29.0 °C / 71 % frame in §5a. Run the main demo
first, then this section. Every command below was run successfully
2026-11-20.

**Do not do this for AWS-01 (Udupi).** Its IF training job fails the
repo's own promotion gate — holdout anomaly rate 0.153 > 0.15, the
only district of the five that fails — so no ACTIVE IF model exists
there and the pipeline shows a Mamba score but no hybrid score. That
is the gate doing its job; do not bypass it. AWS-02's gate passes
(0.095), which is why the ML demo uses AWS-02. See
`ml/MAMBA_README.md` for the numbers.

### M1. Backfill AWS-02 with the district's real hourly history

```bash
cd ~/workspace/skyguard-ai/upstream
DEMO_MODE=true ../venv/bin/python scripts/backfill_demo_history.py \
    --station AWS-02 --district "Dakshina Kannada"
# backfilled 2617 hourly rows for AWS-02 (Dakshina Kannada, 2617 h
# ending 1 h ago; 0 errors)
```

This loads the full Dakshina Kannada district series (the same file
the E1–E4 evaluation uses) with timestamps shifted to end 1 hour ago,
so the IF training job has ≥720 observations over ≥14 days and a live
frame has a 24 h Mamba window. **This is SIMULATED demo history** —
real district values, timestamps shifted to the present. Say so.

### M2. Train the Isolation Forest for AWS-02 (real training job)

```bash
# Get a token (same read -s prompt as §5; password is the one in the
# login box, §4):
read -s -p "Admin password (see login box, §4): " PW; echo
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/admin/login \
  -H 'Content-Type: application/json' \
  -d "{\"username\":\"admin\",\"password\":\"$PW\"}" | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")

curl -s -X POST http://127.0.0.1:8000/api/v1/stations/AWS-02/train \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{}'
# {"success": true, "job_id": 1, "station_id": "AWS-02", ...}
```

The training runs in the background (~1 min). Poll until it completes:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/stations/AWS-02/training-jobs/1/status"
# {"success": true, "job_id": 1, "station_id": "AWS-02",
#  "status": "COMPLETED", "current_stage": "Model Activated",
#  "progress": 100.0, "error_message": null}
```

(Replace `1` with the `job_id` from the train response.) If the
status is `FAILED`, read `error_message` — on this data it completes.

### M3. Register the Mamba model for AWS-02

```bash
cd ~/workspace/skyguard-ai/upstream
DEMO_MODE=true ../venv/bin/python scripts/register_mamba.py \
    --station AWS-02 --district "Dakshina Kannada" --epochs 15
# training Mamba for AWS-02 (Dakshina Kannada): 1500 train windows, ...
# sigma_e: T=0.281 C, RH=1.315 %RH, P=0.505 hPa
# using the station's ACTIVE Isolation Forest from the registry
# validation: 261 IF scores, 261 Mamba scores; tau_M (raw, 95th pct) = 2.30
# alpha=0.05 beta=0.95 tau_hybrid=0.934 val F1=0.418 FPR=0.050
# test temperature_C: Mamba MAE=0.229 vs persistence MAE=0.244
# registered AWS-02_MAMBA_v1_0 -> ml/models/AWS-02/AWS-02_MAMBA_v1_0.json (22s)
```

This trains the Mamba model on the district data (15 epochs, ~22 s),
calibrates the hybrid against the ACTIVE IF model from M2, and
registers it as a CANDIDATE Mamba artifact. The IF model stays the
single ACTIVE row.

### M4. Live frames with Mamba + hybrid scores

Use the district's latest values for the normal frame (24.6 °C,
91.5 %RH, 1009.8 hPa — the last backfilled row). Jitter each frame
slightly (±0.1–0.3 °C / ±1 % / ±0.2 hPa) as in §5c.

```bash
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
curl -s -X POST http://127.0.0.1:8000/api/v1/telemetry/esp32/ingest \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"seq\": 11, \"device_id\": \"esp32-aws02-edge\", \"station_id\": \"AWS-02\", \"timestamp\": \"$TS\",
       \"sensors\": {\"temperature\": {\"value\": 24.6, \"unit\": \"°C\", \"wmo_flag\": 0},
                    \"humidity\": {\"value\": 91.5, \"unit\": \"%\", \"wmo_flag\": 0},
                    \"pressure\": {\"value\": 1009.8, \"unit\": \"hPa\", \"wmo_flag\": 0},
                    \"battery_v\": {\"value\": 12.6, \"unit\": \"V\"}},
       \"derived\": {}, \"edge_ai\": {\"classification\": \"NOMINAL\", \"anomaly_score\": 0.02}}"
```

Expected (verified): HTTP 200, `"status": "ACCEPTED"`,
`cloud_ml.status: "NORMAL"`, fusion ≈ 0.047, and the assessment
carries **`mamba_score` ≈ 0.58, `hybrid_score` ≈ 0.14,
`mamba_errors` ≈ [0.16, 0.57, 0.05], `z_mamba` = 0.0**. (Read the
`assessment` object in the response; the stored assessment at
`GET /stations/AWS-02/assessments/latest` is a subset and does not
carry the Mamba fields.)

Now the spike — same frame with temperature **30.6 °C** (+6 °C,
fresh timestamp):

Expected (verified): HTTP 200, `"status": "ACCEPTED"`,
`cloud_ml.status: "LOCALIZED_ANOMALY_UNCONFIRMED"`,
`root_cause: "THERMAL_SPIKE"`, fusion ≈ 0.999, and the assessment
carries **`mamba_score` ≈ 21.9, `hybrid_score` ≈ 0.998,
`mamba_errors` ≈ [6.16, 0.57, 0.05], `z_mamba` = 1.0**. The Mamba
score jumps from 0.58 to 21.9 because the +6 °C spike is ~22× the
model's temperature error scale (σ_e = 0.28 °C).

An incident is raised. Its `evidence_data` carries the hybrid
evidence — fetch it at `GET /api/v1/incidents?station_id=AWS-02` and
open the incident:

- `evidence_data.final_assessment`: `mamba_score` 21.94,
  `hybrid_score` 0.998.
- `evidence_data.hybrid`: `S_IF` 0.962, `S_M` 1.0, `S_hybrid` 0.998,
  `alpha` 0.05, `beta` 0.95, `tau_hybrid` 0.934, `flag` true.
- `evidence_data.mamba_model`: `model_id` `AWS-02_MAMBA_v1_0`,
  `prediction` [24.48, 92.09, 1009.86], `is_anomaly` true.
- `reason_codes` include `ML_ANOMALY` alongside `THERMAL_SPIKE`.

### M5. What to say on stage (2 minutes)

1. "This is the ML layer. The Mamba model learned the station's
   hourly rhythm from 2,617 hours of its own district data."
2. Send the normal frame: "Mamba score 0.58 — the reading matches
   what the model expected. Hybrid 0.14 — no anomaly."
3. Send the spike: "Mamba score 21.9 — the temperature is 22 times
   the model's normal error. Hybrid 0.998. The incident carries the
   full evidence: both engines' scores, the weights, the threshold."
4. Open the incident and read the `hybrid` block. Do **not** claim
   the hybrid beats the IF model — on the measured E1–E4 evaluation
   it does not (E1 F1 0.560 vs E3 F1 0.493); the honest line is
   "the hybrid is the measured baseline we are improving on."
   See `ml/MAMBA_README.md` for the full numbers, including
   Belagavi, where Mamba is worse than persistence.

## Known limitations (do not present these as working)

- **No trained Isolation Forest model exists for these stations yet
  in the plain §1–§6 flow.** Assessments report
  `ml_model.status: "RULES_ONLY"` and readiness `COLD_START`
  (5 observations at verification time). Training requires at least
  720 clean distinct observations across 14 days
  (`POST /stations/{id}/train`). All scores shown are the rules/fusion
  path, not a fitted model. **The Mamba add-on (§M1–M5) fixes this for
  AWS-02** (backfilled history → trained IF → registered Mamba); AWS-01
  cannot be trained through the normal flow because its holdout
  anomaly rate (0.153) fails the promotion gate (see §M).
- **Fusion scores use default priors**, not field-fitted weights — the
  assessment itself says "Synthetic calibration only; not a field-validated
  probability." Use them to rank incidents, not as probabilities.
- **Peer comparison is incomplete with two stations.** The prototype radius
  is 60 km; Udupi and Dakshina Kannada are ~54 km apart, but a peer only
  counts once it has a recent assessment, so early incidents carry
  `PEERS_INSUFFICIENT` and stay `LOCALIZED_ANOMALY_UNCONFIRMED` rather than
  being confirmed as a sensor fault or a regional event.
- **Flat demo history.** `test_weather.csv` is a single repeated value, so
  the replayed history can itself read as a flatline; treat the seeded
  history as placeholder data, not a real baseline.
- **Fault Lab inject does not affect ESP32 ingest** (see §5b note).

## 7. Stop everything

```bash
# Frontend: Ctrl-C in its terminal, or:
pkill -f "[v]ite"
# Backend: Ctrl-C in its terminal, or:
pkill -f "[u]vicorn backend.app.main:app"
# Confirm nothing is left on the demo ports:
curl -s -o /dev/null -w "backend: %{http_code}\n" --max-time 2 http://127.0.0.1:8000/api/v1/health || echo "backend stopped"
curl -s -o /dev/null -w "frontend: %{http_code}\n" --max-time 2 http://localhost:5173/ || echo "frontend stopped"
```

Ports used: backend **8000**, frontend **5173**.
