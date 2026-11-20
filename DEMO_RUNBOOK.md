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
  below does not depend on it. **For detection to run on live Open-Meteo
  data, start the backend with `WEATHER_POLL_ENABLED=true` (the default
  when the variable is unset): the poller fetches current weather every
  20 s and the observations land in the same telemetry table the models
  train on — so training (M2) trains on the detection population, not on
  ESP32-only data. The demo keeps it off so the demo data stays
  deterministic.**
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
> (The seed rows are tagged synthetic and excluded from model *training* —
> see M2b. The 25.0 °C "normal" here is the live-*scoring* behaviour, not
> the training data.)
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

Log in as a **Station Operator** from the login screen's STATION OPERATOR tab
with username `demo_aws-01` (Udupi) or `demo_aws-02` (Dakshina Kannada) and
password `Station2026`. The operator sees only their own station: Live Cockpit
HUD, Hardware Telemetry, Edge Buffer & Sync, and the Fleet Map. (The seed used
to store the literal string "hash" as these accounts' password hash, which
`verify_password()` can never accept — no operator could log in at all.
`scripts/demo_checklist.py` now seeds a real hash of `Station2026`; an
existing database is never re-seeded, so on an old DB reset the two demo
passwords from Station Credentials, or wipe it (§1) and restart.)

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
   (Data behind this view: `GET /stations/fleet/live`.) The map's base tiles
   come from Esri's CDN and need internet; the pins, radius circle and peer
   vectors are drawn locally and work without tiles. Click a pin to open its
   popup — it stays open while the fleet refreshes every 5 s.
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
  Injecting the same fault twice re-applies the same offset from the
  clean baseline captured before the first inject — the temperature does
  NOT stack (a second SPIKE reads the same ~+8.5 °C, not +17 °C) — and
  the second inject reports **MERGED INTO OPEN INCIDENT** when the
  readings fold into the already-open incident instead of opening a
  duplicate. A **Send 3 Normal Frames to Recover** button sends the §5c
  recovery frames in one click: the frames are built from the station's
  own recent readings (not the spiked live values), so they read NORMAL
  and the incident auto-resolves to `resolved`. The recover button lives
  on an episode banner driven by the backend, so it survives navigating
  away from Fault Lab and back — and it always recovers the episode's
  own station, never another station's open incident.
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
repo's own promotion gate — holdout anomaly rate 0.157 > 0.15, the
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

### M2. Train the models for AWS-02 (one job: Isolation Forest + Mamba)

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

One Train click trains **both** models from the station's own stored
telemetry history — the same history the Isolation Forest job trains
on. That history is what detection runs on: the Open-Meteo poller
writes its observations into the same telemetry table (see the poller
note in §1), so when the poller is on, the models train on live
Open-Meteo data, not on ESP32-only data. The ESP32 only shows how
data is transmitted (simulator → ESP32 → backend); detection runs on
the Open-Meteo observations.

The training runs in the background. Measured on the demo machine with
2,617 backfilled rows (AWS-02): the Isolation Forest part takes **~3–4 s**
end to end (retraining with an incumbent model **~4 s** — a retrain used
to take ~28 s because the incumbent-agreement check re-loaded the model
artifact and ran SHAP for every holdout row; it now loads the artifact
once and scores directly — same decisions, verified identical metrics),
then Mamba trains for **~45 s** (15 epochs over ~1,500 train windows),
so the whole job finishes in **~53 s**. Per-phase breakdown of the IF
part of a retrain: history fetch 0.03 s, climatology + scrubbing
1.8 s, feature generation 0.6 s, forest fit 0.5 s, evaluation 0.4 s,
registration + activation 0.1 s. Udupi takes **~50 s** too (it took
~4 min before the seed-row fix — its training history was 17,617 rows
because the demo seed `test_weather.csv` contributed 15,000 flat hourly
rows; the seed rows are now tagged synthetic and excluded from training,
so Udupi trains on its 2,617 backfilled rows like AWS-02). It still
fails honestly on the IF gate (see M2b) but its Mamba model still
registers. To check the job
from the terminal instead of the UI, poll its status until it completes:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/stations/AWS-02/training-jobs/1/status"
# {"success": true, "job_id": 1, "station_id": "AWS-02",
#  "status": "COMPLETED", "current_stage": "Mamba Registered",
#  "progress": 100.0, "error_message": null}
```

(Replace `1` with the `job_id` from the train response.) If the
status is `FAILED`, read `error_message` — it names the engine that
failed (e.g. `Isolation Forest: ...; Mamba: ...`). A station needs at
least 720 hourly observations before Mamba can train; below that the
job fails honestly with `MAMBA_HISTORY_INSUFFICIENT` instead of
training on too little data.

### M2b. What the audience sees while training runs (live progress)

Do this from the UI, not curl: **Station Training Studio → select the
station → Train Model**. A "TRAINING PIPELINE" card appears under the
upload box and shows the eleven phases with a live state each — DONE
(green tick), IN PROGRESS (spinning, cyan border on the current phase),
PENDING (grey) — plus a progress bar, elapsed time, and the job number.
The same card is visible in **Model Governance** for the selected
station, so the audience never has to hunt for it. It polls the job
status every ~0.6 s; the phases advance as the backend records them:

1. Data Ingested → 2. Data Validated → 3. Data Preprocessed →
4. Features Generated → 5. Training Isolation Forest →
6. Model Evaluation → 7. Model Registered → 8. Model Activated →
9. Training Mamba → 10. Calibrating Mamba → 11. Mamba Registered.

The card stays on screen when the job finishes: COMPLETED reads
"MODELS TRAINED · 11/11 STEPS · 100% · Total ~53 s" (the ~53 s is
~4 s for the Isolation Forest plus ~45 s for Mamba on AWS-02's 2,617
rows; the UI round-trip adds a second or so). Clicking Train
while a job is already running does not start a second job — the
backend allows one running job per station and the UI attaches to the
running one and shows its progress instead.

**The honest failure (show it — it is a feature, not a bug).**
Training AWS-01 (Udupi) takes ~50 s and ends FAILED: the card turns
red at the end, "TRAINING FAILED · 10/11 STEPS · 90.9% · Total ~50s",
and prints the real reason in plain words: `Holdout anomaly rate
15.6% is above the 15% limit, so the model is saved as a candidate but
not activated. The validation data looked more unusual than the training
data; the station needs a longer or more varied history before the model
can activate.` — followed by the gate dict. The card also shows the
per-engine outcome: Isolation Forest trained but not activated, **Mamba
registered** — the gate failure does not block Mamba training. Say so on
stage: the gate did its job. The previous active model (if any) stays in
place; nothing is silently bypassed.

**Why Udupi fails — the measured story, not a guess.** Two different
numbers were seen for this station and they have two different causes;
do not mix them up on stage:
- **1.0 (old, fixed):** on the demo database before the fix, Udupi's
  holdout rate was 1.0 — every holdout row flagged. Cause: the 15,000
  flat rows of `test_weather.csv` (25.0 °C / 60 % / 1010.0 hPa) dominated
  the training history, so the climatology fit collapsed (robust sigma
  0.0) and every real row looked anomalous. Fixed: the seeder now tags
  those rows `synthetic:demo-seed` and the training history excludes
  them (plus a guard that drops long exactly-constant runs). The
  "TRAIN MODEL (17,665 RECORDS)" button counts all stored rows; training
  uses the 2,617 real backfilled observations.
- **0.157 (current, genuine):** with a clean history Udupi's rate is
  0.157 (15.7%) on the live demo database — 0.153 on the unshifted
  fixture. This is real weather, not a bug: the chronological holdout
  (the last 20% of the series) contains two regional monsoon-break dry
  spells, 24–25 Aug and 9–13 Sep 2025, when humidity sat near 83%
  against a ~90% monsoon baseline. 76 of the 80 flagged holdout rows
  are those two spells; without them the rate would be 0.011. Dakshina
  Kannada has the *same* two spells in its holdout and passes at 0.095 —
  Udupi is normally the more humid station (fit mean 90.0% vs 89.3%), so
  the same dip is a bigger deviation there. The gate is doing its job:
  a model must not call more than 15% of clean validation data anomalous.
  **What would change it:** a longer history — more monsoon seasons in
  the training data, so dry spells stop being unprecedented in
  validation. Do not relax the gate, the split, or the validation to
  force a green result.
- **Old un-wiped databases:** if you train on a database built before
  the fix (flat rows untagged), the constant-run guard still prevents
  the 1.0 blow-up — the rate comes back 0.052 and the model activates —
  but the surviving interleaved flat rows widen the fit, so treat the
  wiped-and-reseeded numbers above as the reference. Wipe per §1 before
  presenting.

If the server restarts mid-training, the interrupted job is marked
FAILED on startup ("Training was interrupted by a server restart")
instead of spinning forever — the card shows that reason too.

### M3. Register the Mamba model for AWS-02 (manual alternative — normally done by M2)

Mamba is now trained by the Train button in M2 — the command below is
only the manual alternative (e.g. to re-register with different
epochs, or to train from a district file instead of the station's own
telemetry). It trains the same way (15 epochs, ~22 s), calibrates the
hybrid against the ACTIVE IF model from M2, and registers it as a
CANDIDATE Mamba artifact. The IF model stays the single ACTIVE row.
Do not run it after M2 on the same station unless you want a second
Mamba version registered — the pipeline always uses the latest one.

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

### M6. Model drift tracking — show the model going stale (2 minutes)

Drift answers "does this station's model still match its recent
weather?" It is measured in **residual space** (how far recent
readings sit from what the model learned), not raw values — so a
change of season alone never triggers it. Full design, thresholds and
reasoning: `ml/DRIFT_README.md`.

**Where to look.** **Model Governance → select a station → MODEL
DRIFT panel**: a severity badge in plain words, a per-metric table
with thresholds, 90-day sparklines of PSI and residual mean, and
"Check Drift Now" / "Queue Retraining" buttons (admin only). **Command
Center** shows a `DRIFT: HIGH/MODERATE/LOW/NONE` chip on each station
row. Thresholds: PSI < 0.10 stable · 0.10–0.25 moderate · > 0.25
significant; residual mean shift < 0.3 ok · 0.3–0.7 warn · > 0.7
alert; anomaly rate > 3× the training contamination alerts.

**The demo station (AWS-03, HIGH on purpose).** On a wiped database
AWS-03 does not exist — build it after §3/M1/M2 with the script that
also trains AWS-02:

```bash
cd ~/workspace/skyguard-ai/upstream
DEMO_MODE=true ../venv/bin/python ~/workspace/skyguard-ai/verification/make_shifted_station.py
# station AWS-03 created; backfilled 2617 rows (last 504 h shifted +2.0 C)
# AWS-03: training job FAILED (holdout 35.5% > 15% — honest gate failure,
#   Mamba still registered); AWS-02: COMPLETED; drift computed for all three
```

AWS-03 is the real Dakshina Kannada series with a sustained +2.0 °C
offset on the last 21 days, so the recent window genuinely sits ~5σ
away from what its model learned. Open **Model Governance → AWS-03**:
both models read HIGH — Isolation Forest PSI 7.12, residual mean shift
2.58; Mamba PSI 0.78, error ratio 1.49 — with a red banner saying a
DATA_QUALITY incident was opened and a retraining candidate queued.
Open **Incident Triage**: the `drift:MODEL_DRIFT` incident is there.
Say on stage: "Drift never retrains, activates or deactivates anything
by itself — it queues a candidate and opens an incident, and a human
decides."

**AWS-01 and AWS-02 are HIGH too — and that is real, not a bug.**
Their recent 14 days contain the genuine monsoon-break dry spell of
9–13 Sep (humidity ~83% against a ~90% baseline): IF PSI ~2.4–2.5,
residual mean shift ~1.1, driven by humidity. If the audience asks why
two healthy stations show HIGH, show the per-parameter breakdown in
the panel (humidity moved, temperature and pressure did not) and say:
"Drift means investigate, not the sensor is broken — this one is
regional weather, and the panel shows you which parameter moved so
you can tell the difference."

**To recompute live on stage:** Model Governance → "Check Drift Now"
(admin). It takes a few seconds and re-renders the panel. A nightly
job also computes drift for every station (first run 5 min after the
backend starts, then daily); in DEMO_MODE it only records — it never
queues candidates or opens incidents by itself.

## QC calibration from history

The QC envelope (the station normal range the pipeline flags readings
against) used to be hand-set only — after a §1 reset both stations sit
on wide ADMIN_OVERRIDE values (15–38 °C, 0–100 %RH, 900–1100 hPa,
0 records). The calibration behind it, `calibrate_station_qc`, existed
but was never called. It is wired up now, two ways:

**Manual (demo this):** QC Physics Matrix (admin) → "Calibrate from
history". It derives the envelope from the station's own VALID history
(1st–99th percentiles per variable) and stamps
`HISTORICAL_P01_P99` with the record count. It needs at least **720
usable readings** — the same maturity as IF training — and refuses
plainly below that instead of inventing an envelope. Synthetic seed
rows (`grid_point LIKE 'synthetic:%'`) are excluded, so AWS-01's
15,000 flat seed rows can never dominate its calibration; only its
2,617 real rows count. A variable with no data (wind in the demo
history) gets blank bounds, never a 0–0 range. The audit ledger
records it as `QC_CALIBRATE_FROM_HISTORY`, distinct from a hand edit
(`QC_ENVELOPE_UPDATE` / `ADMIN_OVERRIDE`).

**Automatic:** the first training run for a station with no envelope
at all auto-calibrates from its history. An admin override (or an
earlier calibration) is never overwritten.

**Verified live 2026-11-20:** AWS-02 calibrated from the UI to
T 24.23–29.35 °C, RH 74.47–94.12 %, P 1003.92–1012.37 hPa from 2,617
readings (wind blank). A reading just outside it (30.4 °C, otherwise
median conditions) set z_qc = 1.0. Say on stage: "The normal range is
not a number I typed — it is the 1st to 99th percentile of this
station's own past readings." A hand edit after calibration overwrites
it and is recorded as a manual override; re-calibrating overwrites a
hand edit. The latest write wins, and the ledger shows which it was.

**Model Governance shows the registry truth now.** The model card and
the sidebar Fleet Status read the real model registry: AWS-02 shows
its ACTIVE model (AWS-02_IF_v2.7221a4d39cbb) and "IF ACTIVE"; AWS-01
and AWS-03 show "IF CANDIDATE" — their IF models are registered but
never activated, so Model Governance honestly shows the cold-start
notice for them (AWS-01's card explains why in plain words: holdout
15.6% over the 15% limit, Mamba registered).

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
  anomaly rate (0.157) fails the promotion gate (see §M).
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
