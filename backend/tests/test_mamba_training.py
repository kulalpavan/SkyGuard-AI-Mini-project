"""Mamba training inside the product (backend).

The Train button runs one job that trains the Isolation Forest AND the
Mamba temporal model from the station's own stored telemetry history —
the same history the IF job trains on (which includes the Open-Meteo
observations the poller writes into the telemetry table, plus ESP32 /
simulator frames). No district CSV files are involved: the Mamba step
builds its hourly series from the station's telemetry rows.

  * The combined job registers a Mamba model (CANDIDATE, never ACTIVE)
    and a fresh ingest afterwards carries a real mamba_score.
  * A station without enough hourly history fails honestly with
    MAMBA_HISTORY_INSUFFICIENT — the job reports it instead of training
    on too little data.
  * The job-status endpoint reports the three Mamba phases, and a
    finished job reports current_stage "Mamba Registered" at 100%.

Run:  ../venv/bin/python -m pytest backend/tests/test_mamba_training.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.services.mamba_service import get_mamba_record
from backend.app.services.mamba_training import train_mamba_from_station_history
from backend.app.services.observation_pipeline import observation_pipeline
from backend.app.services.training_service import training_service
from backend.app.storage.database import (
    init_db, insert_telemetry_batch, create_training_job,
    update_training_job, update_training_job_stage, get_training_job)
from ml.mamba_engine import load_district_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "backend" / "tests" / "fixtures"

ALL_STAGES = [
    "Data Ingested", "Data Validated", "Data Preprocessed",
    "Features Generated", "Training Isolation Forest", "Model Evaluation",
    "Model Registered", "Model Activated",
    "Training Mamba", "Calibrating Mamba", "Mamba Registered",
]


def _make_station(sid, name, lat, lon, elev):
    init_db()
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + create_access_token(
        {"sub": "admin", "role": "admin"})}
    resp = client.post("/api/v1/admin/stations", headers=headers, json={
        "station_id": sid, "station_name": name,
        "username": sid.lower().replace("-", ""),
        "password": "test-password", "latitude": lat, "longitude": lon,
        "elevation": elev,
    })
    assert resp.status_code in (200, 201, 409), resp.text
    return client, headers


def _backfill(sid, district):
    stamps, values = load_district_series(str(FIXTURES), district)
    rows = []
    for ts, (t, h, p) in zip(stamps, values):
        when = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        rows.append({"timestamp": when.isoformat(),
                     "temp": float(t), "hum": float(h), "pres": float(p),
                     "wind": None, "rain": None, "battery": 12.6,
                     "signal": -70.0, "raw_payload": {},
                     "qc_flag": "VALID", "grid_point": ""})
    inserted, errors = insert_telemetry_batch(sid, rows)
    assert inserted == len(values)
    return stamps, values


def _ingest(sid, ts_iso, temp, hum, pres):
    return observation_pipeline.process(sid, {
        "timestamp": ts_iso, "temp": float(temp), "hum": float(hum),
        "pres": float(pres)})


def _next_ts(stamps, offset_h):
    last = dt.datetime.fromtimestamp(int(stamps[-1]), tz=dt.timezone.utc)
    return (last + dt.timedelta(hours=offset_h)).isoformat()


def test_combined_job_trains_if_and_mamba_from_station_history():
    """One Train click trains IF and Mamba from the station's own data."""
    sid = "MAMBA-03"
    _make_station(sid, "Dakshina Kannada", 12.87, 74.88, 20.0)
    stamps, values = _backfill(sid, "Dakshina Kannada")

    job = training_service.start_training(sid, custom_version="v1.0")
    training_service.execute_training_job(job["job_id"], sid, "v1.0")
    record = get_training_job(job["job_id"])
    assert record["status"] == "COMPLETED", record.get("error_message")
    import json
    completed = json.loads(record["completed_stages"])
    assert completed == ALL_STAGES, completed

    mamba_rec = get_mamba_record(sid)
    assert mamba_rec is not None, "no Mamba model registered by the job"
    assert mamba_rec["model_type"] == "Mamba"
    assert mamba_rec["status"] == "CANDIDATE"
    assert mamba_rec["model_version"] == "v1.0-mamba"
    assert mamba_rec["model_id"] == "MAMBA-03_MAMBA_v1_0-mamba"

    t, h, p = values[-1]
    normal = _ingest(sid, _next_ts(stamps, 1), t, h, p)
    assert normal["status"] == "ACCEPTED", normal
    a = normal["state"]["final_assessment"]
    assert a["mamba_score"] is not None, "no Mamba score after in-product training"
    assert np.isfinite(a["mamba_score"])
    assert a["hybrid_score"] is not None, "no hybrid score after in-product training"
    spike = _ingest(sid, _next_ts(stamps, 2), t + 6.0, h, p)
    sa = spike["state"]["final_assessment"]
    assert sa["mamba_score"] is not None
    assert sa["mamba_score"] > a["mamba_score"], (
        f"spike Mamba {sa['mamba_score']} not above normal {a['mamba_score']}")


def test_mamba_insufficient_history_fails_honestly():
    """Too little hourly history is a clear failure, not silent training."""
    # Direct: 100 hourly rows are far below the 720 minimum.
    base = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rows = []
    for k in range(100):
        when = base + dt.timedelta(hours=k)
        rows.append({"timestamp": when.isoformat(), "temp": 25.0,
                     "hum": 60.0, "pres": 1013.0, "hour": when.hour})
    with pytest.raises(ValueError, match="MAMBA_HISTORY_INSUFFICIENT"):
        train_mamba_from_station_history("MAMBA-99", rows, 20.0, "v9.9-mamba")

    # Job level: 840 observations across 14 days pass the IF readiness
    # gate, but they cover only 28 distinct hours, so Mamba cannot train.
    sid = "MAMBA-04"
    _make_station(sid, "Sparse Station", 13.0, 74.8, 20.0)
    rng = np.random.default_rng(7)
    rows = []
    for day in range(14):
        for hour in (0, 1):
            for minute in range(0, 60, 2):
                when = base + dt.timedelta(days=day, hours=hour,
                                           minutes=int(minute))
                frac = (when.hour + when.minute / 60.0) / 24.0
                rows.append({
                    "timestamp": when.isoformat(),
                    "temp": float(25 + 3 * np.sin(2 * np.pi * frac)
                                  + rng.normal(0, 0.3)),
                    "hum": float(60 + 5 * np.sin(2 * np.pi * frac)
                                 + rng.normal(0, 2)),
                    "pres": float(1013 + rng.normal(0, 0.5)),
                    "wind": None, "rain": None, "battery": 12.6,
                    "signal": -70.0, "raw_payload": {},
                    "qc_flag": "VALID", "grid_point": ""})
    inserted, errors = insert_telemetry_batch(sid, rows)
    assert inserted == len(rows), f"inserted {inserted}/{len(rows)}"
    job = training_service.start_training(sid, custom_version="v1.0")
    training_service.execute_training_job(job["job_id"], sid, "v1.0")
    record = get_training_job(job["job_id"])
    assert record["status"] == "FAILED", record
    assert "Mamba" in (record["error_message"] or "")
    assert "MAMBA_HISTORY_INSUFFICIENT" in (record["error_message"] or "")
    import json
    completed = json.loads(record["completed_stages"])
    assert "Mamba Registered" not in completed
    assert get_mamba_record(sid) is None


def test_status_endpoint_reports_mamba_phases():
    """The status endpoint carries the Mamba phases and the 11-step total."""
    sid = "MAMBA-03"
    client, headers = _make_station(sid, "Dakshina Kannada", 12.87, 74.88, 20.0)
    job_id = create_training_job(sid, "v9.8")
    update_training_job_stage(
        job_id, "Calibrating Mamba",
        ALL_STAGES[:8] + ["Training Mamba", "Calibrating Mamba"])
    resp = client.get(
        f"/api/v1/stations/{sid}/training-jobs/{job_id}/status",
        headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "RUNNING"
    assert body["current_stage"] == "Calibrating Mamba"
    assert body["completed_stages"][-2:] == ["Training Mamba", "Calibrating Mamba"]
    assert body["progress"] == pytest.approx(90.9, abs=0.05)

    update_training_job(job_id, "COMPLETED", rows_used=2617,
                        completed_stages=ALL_STAGES)
    resp = client.get(
        f"/api/v1/stations/{sid}/training-jobs/{job_id}/status",
        headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETED"
    assert body["current_stage"] == "Mamba Registered"
    assert body["progress"] == 100.0
    assert body["completed_stages"] == ALL_STAGES
