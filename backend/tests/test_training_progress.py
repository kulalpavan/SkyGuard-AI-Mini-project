"""Training progress reliability: incumbent scoring parity + stale-job recovery.

Regression tests for the "training looks stuck" fix:
  * The incumbent-agreement loop used to call score_observation per holdout
    row (artifact load + TreeSHAP each time) — ~25 s on a retrain while the
    UI sat on "Training Isolation Forest". It now loads the artifact once.
    The per-row decisions must be identical to score_observation.
  * A job left RUNNING by a server restart must be marked FAILED at
    startup instead of reporting RUNNING forever.
  * The job-status endpoint must carry started_at / model_version so the
    UI can show elapsed time and which version a job was training.
"""
import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.storage.database import (
    init_db, insert_telemetry_batch, create_training_job, update_training_job,
    get_training_job, fail_stale_training_jobs)
from backend.app.services.training_service import training_service
from ml.mamba_engine import load_district_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "backend" / "tests" / "fixtures"


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


def test_incumbent_flags_match_score_observation():
    """_incumbent_anomaly_flags must decide exactly like score_observation.

    Compares the first 40 holdout rows (the old per-row path takes ~46 ms
    per row, so a full-holdout comparison would add ~25 s to the suite;
    any systematic transform mismatch shows up on the first rows).
    """
    sid = "TRN-02"
    _make_station(sid, "Dakshina Kannada", 12.87, 74.88, 20.0)
    _backfill(sid, "Dakshina Kannada")
    job = training_service.start_training(sid, custom_version="v1.0")
    training_service.execute_training_job(job["job_id"], sid, "v1.0")
    assert get_training_job(job["job_id"])["status"] == "COMPLETED"

    from backend.app.storage.database import get_active_model_record
    incumbent = get_active_model_record(sid)
    assert incumbent is not None

    rows, _ = training_service.history(sid)
    split = int(len(rows) * 0.8)
    holdout = rows[split:][:40]

    fast = training_service._incumbent_anomaly_flags(incumbent, holdout, rows, split)
    slow = []
    for i, row in enumerate(holdout):
        result = training_service.score_observation(
            sid, row, history=rows[max(0, split + i - 12):split + i])
        assert result.get("has_model"), f"row {i} not scored by incumbent"
        slow.append(result["is_anomaly"])

    assert fast == slow, "incumbent flags differ from score_observation"


def test_stale_running_job_marked_failed_on_startup():
    # One RUNNING job per station is enforced by a partial unique index
    # (one_running_training), so the stale and the finished job need
    # different stations.
    sid_stale = "TRN-03"
    _make_station(sid_stale, "Stale Test", 12.0, 75.0, 10.0)
    stale_id = create_training_job(sid_stale, "v9.9")
    sid_done = "TRN-05"
    _make_station(sid_done, "Done Test", 12.0, 75.0, 10.0)
    done_id = create_training_job(sid_done, "v9.8")
    update_training_job(done_id, "COMPLETED", rows_used=10,
                        current_stage="Model Activated",
                        completed_stages=["Data Ingested", "Model Activated"])

    assert get_training_job(stale_id)["status"] == "RUNNING"
    assert fail_stale_training_jobs() >= 1

    stale = get_training_job(stale_id)
    assert stale["status"] == "FAILED"
    assert "restart" in (stale["error_message"] or "").lower()
    assert stale["current_stage"] == "Interrupted by server restart"
    assert stale["completed_at"] is not None

    # A finished job is never touched by the recovery.
    assert get_training_job(done_id)["status"] == "COMPLETED"


def test_status_endpoint_reports_timing_fields():
    sid = "TRN-04"
    client, headers = _make_station(sid, "Status Test", 12.0, 75.0, 10.0)
    job_id = create_training_job(sid, "v9.7")
    resp = client.get(f"/api/v1/stations/{sid}/training-jobs/{job_id}/status",
                      headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "RUNNING"
    assert body["started_at"] is not None
    assert body["model_version"] == "v9.7"
    assert body["job_id"] == job_id
