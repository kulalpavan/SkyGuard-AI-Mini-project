"""QC calibration from history: the envelope must come from the station's
own valid data, never from thin air.

Regression tests for item 9 (QC calibration wiring):
  * calibrate_station_qc() existed but was never called anywhere — dead
    code. The QCRules screen displayed any envelope as CALIBRATED even
    when it was a hand-set ADMIN_OVERRIDE wide value (15-38 C, 0 records).
  * The demo DB showed why the wiring has to be careful: AWS-01's
    telemetry holds 15,000 flat synthetic seed rows (25.0 C / 60% /
    1010 hPa) with qc_flag VALID. A calibration that only filters on
    VALID would build AWS-01's envelope on fake data. Synthetic rows
    (grid_point LIKE 'synthetic:%') must be excluded, the same rule
    training_service.history() applies.
  * Below 720 usable readings (the IF-training TRAINED maturity tier)
    calibration must refuse with a plain 422, not invent an envelope.
  * A variable with no data (e.g. wind in the demo history) gets NULL
    bounds, never a degenerate 0-0 range.
  * Auto-calibration after training must only fill a missing envelope —
    an admin's override is never overwritten.
"""
import datetime as dt

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.storage.database import (
    init_db, get_db, insert_telemetry_batch,
    get_station_qc_config, QC_CALIBRATION_MIN_RECORDS,
)
from backend.app.services.training_service import maybe_auto_calibrate_qc
from backend.app.services.observation_pipeline import observation_pipeline

MIN_RECORDS = QC_CALIBRATION_MIN_RECORDS


def _admin_client(sid):
    init_db()
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + create_access_token(
        {"sub": "admin", "role": "admin"})}
    resp = client.post("/api/v1/admin/stations", headers=headers, json={
        "station_id": sid, "station_name": "QC Calibration Test Station",
        "username": sid.lower().replace("-", ""), "password": "test-password",
        "latitude": 13.34, "longitude": 74.74, "elevation": 27.0,
    })
    assert resp.status_code in (200, 201, 409), resp.text
    return client, headers


def _operator_headers(sid):
    return {"Authorization": "Bearer " + create_access_token(
        {"sub": sid.lower().replace("-", ""), "role": "station_operator",
         "station_id": sid})}


def _seed_history(sid, n, temp_start=20.0, temp_step=0.01,
                  grid_point="test:history", qc_flag="VALID"):
    """n hourly VALID rows ending now: temp ramps, hum/pres cycle."""
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for i in range(n):
        when = now - dt.timedelta(hours=n - i)
        rows.append({
            "timestamp": when.isoformat(),
            "temp": temp_start + i * temp_step,
            "hum": 60.0 + (i % 40) * 0.1,
            "pres": 1010.0 + (i % 20) * 0.1,
            "wind": None, "rain": None, "battery": 12.6,
            "signal": -70.0, "raw_payload": {},
            "qc_flag": qc_flag, "grid_point": grid_point,
        })
    inserted, _ = insert_telemetry_batch(sid, rows)
    assert inserted == n, f"seeded {inserted} of {n}"
    return inserted


def _audit_actions(sid):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT action FROM config_audit WHERE station_id = ? "
            "ORDER BY created_at", (sid,))
        return [r["action"] for r in cur.fetchall()]


def test_calibrate_unknown_station_404_and_operator_forbidden():
    SID = "QCC-01"
    client, headers = _admin_client(SID)
    resp = client.post("/api/v1/admin/stations/NOPE-99/qc/calibrate",
                       headers=headers)
    assert resp.status_code == 404
    resp = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                       headers=_operator_headers(SID))
    assert resp.status_code == 403


def test_calibrate_insufficient_history_422_and_creates_nothing():
    SID = "QCC-02"
    client, headers = _admin_client(SID)
    _seed_history(SID, 100)
    resp = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                       headers=headers)
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "100" in detail and str(MIN_RECORDS) in detail
    assert get_station_qc_config(SID) is None


def test_calibrate_from_history_sets_percentile_envelope():
    SID = "QCC-03"
    client, headers = _admin_client(SID)
    _seed_history(SID, 720, temp_start=20.0, temp_step=0.01)
    resp = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                       headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    cfg = body["config"]
    assert cfg["calibration_method"] == "HISTORICAL_P01_P99"
    assert cfg["calibration_record_count"] == 720
    # Temps run 20.00 .. 27.19; P01 ~= 20.07, P99 ~= 27.12.
    assert abs(cfg["temperature_normal_min"] - 20.07) < 0.02, cfg
    assert abs(cfg["temperature_normal_max"] - 27.12) < 0.02, cfg
    # No wind data in the seeded history: NULL bounds, not 0-0.
    assert cfg["wind_normal_min"] is None
    assert cfg["wind_normal_max"] is None
    assert "QC_CALIBRATE_FROM_HISTORY" in _audit_actions(SID)
    assert body["valid_records"] == 720


def test_calibrate_excludes_synthetic_rows():
    SID = "QCC-04"
    client, headers = _admin_client(SID)
    # 720 flat synthetic rows (the AWS-01 demo-seed shape) plus 720 real
    # rows: the envelope must come from the real rows only.
    _seed_history(SID, 720, temp_start=25.0, temp_step=0.0,
                  grid_point="synthetic:demo-seed")
    _seed_history(SID, 720, temp_start=20.0, temp_step=0.01,
                  grid_point="test:history")
    resp = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                       headers=headers)
    assert resp.status_code == 200, resp.text
    cfg = resp.json()["config"]
    assert cfg["calibration_record_count"] == 720
    assert abs(cfg["temperature_normal_min"] - 20.07) < 0.02, cfg
    assert abs(cfg["temperature_normal_max"] - 27.12) < 0.02, cfg

    # Synthetic rows alone are not enough history at all.
    SID2 = "QCC-05"
    client2, headers2 = _admin_client(SID2)
    _seed_history(SID2, 720, temp_start=25.0, temp_step=0.0,
                  grid_point="synthetic:demo-seed")
    resp = client2.post(f"/api/v1/admin/stations/{SID2}/qc/calibrate",
                        headers=headers2)
    assert resp.status_code == 422, resp.text
    assert get_station_qc_config(SID2) is None


def test_calibrate_twice_bumps_version_and_keeps_audit():
    SID = "QCC-06"
    client, headers = _admin_client(SID)
    _seed_history(SID, 720)
    first = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                        headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["config"]["version"] == 1
    second = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                         headers=headers)
    assert second.status_code == 200, second.text
    assert second.json()["config"]["version"] == 2
    assert second.json()["config"]["calibration_method"] == "HISTORICAL_P01_P99"
    assert _audit_actions(SID).count("QC_CALIBRATE_FROM_HISTORY") == 2


def test_auto_calibrate_only_fills_a_missing_envelope():
    # Enough history, no envelope: the helper calibrates.
    SID = "QCC-07"
    _admin_client(SID)
    _seed_history(SID, 720)
    assert get_station_qc_config(SID) is None
    config = maybe_auto_calibrate_qc(SID)
    assert config is not None
    assert config["calibration_method"] == "HISTORICAL_P01_P99"

    # An admin override is never overwritten.
    SID2 = "QCC-08"
    client, headers = _admin_client(SID2)
    _seed_history(SID2, 720)
    resp = client.put(f"/api/v1/admin/stations/{SID2}/qc", headers=headers, json={
        "temperature_normal_min": 15.0, "temperature_normal_max": 38.0,
        "justification": "admin wide envelope",
    })
    assert resp.status_code == 200, resp.text
    assert maybe_auto_calibrate_qc(SID2) is None
    cfg = get_station_qc_config(SID2)
    assert cfg["calibration_method"] == "ADMIN_OVERRIDE"
    assert cfg["temperature_normal_max"] == 38.0

    # Too little history: nothing is created.
    SID3 = "QCC-09"
    _admin_client(SID3)
    _seed_history(SID3, 100)
    assert maybe_auto_calibrate_qc(SID3) is None
    assert get_station_qc_config(SID3) is None


def test_calibrated_envelope_flags_outside_reading():
    """The calibrated envelope must take effect in the pipeline: a
    reading outside it sets z_qc, the same proof style as item 7."""
    SID = "QCC-10"
    client, headers = _admin_client(SID)
    _seed_history(SID, 720, temp_start=20.0, temp_step=0.01)
    resp = client.post(f"/api/v1/admin/stations/{SID}/qc/calibrate",
                       headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["config"]["temperature_normal_max"] < 28.0

    res = observation_pipeline.process(SID, {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "temperature": 35.0, "humidity": 62.0, "pressure": 1011.0,
        "battery": 12.6, "signal": -70.0,
    })
    assert res["status"] == "ACCEPTED"
    assessment = res["state"]["final_assessment"]
    assert assessment["evidence_vector"]["z_qc"] == 1.0
    assert assessment["quality_state"] != "VALID"
