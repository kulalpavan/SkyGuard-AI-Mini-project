"""QC envelope endpoint: the rule an admin saves must actually take effect.

Regression tests for item 7 (screen verification):
  * PUT /admin/stations/{id}/qc used to return 500 on every call — its SQL
    referenced columns that do not exist in station_qc_config
    (precipitation_normal_max, updated_at) and its config_audit insert used
    columns that do not exist in config_audit (updated_by,
    previous_state_json, new_state_json, justification, timestamp).
    The QCRules screen's "Commit Hard Limits" button hid this by never
    calling the backend at all (it only showed an alert).
  * A saved envelope must change what the pipeline does: a reading outside
    it sets the z_qc evidence flag and the observation is no longer VALID.
"""
import datetime as dt

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.storage.database import init_db
from backend.app.services.observation_pipeline import observation_pipeline

def _admin_client(sid):
    init_db()
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + create_access_token(
        {"sub": "admin", "role": "admin"})}
    resp = client.post("/api/v1/admin/stations", headers=headers, json={
        "station_id": sid, "station_name": "QC Test Station",
        "username": sid.lower().replace("-", ""), "password": "test-password",
        "latitude": 13.34, "longitude": 74.74, "elevation": 27.0,
    })
    assert resp.status_code in (200, 201, 409), resp.text
    return client, headers


def _operator_headers(sid):
    return {"Authorization": "Bearer " + create_access_token(
        {"sub": sid.lower().replace("-", ""), "role": "station_operator",
         "station_id": sid})}


def _ingest(sid, temp, hum=70.0, pres=1012.0, minutes_ago=5):
    ts = (dt.datetime.now(dt.timezone.utc)
          - dt.timedelta(minutes=minutes_ago)).isoformat()
    return observation_pipeline.process(sid, {
        "timestamp": ts, "temperature": temp, "humidity": hum,
        "pressure": pres, "battery": 12.6, "signal": -70.0,
    })


def test_put_creates_envelope_and_get_returns_it():
    SID = "QC-01"
    client, headers = _admin_client(SID)
    resp = client.put(f"/api/v1/admin/stations/{SID}/qc", headers=headers, json={
        "temperature_normal_min": 15.0, "temperature_normal_max": 38.0,
        "humidity_normal_min": 40.0, "humidity_normal_max": 95.0,
        "justification": "test envelope",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    assert resp.json()["config"]["temperature_normal_min"] == 15.0
    assert resp.json()["config"]["calibration_method"] == "ADMIN_OVERRIDE"

    got = client.get(f"/api/v1/stations/{SID}/qc", headers=headers)
    assert got.status_code == 200
    assert got.json()["has_config"] is True
    assert got.json()["config"]["temperature_normal_max"] == 38.0
    assert got.json()["config"]["humidity_normal_min"] == 40.0


def test_put_updates_only_sent_fields():
    SID = "QC-02"
    client, headers = _admin_client(SID)
    client.put(f"/api/v1/admin/stations/{SID}/qc", headers=headers, json={
        "temperature_normal_min": 15.0, "temperature_normal_max": 38.0})
    resp = client.put(f"/api/v1/admin/stations/{SID}/qc", headers=headers, json={
        "humidity_normal_min": 30.0, "humidity_normal_max": 98.0})
    assert resp.status_code == 200, resp.text
    cfg = resp.json()["config"]
    assert cfg["temperature_normal_min"] == 15.0  # kept from the first PUT
    assert cfg["humidity_normal_max"] == 98.0
    assert cfg["version"] == 2  # update bumps the version


def test_put_unknown_station_404_and_operator_forbidden():
    SID = "QC-03"
    client, headers = _admin_client(SID)
    resp = client.put("/api/v1/admin/stations/NOPE-99/qc", headers=headers,
                      json={"temperature_normal_max": 38.0})
    assert resp.status_code == 404
    resp = client.put(f"/api/v1/admin/stations/{SID}/qc",
                      headers=_operator_headers(SID),
                      json={"temperature_normal_max": 38.0})
    assert resp.status_code == 403


def test_envelope_breach_flags_next_observation():
    SID = "QC-04"
    client, headers = _admin_client(SID)
    # A normal reading first: inside the envelope, no history yet.
    ok = _ingest(SID, 27.0)
    assert ok["status"] == "ACCEPTED"
    assert ok["state"]["final_assessment"]["quality_state"] == "VALID"

    # Tighten the envelope around normal conditions, then send a reading
    # far outside it. The pipeline must flag it via the z_qc evidence.
    resp = client.put(f"/api/v1/admin/stations/{SID}/qc", headers=headers, json={
        "temperature_normal_min": 20.0, "temperature_normal_max": 30.0,
        "humidity_normal_min": 50.0, "humidity_normal_max": 90.0,
        "pressure_normal_min": 1000.0, "pressure_normal_max": 1025.0,
    })
    assert resp.status_code == 200, resp.text

    hot = _ingest(SID, 45.0, minutes_ago=2)
    assert hot["status"] == "ACCEPTED"
    assessment = hot["state"]["final_assessment"]
    assert assessment["evidence_vector"]["z_qc"] == 1.0
    assert assessment["quality_state"] != "VALID"


def test_reading_inside_envelope_stays_valid():
    SID = "QC-05"
    client, headers = _admin_client(SID)
    client.put(f"/api/v1/admin/stations/{SID}/qc", headers=headers, json={
        "temperature_normal_min": 20.0, "temperature_normal_max": 30.0,
        "humidity_normal_min": 50.0, "humidity_normal_max": 90.0,
        "pressure_normal_min": 1000.0, "pressure_normal_max": 1025.0,
    })
    res = _ingest(SID, 27.0, hum=70.0, pres=1012.0)
    assert res["status"] == "ACCEPTED"
    assessment = res["state"]["final_assessment"]
    assert assessment["evidence_vector"]["z_qc"] == 0.0
    assert assessment["quality_state"] == "VALID"


def test_envelope_alone_sets_z_qc_on_climatologically_normal_row():
    """The envelope itself must flag a reading, not just ride along with a
    spike. With 2,617 rows of real history behind it, a 27 C / 90% / 1012
    hPa row is climatologically normal (small residuals) but breaches a
    20 C envelope max, so z_qc must be 1.0.

    Designed behaviour, recorded here so nobody "fixes" it by accident: a
    lone envelope breach lifts the fusion score (here > 0.25, up from ~0.05
    with no breach) but does not by itself reach the 0.5 SUSPECT threshold,
    so the verdict stays VALID. The envelope corroborates and amplifies
    other evidence; flipping the verdict on a breach alone would be a
    detection-tuning change, not a QC-screen fix.
    """
    import statistics
    from backend.app.storage.database import insert_telemetry_batch
    from ml.mamba_engine import load_district_series
    from pathlib import Path

    SID = "QC-06"
    client, headers = _admin_client(SID)
    fixtures = Path(__file__).resolve().parent / "fixtures"
    stamps, values = load_district_series(str(fixtures), "Dakshina Kannada")
    rows = []
    for ts, (t, h, p) in zip(stamps, values):
        when = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        rows.append({"timestamp": when.isoformat(),
                     "temp": float(t), "hum": float(h), "pres": float(p),
                     "wind": None, "rain": None, "battery": 12.6,
                     "signal": -70.0, "raw_payload": {},
                     "qc_flag": "VALID", "grid_point": ""})
    inserted, _ = insert_telemetry_batch(SID, rows)
    assert inserted == len(values)

    # Median temperature for the row's hour, so the row is normal by
    # climatology; the envelope max sits 1 C below it.
    row_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30)
    hour_rows = [v for ts, v in zip(stamps, values)
                 if dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).hour == row_time.hour]
    med_t = statistics.median([v[0] for v in hour_rows])
    med_h = statistics.median([v[1] for v in hour_rows])
    med_p = statistics.median([v[2] for v in hour_rows])
    resp = client.put(f"/api/v1/admin/stations/{SID}/qc", headers=headers, json={
        "temperature_normal_min": 15.0, "temperature_normal_max": round(med_t - 1.0, 1),
        "humidity_normal_min": 50.0, "humidity_normal_max": 98.0,
        "pressure_normal_min": 1000.0, "pressure_normal_max": 1025.0,
    })
    assert resp.status_code == 200, resp.text

    res = observation_pipeline.process(SID, {
        "timestamp": row_time.isoformat(),
        "temperature": med_t, "humidity": med_h, "pressure": med_p,
        "battery": 12.6, "signal": -70.0,
    })
    assert res["status"] == "ACCEPTED"
    assessment = res["state"]["final_assessment"]
    assert assessment["evidence_vector"]["z_qc"] == 1.0
    residuals = assessment.get("residuals") or []
    assert residuals and max(abs(v) for v in residuals) < 3, residuals
    assert assessment["fusion"]["score"] > 0.25
    assert assessment["quality_state"] == "VALID"  # lone breach: flagged, not flipped
