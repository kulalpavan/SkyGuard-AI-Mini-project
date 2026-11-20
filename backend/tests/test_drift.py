"""SkyGuard-AI — model drift tracking tests (spec §21, §28.5).

Unit level (T-DRF-01..04, pure drift_engine, no DB):
  01 known-shifted distribution -> PSI in the HIGH band
  02 seasonal-only change -> NO drift in residual space (the §21.2
     key test: raw values move with the season, residuals stay ~N(0,1))
  03 injected calibration drift -> residual mean shift above threshold
  04 severity mapping across bands for every metric

Integration level (disposable DB, real district fixture):
  05 HIGH drift queues a retraining candidate and opens a
     DATA_QUALITY incident (shifted-tail station)
  06 on-demand compute + read APIs work end to end on a trained
     station, including the Mamba model
  07 INSUFFICIENT_DATA when the recent window is too short
  08 RBAC: an operator cannot compute drift; unknown station -> 404
  09 training stores the drift reference in the model card
  10 migration v4 is idempotent (init_db twice, columns present)

The district values are the real supplied NOAA files via the committed
fixture (backend/tests/fixtures/). Mamba is registered with 2 epochs
here (wiring, not accuracy), as in test_mamba_pipeline.py.

Run:  ../venv/bin/python -m pytest backend/tests/test_drift.py
"""

import datetime as dt
import importlib.util
import json
import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.storage.database import init_db, get_db, insert_telemetry_batch
from backend.app.services.training_service import training_service
from ml.drift_engine import (
    deciles, compute_psi, ks_statistic_from_deciles, severity_for_metric,
    overall_severity, plain_words,
)
from ml.mamba_engine import load_district_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------
# Unit: drift_engine
# --------------------------------------------------------------------------

def _normal_samples(n, mean=0.0, std=1.0, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(mean, std) for _ in range(n)]


def test_psi_identical_distributions_near_zero():
    ref = deciles(_normal_samples(2000, seed=1))
    psi = compute_psi(ref, _normal_samples(500, seed=2))
    assert psi is not None and psi < 0.05


def test_psi_known_shift_in_high_band():
    # T-DRF-01: a +1 sigma mean shift is a large distribution change.
    ref = deciles(_normal_samples(2000, seed=1))
    psi = compute_psi(ref, _normal_samples(500, mean=1.0, seed=3))
    assert psi is not None and psi > 0.25


def test_psi_moderate_shift_in_moderate_band():
    ref = deciles(_normal_samples(2000, seed=1))
    psi = compute_psi(ref, _normal_samples(500, mean=0.45, seed=4))
    assert psi is not None and 0.10 < psi < 0.25


def test_seasonal_only_change_is_not_drift():
    # T-DRF-02 (§21.2): the raw season moves, the residuals do not.
    # Reference residuals are N(0,1) from the training season; current
    # residuals are still N(0,1) even though the raw values behind them
    # sit 4 degrees warmer (a new season). PSI and mean shift stay in
    # the NONE band — raw-space PSI would have fired here.
    ref = deciles(_normal_samples(2000, seed=11))
    cur = _normal_samples(336, seed=12)
    assert compute_psi(ref, cur) < 0.10
    mean = sum(cur) / len(cur)
    assert abs(mean) < 0.3
    assert severity_for_metric("psi", compute_psi(ref, cur)) in ("NONE", "LOW")
    assert severity_for_metric("residual_mean_shift", abs(mean)) == "NONE"


def test_calibration_drift_raises_mean_shift():
    # T-DRF-03: a sensor drifting +0.8 sigma reads consistently high.
    ref = deciles(_normal_samples(2000, seed=21))
    cur = _normal_samples(336, mean=0.8, seed=22)
    mean = sum(cur) / len(cur)
    assert abs(mean) > 0.7
    assert severity_for_metric("residual_mean_shift", abs(mean)) == "HIGH"


def test_severity_mapping_bands():
    # T-DRF-04: every metric's bands, including the LOW early warning.
    assert severity_for_metric("psi", 0.04) == "NONE"
    assert severity_for_metric("psi", 0.07) == "LOW"
    assert severity_for_metric("psi", 0.15) == "MODERATE"
    assert severity_for_metric("psi", 0.30) == "HIGH"
    assert severity_for_metric("residual_mean_shift", 0.25) == "LOW"
    assert severity_for_metric("residual_mean_shift", 0.5) == "MODERATE"
    assert severity_for_metric("residual_mean_shift", 0.9) == "HIGH"
    assert severity_for_metric("variance_ratio_deviation", 1.5) == "NONE"
    assert severity_for_metric("variance_ratio_deviation", 3.0) == "MODERATE"
    assert severity_for_metric("variance_ratio_deviation", 5.0) == "HIGH"
    assert severity_for_metric("anomaly_rate_multiple", 1.5) == "NONE"
    assert severity_for_metric("anomaly_rate_multiple", 2.5) == "MODERATE"
    assert severity_for_metric("anomaly_rate_multiple", 4.0) == "HIGH"
    assert severity_for_metric("ks_statistic", 0.05) == "NONE"
    assert severity_for_metric("ks_statistic", 0.25) == "MODERATE"
    assert severity_for_metric("ks_statistic", 0.4) == "HIGH"
    assert severity_for_metric("feature_availability", 0.99) == "NONE"
    assert severity_for_metric("feature_availability", 0.9) == "LOW"
    assert severity_for_metric("feature_availability", 0.7) == "MODERATE"
    assert severity_for_metric("feature_availability", 0.4) == "HIGH"
    assert severity_for_metric("mamba_error_ratio", 1.1) == "NONE"
    assert severity_for_metric("mamba_error_ratio", 1.7) == "MODERATE"
    assert severity_for_metric("mamba_error_ratio", 2.5) == "HIGH"
    assert severity_for_metric("psi", None) is None
    assert overall_severity(["NONE", "LOW", "MODERATE"]) == "MODERATE"
    assert overall_severity(["NONE", "NONE"]) == "NONE"
    assert overall_severity([]) == "NONE"


def test_plain_words_names_the_worst_sign():
    text = plain_words("HIGH", "residual_mean_shift", 1.2)
    assert "Needs retraining" in text
    assert "consistently off" in text


# --------------------------------------------------------------------------
# Integration helpers
# --------------------------------------------------------------------------

def _admin_client():
    init_db()
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + create_access_token(
        {"sub": "admin", "role": "admin"})}
    return client, headers


def _make_station(client, headers, sid, name, lat, lon, elev=10.0):
    resp = client.post("/api/v1/admin/stations", headers=headers, json={
        "station_id": sid, "station_name": name,
        "username": sid.lower().replace("-", ""),
        "password": "test-password", "latitude": lat, "longitude": lon,
        "elevation": elev,
    })
    assert resp.status_code in (200, 201, 409), resp.text


def _backfill(sid, district, shift_temp=0.0, shift_from_hour=None):
    """Insert the full district series (2,617 hourly rows).

    shift_temp/shift_from_hour optionally add a sustained offset to the
    tail of the series — a genuine regime change the drift monitor
    should catch, built from real data rather than invented values.
    """
    stamps, values = load_district_series(str(DATA_DIR), district)
    rows = []
    for idx, (ts, (t, h, p)) in enumerate(zip(stamps, values)):
        when = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        temp = float(t) + (shift_temp if shift_from_hour is not None and idx >= shift_from_hour else 0.0)
        rows.append({"timestamp": when.isoformat(),
                     "temp": temp, "hum": float(h), "pres": float(p),
                     "grid_point": "", "qc_flag": "VALID"})
    inserted, errors = insert_telemetry_batch(sid, rows)
    assert errors == 0, f"backfill errors: {errors}"
    assert inserted == len(rows), f"backfill inserted {inserted}/{len(rows)}"
    return rows


def _train_if(sid, version):
    rows, raw_count = training_service.history(sid)
    from ml.feature_engine import readiness
    tier = readiness(rows)["tier"]
    assert tier in ("TRAINED", "MATURE"), tier
    bundle = training_service.train_if_model(
        sid, rows, raw_count, version, 10.0, tier)
    return bundle


# --------------------------------------------------------------------------
# Integration: drift APIs
# --------------------------------------------------------------------------

def test_training_stores_drift_reference():
    # T-DRF-09: the model card carries its training-time reference.
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-09", "Drift Ref Station", 12.87, 74.88)
    _backfill("DRF-09", "Dakshina Kannada")
    bundle = _train_if("DRF-09", "v2.driftref")
    ref = bundle["card"]["drift_reference"]
    assert set(ref["residual_deciles"]) == {"temperature", "humidity", "pressure"}
    assert len(ref["residual_deciles"]["temperature"]) == 9
    assert len(ref["if_score_deciles"]) == 9
    assert 0 < ref["anomaly_rate_reference"] < 0.2
    assert ref["window"]["start"] < ref["window"]["end"]


def test_compute_and_read_drift_end_to_end():
    # T-DRF-06 (IF part): on-demand compute stores rows and the read
    # APIs return them. Severity is asserted only to be a valid value:
    # the real DK tail contains a genuine dry spell, so the honest
    # result may be MODERATE or HIGH — the test pins the wiring, not
    # the weather.
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-06", "Drift E2E Station", 12.87, 74.88)
    _backfill("DRF-06", "Dakshina Kannada")
    bundle = _train_if("DRF-06", "v2.drifte2e")
    training_service.activate("DRF-06", "v2.drifte2e")

    resp = client.post("/api/v1/stations/DRF-06/drift/compute", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["overall_severity"] in ("NONE", "LOW", "MODERATE", "HIGH")
    assert body["plain_words"]
    if_model = next(m for m in body["models"] if m["model_type"] == "IsolationForest")
    assert if_model["model_version"] == "v2.drifte2e"
    assert if_model["metrics"]["anomaly_rate"] is not None
    assert if_model["metrics"]["feature_availability"] is not None
    assert if_model["metrics"]["feature_availability"] > 0.95
    assert if_model["metric_entries"], "metric table must not be empty"

    latest = client.get("/api/v1/stations/DRF-06/drift/latest", headers=headers)
    assert latest.status_code == 200, latest.text
    assert "IsolationForest" in latest.json()["latest"]
    history = client.get("/api/v1/stations/DRF-06/drift?days=90", headers=headers)
    assert history.status_code == 200, history.text
    assert len(history.json()["history"]) >= 1
    assert "psi" in history.json()["thresholds"]
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM drift_metrics WHERE station_id = 'DRF-06'")
        assert cur.fetchone()["n"] >= 1


def test_high_drift_queues_candidate_and_incident():
    # T-DRF-05: a sustained +3 C offset on the tail of the series is a
    # real regime change vs the training reference. Expect HIGH IF
    # drift, a queued candidate and one open DATA_QUALITY incident —
    # and no duplicate on a second compute (24 h dedup).
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-05", "Drift High Station", 12.87, 74.88)
    _backfill("DRF-05", "Dakshina Kannada", shift_temp=3.0, shift_from_hour=2617 - 504)
    _train_if("DRF-05", "v2.drifthigh")

    resp = client.post("/api/v1/stations/DRF-05/drift/compute", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["overall_severity"] == "HIGH", body
    if_model = next(m for m in body["models"] if m["model_type"] == "IsolationForest")
    assert if_model["severity"] == "HIGH"
    assert if_model["metrics"]["residual_mean_shift"] > 0.7
    assert if_model["metrics"]["psi_temperature"] is not None
    assert if_model["metrics"]["psi_temperature"] > 0.25
    assert if_model["triggered_candidate"] is True

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT * FROM retraining_candidates
                       WHERE station_id = 'DRF-05' AND status = 'QUEUED'""")
        candidates = cur.fetchall()
        assert len(candidates) == 1
        assert candidates[0]["priority"] == "HIGH"
        assert "DRIFT_HIGH" in (candidates[0]["trigger_reason"] or "")
        cur.execute("""SELECT * FROM incidents WHERE station_id = 'DRF-05'
                       AND variable = 'drift:MODEL_DRIFT' AND status = 'open'""")
        incidents = cur.fetchall()
        assert len(incidents) == 1
        assert incidents[0]["quality_state"] == "DATA_QUALITY"

    # Second compute: no duplicate candidate or incident.
    resp2 = client.post("/api/v1/stations/DRF-05/drift/compute", headers=headers)
    assert resp2.status_code == 200, resp2.text
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT COUNT(*) AS n FROM retraining_candidates
                       WHERE station_id = 'DRF-05' AND status = 'QUEUED'""")
        assert cur.fetchone()["n"] == 1
        cur.execute("""SELECT COUNT(*) AS n FROM incidents WHERE station_id = 'DRF-05'
                       AND variable = 'drift:MODEL_DRIFT' AND status = 'open'""")
        assert cur.fetchone()["n"] == 1


def test_mamba_drift_measured():
    # T-DRF-06 (Mamba part): a registered Mamba model gets its own
    # drift result with prediction-error metrics.
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-07", "Drift Mamba Station", 12.87, 74.88)
    _backfill("DRF-07", "Dakshina Kannada")
    _train_if("DRF-07", "v2.driftmamba")
    mod = _load_register_mamba()
    mod.register_station("DRF-07", "Dakshina Kannada", str(DATA_DIR),
                         version="v1.0-drift", epochs=2)

    resp = client.post("/api/v1/stations/DRF-07/drift/compute", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    mamba = next((m for m in body["models"] if m["model_type"] == "Mamba"), None)
    assert mamba is not None, body
    assert mamba["severity"] in ("NONE", "LOW", "MODERATE", "HIGH", "INSUFFICIENT_DATA")
    assert mamba["metrics"].get("mamba_error_ratio") is not None
    assert mamba["metrics"].get("anomaly_rate") is not None
    assert mamba["detail"]["windows_scored"] >= 50


def _load_register_mamba():
    spec = importlib.util.spec_from_file_location(
        "register_mamba", REPO_ROOT / "scripts" / "register_mamba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_insufficient_data_when_window_too_short():
    # T-DRF-07: a station with a model but only 5 days of recent
    # telemetry reports INSUFFICIENT_DATA, never a false NONE.
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-08", "Drift Short Station", 12.87, 74.88)
    _backfill("DRF-08", "Dakshina Kannada")
    _train_if("DRF-08", "v2.driftshort")
    with get_db() as conn:
        cur = conn.cursor()
        # Keep only the last 5 days: delete everything older than 5
        # days before the newest row.
        cur.execute("""DELETE FROM telemetry WHERE station_id = 'DRF-08'
                       AND timestamp < datetime(
                           (SELECT MAX(timestamp) FROM telemetry
                            WHERE station_id = 'DRF-08'), '-5 days')""")
    resp = client.post("/api/v1/stations/DRF-08/drift/compute", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["overall_severity"] == "INSUFFICIENT_DATA", body
    assert body["models"], "the model row is still recorded"
    assert body["models"][0]["severity"] == "INSUFFICIENT_DATA"


def test_drift_rbac_and_unknown_station():
    # T-DRF-08: operators cannot trigger a compute; unknown station 404s.
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-10", "Drift RBAC Station", 12.87, 74.88)
    op_headers = {"Authorization": "Bearer " + create_access_token(
        {"sub": "drf10", "role": "station_operator", "station_id": "DRF-10"})}
    resp = client.post("/api/v1/stations/DRF-10/drift/compute", headers=op_headers)
    assert resp.status_code in (401, 403), resp.text
    resp = client.post("/api/v1/stations/NOPE-99/drift/compute", headers=headers)
    assert resp.status_code == 404, resp.text
    resp = client.get("/api/v1/stations/NOPE-99/drift/latest", headers=headers)
    assert resp.status_code == 404, resp.text


def test_manual_queue_retraining():
    client, headers = _admin_client()
    _make_station(client, headers, "DRF-11", "Drift Queue Station", 12.87, 74.88)
    _backfill("DRF-11", "Dakshina Kannada")
    _train_if("DRF-11", "v2.driftqueue")
    resp = client.post("/api/v1/stations/DRF-11/drift/queue-retraining",
                       headers=headers,
                       json={"model_version": "v2.driftqueue",
                             "trigger_reason": "MANUAL_DRIFT_REVIEW",
                             "priority": "LOW"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["created"] is True
    resp2 = client.post("/api/v1/stations/DRF-11/drift/queue-retraining",
                        headers=headers,
                        json={"model_version": "v2.driftqueue"})
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["created"] is False  # dedup within 24 h
    resp3 = client.post("/api/v1/stations/DRF-11/drift/queue-retraining",
                        headers=headers, json={})
    assert resp3.status_code == 400, resp3.text


def test_migration_v4_idempotent():
    # T-DRF-10: init_db twice leaves the drift columns in place and
    # schema_migrations records version 4.
    init_db()
    init_db()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(drift_metrics)")
        cols = {r["name"] for r in cur.fetchall()}
        for expected in ("model_version", "model_type", "severity",
                         "psi_temperature", "residual_mean_shift",
                         "ks_statistic", "feature_availability",
                         "metrics_json", "is_partial"):
            assert expected in cols, f"missing drift_metrics.{expected}"
        cur.execute("PRAGMA table_info(retraining_candidates)")
        cand_cols = {r["name"] for r in cur.fetchall()}
        for expected in ("trigger_reason", "priority", "status"):
            assert expected in cand_cols
        cur.execute("SELECT version FROM schema_migrations WHERE version = 4")
        assert cur.fetchone() is not None
