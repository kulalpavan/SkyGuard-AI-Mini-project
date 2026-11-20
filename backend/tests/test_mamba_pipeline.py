"""SkyGuard-AI — Mamba wiring integration tests (backend).

Verifies, against a disposable database, that a registered Mamba model
is actually used by the real scoring path:
  * MAMBA-02 / Dakshina Kannada values — full hybrid: the IF training
    job activates (holdout anomaly rate 0.095, gate <= 0.15), the Mamba
    model registers against that ACTIVE forest, and a fresh ingest
    carries mamba_score / mamba_errors / hybrid_score plus a z_mamba
    evidence term. A spike frame scores higher on Mamba and hybrid.
  * MAMBA-01 / Udupi values — Mamba-only: the IF training job's
    promotion gate FAILS on the Udupi district data (holdout anomaly
    rate 0.153 > 0.15, measured 2026-11-20 — the only district of the
    five that fails), so no ACTIVE IF model exists. The Mamba model
    still registers (fresh-forest fallback) and the pipeline still
    scores it (mamba_score + z_mamba present, hybrid_score None). This
    is the repo's own gate doing its job, not a test failure — see
    ml/MAMBA_README.md.

The district values are real supplied data; the Mamba models are
trained on the same district files. The tests assert wiring, not
accuracy. Mamba is trained with few epochs here (wiring, not accuracy).

Run:  ../venv/bin/python -m pytest backend/tests/test_mamba_pipeline.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.storage.database import (
    init_db, get_active_model_record, insert_telemetry_batch)
from backend.app.services.observation_pipeline import observation_pipeline
from backend.app.services.training_service import training_service
from ml.mamba_engine import load_district_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT.parent / "data"


def _load_register_mamba():
    spec = importlib.util.spec_from_file_location(
        "register_mamba", REPO_ROOT / "scripts" / "register_mamba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    # 409 = already exists (another test in this shared disposable DB
    # created it, or this file ran twice); the station is usable as-is.
    assert resp.status_code in (200, 201, 409), resp.text
    return client, headers


def _backfill(sid, district):
    """Insert the FULL district series (2,617 hourly rows)."""
    stamps, values = load_district_series(str(DATA_DIR), district)
    import datetime as dt
    rows = []
    for ts, (t, h, p) in zip(stamps, values):
        when = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        rows.append({"timestamp": when.isoformat(),
                     "temp": float(t), "hum": float(h), "pres": float(p),
                     "wind": None, "rain": None, "battery": 12.6,
                     "signal": -70.0, "raw_payload": {},
                     "qc_flag": "VALID", "grid_point": ""})
    inserted, errors = insert_telemetry_batch(sid, rows)
    assert inserted == len(values), \
        f"backfill inserted {inserted}/{len(values)} (errors {errors})"
    return stamps, values


def _ingest(sid, ts_iso, temp, hum, pres):
    return observation_pipeline.process(sid, {
        "timestamp": ts_iso, "temp": float(temp), "hum": float(hum),
        "pres": float(pres)})


def _next_ts(stamps, offset_h):
    import datetime as dt
    last = dt.datetime.fromtimestamp(int(stamps[-1]), tz=dt.timezone.utc)
    return (last + dt.timedelta(hours=offset_h)).isoformat()


def test_hourly_window_skips_rapid_frames():
    """_hourly_window picks the most recent 24 hourly-spaced rows,
    skipping rapid live frames sent seconds apart (the demo case) and
    falling back to the on-grid hourly history when a live frame is
    off the hourly grid."""
    from backend.app.services.mamba_service import _hourly_window, WINDOW
    import datetime as dt
    base = dt.datetime(2026, 11, 20, tzinfo=dt.timezone.utc)
    rows = []
    for k in range(30):
        when = base - dt.timedelta(hours=30 - k)
        rows.append({"timestamp": when.isoformat(), "temp": 25.0,
                     "hum": 60.0, "pres": 1010.0})
    # Two rapid live frames, seconds apart, after the hourly history.
    rows.append({"timestamp": (base + dt.timedelta(seconds=10)).isoformat(),
                 "temp": 25.1, "hum": 60.0, "pres": 1010.0})
    rows.append({"timestamp": (base + dt.timedelta(seconds=25)).isoformat(),
                 "temp": 25.2, "hum": 60.0, "pres": 1010.0})
    window = _hourly_window(rows)
    assert len(window) == WINDOW, f"window has {len(window)} rows"
    stamps = [dt.datetime.fromisoformat(r["timestamp"]) for r in window]
    gaps = [(stamps[i + 1] - stamps[i]).total_seconds()
            for i in range(len(stamps) - 1)]
    assert all(abs(g - 3600) <= 120 for g in gaps), f"gaps not hourly: {gaps}"
    # The most recent rapid frame is the window's last row; the earlier
    # rapid frame (15 s before it) was skipped.
    assert window[-1]["temp"] == 25.2
    assert all(r["temp"] != 25.1 for r in window)

    # Off-grid live frame: sent at 12:34 when history is on the hour.
    # It cannot anchor a window, so the window falls back to the last
    # on-hour row (11:00) and runs back hourly from there.
    rows_off = list(rows[:30])
    rows_off.append({"timestamp": (base + dt.timedelta(minutes=34)).isoformat(),
                     "temp": 26.0, "hum": 60.0, "pres": 1010.0})
    window_off = _hourly_window(rows_off)
    assert len(window_off) == WINDOW, f"off-grid window has {len(window_off)} rows"
    stamps_off = [dt.datetime.fromisoformat(r["timestamp"]) for r in window_off]
    gaps_off = [(stamps_off[i + 1] - stamps_off[i]).total_seconds()
                for i in range(len(stamps_off) - 1)]
    assert all(abs(g - 3600) <= 120 for g in gaps_off)
    assert window_off[-1]["timestamp"] == rows[29]["timestamp"], (
        "off-grid window should end at the last on-hour row")
    assert all(r["temp"] != 26.0 for r in window_off)


def test_mamba_hybrid_wired_into_pipeline():
    """MAMBA-02 (Dakshina Kannada values): IF activates, Mamba
    registers, hybrid appears. Station ids are test-only (MAMBA-*)
    so these tests do not collide with other tests' AWS-01/AWS-02
    fixtures in the shared disposable database."""
    sid = "MAMBA-02"
    _make_station(sid, "Dakshina Kannada", 12.87, 74.88, 20.0)
    stamps, values = _backfill(sid, "Dakshina Kannada")
    reg = _load_register_mamba()

    job = training_service.start_training(sid, custom_version="v1.0")
    training_service.execute_training_job(job["job_id"], sid, "v1.0")
    assert get_active_model_record(sid) is not None, "IF model not active"

    result = reg.register_station(sid, "Dakshina Kannada", str(DATA_DIR),
                                  version="v1.0", epochs=5)
    assert Path(result["path"]).exists()
    assert result["if_model"] == "ACTIVE registry model"

    t, h, p = values[-1]
    normal = _ingest(sid, _next_ts(stamps, 1), t, h, p)
    assert normal["status"] == "ACCEPTED", normal
    a = normal["state"]["final_assessment"]
    assert a["mamba_score"] is not None, "no Mamba score in assessment"
    assert np.isfinite(a["mamba_score"])
    assert a["mamba_errors"] is not None and len(a["mamba_errors"]) == 3
    assert a["hybrid_score"] is not None, "no hybrid score in assessment"
    assert 0.0 <= a["hybrid_score"] <= 1.0
    assert a["evidence_vector"]["z_mamba"] is not None, "no z_mamba evidence"
    assert normal["state"]["mamba_model"]["has_model"] is True

    spike = _ingest(sid, _next_ts(stamps, 2), t + 6.0, h, p)
    assert spike["status"] == "ACCEPTED", spike
    sa = spike["state"]["final_assessment"]
    assert sa["mamba_score"] is not None
    assert sa["mamba_score"] > a["mamba_score"], (
        f"spike Mamba {sa['mamba_score']} not above normal {a['mamba_score']}")
    assert sa["mamba_errors"][0] > 3.0, "spike temperature error too small"
    assert sa["hybrid_score"] > a["hybrid_score"], (
        "spike hybrid score did not rise")


def test_mamba_without_active_if():
    """MAMBA-01 (Udupi values): no ACTIVE IF model (the IF promotion
    gate fails on the Udupi district data), but the Mamba model still
    registers and the pipeline still scores it."""
    sid = "MAMBA-01"
    _make_station(sid, "Udupi", 13.34, 74.74, 27.0)
    stamps, values = _backfill(sid, "Udupi")
    reg = _load_register_mamba()
    assert get_active_model_record(sid) is None

    result = reg.register_station(sid, "Udupi", str(DATA_DIR),
                                  version="v1.0", epochs=5)
    assert Path(result["path"]).exists()
    assert result["if_model"] == "fresh forest (district data)"

    t, h, p = values[-1]
    normal = _ingest(sid, _next_ts(stamps, 1), t, h, p)
    assert normal["status"] == "ACCEPTED", normal
    a = normal["state"]["final_assessment"]
    assert a["mamba_score"] is not None, "no Mamba score in assessment"
    assert np.isfinite(a["mamba_score"])
    assert a["hybrid_score"] is None, (
        "hybrid should be None without an ACTIVE IF model")
    assert a["evidence_vector"]["z_mamba"] is not None, "no z_mamba evidence"
    assert normal["state"]["mamba_model"]["has_model"] is True
