"""Udupi gate regression tests.

Root causes, both measured 2026-11-20 (see verification/udupi_diagnosis.py):
  * The live demo DB's Udupi holdout rate was 1.0 because the 15,000 flat
    rows of test_weather.csv (25.0 C / 60% / 1010.0 hPa, loaded by
    scripts/demo_checklist.py) dominated the training history: the
    climatology fit collapsed to robust_sigma 0.0 and every real holdout
    row scored anomalous. Those rows are replay fixture, not observations,
    so the seeder now tags them grid_point 'synthetic:demo-seed' (excluded
    from the training history by the existing 'synthetic:*' rule) and
    history() additionally drops long exactly-constant runs.
  * With a clean history Udupi's rate is 0.153 — genuine, not a bug: two
    regional monsoon-break dry spells (24-25 Aug, 9-13 Sep 2025) sit in the
    chronological holdout; 76 of the 80 flagged rows are those spells, and
    without them the rate is 0.011. The 0.15 gate must keep rejecting it.
"""
import datetime as dt
import importlib.util
import json
from pathlib import Path

from backend.app.storage.database import (
    get_db, init_db, insert_telemetry_batch)
from backend.app.services.training_service import training_service
from ml.mamba_engine import load_district_series

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_demo_checklist():
    spec = importlib.util.spec_from_file_location(
        "demo_checklist", REPO_ROOT / "scripts" / "demo_checklist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_station(sid, lat=13.34, lon=74.74, elev=27.0):
    init_db()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM stations WHERE station_id = ?", (sid,))
        if cur.fetchone():
            return
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        cur.execute(
            """INSERT INTO stations (station_id, station_name, username, password_hash,
               latitude, longitude, elevation, region, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, "Test", f"u_{sid.lower()}", "hash", lat, lon, elev,
             "Coastal Karnataka", "ACTIVE", now, now))
        conn.commit()


def _fixture_rows(district, grid_point):
    stamps, values = load_district_series(str(FIXTURES), district)
    rows = []
    for ts, (t, h, p) in zip(stamps, values):
        when = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
        rows.append({"timestamp": when.isoformat(), "temp": float(t), "hum": float(h),
                     "pres": float(p), "wind": None, "rain": None, "battery": 12.6,
                     "signal": -70.0, "raw_payload": {}, "qc_flag": "VALID",
                     "grid_point": grid_point})
    return rows


def _flat_rows(n=100, start="2026-01-01T00:00:00+00:00", grid_point=""):
    base = dt.datetime.fromisoformat(start)
    return [{"timestamp": (base + dt.timedelta(hours=k)).isoformat(),
             "temp": 25.0, "hum": 60.0, "pres": 1010.0, "wind": 12.0, "rain": 0.0,
             "battery": 12.6, "signal": -70.0, "raw_payload": {},
             "qc_flag": "VALID", "grid_point": grid_point}
            for k in range(n)]


def _run_if_job(sid):
    job = training_service.start_training(sid, custom_version="v1.0")
    training_service.execute_training_job(job["job_id"], sid, "v1.0")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT metrics, status FROM model_registry "
                    "WHERE station_id = ? AND model_version = 'v1.0'", (sid,))
        reg = cur.fetchone()
        cur.execute("SELECT status, error_message FROM training_jobs WHERE id = ?",
                    (job["job_id"],))
        jobrow = cur.fetchone()
    return json.loads(reg["metrics"]), reg["status"], jobrow


def test_history_excludes_synthetic_seed_rows():
    """Rows the seeder tags synthetic:demo-seed never reach training."""
    _make_station("UD-01")
    insert_telemetry_batch("UD-01", _flat_rows(300, grid_point="synthetic:demo-seed"))
    insert_telemetry_batch("UD-01", _fixture_rows("Udupi", "demo:udupi"))
    rows, raw = training_service.history("UD-01")
    assert len(rows) == 2617, f"history has {len(rows)} rows, expected only the 2,617 fixture rows"
    assert all(r["grid_point"] != "synthetic:demo-seed" for r in rows)


def test_history_excludes_long_constant_runs():
    """Legacy flat rows (untagged, grid_point='') in one long constant run
    are dropped from the training history too."""
    _make_station("UD-02")
    # Flat block first (consecutive hours), fixture rows after it so the
    # constant run is unbroken in the sorted history.
    insert_telemetry_batch("UD-02", _flat_rows(300, start="2025-01-01T00:00:00+00:00"))
    insert_telemetry_batch("UD-02", _fixture_rows("Udupi", "demo:udupi"))
    rows, raw = training_service.history("UD-02")
    assert len(rows) == 2617, f"history has {len(rows)} rows, expected only the 2,617 fixture rows"
    assert not any(r["temp"] == 25.0 and r["hum"] == 60.0 and r["pres"] == 1010.0
                   for r in rows)


def test_demo_checklist_tags_seed_rows_synthetic(tmp_path):
    """The seeder labels test_weather.csv rows as synthetic replay fixture."""
    _make_station("UD-03")
    csv_path = tmp_path / "flat.csv"
    csv_path.write_text(
        "station_id,date,temp,hum,pres,wind,rain\n"
        "UD-03,2026-01-01T00:00:00+00:00,25.0,60.0,1010.0,12.0,0.0\n"
        "UD-03,2026-01-01T01:00:00+00:00,25.0,60.0,1010.0,12.0,0.0\n")
    mod = _load_demo_checklist()
    mod.load_telemetry_from_csv(csv_path)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT grid_point FROM telemetry WHERE station_id = 'UD-03'")
        points = {r["grid_point"] for r in cur.fetchall()}
    assert points == {"synthetic:demo-seed"}, points


def test_flat_seed_rows_do_not_dominate_training(tmp_path):
    """End to end: the real 15,000 flat seed rows plus the real backfill
    no longer produce a 1.0 holdout rate. The rate must come back at the
    clean-history value (~0.153) and the gate must still fail honestly.

    Two determinism traps, both measured 2026-11-20 (CI run 44768593732):
      * The fixture must be UNSHIFTED. The climatology model uses
        day-of-year, so shifting the fixture to the current hour makes
        the holdout rate wobble by a few rows around the 0.15 line —
        measured 0.1450 / 0.1508 / 0.1450 for the same series shifted
        to 16:00 / 10:00 / 04:00 UTC — and this test flipped with the
        clock (passed 15:54 UTC, failed 16:01 UTC).
      * The seed rows must land on a TEST-ONLY station (UD-04), not
        AWS-01. The disposable DB is shared, and AWS-01 accumulates
        stray rows from other tests — test_timestamp_parsing uploads
        one row for AWS-01 once the ESP32 security tests have created
        the station. One extra row moves the 80/20 split by one and
        flips this boundary test (reproduced: esp32 + parsing + this
        file fails; esp32 + this file passes; parsing + this file
        passes because the upload itself errors without AWS-01)."""
    # The real demo file has no station_id column, so the seeder would
    # default to AWS-01. Retarget a copy at UD-04: same 15,000 real
    # flat rows, private history.
    csv_path = tmp_path / "flat-seed.csv"
    with (REPO_ROOT / "test_weather.csv").open() as src:
        lines = src.readlines()
    csv_path.write_text(
        "station_id," + lines[0]
        + "".join(f"UD-04,{line}" for line in lines[1:]))
    _make_station("UD-04")
    mod = _load_demo_checklist()
    mod.load_telemetry_from_csv(csv_path)
    insert_telemetry_batch(
        "UD-04", _fixture_rows("Udupi", "demo:udupi"))
    metrics, status, jobrow = _run_if_job("UD-04")
    rate = metrics["anomaly_rate"]
    assert 0.10 < rate < 0.25, f"holdout rate {rate} — flat seed rows still dominate the fit"
    assert status == "CANDIDATE", f"model status {status}, expected CANDIDATE"
    assert jobrow["status"] == "FAILED"
    assert "above the 15% limit" in (jobrow["error_message"] or ""), jobrow["error_message"]


def test_gate_still_rejects_genuine_high_holdout_rate():
    """Fixture-only Udupi: the gate still rejects the genuine 0.153 rate.
    Nothing about the fix weakens it."""
    _make_station("UD-05")
    insert_telemetry_batch("UD-05", _fixture_rows("Udupi", "demo:udupi"))
    metrics, status, jobrow = _run_if_job("UD-05")
    rate = metrics["anomaly_rate"]
    assert 0.10 < rate < 0.25, f"holdout rate {rate} — expected the genuine ~0.153"
    assert status == "CANDIDATE"
    assert jobrow["status"] == "FAILED"
    assert "above the 15% limit" in (jobrow["error_message"] or "")
