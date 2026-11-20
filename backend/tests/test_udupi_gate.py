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


def _fixture_rows(district, grid_point, shift_to_present=False):
    stamps, values = load_district_series(str(FIXTURES), district)
    rows = []
    if shift_to_present:
        now = dt.datetime.now(dt.timezone.utc).replace(
            minute=0, second=0, microsecond=0)
        n = len(values)
        for k, (t, h, p) in enumerate(values):
            when = now - dt.timedelta(hours=n - k)
            rows.append({"timestamp": when.isoformat(), "temp": float(t), "hum": float(h),
                         "pres": float(p), "wind": None, "rain": None, "battery": 12.6,
                         "signal": -70.0, "raw_payload": {}, "qc_flag": "VALID",
                         "grid_point": grid_point})
    else:
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


def test_flat_seed_rows_do_not_dominate_training():
    """End to end: the real 15,000 flat seed rows plus the real backfill
    no longer produce a 1.0 holdout rate. The rate must come back at the
    clean-history value (~0.153) and the gate must still fail honestly."""
    # The real demo file has no station_id column, so the seeder defaults
    # to AWS-01 — the real demo station. Use it here, with the fixture
    # rows on the same station, exactly like the live demo DB.
    _make_station("AWS-01")
    mod = _load_demo_checklist()
    mod.load_telemetry_from_csv(REPO_ROOT / "test_weather.csv")
    insert_telemetry_batch(
        "AWS-01", _fixture_rows("Udupi", "demo:udupi", shift_to_present=True))
    metrics, status, jobrow = _run_if_job("AWS-01")
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
