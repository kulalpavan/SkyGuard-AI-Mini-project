"""Backfill simulated demo history for a station from district data.

The demo needs enough hourly history for two things:
  * the Isolation Forest training job (readiness tier TRAINED needs
    >= 720 observations over >= 14 days), and
  * the Mamba window (24 consecutive hourly rows before a live frame).

`test_weather.csv` (used by scripts/demo_checklist.py) is one flat
value, so it cannot serve either purpose for the ML demo. This script
instead loads the last N hours of a real district series (the same
files the E1-E4 evaluation uses) and inserts them with timestamps
ending one hour ago, so a live frame sent "now" has a proper 24 h
window and a trained model behind it.

The values are real district values; the TIMESTAMPS are shifted to
the present for the demo. That makes this SIMULATED demo history, not
live sensor data — say so on stage, and see DEMO_RUNBOOK.md.

Usage (from the repo root, with the backend env as in the runbook):
    DEMO_MODE=true ../venv/bin/python scripts/backfill_demo_history.py \\
        --station AWS-01 --district Udupi --hours 800

Idempotent: re-running inserts nothing new (the telemetry table
ignores duplicate (station_id, timestamp, grid_point) rows).
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.storage.database import insert_telemetry_batch  # noqa: E402
from ml.mamba_engine import load_district_series  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--station", required=True,
                    help="station id, e.g. AWS-01")
    ap.add_argument("--district", required=True,
                    help="district name in the data files, e.g. Udupi")
    ap.add_argument("--hours", type=int, default=0,
                    help="hourly rows to backfill (default 0 = the whole "
                         "district series, 2,617 h). Use the whole series: "
                         "the IF training promotion gate (holdout anomaly "
                         "rate <= 0.15) fails on a short 800 h backfill "
                         "(rate 0.350) and still fails on the full series "
                         "for Udupi (rate 0.153, measured 2026-11-20 — two "
                         "real dry spells sit in the chronological holdout; "
                         "see DEMO_RUNBOOK.md M2b). An earlier note claimed "
                         "0.139 here; that was wrong.")
    ap.add_argument("--data-dir", default=None,
                    help="district CSV directory (default: ../skyguard-ai/data)")
    args = ap.parse_args()

    data_dir = args.data_dir or str(
        Path(__file__).resolve().parent.parent.parent / "data")
    stamps, values = load_district_series(data_dir, args.district)
    hours = args.hours or len(values)
    if len(values) < hours:
        raise SystemExit(
            f"{args.district}: only {len(values)} hourly points, "
            f"need {hours}")
    stamps, values = stamps[-hours:], values[-hours:]

    # Timestamps: hourly, ending one hour ago (so a live frame sent now
    # has a complete 24 h window immediately before it).
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0,
                                                  microsecond=0)
    rows = []
    for k, (t, h, p) in enumerate(values):
        when = now - dt.timedelta(hours=hours - k)
        rows.append({
            "timestamp": when.isoformat(),
            "temp": float(t), "hum": float(h), "pres": float(p),
            "wind": None, "rain": None, "battery": 12.6, "signal": -70.0,
            "raw_payload": {"source": "simulated demo backfill",
                            "district": args.district,
                            "note": "real district values, timestamps "
                                    "shifted to the present for the demo"},
            "qc_flag": "VALID",
            "grid_point": f"demo:{args.district.lower()}",
        })
    inserted, errors = insert_telemetry_batch(args.station, rows)
    print(f"backfilled {inserted} hourly rows for {args.station} "
          f"({args.district}, {hours} h ending 1 h ago; "
          f"{errors} errors)")
    print("SIMULATED demo history — not live sensor data.")


if __name__ == "__main__":
    main()
