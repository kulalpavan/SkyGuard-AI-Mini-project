#!/usr/bin/env python3
"""Register a trained Mamba model for a station in the backend registry.

Trains the Mamba temporal predictor on a district's supplied data
(chronological 60/10/30 split, same as scripts/evaluate_e1_e4.py),
calibrates sigma_e, the ECDFs and the hybrid weights on the validation
split, and registers the artifact with model_type 'Mamba' (status
CANDIDATE — the Isolation Forest keeps the single ACTIVE row).

The Isolation Forest side reuses the station's ACTIVE IF model from the
registry when one exists (so the hybrid calibration matches the deployed
forest); otherwise it trains a fresh forest from the district data.

Usage (from the repo root, backend venv):
  DEMO_MODE=true WEATHER_POLL_ENABLED=false \
      ../venv/bin/python scripts/register_mamba.py \
      --station AWS-01 --district Udupi --epochs 15

Writes nothing to the original repo and never touches the origin remote.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml.mamba_engine import (MambaEncoder, load_district_series,
                             standardise, DISTRICT_ELEVATION_M,
                             consecutive_windows)
from ml.hybrid_engine import (HybridScorer, train_if_forest,
                              if_raw_score, rows_from_series,
                              score_validation_points)
from ml.fault_injection import inject, inject_pair
from backend.app.services.mamba_service import mamba_service
from backend.app.services.model_storage import model_storage_service
from backend.app.storage.database import get_active_model_record
from ml.station_adaptive_pipeline import IsolationForest

WINDOW = 24


def load_active_if(station_id):
    """Load the ACTIVE Isolation Forest (forest, climatology, stats)."""
    record = get_active_model_record(station_id)
    if not record:
        return None
    artifact = model_storage_service.load_artifact(record["model_location"])
    if not artifact or artifact.get("schema_version", 1) < 2:
        return None
    forest = IsolationForest.from_dict(artifact["model_weights"])
    clim = artifact.get("climatology")
    stats = artifact["model_card"]["normalization_stats"]
    if clim is None or stats is None:
        return None
    return forest, clim, stats


def register_station(station, district, data_dir, version="v1.0",
                     epochs=15, d_model=32, state_dim=8):
    """Train a Mamba model for a station and register it.

    Used by the CLI (main) and by the backend integration test. Returns
    a dict with the artifact path, the test-split metrics, the hybrid
    calibration and the trained Mamba model.
    """
    station = station.strip().upper()
    t0 = time.time()
    stamps, values = load_district_series(data_dir, district)
    n = len(values)
    n_train, n_val = int(0.6 * n), int(0.1 * n)
    lo_va, hi_va = n_train, n_train + n_val
    lo_te = hi_va
    mean, std = standardise(values[:n_train])
    z = (values - mean) / std

    # --- Train Mamba ---
    Xtr, Ytr = consecutive_windows(stamps[:n_train], z[:n_train], WINDOW)
    Xva, Yva = consecutive_windows(stamps[lo_va:hi_va], z[lo_va:hi_va], WINDOW)
    mamba = MambaEncoder(d_model=d_model, state_dim=state_dim, seed=42)
    mamba.mean, mamba.std = mean, std
    print(f"training Mamba for {station} ({district}): "
          f"{len(Xtr)} train windows, {len(Xva)} val windows")
    mamba.train(Xtr, Ytr, Xva, Yva, epochs=epochs, batch=64, lr=0.003,
                log=print)
    va_out, _ = mamba.forward(Xva)
    signed_va = (va_out[:, -1, :] - Yva[:, -1, :]) * std
    mamba.calibrate_sigma_e_signed(signed_va)
    print(f"sigma_e: T={mamba.sigma_e[0]:.3f} C, RH={mamba.sigma_e[1]:.3f} %RH, "
          f"P={mamba.sigma_e[2]:.3f} hPa")

    # --- Isolation Forest side ---
    rows = rows_from_series(stamps, values)
    active = load_active_if(station)
    if active:
        forest, clim, stats = active
        print("using the station's ACTIVE Isolation Forest from the registry")
    else:
        print("no ACTIVE Isolation Forest; training a fresh forest from the "
              "district data (register the station's own IF model first for "
              "a production hybrid)")
        forest, clim, stats = train_if_forest(
            rows[:n_train], elevation_m=DISTRICT_ELEVATION_M[district])

    # --- Validation calibration (clean) ---
    if_raw, mamba_raw = score_validation_points(
        mamba, forest, clim, stats, rows, values, lo_va, hi_va, mean, std)
    ok_if = if_raw[np.isfinite(if_raw)]
    ok_m = mamba_raw[np.isfinite(mamba_raw)]
    tau_m_raw = float(np.quantile(ok_m, 0.95))
    print(f"validation: {len(ok_if)} IF scores, {len(ok_m)} Mamba scores; "
          f"tau_M (raw, 95th pct) = {tau_m_raw:.2f}")

    # --- Alpha calibration on an injected validation copy ---
    if district in ("Udupi", "Dakshina Kannada"):
        partner = "Dakshina Kannada" if district == "Udupi" else "Udupi"
        _, pvals = load_district_series(data_dir, partner)
        (vu, lu, _), (vk, lk, _) = inject_pair(
            values[lo_va:hi_va].copy(), pvals[lo_va:hi_va].copy())
        inj_vals, inj_labels = (vu, lu) if district == "Udupi" else (vk, lk)
    else:
        inj_vals, inj_labels, _ = inject(values[lo_va:hi_va].copy())
    inj_rows = rows_from_series(stamps[lo_va:hi_va], inj_vals)
    inj_full_values = values.copy()
    inj_full_values[lo_va:hi_va] = inj_vals
    inj_full_rows = rows[:lo_va] + inj_rows + rows[hi_va:]
    if_inj, m_inj = score_validation_points(
        mamba, forest, clim, stats, inj_full_rows, inj_full_values,
        lo_va, hi_va, mean, std)
    vi = np.isfinite(if_inj) & np.isfinite(m_inj)
    scorer = HybridScorer(station)
    scorer.calibrate(ok_if, ok_m)
    res = scorer.calibrate_alpha(if_inj[vi], m_inj[vi], inj_labels[vi],
                                 clean_if_raw=if_raw,
                                 clean_mamba_raw=mamba_raw)
    print(f"alpha={res['alpha']:.2f} beta={res['beta']:.2f} "
          f"tau_hybrid={res['tau_hybrid']:.3f} val F1={res['f1']:.3f} "
          f"FPR={res['fpr']:.3f}")

    # --- Test errors vs persistence (same test split, normal points) ---
    Xte, Yte = consecutive_windows(stamps[lo_te:], z[lo_te:], WINDOW)
    te_out, _ = mamba.forward(Xte)
    signed_te = (te_out[:, -1, :] - Yte[:, -1, :]) * std
    pers = np.abs(values[lo_te + WINDOW:lo_te + WINDOW + len(Xte)]
                  - values[lo_te + WINDOW - 1:lo_te + WINDOW - 1 + len(Xte)])
    metrics = {}
    for i, name in enumerate(["temperature_C", "humidity_pct", "pressure_hPa"]):
        e = signed_te[:, i]
        metrics[name] = {
            "mamba_MAE": float(np.mean(np.abs(e))),
            "mamba_RMSE": float(np.sqrt(np.mean(e ** 2))),
            "persistence_MAE": float(np.mean(pers[:, i])),
            "persistence_RMSE": float(np.sqrt(np.mean(pers[:, i] ** 2))),
        }
        print(f"test {name}: Mamba MAE={metrics[name]['mamba_MAE']:.3f} "
              f"vs persistence MAE={metrics[name]['persistence_MAE']:.3f}")

    # --- Build + register the artifact ---
    model_id = f"{station}_MAMBA_{version.replace('.', '_')}"
    calibration = {
        "tau_M_raw": tau_m_raw,
        "alpha": res["alpha"], "beta": res["beta"],
        "tau_hybrid": res["tau_hybrid"],
        "if_val_raw": ok_if.tolist(), "mamba_val_raw": ok_m.tolist(),
    }
    training_info = {
        "district": district, "source": "supplied district file",
        "train_range": [rows[0]["timestamp"], rows[n_train - 1]["timestamp"]],
        "n_train_windows": int(len(Xtr)), "n_val_windows": int(len(Xva)),
        "epochs": epochs, "d_model": d_model,
        "state_dim": state_dim, "window": WINDOW,
        "if_model": "ACTIVE registry model" if active else "fresh forest (district data)",
    }
    artifact = mamba_service.build_mamba_artifact(
        station, model_id, version, mamba, calibration, training_info)
    path = mamba_service.register_mamba_model(
        station, artifact, version, len(Xtr), metrics)
    print(f"registered {model_id} -> {path} ({time.time() - t0:.0f}s)")
    return {"path": path, "model_id": model_id, "metrics": metrics,
            "calibration": calibration, "mamba": mamba,
            "if_model": "ACTIVE registry model" if active
                        else "fresh forest (district data)"}


def main():
    ap = argparse.ArgumentParser(description="Train and register a Mamba model")
    ap.add_argument("--station", required=True, help="e.g. AWS-01")
    ap.add_argument("--district", required=True,
                    help="e.g. Udupi (must match the station)")
    ap.add_argument("--data-dir", default=None,
                    help="district CSV directory (default: the data dir "
                         "next to this repo, ~/workspace/skyguard-ai/data)")
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--state-dim", type=int, default=8)
    args = ap.parse_args()
    data_dir = args.data_dir or str(
        Path(__file__).resolve().parent.parent.parent / "data")
    register_station(args.station, args.district, data_dir,
                     version=args.version, epochs=args.epochs,
                     d_model=args.d_model, state_dim=args.state_dim)


if __name__ == "__main__":
    main()
