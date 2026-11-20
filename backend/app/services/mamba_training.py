"""Backend service: train a Mamba temporal model from a station's own telemetry.

This is the in-product Mamba training path behind the Train button. It is
the same method as the registration CLI (scripts/register_mamba.py) —
chronological 60/10/30 split, 15 epochs, sigma_e calibration on validation,
ECDF + hybrid-weight (alpha) calibration on clean and injected validation
copies, artifact built and registered exactly the same way (CANDIDATE, never
ACTIVE) — but the data comes from the station's stored telemetry history
(the same history the Isolation Forest job trains on), not from a district
CSV file. That history is what detection runs on: the Open-Meteo poller
writes its observations into the same telemetry table (via the observation
pipeline), so training on it trains on the detection population.

Minimum history: at least 720 hourly observations (30 days at hourly
cadence). Below that the validation region is too small for the injected
fault schedule to place scenarios, so the hybrid-weight calibration would
have nothing to calibrate on. A station that does not meet it fails
honestly with MAMBA_HISTORY_INSUFFICIENT instead of training on too
little data.
"""

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover - backend boots without numpy
    np = None

from backend.app.services.mamba_service import (
    WINDOW, build_mamba_artifact, register_mamba_model)
from ml.fault_injection import inject
from ml.feature_engine import instant
from ml.hybrid_engine import (
    HybridScorer, rows_from_series, score_validation_points,
    train_if_forest)
from ml.mamba_engine import MambaEncoder, consecutive_windows, standardise

MIN_HOURLY_POINTS = 720
MIN_TRAIN_WINDOWS = 100
MIN_VAL_WINDOWS = 10
EPOCHS = 15
D_MODEL = 32
STATE_DIM = 8


def hourly_series_from_rows(rows):
    """Collapse station telemetry rows to one mean value per UTC hour.

    Returns (stamps, values) where stamps are epoch seconds (hour floors,
    ascending) and values is an (n, 3) float array [T, RH, P]. Rows with
    missing or non-finite core values are skipped. Hours with no rows are
    simply absent — windows straddling them are dropped downstream by
    consecutive_windows, and validation scoring skips them too.
    """
    buckets = {}
    for row in rows:
        try:
            ts = instant(row["timestamp"])
            vals = (float(row["temp"]), float(row["hum"]),
                    float(row["pres"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in vals):
            continue
        hour = ts.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append(vals)
    stamps = sorted(buckets)
    values = np.array(
        [[float(np.mean([v[i] for v in buckets[h]])) for i in range(3)]
         for h in stamps], dtype=np.float64)
    epochs = np.array([int(h.timestamp()) for h in stamps], dtype=np.int64)
    return epochs, values


def train_mamba_from_station_history(
        station_id, rows, elevation_m, version, forest_bundle=None,
        epochs=EPOCHS, d_model=D_MODEL, state_dim=STATE_DIM, stage=None):
    """Train and register a Mamba model from a station's telemetry history.

    rows: the station's clean history rows (as training_service.history()
    returns them). elevation_m: the station's elevation (fresh-forest
    fallback). version: the Mamba model version string. forest_bundle:
    optional (forest, climatology, stats) from the Isolation Forest job
    that just ran — the hybrid calibration then uses the exact forest the
    job trained; otherwise a fresh forest is trained from the station's
    own history. stage: optional callback called with each completed
    phase name ('Training Mamba', 'Calibrating Mamba',
    'Mamba Registered'), mirroring the training job's stage recording.

    Returns the same result dict shape as scripts/register_mamba.py's
    register_station: path, model_id, metrics, calibration, mamba,
    if_model.
    """
    if np is None:
        raise ValueError(
            "MAMBA_NUMPY_UNAVAILABLE: numpy is required to train Mamba")
    sid = station_id.strip().upper()
    stamps, values = hourly_series_from_rows(rows)
    n = len(values)
    if n < MIN_HOURLY_POINTS:
        raise ValueError(
            "MAMBA_HISTORY_INSUFFICIENT: need at least "
            f"{MIN_HOURLY_POINTS} hourly observations (30 days at hourly "
            f"cadence) to train Mamba; station has {n} hourly points")
    n_train, n_val = int(0.6 * n), int(0.1 * n)
    lo_va, hi_va = n_train, n_train + n_val
    lo_te = hi_va
    mean, std = standardise(values[:n_train])
    z = (values - mean) / std

    # --- Train Mamba (same defaults as the registration CLI) ---
    Xtr, Ytr = consecutive_windows(stamps[:n_train], z[:n_train], WINDOW)
    Xva, Yva = consecutive_windows(stamps[lo_va:hi_va], z[lo_va:hi_va], WINDOW)
    if len(Xtr) < MIN_TRAIN_WINDOWS or len(Xva) < MIN_VAL_WINDOWS:
        raise ValueError(
            "MAMBA_HISTORY_INSUFFICIENT: too few consecutive hourly "
            f"windows (train {len(Xtr)}, validation {len(Xva)}); the "
            "station history has gaps or too little hourly coverage")
    mamba = MambaEncoder(d_model=d_model, state_dim=state_dim, seed=42)
    mamba.mean, mamba.std = mean, std
    mamba.train(Xtr, Ytr, Xva, Yva, epochs=epochs, batch=64, lr=0.003)
    va_out, _ = mamba.forward(Xva)
    signed_va = (va_out[:, -1, :] - Yva[:, -1, :]) * std
    mamba.calibrate_sigma_e_signed(signed_va)
    if stage:
        stage("Training Mamba")

    # --- Isolation Forest side ---
    series_rows = rows_from_series(stamps, values)
    if forest_bundle is not None:
        forest, clim, stats = forest_bundle
        if_model = "job Isolation Forest"
    else:
        forest, clim, stats = train_if_forest(
            series_rows[:n_train], elevation_m=elevation_m)
        if_model = "fresh forest (station telemetry)"

    # --- Validation calibration (clean) ---
    if_raw, mamba_raw = score_validation_points(
        mamba, forest, clim, stats, series_rows, values,
        lo_va, hi_va, mean, std, window=WINDOW, stamps=stamps)
    ok_if = if_raw[np.isfinite(if_raw)]
    ok_m = mamba_raw[np.isfinite(mamba_raw)]
    if len(ok_if) == 0 or len(ok_m) == 0:
        raise ValueError(
            "MAMBA_CALIBRATION_FAILED: no usable validation scores "
            f"(IF {len(ok_if)}, Mamba {len(ok_m)}); the validation region "
            "has too many gaps or missing values")
    tau_m_raw = float(np.quantile(ok_m, 0.95))

    # --- Alpha calibration on an injected validation copy ---
    inj_vals, inj_labels, _ = inject(values[lo_va:hi_va].copy())
    inj_rows = rows_from_series(stamps[lo_va:hi_va], inj_vals)
    inj_full_values = values.copy()
    inj_full_values[lo_va:hi_va] = inj_vals
    inj_full_rows = series_rows[:lo_va] + inj_rows + series_rows[hi_va:]
    if_inj, m_inj = score_validation_points(
        mamba, forest, clim, stats, inj_full_rows, inj_full_values,
        lo_va, hi_va, mean, std, window=WINDOW, stamps=stamps)
    vi = np.isfinite(if_inj) & np.isfinite(m_inj)
    scorer = HybridScorer(sid)
    scorer.calibrate(ok_if, ok_m)
    res = scorer.calibrate_alpha(if_inj[vi], m_inj[vi], inj_labels[vi],
                                 clean_if_raw=if_raw,
                                 clean_mamba_raw=mamba_raw)
    if stage:
        stage("Calibrating Mamba")

    # --- Test errors vs persistence (same test split, normal points) ---
    Xte, Yte = consecutive_windows(stamps[lo_te:], z[lo_te:], WINDOW)
    te_out, _ = mamba.forward(Xte)
    signed_te = (te_out[:, -1, :] - Yte[:, -1, :]) * std
    # Persistence error per window, aligned to each window's own last
    # step (identical to the CLI's positional form on consecutive data,
    # and still correct when the test region has gaps).
    pers = np.abs(Yte[:, -1, :] - Xte[:, -1, :]) * std
    metrics = {}
    for i, name in enumerate(["temperature_C", "humidity_pct", "pressure_hPa"]):
        e = signed_te[:, i]
        metrics[name] = {
            "mamba_MAE": float(np.mean(np.abs(e))),
            "mamba_RMSE": float(np.sqrt(np.mean(e ** 2))),
            "persistence_MAE": float(np.mean(pers[:, i])),
            "persistence_RMSE": float(np.sqrt(np.mean(pers[:, i] ** 2))),
        }

    # --- Build + register the artifact (same format as the CLI) ---
    model_id = f"{sid}_MAMBA_{version.replace('.', '_')}"
    calibration = {
        "tau_M_raw": tau_m_raw,
        "alpha": res["alpha"], "beta": res["beta"],
        "tau_hybrid": res["tau_hybrid"],
        "if_val_raw": ok_if.tolist(), "mamba_val_raw": ok_m.tolist(),
    }
    training_info = {
        "station": sid, "source": "station telemetry",
        "train_range": [series_rows[0]["timestamp"],
                        series_rows[n_train - 1]["timestamp"]],
        "n_train_windows": int(len(Xtr)), "n_val_windows": int(len(Xva)),
        "epochs": epochs, "d_model": d_model,
        "state_dim": state_dim, "window": WINDOW,
        "if_model": if_model,
    }
    artifact = build_mamba_artifact(
        sid, model_id, version, mamba, calibration, training_info)
    path = register_mamba_model(sid, artifact, version, len(Xtr), metrics)
    if stage:
        stage("Mamba Registered")
    return {"path": path, "model_id": model_id, "metrics": metrics,
            "calibration": calibration, "mamba": mamba,
            "if_model": if_model}
