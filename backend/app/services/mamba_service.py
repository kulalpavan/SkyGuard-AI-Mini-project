"""Backend service: Mamba temporal model registry, scoring, and hybrid wiring.

A Mamba model is a *companion* to the station's Isolation Forest model,
not a replacement. The model_registry enforces a single ACTIVE model per
station (unique partial index), and that row is the Isolation Forest, so
Mamba artifacts are registered with model_type 'Mamba' and status
'CANDIDATE' (never ACTIVE). The pipeline looks the Mamba model up by
station + model_type and uses the latest non-archived row.

Artifact (stored via model_storage_service, sha256-verified, immutable):
  schema_version 2, model_type 'mamba', model_card, model_weights
  (MambaEncoder.to_artifact_dict: params, mean, std, sigma_e, tau_M),
  hybrid calibration (alpha, beta, tau_hybrid, if_val_raw, mamba_val_raw).

The pipeline calls score_observation() with the preceding observations;
the window is the last 24 hourly-spaced rows (3600 s gaps, small
tolerance). Rows that arrive too close together — rapid live frames
sent seconds apart, or a sensor burst — are skipped when selecting the
window, so the model always scores on the hourly history it was
trained on. If no 24-row hourly window exists the result reports
has_model=True with status INSUFFICIENT_WINDOW and no score, rather
than scoring on data the model was not trained for.
"""

import datetime as dt
import json
import math

# NumPy is imported lazily: the backend must still boot on a machine
# without it (the repo's requirements.txt has no NumPy; Mamba scoring
# reports NUMPY_UNAVAILABLE there instead of breaking the import chain).
try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None

from backend.app.storage.database import get_db
from backend.app.services.model_storage import model_storage_service
from ml.feature_engine import instant

MAMBA_MODEL_TYPE = "Mamba"
WINDOW = 24
GAP_TOLERANCE_S = 120

_cache = {}


def get_mamba_record(station_id):
    """Latest non-archived Mamba registry row for a station, or None."""
    sid = station_id.strip().upper()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM model_registry
               WHERE station_id = ? AND model_type = ? AND status != 'ARCHIVED'
               ORDER BY id DESC LIMIT 1""",
            (sid, MAMBA_MODEL_TYPE))
        return cur.fetchone()


def load_mamba_model(station_id):
    """Load (encoder, artifact, hybrid_scorer) for a station, cached.

    Returns (None, None, None) when no Mamba model is registered.
    The hybrid scorer is built only when the artifact carries the
    validation calibration arrays (if_val_raw / mamba_val_raw).
    """
    if np is None:
        return None, None, None
    from ml.mamba_engine import MambaEncoder
    from ml.hybrid_engine import HybridScorer

    record = get_mamba_record(station_id)
    if not record:
        return None, None, None
    location = record["model_location"]
    if location in _cache:
        return _cache[location]
    artifact = model_storage_service.load_artifact(location)
    if not artifact or artifact.get("model_type") != "mamba":
        return None, None, None
    encoder = MambaEncoder.from_artifact_dict(artifact["model_weights"])
    scorer = None
    hybrid = artifact.get("hybrid") or {}
    if hybrid.get("if_val_raw") and hybrid.get("mamba_val_raw"):
        scorer = HybridScorer(station_id)
        scorer.calibrate(np.asarray(hybrid["if_val_raw"], dtype=float),
                         np.asarray(hybrid["mamba_val_raw"], dtype=float))
        scorer.alpha = float(hybrid.get("alpha", 0.5))
        scorer.beta = round(1.0 - scorer.alpha, 2)
        scorer.tau_hybrid = float(hybrid.get("tau_hybrid", 0.95))
    _cache[location] = (encoder, artifact, scorer)
    return _cache[location]


def clear_cache():
    _cache.clear()


def _hourly_window(rows):
    """Most recent 24 rows forming an hourly-spaced sequence (oldest first).

    Tries each row from the most recent backwards as the window's last
    row, and extends backwards keeping only rows ~3600 s apart (within
    GAP_TOLERANCE_S). Rows that arrive too close together — rapid live
    frames sent seconds apart — are skipped, and a live frame whose
    timestamp is off the hourly grid (e.g. sent at 12:34 when the
    history is on the hour) cannot anchor a window, so the search falls
    back to the most recent on-grid hourly row. Returns fewer than
    WINDOW rows (possibly empty) when no 24-row hourly sequence exists.
    """
    stamps = []
    for row in rows:
        try:
            stamps.append(instant(row["timestamp"]))
        except (KeyError, TypeError, ValueError):
            stamps.append(None)
    n = len(rows)
    for end in range(n - 1, WINDOW - 2, -1):
        if stamps[end] is None:
            continue
        window = [rows[end]]
        prev = stamps[end]
        j = end - 1
        while len(window) < WINDOW and j >= 0:
            if stamps[j] is None:
                j -= 1
                continue
            gap = (prev - stamps[j]).total_seconds()
            if abs(gap - 3600) <= GAP_TOLERANCE_S:
                window.append(rows[j])
                prev = stamps[j]
            j -= 1
        if len(window) == WINDOW:
            window.reverse()
            return window
    return []


def score_observation(station_id, observation, history=None):
    """Score one observation with the station's Mamba model.

    observation: dict with temp/hum/pres/timestamp. history: preceding
    observations in chronological order (oldest first); the last 24
    hourly ones form the window. Returns a dict with has_model, score
    (raw S_M = max(e / sigma_e)), errors [e_T, e_H, e_P] in original
    units, prediction, threshold (tau_M on the raw scale) and is_anomaly.
    """
    absent = {"station_id": station_id, "has_model": False,
              "status": "MAMBA_NOT_REGISTERED", "score": None,
              "errors": None, "prediction": None, "is_anomaly": False}
    if np is None:
        return {**absent, "status": "NUMPY_UNAVAILABLE",
                "reason": "numpy is required for Mamba scoring"}
    encoder, artifact, _ = load_mamba_model(station_id)
    if encoder is None:
        return absent
    rows = [r for r in (history or []) if r is not None]
    window_rows = _hourly_window(rows)
    if len(window_rows) < WINDOW:
        return {**absent, "has_model": True, "status": "INSUFFICIENT_WINDOW",
                "reason": f"need {WINDOW} hourly history rows, have "
                          f"{len(window_rows)} hourly-spaced rows"}
    try:
        stamps = [instant(r["timestamp"]) for r in window_rows]
        gaps = [(stamps[i + 1] - stamps[i]).total_seconds()
                for i in range(len(stamps) - 1)]
        if any(abs(g - 3600) > GAP_TOLERANCE_S for g in gaps):
            return {**absent, "has_model": True, "status": "NON_HOURLY_WINDOW",
                    "reason": "history is not consecutive hourly data"}
        vals = np.array([[float(r["temp"]), float(r["hum"]), float(r["pres"])]
                         for r in window_rows], dtype=float)
        if not np.isfinite(vals).all():
            return {**absent, "has_model": True, "status": "WINDOW_HAS_MISSING",
                    "reason": "window contains missing values"}
    except (KeyError, TypeError, ValueError) as error:
        return {**absent, "has_model": True, "status": "WINDOW_UNUSABLE",
                "reason": str(error)}
    z = (vals - encoder.mean) / encoder.std
    pred_std = encoder.predict_next(z)
    pred = pred_std * encoder.std + encoder.mean
    observed = np.array([float(observation["temp"]), float(observation["hum"]),
                         float(observation["pres"])], dtype=float)
    errors = np.abs(observed - pred)
    raw = float(encoder.temporal_score(errors))
    tau = artifact.get("model_weights", {}).get("tau_M")
    tau = float(tau) if tau is not None else None
    return {"station_id": station_id, "has_model": True,
            "model_id": artifact.get("model_card", {}).get("model_id"),
            "status": "ANOMALY" if (tau is not None and raw > tau) else "NORMAL",
            "score": raw, "errors": [float(e) for e in errors],
            "prediction": [float(p) for p in pred],
            "threshold": tau, "is_anomaly": bool(tau is not None and raw > tau)}


def combine_scores(station_id, if_raw, mamba_raw):
    """Calibrated hybrid score from the two raw engine scores.

    Returns None when the station's Mamba artifact has no hybrid
    calibration. Otherwise returns the HybridScorer.score() dict.
    """
    if np is None:
        return None
    _, _, scorer = load_mamba_model(station_id)
    if scorer is None:
        return None
    return scorer.score(float(if_raw), float(mamba_raw))


def build_mamba_artifact(station_id, model_id, version, encoder, calibration,
                         training_info):
    """Assemble the registry artifact dict for a trained encoder.

    calibration: dict with sigma_e (already on the encoder), tau_M_raw,
    alpha, beta, tau_hybrid, if_val_raw, mamba_val_raw.
    training_info: dict describing the training data (district, ranges,
    n_windows, metrics).
    """
    weights = encoder.to_artifact_dict()
    weights["tau_M"] = float(calibration["tau_M_raw"])
    artifact = {
        "schema_version": 2,
        "model_type": "mamba",
        "model_card": {
            "model_id": model_id,
            "station_id": station_id,
            "version": version,
            "algorithm": "Mamba selective state-space temporal predictor",
            "d_model": encoder.d,
            "state_dim": encoder.N,
            "window": WINDOW,
            "readiness_tier": "TRAINED",
            "trained_on": training_info,
            "metrics": {
                "sigma_e": [float(v) for v in encoder.sigma_e],
                "tau_M_raw": float(calibration["tau_M_raw"]),
                "alpha": float(calibration["alpha"]),
                "beta": float(calibration["beta"]),
                "tau_hybrid": float(calibration["tau_hybrid"]),
            },
        },
        "model_weights": weights,
        "hybrid": {
            "alpha": float(calibration["alpha"]),
            "beta": float(calibration["beta"]),
            "tau_hybrid": float(calibration["tau_hybrid"]),
            "if_val_raw": [float(v) for v in calibration["if_val_raw"]],
            "mamba_val_raw": [float(v) for v in calibration["mamba_val_raw"]],
        },
    }
    return artifact


def register_mamba_model(station_id, artifact, version, training_rows, metrics):
    """Persist a Mamba artifact and register it (status CANDIDATE).

    The Mamba model is a companion model: it is never set ACTIVE (the
    unique partial index allows one ACTIVE model per station, which is
    the Isolation Forest). The pipeline uses the latest non-archived
    Mamba row. Returns the artifact path.
    """
    sid = station_id.strip().upper()
    model_id = artifact["model_card"]["model_id"]
    path = model_storage_service.save_artifact(sid, model_id, artifact)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO model_registry (station_id,model_id,model_version,
                 model_type,model_location,feature_schema,training_rows,
                 threshold,contamination_rate,sha256,status,
                 training_started_at,training_completed_at,metrics,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, model_id, version, MAMBA_MODEL_TYPE, path,
             json.dumps(["temperature_C", "humidity_pct", "pressure_hPa"]),
             int(training_rows), float(artifact["model_card"]["metrics"]["tau_M_raw"]),
             0.05, artifact["sha256"], "CANDIDATE", now, now,
             json.dumps(metrics), now))
    clear_cache()
    return path


class _MambaService:
    """Thin object wrapper so callers can use mamba_service.score_observation."""

    score_observation = staticmethod(score_observation)
    combine_scores = staticmethod(combine_scores)
    load_mamba_model = staticmethod(load_mamba_model)
    get_mamba_record = staticmethod(get_mamba_record)
    register_mamba_model = staticmethod(register_mamba_model)
    build_mamba_artifact = staticmethod(build_mamba_artifact)
    clear_cache = staticmethod(clear_cache)


mamba_service = _MambaService()
