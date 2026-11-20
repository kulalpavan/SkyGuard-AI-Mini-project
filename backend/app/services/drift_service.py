"""Model drift tracking (spec §21, §28.5) — measurement, storage, actions.

What it does
------------
For each station and each registered model type (Isolation Forest and
Mamba), once per computation it compares the last 14 days of the
station's own telemetry against the model's training-time reference
and records six metrics (spec §21.3) plus a severity:

  IF:     PSI per residual (T/H/P), residual mean shift, residual
          variance ratio, anomaly rate vs training contamination,
          KS statistic vs the stored training score deciles, feature
          availability.
  Mamba:  PSI / KS of the temporal score S_M vs the stored validation
          scores, mean prediction error vs the calibrated sigma_e,
          anomaly rate vs the 5% calibration rate, feature availability.

The window is anchored at the station's NEWEST observation, not at
wall-clock "now": the demo history is backfilled (timestamps shifted
to the present), and a live station's newest observation is "now"
anyway, so both cases behave the same.

What it deliberately does NOT do
---------------------------------
* No automatic retraining and no automatic deactivation. A MODERATE
  result queues a retraining CANDIDATE (a human approves and trains
  it); a HIGH result additionally opens a DATA_QUALITY incident so an
  operator sees it. Drift is a signal for a human decision.
* No confidence reduction. Spec §21.4 also reduces the station's
  confidence multiplier to 0.8 on HIGH drift; that would silently
  change detection behaviour, so it is NOT implemented here. The
  severity, the numbers and the incident are the alert.
* No raw-space drift. See ml/drift_engine.py: a monsoon onset moves
  raw T/H/P a lot while residuals stay put, so raw-space PSI would
  false-alarm twice a year. A sustained regional weather pattern can
  still move residuals (the Udupi dry spells did exactly that), so a
  MODERATE/HIGH result means "investigate", not "the sensor is broken"
  — the panel shows the per-parameter breakdown so the operator can
  see whether it is humidity-driven weather or a single-sensor offset.

References
----------
A model trained after this change carries its training-time reference
in the model card (``drift_reference``: residual deciles, score
deciles, residual variances, contamination). For older models the
reference is rebuilt from the training window still in the station's
telemetry (card ``training_data`` start/end); when that window is gone
the computation is marked PARTIAL — PSI/KS are skipped and the rest
uses the theoretical residual reference (mean 0, variance 1).
"""

import datetime as dt
import json
import logging
import statistics
import uuid

from backend.app.storage.database import (
    get_db, fetch_historical_telemetry,
    list_station_models,
)
from backend.app.services.model_storage import model_storage_service
from backend.app.services.mamba_service import load_mamba_model, MAMBA_MODEL_TYPE
from ml.drift_engine import (
    deciles, compute_psi, ks_statistic_from_deciles, residual_stats,
    severity_for_metric, overall_severity, plain_words,
)
from ml.feature_engine import feature_engine, instant, valid_core

logger = logging.getLogger("skyguard.drift")

WINDOW_DAYS = 14
MIN_DAYS = 7
MIN_ROWS = 100
MIN_MAMBA_WINDOWS = 50
# E|N(0,1)| — the mean absolute standardised error a correctly
# calibrated model shows on validation data.
EXPECTED_ABS_NORMAL = 0.7978845608028654
# Anomaly-rate alerts also need an absolute floor, so a 4% rate on a
# 1% training reference (4x) does not look like an alert.
ANOMALY_RATE_FLOOR = 0.05
PARAMS = ("temperature", "humidity", "pressure")
PARAM_LABELS = {"temperature": "Temperature", "humidity": "Humidity", "pressure": "Pressure"}


# --------------------------------------------------------------------------
# Reference handling
# --------------------------------------------------------------------------

def _reference_from_card(card):
    ref = (card or {}).get("drift_reference") or {}
    if not ref.get("residual_deciles"):
        return None
    return ref


def _rebuild_reference(station_id, record, artifact, history_rows):
    """Rebuild the training reference from the training window in telemetry.

    Returns (reference, partial, reason). The reference is the same
    shape as the card's drift_reference, computed the same way training
    does: residuals on the cleaned fit partition (first 80% of the
    training window, rows with max|z| > 8 scrubbed), IF scores on the
    holdout (last 20%).
    """
    card = artifact.get("model_card") or {}
    training_data = card.get("training_data") or {}
    start, end = training_data.get("start"), training_data.get("end")
    if not start or not end:
        return None, True, "no training window recorded on the model card"
    window = [r for r in history_rows if start <= r["timestamp"] <= end]
    if len(window) < 100:
        return None, True, "training window is no longer in stored telemetry"
    split = int(len(window) * 0.8)
    fit, holdout = window[:split], window[split:]
    climate = artifact.get("climatology")
    stats = card.get("normalization_stats")
    if not climate or not stats:
        return None, True, "model artifact has no climatology"
    from ml.station_adaptive_pipeline import IsolationForest
    forest = IsolationForest.from_dict(artifact["model_weights"])
    cleaned = [r for r in fit
               if max(abs(z) for z in feature_engine.residuals(r, climate)) <= 8]
    if len(cleaned) < 72:
        return None, True, "too few clean fit rows to rebuild a reference"
    res = {p: [feature_engine.residuals(r, climate)[i] for r in cleaned]
           for i, p in enumerate(PARAMS)}
    scores = []
    for i, r in enumerate(holdout):
        vec = feature_engine.transform(
            r, climate, stats, holdout[max(0, i - 12):i])
        scores.append(forest.score_sample(vec))
    return {
        "residual_deciles": {p: deciles(res[p]) for p in PARAMS},
        "residual_var_ref": {p: residual_stats(res[p])[1] or 1.0 for p in PARAMS},
        "if_score_deciles": deciles(scores),
        "anomaly_rate_reference": record.get("contamination_rate") or 0.05,
        "window": {"start": start, "end": end},
    }, False, ""


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def _store_drift_row(station_id, result):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    m = result["metrics"]
    # The row keeps the headline numbers in columns (queryable) and the
    # full metric detail — per-parameter breakdown, per-metric
    # severities, metric table, plain-words sentence — in metrics_json,
    # so the UI can render history without recomputing anything.
    stored = {
        "metrics": result.get("metrics") or {},
        "detail": result.get("detail") or {},
        "metric_entries": result.get("metric_entries") or [],
        "plain_words": result.get("plain_words") or "",
        "worst_metric": result.get("worst_metric") or "",
        "reason": result.get("reason") or "",
        "partial_reason": result.get("partial_reason") or "",
    }
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO drift_metrics (station_id, model_version, model_type,
                 window_days, computed_at, recorded_at, psi_temperature, psi_humidity,
                 psi_pressure, residual_mean_shift, residual_variance_ratio,
                 anomaly_rate, anomaly_rate_reference, ks_statistic,
                 feature_availability, severity, triggered_candidate,
                 metrics_json, is_partial)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (station_id, result["model_version"], result["model_type"],
             result["window_days"], now, now,
             m.get("psi_temperature"), m.get("psi_humidity"), m.get("psi_pressure"),
             m.get("residual_mean_shift"), m.get("residual_variance_ratio"),
             m.get("anomaly_rate"), m.get("anomaly_rate_reference"),
             m.get("ks_statistic"), m.get("feature_availability"),
             result["severity"], 1 if result.get("triggered_candidate") else 0,
             json.dumps(stored, default=str),
             1 if result.get("is_partial") else 0))
    return now


def latest_drift(station_id):
    """Latest computed drift row per model type, or {} when never computed."""
    sid = station_id.strip().upper()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM drift_metrics WHERE station_id = ?
               ORDER BY computed_at DESC""", (sid,))
        rows = cur.fetchall()
    out = {}
    for row in rows:
        key = row.get("model_type") or "IsolationForest"
        if key not in out:
            out[key] = _row_to_api(row)
    return out


def drift_history(station_id, days=90):
    """Drift rows for the sparkline, oldest first, limited to `days`."""
    sid = station_id.strip().upper()
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM drift_metrics WHERE station_id = ? AND computed_at >= ?
               ORDER BY computed_at ASC""", (sid, cutoff))
        return [_row_to_api(r) for r in cur.fetchall()]


def _row_to_api(row):
    try:
        stored = json.loads(row.get("metrics_json") or "{}")
    except (TypeError, ValueError):
        stored = {}
    if not isinstance(stored, dict):
        stored = {}
    # Rows written before the metrics_json enrichment carry the bare
    # detail dict; rows written after carry metrics/detail/entries.
    metrics = stored.get("metrics") or {}
    detail = stored.get("detail") or (stored if "metrics" not in stored else {})
    return {
        "id": row.get("id"),
        "station_id": row.get("station_id"),
        "model_version": row.get("model_version"),
        "model_type": row.get("model_type") or "IsolationForest",
        "window_days": row.get("window_days"),
        "computed_at": row.get("computed_at"),
        "psi_temperature": row.get("psi_temperature"),
        "psi_humidity": row.get("psi_humidity"),
        "psi_pressure": row.get("psi_pressure"),
        "residual_mean_shift": row.get("residual_mean_shift"),
        "residual_variance_ratio": row.get("residual_variance_ratio"),
        "anomaly_rate": row.get("anomaly_rate"),
        "anomaly_rate_reference": row.get("anomaly_rate_reference"),
        "ks_statistic": row.get("ks_statistic"),
        "feature_availability": row.get("feature_availability"),
        "severity": row.get("severity"),
        "triggered_candidate": bool(row.get("triggered_candidate")),
        "is_partial": bool(row.get("is_partial")),
        "metrics": metrics,
        "metric_entries": stored.get("metric_entries") or [],
        "plain_words": stored.get("plain_words") or "",
        "worst_metric": stored.get("worst_metric") or "",
        "reason": stored.get("reason") or "",
        "partial_reason": stored.get("partial_reason") or "",
        "detail": detail,
    }


def list_retraining_candidates(station_id, status=None):
    sid = station_id.strip().upper()
    with get_db() as conn:
        cur = conn.cursor()
        if status:
            cur.execute(
                """SELECT * FROM retraining_candidates
                   WHERE station_id = ? AND status = ?
                   ORDER BY created_at DESC""", (sid, status))
        else:
            cur.execute(
                """SELECT * FROM retraining_candidates WHERE station_id = ?
                   ORDER BY created_at DESC""", (sid,))
        return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------
# Actions (human decision support — never automatic training/activation)
# --------------------------------------------------------------------------

def queue_retraining_candidate(station_id, model_version, trigger_reason,
                               priority, actor=None):
    """Queue a retraining candidate (status QUEUED).

    Dedup: an already-queued candidate for the same station + model
    version in the last 24 h is returned instead of a duplicate. A
    candidate is a queue entry only — training it still needs a human
    to start the job, and activation still needs the promotion gate.
    """
    sid = station_id.strip().upper()
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = (now - dt.timedelta(hours=24)).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM retraining_candidates
               WHERE station_id = ? AND model_version = ? AND status = 'QUEUED'
                 AND created_at >= ? ORDER BY created_at DESC LIMIT 1""",
            (sid, model_version, cutoff))
        existing = cur.fetchone()
        if existing:
            return dict(existing), False
        cur.execute(
            """INSERT INTO retraining_candidates
                 (station_id, model_version, trigger_reason, priority, status,
                  gate_status, gate_results, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (sid, model_version, trigger_reason, priority, "QUEUED",
             "PENDING", json.dumps({"triggered_by": actor or "drift-monitor"}),
             now.isoformat()))
        candidate_id = cur.lastrowid
        cur.execute("SELECT * FROM retraining_candidates WHERE id = ?",
                    (candidate_id,))
        return dict(cur.fetchone()), True


def create_drift_incident(station_id, result):
    """Open a DATA_QUALITY incident for a HIGH drift result, once.

    Dedup: an already-open drift incident for the station is returned
    instead of a duplicate. The incident is the record the operator
    sees; it does not change detection behaviour.
    """
    sid = station_id.strip().upper()
    station_name = sid
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT station_name FROM stations WHERE station_id = ?",
                    (sid,))
        row = cur.fetchone()
        if row:
            station_name = row.get("station_name") or sid
        cur.execute(
            """SELECT id FROM incidents WHERE station_id = ?
               AND variable = 'drift:MODEL_DRIFT' AND status = 'open'""",
            (sid,))
        existing = cur.fetchone()
        if existing:
            return existing["id"], False
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        incident_id = "INC-" + uuid.uuid4().hex
        explanation = result.get("plain_words") or (
            f"Model drift HIGH for {result['model_type']} "
            f"{result['model_version']}: the model no longer matches "
            "recent weather. Retrain before trusting its verdicts.")
        cur.execute(
            """INSERT INTO incidents (id, station_id, station_name, variable,
                 severity, fault_risk, quality_state, reason_codes, explanation,
                 recommended_actions, evidence_ids, evidence_data, status,
                 created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (incident_id, sid, station_name, "drift:MODEL_DRIFT", "high",
             0.8, "DATA_QUALITY", json.dumps(["MODEL_DRIFT"]), explanation,
             json.dumps(["Open Model Governance and review the drift panel",
                         "Queue a retraining from the drift panel",
                         "Train a new model version from the Training Studio"]),
             json.dumps([]),
             json.dumps({"drift": result.get("metrics") or {},
                         "severity": result["severity"]}),
             "open", now, now))
        return incident_id, True


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def _feature_availability(station_id, window_start, window_end):
    """Fraction of recent observations with complete core values.

    Measured on the raw telemetry in the window (excluding the
    synthetic replay fixture), before any cleaning: a falling
    availability means the upstream feed is decaying, not the model.
    """
    rows = fetch_historical_telemetry(station_id, limit=100000)
    start_dt, end_dt = instant(window_start), instant(window_end)
    total = 0
    valid = 0
    for row in rows:
        if str(row.get("grid_point", "")).startswith("synthetic:"):
            continue
        try:
            ts = instant(row.get("timestamp"))
        except (ValueError, TypeError):
            continue
        if not (start_dt <= ts <= end_dt):
            continue
        total += 1
        if valid_core({"temp": row.get("temp"), "hum": row.get("hum"),
                       "pres": row.get("pres")}):
            valid += 1
    if total == 0:
        return None, 0
    return valid / total, total


def _metric_entries(metrics):
    """Uniform metric list for the API/UI: key, label, value, threshold, severity.

    Built from the model's own metrics dict, so each model type shows
    exactly the metrics it measures (IF: PSI, mean shift, variance
    ratio, anomaly rate, KS, availability; Mamba: PSI/KS on S_M, error
    ratio, anomaly rate, availability). Keys a model does not measure
    are simply absent.
    """
    entries = []
    spec = [
        ("psi", "PSI — distribution shape shift", "PSI < 0.10 stable · 0.10–0.25 moderate · > 0.25 significant"),
        ("residual_mean_shift", "Residual mean shift — bias", "|mean z| < 0.3 ok · 0.3–0.7 warn · > 0.7 alert"),
        ("variance_ratio_deviation", "Residual variance ratio — noise / flatline", "0.5–2.0 ok · beyond 4× alert"),
        ("anomaly_rate_multiple", "Anomaly rate vs training", "≤ 2× ok · 2–3× warn · > 3× alert"),
        ("ks_statistic", "KS statistic — score shift", "D > 0.2 warn · > 0.35 alert"),
        ("feature_availability", "Feature availability — complete observations", "≥ 0.95 ok · < 0.8 warn · < 0.5 alert"),
        ("mamba_error_ratio", "Mamba error ratio — prediction miss vs calibration", "< 1.25× ok · 1.5–2× warn · > 2× alert"),
    ]
    for key, label, threshold in spec:
        if metrics is None or key not in metrics:
            continue
        value = metrics[key]
        entries.append({
            "key": key, "label": label, "value": value,
            "threshold": threshold,
            "severity": severity_for_metric(key, value),
        })
    return entries


def _compute_if_drift(station_id, record, history_rows, window_rows,
                      window_start, window_end, availability):
    """Drift of the Isolation Forest model over the current window."""
    from ml.station_adaptive_pipeline import IsolationForest

    artifact = model_storage_service.load_artifact(record["model_location"])
    if not artifact:
        return {"model_type": "IsolationForest",
                "model_version": record["model_version"],
                "severity": "INSUFFICIENT_DATA",
                "reason": "model artifact unavailable",
                "metrics": {}, "detail": {}}
    card = artifact.get("model_card") or {}
    ref = _reference_from_card(card)
    partial, ref_reason = False, ""
    if ref is None:
        ref, partial, ref_reason = _rebuild_reference(
            station_id, record, artifact, history_rows)
    if ref is None:
        ref = {"residual_deciles": {}, "residual_var_ref": {p: 1.0 for p in PARAMS},
               "if_score_deciles": [], "anomaly_rate_reference":
               record.get("contamination_rate") or 0.05}
        partial, ref_reason = True, ref_reason or "no usable training reference"
    climate, stats = artifact["climatology"], card["normalization_stats"]
    forest = IsolationForest.from_dict(artifact["model_weights"])
    threshold = float(record.get("threshold") or forest.threshold)

    # Residuals and scores for the window rows, with 12-row context
    # from the full history (same transform as live scoring).
    start_idx = next((i for i, r in enumerate(history_rows)
                      if r["timestamp"] >= window_start), len(history_rows))
    residuals = {p: [] for p in PARAMS}
    scores = []
    for j, row in enumerate(window_rows):
        i = start_idx + j
        z = feature_engine.residuals(row, climate)
        for k, p in enumerate(PARAMS):
            residuals[p].append(z[k])
        vec = feature_engine.transform(
            row, climate, stats, history_rows[max(0, i - 12):i])
        scores.append(forest.score_sample(vec))

    metrics = {}
    per_param = {}
    psi_values = {}
    for k, p in enumerate(PARAMS):
        mean, var = residual_stats(residuals[p])
        psi = compute_psi((ref.get("residual_deciles") or {}).get(p), residuals[p])
        psi_values[p] = psi
        per_param[p] = {
            "residual_mean": mean,
            "residual_variance": var,
            "variance_ratio": (var / ref["residual_var_ref"][p]
                               if ref["residual_var_ref"].get(p) else None),
            "psi": psi,
        }
    mean_shift = max(abs(per_param[p]["residual_mean"] or 0) for p in PARAMS)
    var_ratios = [per_param[p]["variance_ratio"] for p in PARAMS
                  if per_param[p]["variance_ratio"]]
    var_dev = max((max(r, 1.0 / r) if r and r > 0 else 1.0) for r in var_ratios) if var_ratios else None
    psi_max = max((v for v in psi_values.values() if v is not None), default=None)
    anomaly_rate = (sum(1 for s in scores if s >= threshold) / len(scores)) if scores else None
    ref_rate = ref.get("anomaly_rate_reference") or 0.05
    rate_multiple = (anomaly_rate / ref_rate) if (anomaly_rate is not None and ref_rate > 0) else None
    ks = ks_statistic_from_deciles(ref.get("if_score_deciles"), scores)

    metric_values = {
        "psi": psi_max,
        "residual_mean_shift": mean_shift,
        "variance_ratio_deviation": var_dev,
        "anomaly_rate_multiple": rate_multiple,
        "ks_statistic": ks,
        "feature_availability": availability,
    }
    # The anomaly-rate alert needs the absolute floor as well: a 4%
    # rate on a 1% reference is 4x but only a handful of rows.
    severities = {}
    for key, value in metric_values.items():
        sev = severity_for_metric(key, value)
        if key == "anomaly_rate_multiple" and sev == "HIGH" and (anomaly_rate or 0) <= ANOMALY_RATE_FLOOR:
            sev = "MODERATE"
        severities[key] = sev
    severity = overall_severity(list(severities.values()))
    worst = max(severities, key=lambda k: {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}[severities[k] or "NONE"])
    worst_value = metric_values.get(worst)
    return {
        "model_type": "IsolationForest",
        "model_version": record["model_version"],
        "model_status": record.get("status"),
        "severity": severity,
        "plain_words": plain_words(severity, worst, worst_value),
        "worst_metric": worst,
        "is_partial": partial,
        "partial_reason": ref_reason if partial else "",
        "metrics": {
            "psi_temperature": psi_values.get("temperature"),
            "psi_humidity": psi_values.get("humidity"),
            "psi_pressure": psi_values.get("pressure"),
            "psi": psi_max,
            "residual_mean_shift": mean_shift,
            "residual_variance_ratio": (max(var_ratios) if var_ratios else None),
            "variance_ratio_deviation": var_dev,
            "anomaly_rate": anomaly_rate,
            "anomaly_rate_reference": ref_rate,
            "anomaly_rate_multiple": rate_multiple,
            "ks_statistic": ks,
            "feature_availability": availability,
            "score_mean": statistics.mean(scores) if scores else None,
            "score_std": statistics.pstdev(scores) if len(scores) > 1 else None,
        },
        "detail": {
            "per_parameter": per_param,
            "metric_severities": severities,
            "window_rows": len(window_rows),
            "threshold": threshold,
        },
    }


def _compute_mamba_drift(station_id, record, window_rows, availability):
    """Drift of the Mamba temporal model over the current window."""
    try:
        import numpy as np
    except ImportError:
        return {"model_type": "Mamba", "model_version": record["model_version"],
                "severity": "INSUFFICIENT_DATA",
                "reason": "numpy unavailable for Mamba scoring",
                "metrics": {}, "detail": {}}
    from backend.app.services.mamba_training import hourly_series_from_rows
    from ml.mamba_engine import consecutive_windows

    encoder, artifact, _ = load_mamba_model(station_id)
    if encoder is None or artifact is None:
        return {"model_type": "Mamba", "model_version": record["model_version"],
                "severity": "INSUFFICIENT_DATA",
                "reason": "Mamba artifact unavailable",
                "metrics": {}, "detail": {}}
    stamps, values = hourly_series_from_rows(window_rows)
    if len(values) < 25:
        return {"model_type": "Mamba", "model_version": record["model_version"],
                "severity": "INSUFFICIENT_DATA",
                "reason": f"only {len(values)} hourly points in the window",
                "metrics": {}, "detail": {}}
    z = (values - encoder.mean) / encoder.std
    X, Y = consecutive_windows(stamps, z, 24)
    if len(X) < MIN_MAMBA_WINDOWS:
        return {"model_type": "Mamba", "model_version": record["model_version"],
                "severity": "INSUFFICIENT_DATA",
                "reason": f"only {len(X)} consecutive 24 h windows in the window",
                "metrics": {}, "detail": {}}
    out, _ = encoder.forward(X)
    errors_orig = np.abs((out[:, -1, :] - Y[:, -1, :]) * encoder.std)
    sigma_e = np.asarray(encoder.sigma_e, dtype=float)
    norm_errors = errors_orig / sigma_e
    s_m = np.max(norm_errors, axis=-1)
    tau_m = float(artifact.get("model_weights", {}).get("tau_M") or 0.0) or None

    hybrid = artifact.get("hybrid") or {}
    ref_scores = hybrid.get("mamba_val_raw") or []
    ref_deciles = deciles(ref_scores)
    psi = compute_psi(ref_deciles, s_m.tolist())
    ks = ks_statistic_from_deciles(ref_deciles, s_m.tolist())
    mean_abs_norm = float(np.mean(norm_errors))
    error_ratio = mean_abs_norm / EXPECTED_ABS_NORMAL
    anomaly_rate = float(np.mean(s_m > tau_m)) if tau_m else None
    ref_rate = 0.05  # tau_M is the 95th percentile of validation S_M by construction
    rate_multiple = (anomaly_rate / ref_rate) if anomaly_rate is not None else None

    metric_values = {
        "psi": psi,
        "ks_statistic": ks,
        "mamba_error_ratio": error_ratio,
        "anomaly_rate_multiple": rate_multiple,
        "feature_availability": availability,
    }
    severities = {}
    for key, value in metric_values.items():
        sev = severity_for_metric(key, value)
        if key == "anomaly_rate_multiple" and sev == "HIGH" and (anomaly_rate or 0) <= ANOMALY_RATE_FLOOR:
            sev = "MODERATE"
        severities[key] = sev
    severity = overall_severity(list(severities.values()))
    worst = max(severities, key=lambda k: {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}[severities[k] or "NONE"])
    per_param = {}
    for i, p in enumerate(PARAMS):
        per_param[p] = {
            "mean_abs_error": float(np.mean(errors_orig[:, i])),
            "mean_abs_normalised_error": float(np.mean(norm_errors[:, i])),
            "sigma_e": float(sigma_e[i]),
        }
    return {
        "model_type": "Mamba",
        "model_version": record["model_version"],
        "model_status": record.get("status"),
        "severity": severity,
        "plain_words": plain_words(severity, worst, metric_values.get(worst)),
        "worst_metric": worst,
        "is_partial": not bool(ref_deciles),
        "partial_reason": "" if ref_deciles else "no validation scores stored on the Mamba artifact; PSI/KS skipped",
        "metrics": {
            "psi": psi,
            "psi_mamba_score": psi,
            "ks_statistic": ks,
            "ks_mamba_score": ks,
            "mamba_error_ratio": error_ratio,
            "mean_abs_normalised_error": mean_abs_norm,
            "anomaly_rate": anomaly_rate,
            "anomaly_rate_reference": ref_rate,
            "anomaly_rate_multiple": rate_multiple,
            "feature_availability": availability,
            "tau_M": tau_m,
        },
        "detail": {
            "per_parameter": per_param,
            "metric_severities": severities,
            "window_rows": len(window_rows),
            "windows_scored": int(len(X)),
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def compute_station_drift(station_id, window_days=WINDOW_DAYS, perform_actions=True):
    """Compute, store and (optionally) act on drift for one station.

    Returns {"station_id", "window": {...}, "models": [per-type result],
    "overall_severity", "plain_words"}. A result with severity
    INSUFFICIENT_DATA is still stored, so the panel can say why there
    is no verdict instead of showing nothing.
    """
    from backend.app.services.training_service import training_service

    sid = station_id.strip().upper()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT station_id FROM stations WHERE station_id = ?", (sid,))
        if not cur.fetchone():
            raise ValueError("Station not found")
    history_rows, _ = training_service.history(sid)
    if not history_rows:
        return {"station_id": sid, "window_days": window_days, "models": [],
                "overall_severity": "INSUFFICIENT_DATA",
                "plain_words": "No telemetry history yet, so drift cannot be judged.",
                "reason": "no telemetry history"}
    max_ts = instant(history_rows[-1]["timestamp"])
    window_start_dt = max_ts - dt.timedelta(days=window_days)
    window_start, window_end = window_start_dt.isoformat(), max_ts.isoformat()
    window_rows = [r for r in history_rows if r["timestamp"] >= window_start]
    days = len({instant(r["timestamp"]).date() for r in window_rows})
    availability, _ = _feature_availability(sid, window_start, window_end)

    models = list_station_models(sid)
    if_record = next((m for m in models if m.get("model_type") == "IsolationForest"
                      and m.get("status") == "ACTIVE"), None) or next(
        (m for m in models if m.get("model_type") == "IsolationForest"
         and m.get("status") != "ARCHIVED"), None)
    mamba_record = next((m for m in models if m.get("model_type") == MAMBA_MODEL_TYPE
                         and m.get("status") != "ARCHIVED"), None)

    results = []
    if days < MIN_DAYS or len(window_rows) < MIN_ROWS:
        reason = (f"only {days} days / {len(window_rows)} rows in the last "
                  f"{window_days} days; need at least {MIN_DAYS} days and "
                  f"{MIN_ROWS} rows")
        for record, model_type in ((if_record, "IsolationForest"),
                                   (mamba_record, "Mamba")):
            if not record:
                continue
            result = {"model_type": model_type, "model_version": record["model_version"],
                      "model_status": record.get("status"), "severity": "INSUFFICIENT_DATA",
                      "plain_words": plain_words("INSUFFICIENT_DATA"),
                      "reason": reason, "metrics": {}, "detail": {"reason": reason},
                      "is_partial": False}
            result["window_days"] = window_days
            _store_drift_row(sid, result)
            results.append(result)
        return {"station_id": sid, "window_days": window_days,
                "window": {"start": window_start, "end": window_end},
                "models": results, "overall_severity": "INSUFFICIENT_DATA",
                "plain_words": plain_words("INSUFFICIENT_DATA"),
                "reason": reason}

    if if_record:
        results.append(_compute_if_drift(
            sid, if_record, history_rows, window_rows,
            window_start, window_end, availability))
    if mamba_record:
        results.append(_compute_mamba_drift(
            sid, mamba_record, window_rows, availability))
    if not results:
        return {"station_id": sid, "window_days": window_days,
                "window": {"start": window_start, "end": window_end},
                "models": [], "overall_severity": "NO_MODEL",
                "plain_words": plain_words("NO_MODEL"),
                "reason": "no registered model for this station"}

    for result in results:
        result["window_days"] = window_days
        result["metric_entries"] = _metric_entries(result["metrics"])
        if perform_actions:
            triggered = False
            if result["severity"] in ("MODERATE", "HIGH"):
                priority = "HIGH" if result["severity"] == "HIGH" else "LOW"
                _, created = queue_retraining_candidate(
                    sid, result["model_version"],
                    f"DRIFT_{result['severity']}", priority,
                    actor="drift-monitor")
                triggered = triggered or created
                if result["severity"] == "HIGH":
                    create_drift_incident(sid, result)
            result["triggered_candidate"] = triggered
        _store_drift_row(sid, result)

    overall = overall_severity([r["severity"] for r in results
                                if r["severity"] not in ("INSUFFICIENT_DATA", "NO_MODEL")])
    if all(r["severity"] == "INSUFFICIENT_DATA" for r in results):
        overall = "INSUFFICIENT_DATA"
    worst = max(results, key=lambda r: {"NONE": 0, "LOW": 1, "MODERATE": 2,
                                        "HIGH": 3, "INSUFFICIENT_DATA": -1}.get(r["severity"], 0))
    return {"station_id": sid, "window_days": window_days,
            "window": {"start": window_start, "end": window_end},
            "models": results, "overall_severity": overall,
            "plain_words": worst.get("plain_words") or plain_words(overall)}


def compute_all_stations(perform_actions=True):
    """Nightly job helper: compute drift for every station with models."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT station_id FROM stations WHERE status = 'ACTIVE'")
        stations = [r["station_id"] for r in cur.fetchall()]
    outcomes = {}
    for sid in stations:
        try:
            outcomes[sid] = compute_station_drift(
                sid, perform_actions=perform_actions)["overall_severity"]
        except Exception as error:  # never let one station stop the nightly run
            logger.exception("Drift computation failed for %s", sid)
            outcomes[sid] = f"ERROR: {error}"
    return outcomes
