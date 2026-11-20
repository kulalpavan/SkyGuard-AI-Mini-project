"""Model drift metrics — pure Python, no new dependency (spec §21).

Drift is measured in RESIDUAL space, not raw space (§21.2): the
climatology already removes seasonal and diurnal structure, so a stable
sensor with a correct model produces residuals that are approximately
zero-mean and unit-variance in every season. A shift in the residual
distribution therefore means the sensor or the model changed — not
that the season did. A naive PSI on raw T/H/P would fire on every
monsoon onset and withdrawal; this module must never do that.

Metrics (spec §21.3, 14-day rolling window vs the training reference):
  PSI on residuals, residual mean shift, residual variance ratio,
  anomaly rate vs training contamination, KS statistic vs the stored
  training score deciles, feature availability.

Thresholds: the MODERATE/HIGH bands follow §21.3 exactly. The LOW
band is this module's own early-warning extension (the spec only
defines stable/moderate/significant); it changes nothing about
detection — drift observes, it never retunes scoring or gates.
"""

import math

SEVERITY_ORDER = ("NONE", "LOW", "MODERATE", "HIGH", "INSUFFICIENT_DATA")

# metric -> (none_max, low_max, moderate_max) — value at or below the
# bound is that severity; above moderate_max is HIGH. A bound of None
# means that severity level does not exist for the metric.
# Bands follow SKYGUARD-AI-PRODUCTION-SPEC.md §21.3; LOW is added here
# as an early warning (see the module docstring).
THRESHOLDS = {
    # PSI: <0.10 stable, 0.10-0.25 moderate, >0.25 significant (§21.3).
    "psi": {"none_max": 0.05, "low_max": 0.10, "moderate_max": 0.25},
    # Residual mean shift |mean(z)|: <0.3 ok, 0.3-0.7 warn, >0.7 alert.
    "residual_mean_shift": {"none_max": 0.2, "low_max": 0.3, "moderate_max": 0.7},
    # Variance ratio, expressed as the worst deviation factor
    # max(r, 1/r) so both noise growth and flatlining are caught:
    # 0.5-2.0 ok (§21.3); beyond 4x is a broken sensor or a dead one.
    "variance_ratio_deviation": {"none_max": 2.0, "low_max": 2.5, "moderate_max": 4.0},
    # Anomaly rate as a multiple of the training contamination:
    # >3x is an alert (§21.3); 2-3x is an early warning.
    "anomaly_rate_multiple": {"none_max": 2.0, "low_max": None, "moderate_max": 3.0},
    # KS statistic D: >0.2 warns (§21.3).
    "ks_statistic": {"none_max": 0.1, "low_max": 0.2, "moderate_max": 0.35},
    # Feature availability: <0.8 warns (§21.3).
    # (Bands are inverted: lower is worse.)
    "feature_availability": {"none_min": 0.95, "low_min": 0.8, "moderate_min": 0.5},
    # Mamba only: mean |prediction error| / sigma_e, divided by
    # E|N(0,1)| = 0.798 — the value a correctly calibrated model shows
    # on validation data. A sensor drifting away from the model makes
    # its predictions miss by more; 2x the calibrated error is an alert.
    "mamba_error_ratio": {"none_max": 1.25, "low_max": 1.5, "moderate_max": 2.0},
}

# Plain-words meaning per severity, shown in the UI next to the badge.
SEVERITY_WORDS = {
    "NONE": "Stable — the model still matches recent weather.",
    "LOW": "Slight shift — worth watching, nothing to do yet.",
    "MODERATE": "Drifting — the model no longer matches recent weather well; queue a retraining.",
    "HIGH": "Needs retraining — the model has moved well away from recent weather; retrain before trusting its verdicts.",
    "INSUFFICIENT_DATA": "Not enough recent data to judge drift yet.",
    "NO_MODEL": "No model registered yet, so there is nothing to compare against.",
}


def deciles(values):
    """9 cut points splitting sorted values into 10 equal bins.

    Returns [] when there are too few values to define deciles.
    """
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if len(vals) < 20:
        return []
    cuts = []
    for k in range(1, 10):
        pos = k * (len(vals) - 1) / 10.0
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        cuts.append(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))
    return cuts


def compute_psi(ref_deciles, cur_values):
    """Population Stability Index of current values vs reference deciles.

    10 fixed bins from the training deciles; PSI = sum over bins of
    (p_cur - p_ref) * ln(p_cur / p_ref). Reference bins hold 10% each
    by construction. Proportions are floored at 1e-4 so an empty bin
    contributes a finite penalty instead of exploding. Returns None
    when the reference deciles are missing (caller marks partial).
    """
    if not ref_deciles or len(ref_deciles) != 9:
        return None
    cur = [float(v) for v in cur_values if v is not None and math.isfinite(v)]
    if not cur:
        return None
    counts = [0] * 10
    for v in cur:
        b = 0
        while b < 9 and v > ref_deciles[b]:
            b += 1
        counts[b] += 1
    psi = 0.0
    for count in counts:
        p_cur = max(count / len(cur), 1e-4)
        p_ref = 0.1
        psi += (p_cur - p_ref) * math.log(p_cur / p_ref)
    return psi


def ks_statistic_from_deciles(ref_deciles, cur_values):
    """Two-sample KS statistic D against the stored training deciles.

    Evaluated at the 9 decile cuts only (the raw training set is not
    retained — §21.3): D = max |F_cur(cut) - k/10|. Returns None when
    the reference deciles are missing (caller marks partial).
    """
    if not ref_deciles or len(ref_deciles) != 9:
        return None
    cur = sorted(float(v) for v in cur_values if v is not None and math.isfinite(v))
    if not cur:
        return None
    d = 0.0
    for k, cut in enumerate(ref_deciles, start=1):
        f_cur = sum(1 for v in cur if v <= cut) / len(cur)
        d = max(d, abs(f_cur - k / 10.0))
    return d


def residual_stats(values):
    """(mean, variance) of a residual series; (None, None) when empty."""
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, var


def severity_for_metric(name, value):
    """Severity for one metric value, or None when the metric is absent.

    ``value`` for anomaly rate is the multiple of the training
    contamination; for variance ratio it is the deviation factor
    max(r, 1/r); for feature availability lower is worse.
    """
    if value is None:
        return None
    band = THRESHOLDS[name]
    if "none_min" in band:  # inverted metric: lower is worse
        if value >= band["none_min"]:
            return "NONE"
        if value >= band["low_min"]:
            return "LOW"
        if value >= band["moderate_min"]:
            return "MODERATE"
        return "HIGH"
    if value <= band["none_max"]:
        return "NONE"
    if band["low_max"] is not None and value <= band["low_max"]:
        return "LOW"
    if value <= band["moderate_max"]:
        return "MODERATE"
    return "HIGH"


def overall_severity(metric_severities):
    """Max severity over the metrics that were computed (§21.4)."""
    rank = {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
    present = [s for s in metric_severities if s in rank]
    if not present:
        return "NONE"
    return max(present, key=lambda s: rank[s])


def plain_words(severity, worst_metric=None, worst_value=None):
    """One plain sentence for the UI badge (§21.5)."""
    base = SEVERITY_WORDS.get(severity, SEVERITY_WORDS["NONE"])
    if worst_metric and severity in ("LOW", "MODERATE", "HIGH") and worst_value is not None:
        labels = {
            "psi": "the shape of recent readings has moved away from training",
            "residual_mean_shift": "recent readings sit consistently off the model's expectation",
            "variance_ratio_deviation": "recent readings are much noisier (or flatter) than in training",
            "anomaly_rate_multiple": "the model is flagging far more often than it did in training",
            "ks_statistic": "the model's scores have shifted from training",
            "feature_availability": "too many recent observations are incomplete",
            "mamba_error_ratio": "the temporal model's predictions are missing by much more than in training",
        }
        return f"{base} Worst sign: {labels.get(worst_metric, worst_metric)}."
    return base


def threshold_table():
    """Threshold bands for the API/UI, so the numbers are never hidden."""
    return {
        name: {k: v for k, v in band.items()}
        for name, band in THRESHOLDS.items()
    }
