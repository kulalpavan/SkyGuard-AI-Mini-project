#!/usr/bin/env python3
"""SkyGuard-AI — hybrid scoring and fusion tests.

Covers:
1. ECDF calibration maps validation scores into [0, 1] and propagates NaN.
2. Alpha grid search picks weights with alpha + beta = 1 and FPR <= 0.05.
3. Hybrid score is a convex combination of the calibrated scores.
4. Three-way classification follows paper Eqs. 18-23 (ordering matters).
5. Nearby-anomaly detection (Algorithm 4): radius, window, same sign.
6. Hard QC flags physically impossible values.

Run:  python3 ml/test_hybrid_engine.py
  or:  python3 -m pytest ml/test_hybrid_engine.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.hybrid_engine import (ECDFCalibrator, HybridScorer,
                              classify_three_way, nearby_anomaly,
                              hard_qc_violation)


def test_ecdf_calibration():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 500)
    cal = ECDFCalibrator().fit(vals)
    out = cal.transform(vals)
    assert (out >= 0).all() and (out <= 1).all()
    # Median of the calibration set maps to ~0.5.
    assert abs(float(cal.transform(np.median(vals))) - 0.5) < 0.05
    # A value far above everything maps to 1; far below to 0.
    assert float(cal.transform(10.0)) == 1.0
    assert float(cal.transform(-10.0)) == 0.0
    # NaN stays NaN (a missing engine score must not become "maximally anomalous").
    assert np.isnan(float(cal.transform(np.nan)))
    print("  ECDF calibration: OK")


def test_alpha_grid_search():
    rng = np.random.default_rng(1)
    sc = HybridScorer("TEST")
    # IF is informative, Mamba is noise: alpha should favour IF.
    if_normal = rng.normal(0.4, 0.1, 400)
    if_fault = rng.normal(0.8, 0.1, 100)
    m_normal = rng.normal(0.5, 0.2, 400)
    m_fault = rng.normal(0.5, 0.2, 100)
    sc.calibrate(if_normal, m_normal)
    res = sc.calibrate_alpha(
        np.concatenate([if_normal, if_fault]),
        np.concatenate([m_normal, m_fault]),
        np.concatenate([np.zeros(400, dtype=int), np.ones(100, dtype=int)]))
    assert abs(res["alpha"] + res["beta"] - 1.0) < 1e-9
    assert res["fpr"] <= 0.05 + 1e-9, f"FPR {res['fpr']} exceeds constraint"
    assert res["alpha"] > 0.5, f"alpha should favour the informative engine, got {res['alpha']}"
    assert res["f1"] > 0.5, f"F1 too low: {res['f1']}"
    print(f"  alpha grid search: OK (alpha={res['alpha']}, F1={res['f1']:.3f}, "
          f"FPR={res['fpr']:.3f})")


def test_hybrid_score_is_convex_combination():
    sc = HybridScorer("TEST")
    rng = np.random.default_rng(2)
    sc.calibrate(rng.normal(0.5, 0.1, 300), rng.normal(0.5, 0.1, 300))
    sc.alpha, sc.beta, sc.tau_hybrid = 0.7, 0.3, 0.95
    r = sc.score(0.6, 0.9)
    expected = 0.7 * r["S_IF"] + 0.3 * r["S_M"]
    assert abs(r["S_hybrid"] - expected) < 1e-12
    assert 0.0 <= r["S_IF"] <= 1.0 and 0.0 <= r["S_M"] <= 1.0
    assert 0.0 <= r["S_hybrid"] <= 1.0
    assert r["flag"] == (r["S_hybrid"] > 0.95)
    print("  hybrid score combination: OK")


def test_three_way_classification_ordering():
    # q = 1 beats everything.
    assert classify_three_way(1, True, 1, 1, 1)[0] == "LOCAL_SENSOR_FAULT"
    # No flag -> NORMAL even with neighbour evidence.
    assert classify_three_way(0, False, 1, 1, 1)[0] == "NORMAL"
    # Nearby anomaly -> REGIONAL.
    assert classify_three_way(0, True, 1, 0, 0)[0] == "REGIONAL_WEATHER_EVENT"
    # Front consistency without a neighbour flag -> REGIONAL.
    assert classify_three_way(0, True, 0, 1, 0)[0] == "REGIONAL_WEATHER_EVENT"
    # Degraded health, no other evidence -> LOCAL.
    assert classify_three_way(0, True, 0, 0, 1)[0] == "LOCAL_SENSOR_FAULT"
    # Unexplained single-station anomaly -> LOCAL, flagged for review.
    label, reason = classify_three_way(0, True, 0, 0, 0)
    assert label == "LOCAL_SENSOR_FAULT" and "review" in reason
    print("  three-way classification ordering: OK")


def test_nearby_anomaly():
    # Udupi (13.34, 74.74) to Dakshina Kannada (12.87, 74.88): ~54 km.
    n, delay = nearby_anomaly((13.34, 74.74), (12.87, 74.88),
                              [False, False, True, False], [0, 0, -1, 0], -1)
    assert n == 1 and delay == 2, f"expected n=1 delay=2, got {n}, {delay}"
    # Opposite sign at the neighbour does not count.
    n, _ = nearby_anomaly((13.34, 74.74), (12.87, 74.88),
                          [True], [1], -1)
    assert n == 0
    # Too far away does not count.
    n, _ = nearby_anomaly((13.34, 74.74), (12.0, 77.0),
                          [True], [-1], -1)
    assert n == 0
    # Outside the propagation window does not count.
    n, _ = nearby_anomaly((13.34, 74.74), (12.87, 74.88),
                          [False] * 7 + [True], [0] * 7 + [-1], -1)
    assert n == 0
    print("  nearby-anomaly detection: OK")


def test_hard_qc():
    assert hard_qc_violation(25.0, 80.0, 1009.0) == 0
    assert hard_qc_violation(70.0, 80.0, 1009.0) == 1   # impossible temperature
    assert hard_qc_violation(25.0, 120.0, 1009.0) == 1  # impossible humidity
    assert hard_qc_violation(25.0, 80.0, 200.0) == 1    # impossible pressure
    assert hard_qc_violation(float("nan"), 80.0, 1009.0) == 1
    print("  hard QC: OK")


def run_tests():
    print("=" * 70)
    print("SKYGUARD-AI: HYBRID ENGINE TEST SUITE")
    print("=" * 70)
    test_ecdf_calibration()
    test_alpha_grid_search()
    test_hybrid_score_is_convex_combination()
    test_three_way_classification_ordering()
    test_nearby_anomaly()
    test_hard_qc()
    print("ALL HYBRID TESTS PASSED")


if __name__ == "__main__":
    run_tests()
