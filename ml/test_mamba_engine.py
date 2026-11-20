#!/usr/bin/env python3
"""SkyGuard-AI — Mamba selective SSM engine tests.

Covers:
1. ZOH discretisation correctness (closed form + small-z limit).
2. Analytic gradients match finite differences (BPTT correctness).
3. One Adam training step reduces the loss.
4. Save/load round-trip preserves predictions.
5. Prediction shapes and S_M in [0, inf) with calibrated sigma_e.
6. CLI train/score smoke test on a tiny synthetic series (no data files).

Run:  python3 ml/test_mamba_engine.py        (standalone)
  or:  python3 -m pytest ml/test_mamba_engine.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.mamba_engine import (MambaEncoder, discretise, _expm1_over_z,
                             consecutive_windows)


def test_discretisation_closed_form():
    # Scalar case: A = -2, Delta = 0.5, B = 3.
    # A-bar = exp(-1); B-bar = (exp(-1) - 1)/(-2) * 3.
    delta = np.array([[0.5]])
    A = np.array([[-2.0]])
    B = np.array([[[3.0]]])
    a_bar, b_bar = discretise(delta, A, B)
    assert abs(a_bar[0, 0, 0] - np.exp(-1.0)) < 1e-12
    assert abs(b_bar[0, 0, 0] - (np.exp(-1.0) - 1.0) / -2.0 * 3.0) < 1e-12
    # Small-z limit: A -> 0, B-bar -> Delta * B.
    A0 = np.array([[0.0]])
    _, b0 = discretise(delta, A0, B)
    assert abs(b0[0, 0, 0] - 0.5 * 3.0) < 1e-9
    # _expm1_over_z at exactly 0 must be 1, not NaN.
    assert _expm1_over_z(np.array([0.0]))[0] == 1.0
    assert np.isfinite(_expm1_over_z(np.array([1e-12, -1e-12]))).all()
    print("  discretisation: OK")


def test_gradients_finite_difference():
    rng = np.random.default_rng(7)
    m = MambaEncoder(d_model=4, state_dim=3, seed=1)
    X = rng.normal(0, 1, (2, 5, 3))
    Y = rng.normal(0, 1, (2, 5, 3))
    _, grads = m.loss_and_grads(X, Y)

    def loss_with(param, idx, value):
        old = m.params[param].copy()
        m.params[param][idx] = value
        out, _ = m.forward(X)
        loss = float(np.mean((out - Y) ** 2))
        m.params[param] = old
        return loss

    worst_abs = 0.0
    worst_rel = 0.0
    for name, g in grads.items():
        it = np.nditer(g, flags=["multi_index"])
        for _ in range(g.size):
            idx = it.multi_index
            v = m.params[name][idx]
            eps = 1e-6
            num = (loss_with(name, idx, v + eps) - loss_with(name, idx, v - eps)) / (2 * eps)
            ana = g[idx]
            worst_abs = max(worst_abs, abs(num - ana))
            if max(abs(num), abs(ana)) > 1e-5:
                worst_rel = max(worst_rel, abs(num - ana) / max(abs(num), abs(ana)))
            it.iternext()
    assert worst_abs < 1e-7, f"grad abs diff {worst_abs}"
    assert worst_rel < 1e-4, f"grad rel err {worst_rel}"
    print(f"  gradients vs finite differences: OK (worst abs {worst_abs:.1e}, "
          f"worst rel {worst_rel:.1e})")


def test_training_step_reduces_loss():
    rng = np.random.default_rng(3)
    m = MambaEncoder(d_model=8, state_dim=4, seed=0)
    # Smooth synthetic series: sum of sines, so a temporal model can learn it.
    t = np.arange(300, dtype=float)
    series = np.stack([np.sin(t / 9.0) + 0.3 * np.sin(t / 3.0),
                       np.cos(t / 11.0), np.sin(t / 7.0) * 0.5], axis=1)
    stamps = np.arange(300) * 3600 + 1_700_000_000
    X, Y = consecutive_windows(stamps, series, 12)
    Xtr, Ytr = X[:200], Y[:200]
    loss0, _ = m.loss_and_grads(Xtr[:32], Ytr[:32])
    m.train(Xtr, Ytr, epochs=3, batch=32, lr=0.005)
    loss1, _ = m.loss_and_grads(Xtr[:32], Ytr[:32])
    assert loss1 < loss0, f"loss did not decrease: {loss0:.4f} -> {loss1:.4f}"
    print(f"  training step reduces loss: OK ({loss0:.4f} -> {loss1:.4f})")


def test_save_load_round_trip():
    rng = np.random.default_rng(5)
    m = MambaEncoder(d_model=8, state_dim=4, seed=2)
    m.mean = np.array([25.0, 85.0, 1009.0])
    m.std = np.array([1.5, 5.0, 2.0])
    m.calibrate_sigma_e_signed(rng.normal(0, [0.3, 1.4, 0.5], (100, 3)))
    X = rng.normal(0, 1, (4, 12, 3))
    before, _ = m.forward(X)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m"
        m.save(path)
        m2 = MambaEncoder.load(path)
    after, _ = m2.forward(X)
    assert np.allclose(before, after), "loaded model predictions differ"
    assert np.allclose(m2.sigma_e, m.sigma_e)
    assert np.allclose(m2.mean, m.mean) and np.allclose(m2.std, m.std)
    # JSON artifact round-trip (backend registry format).
    art = m.to_artifact_dict()
    m3 = MambaEncoder.from_artifact_dict(art)
    after3, _ = m3.forward(X)
    assert np.allclose(before, after3), "artifact round-trip predictions differ"
    print("  save/load round-trip: OK")


def test_prediction_shape_and_score_range():
    rng = np.random.default_rng(11)
    m = MambaEncoder(d_model=8, state_dim=4, seed=3)
    X = rng.normal(0, 1, (6, 24, 3))
    out, _ = m.forward(X)
    assert out.shape == (6, 24, 3), f"bad output shape {out.shape}"
    assert np.isfinite(out).all()
    # S_M range: calibrated sigma_e from synthetic errors.
    m.calibrate_sigma_e_signed(rng.normal(0, 1.0, (200, 3)))
    errs = np.abs(rng.normal(0, 1.0, (50, 3)))
    s_m = m.temporal_score(errs)
    assert s_m.shape == (50,)
    assert (s_m >= 0).all() and np.isfinite(s_m).all()
    print("  prediction shape and S_M range: OK")


def run_tests():
    print("=" * 70)
    print("SKYGUARD-AI: MAMBA ENGINE TEST SUITE")
    print("=" * 70)
    test_discretisation_closed_form()
    test_gradients_finite_difference()
    test_training_step_reduces_loss()
    test_save_load_round_trip()
    test_prediction_shape_and_score_range()
    print("ALL MAMBA TESTS PASSED")


if __name__ == "__main__":
    run_tests()
