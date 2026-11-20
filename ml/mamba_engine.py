#!/usr/bin/env python3
"""SkyGuard-AI — Mamba selective state-space temporal predictor (NumPy only).

Implements the temporal detection engine specified in Section VIII of the
SkyGuard-AI manuscript (manuscript-draft.md):

* Continuous-time SSM h'(t) = A h(t) + B u(t), y(t) = C h(t) + D u(t),
  discretised by zero-order hold: A-bar = exp(Delta*A),
  B-bar = (Delta*A)^(-1) (exp(Delta*A) - I) Delta*B, with a Taylor series
  for the (exp(z) - 1)/z factor when |z| is small.
* Selective mechanism (Eq. 8): B_t, C_t and Delta_t are input-dependent
  linear projections of the projected input u_t, with Delta_t > 0 via
  softplus. A is per-channel, A = -exp(log_A) < 0, so the state is stable.
* Sliding windows of length L over the standardised (T, RH, P) stream;
  a next-step prediction head maps the final state to x-hat_{t+1}.
* Training by gradient descent (Adam, implemented here) with backprop
  through the recurrence written out by hand. The analytic gradients are
  verified against finite differences in ml/test_mamba_engine.py.
* Sensor-wise prediction errors e_T, e_H, e_P (original units) and the
  temporal anomaly score S_M = max(e / sigma_e) (phi = max, fixed before
  the experiments, per Eq. 13). sigma_e is the per-variable standard
  deviation of the prediction error on normal validation data.

Pure NumPy: no torch, no new dependencies, matching the repo's
zero-ML-library style. Training runs on CPU; inference is a linear-time
recurrence, so per-observation streaming scoring is cheap.

Data contract: windows are built from a district-mean hourly series with
consecutive timestamps exactly 3600 s apart (windows never straddle the
one-day break in the supplied data). Standardisation uses the TRAIN split
statistics only.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data helpers (shared with scripts/evaluate_e1_e4.py)
# ---------------------------------------------------------------------------

DISTRICT_FILES = {
    "Udupi": "udupi_weather.csv",
    "Dakshina Kannada": "dakshina_kannada_weather.csv",
    "Belagavi": "belagavi_weather.csv",
    "Bengaluru Urban": "bengaluru_urban_weather.csv",
    "Mysore": "mysuru_weather.csv",
}

# Approximate district-headquarters elevations (m), used only for the
# elevation-adjusted pressure residual feature of the Isolation Forest side.
# Documented as approximate; they do not affect the Mamba model.
DISTRICT_ELEVATION_M = {
    "Udupi": 10.0,
    "Dakshina Kannada": 20.0,
    "Belagavi": 760.0,
    "Bengaluru Urban": 920.0,
    "Mysore": 760.0,
}

# Prototype station coordinates (paper Section IX.B).
DISTRICT_COORDS = {
    "Udupi": (13.34, 74.74),
    "Dakshina Kannada": (12.87, 74.88),
}

VARIABLES = ("temperature_C", "humidity_pct", "pressure_hPa")


def load_district_series(data_dir, district):
    """Load the district-mean hourly (T, RH, P) series for one district.

    Returns (timestamps_epoch, values) where values is an (n, 3) float
    array of [temperature_C, humidity_pct, pressure_hPa] and timestamps
    are sorted ascending. Uses only the CSV columns present in the
    supplied files; no pandas required.
    """
    import csv

    path = Path(data_dir) / DISTRICT_FILES[district]
    by_ts = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["timestamp_epoch"])
            vals = (float(row["t2m_deg_c"]),
                    float(row["relative_humidity_pct"]),
                    float(row["msl_hpa"]))
            by_ts.setdefault(ts, []).append(vals)
    stamps = sorted(by_ts)
    values = np.array([[np.mean([v[i] for v in by_ts[t]]) for i in range(3)]
                       for t in stamps], dtype=np.float64)
    return np.array(stamps, dtype=np.int64), values


def consecutive_windows(stamps, values, window):
    """Build (input, target) window pairs from a series.

    Each example is L+1 consecutive hourly points with no gaps:
    inputs = points [0..L-1], targets = points [1..L]. Returns
    (X, Y) with shapes (n, L, 3), (n, L, 3). Windows straddling a gap
    (gap != 3600 s) are dropped.
    """
    gaps = np.diff(stamps) == 3600
    # A window starting at i needs gaps[i..i+L-1] all True (L gaps for L+1 points).
    ok = np.convolve(gaps.astype(np.int64), np.ones(window, dtype=np.int64),
                     mode="valid") == window
    starts = np.nonzero(ok)[0]
    X = np.stack([values[s:s + window] for s in starts])
    Y = np.stack([values[s + 1:s + window + 1] for s in starts])
    return X, Y


def standardise(train_values):
    """Per-variable mean/std from training values only."""
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _softplus(s):
    return np.logaddexp(0.0, s)


def _sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -50.0, 50.0)))


def _expm1_over_z(z):
    """f(z) = (exp(z) - 1) / z, with a Taylor series near z = 0."""
    f = np.empty_like(z)
    small = np.abs(z) < 1e-6
    big = ~small
    f[big] = np.expm1(z[big]) / z[big]
    zs = z[small]
    f[small] = 1.0 + zs / 2.0 + zs ** 2 / 6.0 + zs ** 3 / 24.0
    return f


def _expm1_over_z_deriv(z):
    """f'(z) for f(z) = (exp(z) - 1) / z, Taylor series near z = 0."""
    fp = np.empty_like(z)
    small = np.abs(z) < 1e-6
    big = ~small
    zb = z[big]
    fp[big] = (zb * np.exp(zb) - np.expm1(zb)) / zb ** 2
    zs = z[small]
    fp[small] = 0.5 + zs / 3.0 + zs ** 2 / 8.0 + zs ** 3 / 30.0
    return fp


def discretise(delta, A, B):
    """Zero-order-hold discretisation for one step.

    delta: (..., d), A: (d, N), B: (..., d, N).
    Returns (A_bar, B_bar) with shapes (..., d, N).
    A-bar = exp(delta * A); B-bar = delta * B * (exp(z) - 1)/z, z = delta*A.
    """
    z = delta[..., None] * A
    a_bar = np.exp(z)
    b_bar = delta[..., None] * B * _expm1_over_z(z)
    return a_bar, b_bar


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MambaEncoder:
    """Selective SSM next-step predictor for (T, RH, P) weather series.

    One selective SSM block: input projection -> selective SSM -> linear
    prediction head. Per-channel state of dimension N.
    """

    def __init__(self, d_model=32, state_dim=8, seed=42):
        self.d = int(d_model)
        self.N = int(state_dim)
        rng = np.random.default_rng(seed)
        d, N = self.d, self.N
        self.params = {
            "W_in": rng.normal(0.0, 1.0 / math.sqrt(3.0), (d, 3)),
            "b_in": np.zeros(d),
            "W_B": rng.normal(0.0, 0.1, (d * N, d)),
            "b_B": np.zeros(d * N),
            "W_C": rng.normal(0.0, 0.1, (d * N, d)),
            "b_C": np.zeros(d * N),
            "W_d": rng.normal(0.0, 0.1, (d, d)),
            "b_d": np.full(d, -2.0),
            # A = -exp(log_A); log-uniform in [0.1, 3.0] gives decay times
            # from ~0.3 h to ~10 h at Delta ~ 1.
            "log_A": np.log(rng.uniform(0.1, 3.0, (d, N))),
            "D": np.ones(d),
            "W_out": rng.normal(0.0, 1.0 / math.sqrt(d), (3, d)),
            "b_out": np.zeros(3),
        }
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        self.sigma_e = None      # per-variable error scale (original units)
        self.tau_M = None        # threshold on calibrated S_M (set by caller)
        self._adam_m = None
        self._adam_v = None
        self._adam_t = 0

    # -- properties ------------------------------------------------------
    @property
    def A(self):
        return -np.exp(self.params["log_A"])

    # -- forward ---------------------------------------------------------
    def forward(self, X, cache=False):
        """Forward pass over a batch of standardised windows.

        X: (B, L, 3) standardised inputs. Returns (out, cache) where out is
        (B, L, 3): out[:, t] is the prediction of the observation AFTER
        X[:, t] (i.e. the target at step t). If cache=False, cache is None.
        """
        p = self.params
        B, L, _ = X.shape
        d, N = self.d, self.N
        u = X @ p["W_in"].T + p["b_in"]                       # (B, L, d)
        h = np.zeros((B, d, N))
        out = np.empty((B, L, 3))
        caches = [] if cache else None
        A = self.A                                            # (d, N)
        for t in range(L):
            u_t = u[:, t, :]                                  # (B, d)
            s_pre = u_t @ p["W_d"].T + p["b_d"]               # (B, d)
            delta = _softplus(s_pre)                          # (B, d)
            B_t = (u_t @ p["W_B"].T + p["b_B"]).reshape(B, d, N)
            C_t = (u_t @ p["W_C"].T + p["b_C"]).reshape(B, d, N)
            z = delta[:, :, None] * A[None, :, :]            # (B, d, N)
            a_bar = np.exp(z)
            f = _expm1_over_z(z)
            b_bar = delta[:, :, None] * B_t * f
            h_prev = h
            h = a_bar * h_prev + b_bar * u_t[:, :, None]
            y = np.sum(C_t * h, axis=2) + p["D"][None, :] * u_t
            out[:, t, :] = y @ p["W_out"].T + p["b_out"]
            if cache:
                caches.append({
                    "x_t": X[:, t, :], "u_t": u_t, "s_pre": s_pre,
                    "delta": delta, "B_t": B_t, "C_t": C_t, "z": z,
                    "f": f, "fprime": _expm1_over_z_deriv(z),
                    "a_bar": a_bar, "b_bar": b_bar,
                    "h_prev": h_prev, "h": h, "y": y,
                })
        return out, caches

    def loss_and_grads(self, X, Y):
        """Mean squared error and its gradients wrt all parameters."""
        p = self.params
        B, L, _ = X.shape
        out, caches = self.forward(X, cache=True)
        err = out - Y
        count = B * L * 3
        loss = float(np.mean(err ** 2))
        dout = 2.0 * err / count                              # (B, L, 3)
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros((B, self.d, self.N))
        A = self.A
        for t in range(L - 1, -1, -1):
            c = caches[t]
            dout_t = dout[:, t, :]
            grads["W_out"] += dout_t.T @ c["y"]
            grads["b_out"] += dout_t.sum(axis=0)
            dy = dout_t @ p["W_out"]                          # (B, d)
            dC_t = dy[:, :, None] * c["h"]
            dh = dy[:, :, None] * c["C_t"] + dh_next
            grads["D"] += np.sum(dy * c["u_t"], axis=0)
            du = dy * p["D"][None, :]
            # h = a_bar * h_prev + b_bar * u
            dA_bar = dh * c["h_prev"]
            dB_bar = dh * c["u_t"][:, :, None]
            dh_next = dh * c["a_bar"]
            du += np.sum(dh * c["b_bar"], axis=2)
            # a_bar = exp(z); b_bar = delta * B_t * f(z)
            dz = dA_bar * c["a_bar"]
            dB_t = dB_bar * (c["delta"][:, :, None] * c["f"])
            dg = dB_bar * c["B_t"]
            dDelta = np.sum(dg * c["f"], axis=2)
            dz += dg * c["delta"][:, :, None] * c["fprime"]
            dA_step = np.sum(dz * c["delta"][:, :, None], axis=0)
            grads["log_A"] += dA_step * A
            dDelta += np.sum(dz * A[None, :, :], axis=2)
            # delta = softplus(s_pre)
            ds = dDelta * _sigmoid(c["s_pre"])
            grads["W_d"] += ds.T @ c["u_t"]
            grads["b_d"] += ds.sum(axis=0)
            du += ds @ p["W_d"]
            # B_t, C_t projections
            dB_flat = dB_t.reshape(B, -1)
            grads["W_B"] += dB_flat.T @ c["u_t"]
            grads["b_B"] += dB_flat.sum(axis=0)
            du += dB_flat @ p["W_B"]
            dC_flat = dC_t.reshape(B, -1)
            grads["W_C"] += dC_flat.T @ c["u_t"]
            grads["b_C"] += dC_flat.sum(axis=0)
            du += dC_flat @ p["W_C"]
            # u = x @ W_in.T + b_in
            grads["W_in"] += du.T @ c["x_t"]
            grads["b_in"] += du.sum(axis=0)
        return loss, grads

    # -- training --------------------------------------------------------
    def train(self, Xtr, Ytr, Xval=None, Yval=None, epochs=15, batch=64,
              lr=0.003, seed=0, log=None):
        """Train with Adam on standardised windows. Returns history dict.

        If validation windows are given, the best-validation-loss
        parameters are kept (checkpoint restore) and reported.
        """
        rng = np.random.default_rng(seed)
        n = Xtr.shape[0]
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t = 0
        best_val = math.inf
        best_params = None
        history = {"train_loss": [], "val_loss": []}
        for ep in range(epochs):
            order = rng.permutation(n)
            ep_loss, ep_count = 0.0, 0
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                loss, grads = self.loss_and_grads(Xtr[idx], Ytr[idx])
                self._adam_step(grads, lr)
                ep_loss += loss * len(idx)
                ep_count += len(idx)
            history["train_loss"].append(ep_loss / ep_count)
            if Xval is not None:
                val_out, _ = self.forward(Xval)
                val_loss = float(np.mean((val_out - Yval) ** 2))
                history["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_params = {k: v.copy() for k, v in self.params.items()}
                if log:
                    log(f"epoch {ep + 1}/{epochs} train_mse={history['train_loss'][-1]:.4f} "
                        f"val_mse={val_loss:.4f}")
            elif log:
                log(f"epoch {ep + 1}/{epochs} train_mse={history['train_loss'][-1]:.4f}")
        if best_params is not None:
            self.params = best_params
            history["best_val_loss"] = best_val
        return history

    def _adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        for k in self.params:
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * g * g
            m_hat = self._adam_m[k] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[k] / (1 - beta2 ** self._adam_t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # -- inference / scoring ----------------------------------------------
    def predict_next(self, window_std):
        """Predict the next standardised observation after a window.

        window_std: (L, 3) standardised. Returns (3,) standardised.
        """
        out, _ = self.forward(window_std[None, :, :])
        return out[0, -1, :]

    def score_windows(self, X_std):
        """Per-step prediction errors (standardised units) for a batch.

        X_std: (B, L, 3) standardised inputs. Returns (B, L, 3) of
        out - target where target is the next observation after each step;
        the caller must supply targets separately. See score_series().
        """
        out, _ = self.forward(X_std)
        return out

    def score_series(self, values, mean, std, window=24):
        """Score every point of a raw series with a causal window.

        values: (n, 3) raw. Returns (errors, scored_idx) where errors is
        (m, 3) of |observed - predicted| in ORIGINAL units and scored_idx
        the indices of the scored points (each needs a full window of
        consecutive hours before it; windows straddling a gap are skipped
        by the caller building windows).
        """
        n = len(values)
        errs = []
        idxs = []
        z = (values - mean) / std
        for i in range(window, n):
            w = z[i - window:i]
            pred_std = self.predict_next(w)
            pred = pred_std * std + mean
            errs.append(np.abs(values[i] - pred))
            idxs.append(i)
        return np.array(errs), np.array(idxs)

    def calibrate_sigma_e(self, val_errors):
        """Set sigma_e from normal validation errors (original units).

        val_errors: (..., 3) absolute prediction errors on normal data.
        sigma_e is the per-variable standard deviation of the SIGNED
        errors; callers passing absolute errors get the std of |e|, so
        prefer calibrate_sigma_e_signed when signed errors are available.
        """
        e = np.asarray(val_errors, dtype=np.float64).reshape(-1, 3)
        self.sigma_e = np.std(e, axis=0)
        self.sigma_e = np.where(self.sigma_e > 0, self.sigma_e, 1.0)
        return self.sigma_e

    def calibrate_sigma_e_signed(self, signed_errors):
        """Set sigma_e from SIGNED normal validation errors (original units)."""
        e = np.asarray(signed_errors, dtype=np.float64).reshape(-1, 3)
        self.sigma_e = np.std(e, axis=0)
        self.sigma_e = np.where(self.sigma_e > 0, self.sigma_e, 1.0)
        return self.sigma_e

    def temporal_score(self, errors):
        """S_M = max(e / sigma_e) over the three variables (phi = max).

        errors: (..., 3) absolute prediction errors in original units.
        Returns S_M in [0, inf); calibration to [0, 1] is done by the
        hybrid engine via the empirical CDF on validation data.
        """
        if self.sigma_e is None:
            raise ValueError("sigma_e not calibrated; call calibrate_sigma_e first")
        return np.max(errors / self.sigma_e, axis=-1)

    # -- persistence -------------------------------------------------------
    def to_artifact_dict(self):
        """JSON-serialisable artifact (for the backend model registry)."""
        return {
            "model_type": "mamba",
            "d_model": self.d,
            "state_dim": self.N,
            "params": {k: v.tolist() for k, v in self.params.items()},
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "sigma_e": None if self.sigma_e is None else self.sigma_e.tolist(),
            "tau_M": self.tau_M,
        }

    @classmethod
    def from_artifact_dict(cls, artifact):
        m = cls(d_model=artifact["d_model"], state_dim=artifact["state_dim"])
        m.params = {k: np.array(v, dtype=np.float64)
                    for k, v in artifact["params"].items()}
        m.mean = np.array(artifact["mean"], dtype=np.float64)
        m.std = np.array(artifact["std"], dtype=np.float64)
        m.sigma_e = (None if artifact.get("sigma_e") is None
                     else np.array(artifact["sigma_e"], dtype=np.float64))
        m.tau_M = artifact.get("tau_M")
        return m

    def save(self, path):
        """Save parameters (.npz) plus metadata (.json) next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"),
                 **{k: v for k, v in self.params.items()})
        meta = {"d_model": self.d, "state_dim": self.N,
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "sigma_e": None if self.sigma_e is None else self.sigma_e.tolist(),
                "tau_M": self.tau_M}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return str(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        m = cls(d_model=meta["d_model"], state_dim=meta["state_dim"])
        data = np.load(path.with_suffix(".npz"))
        m.params = {k: np.array(data[k], dtype=np.float64) for k in data.files}
        m.mean = np.array(meta["mean"], dtype=np.float64)
        m.std = np.array(meta["std"], dtype=np.float64)
        m.sigma_e = (None if meta.get("sigma_e") is None
                     else np.array(meta["sigma_e"], dtype=np.float64))
        m.tau_M = meta.get("tau_M")
        return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _split_windows(X, Y, train_frac=0.7):
    n = X.shape[0]
    k = int(n * train_frac)
    return X[:k], Y[:k], X[k:], Y[k:]


def cmd_train(args):
    stamps, values = load_district_series(args.data_dir, args.district)
    mean, std = standardise(values[:int(0.6 * len(values))])
    z = (values - mean) / std
    # chronological split on the window starts: train 60%, val 10%, test 30%
    X, Y = consecutive_windows(stamps, z, args.window)
    n = X.shape[0]
    n_train = int(0.6 * n)
    n_val = int(0.1 * n)
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xval, Yval = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    Xte, Yte = X[n_train + n_val:], Y[n_train + n_val:]
    print(f"{args.district}: {n} windows (train {len(Xtr)}, val {len(Xval)}, test {len(Xte)})")
    model = MambaEncoder(d_model=args.d_model, state_dim=args.state_dim, seed=42)
    model.mean, model.std = mean, std
    history = model.train(Xtr, Ytr, Xval, Yval, epochs=args.epochs,
                          batch=args.batch, lr=args.lr, log=print)
    # validation errors in original units -> sigma_e
    val_out, _ = model.forward(Xval)
    signed = (val_out - Yval) * std
    model.calibrate_sigma_e_signed(signed)
    print(f"sigma_e (val): T={model.sigma_e[0]:.3f} C, RH={model.sigma_e[1]:.3f} %RH, "
          f"P={model.sigma_e[2]:.3f} hPa")
    te_out, _ = model.forward(Xte)
    signed_te = (te_out - Yte) * std
    for i, name in enumerate(["T (C)", "RH (%RH)", "P (hPa)"]):
        e = signed_te[:, :, i]
        print(f"test {name}: bias={e.mean():+.3f} MAE={np.abs(e).mean():.3f} "
              f"RMSE={np.sqrt((e ** 2).mean()):.3f}")
    out_path = Path(args.out) / f"{args.district.replace(' ', '_')}_mamba_v1"
    model.save(out_path)
    print(f"saved {out_path}.npz/.json")


def cmd_score(args):
    stamps, values = load_district_series(args.data_dir, args.district)
    model = MambaEncoder.load(args.model)
    errs, idxs = model.score_series(values, model.mean, model.std,
                                    window=args.window)
    print(f"{args.district}: scored {len(idxs)} points")
    for i, name in enumerate(["T (C)", "RH (%RH)", "P (hPa)"]):
        e = errs[:, i]
        print(f"  {name}: MAE={e.mean():.3f}")
    if model.sigma_e is not None:
        s_m = model.temporal_score(errs)
        print(f"  S_M (raw, max e/sigma_e): median={np.median(s_m):.2f} "
              f"p95={np.percentile(s_m, 95):.2f} max={s_m.max():.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mamba selective SSM temporal predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--district", required=True)
    p.add_argument("--data-dir", default="workspace/skyguard-ai/data")
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--state-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--out", default="ml/models")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("score")
    p.add_argument("--district", required=True)
    p.add_argument("--data-dir", default="workspace/skyguard-ai/data")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=24)
    p.set_defaults(func=cmd_score)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
