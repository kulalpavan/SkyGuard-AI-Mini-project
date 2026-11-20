# SkyGuard-AI — Mamba Temporal Engine and Hybrid Scoring

This is the Mamba half of SkyGuard-AI: a selective state-space temporal
predictor (NumPy only, no torch, no new dependencies — matching the
repo's zero-ML-library style), the hybrid Isolation Forest + Mamba score
from the paper (Section IX), the E1–E4 evaluation harness, and the
backend wiring that puts a Mamba score into the real pipeline output.

Every number below was **measured by running the code** on the five
supplied district files (hourly gridded records, 2,617 timestamps
2025-05-26 → 2025-09-13). Nothing here is invented. Where Mamba
underperforms, that is stated plainly.

## What was built

| File | What it is |
| --- | --- |
| `ml/mamba_engine.py` | Selective state-space predictor. Input-dependent B, C, Δ projections; ZOH discretisation (Ā = exp(ΔA), B̄ = ΔB·(exp(z)−1)/z with a Taylor limit for \|z\| < 1e-6); recurrence h_t = Āh_{t−1} + B̄u_t, y_t = C_t h_t + D u_t; Adam in NumPy with hand-written BPTT; next-step prediction of (T, RH, P) from 24 h windows (d_model=32, state_dim=8); per-variable error scale σ_e from validation; S_M = max(e/σ_e); save/load to `.npz` + `.json`. CLI: `python -m ml.mamba_engine train/score`. |
| `ml/hybrid_engine.py` | Hybrid score S_hybrid = αS_IF + βS_M (α+β=1) with ECDF calibration (Eq. 14; NaN propagates — a missing engine score never becomes maximally anomalous), α grid search 0–1 step 0.05 maximising F1 subject to FPR ≤ 0.05, per-station calibration, and the three-way outcome (NORMAL / LOCAL_SENSOR_FAULT / REGIONAL_WEATHER_EVENT) per Eqs. 18–23 / Algorithm 5, plus nearby-anomaly detection (60 km radius, 6 h propagation window, same-sign check). |
| `scripts/evaluate_e1_e4.py` | E1–E4 harness. Chronological splits (train 60%, validation 10%, test 30%); fault injection into held-out data only (spikes, drift, stuck, missing, gradual degradation; regional events and sea-breeze hard negatives for the Udupi↔DK pair); raw point-wise precision/recall/F1 (no point adjustment), FPR, detection delay; Mamba next-step MAE/RMSE vs persistence and AR(1) on the same clean test points. Writes `results/e1_e4_results.json`. |
| `scripts/register_mamba.py` | Trains a Mamba model for a station, calibrates (σ_e, ECDFs, α, τ_hybrid, τ_M), and registers the artifact (model_type `Mamba`, status CANDIDATE — the IF model keeps the single ACTIVE row). Reuses the ACTIVE IF model when one exists; otherwise trains a fresh forest from the district data. |
| `backend/app/services/mamba_service.py` | Mamba registry/scoring/hybrid service. Loads the latest non-archived Mamba row, caches it, scores one observation from the 24 preceding hourly observations. The window is the most recent 24 hourly-spaced rows: rapid live frames (seconds apart) are skipped, and an off-grid live frame falls back to the most recent on-grid hourly row — so the model always scores on the hourly history it was trained on. Returns raw S_M, errors, prediction, τ_M (raw, 95th pct of validation S_M), is_anomaly. NumPy is imported lazily; without it the backend still boots and Mamba reports NUMPY_UNAVAILABLE. |
| `backend/app/services/observation_pipeline.py` | Wiring: after the IF scoring, Mamba is scored under the same TRAINED/MATURE readiness gate; if both engines scored, the hybrid is computed. Evidence gains `z_mamba`; the assessment gains `mamba_score`, `mamba_errors`, `mamba_prediction`, `mamba_status`, `hybrid_score`, `hybrid_contributions`; state gains `mamba_model` and `hybrid`. `fuse()` drops the `z_mamba` weight when no Mamba channel is present, so pre-Mamba behaviour is identical. |
| `scripts/backfill_demo_history.py` | Backfills a station with a district's full hourly series (timestamps shifted to end 1 h ago) so the demo has ≥720 observations for IF training and a 24 h Mamba window. Labelled SIMULATED demo history. |

## How to train and evaluate

```bash
# E1–E4 harness (all five districts, ~3 min on this VM)
python3 scripts/evaluate_e1_e4.py --data-dir ~/workspace/skyguard-ai/data \
    --out results/e1_e4_results.json --epochs 15

# Register a Mamba model for a station (needs the backend venv for numpy)
DEMO_MODE=true ../venv/bin/python scripts/register_mamba.py \
    --station AWS-02 --district "Dakshina Kannada" --epochs 15

# Mamba CLI
python3 -m ml.mamba_engine train --district Udupi --epochs 15
python3 -m ml.mamba_engine score --model ml/models/AWS-02/AWS-02_MAMBA_v1_0.npz
```

## Measured results (E1–E4, `results/e1_e4_results.json`)

Test set: latest 30% (786 h per district), injected faults. Raw point-wise
metrics, no point adjustment.

| Config | Precision | Recall | F1 | FPR | Mean delay | Instances |
| --- | --- | --- | --- | --- | --- | --- |
| E1 IF only | 0.611 | 0.517 | **0.560** | 0.132 | 3.50 h | 62/66 |
| E2 Mamba only | 0.582 | 0.377 | 0.457 | 0.108 | 5.72 h | 63/66 |
| E3 IF+Mamba | 0.575 | 0.431 | 0.493 | 0.128 | 4.09 h | 63/66 |
| E4 +QC+fusion | 0.575 | 0.431 | 0.493 | 0.128 | 4.09 h | 63/66 |

E4's detection is identical to E3 by design — E4 adds the three-way
classification, not a new detection flag. Its value is the
classification: **accuracy 0.580** on 842 detected anomaly points, and
sea-breeze hard negatives blamed as sensor faults **5/16** points
(the prototype's own 0/6 figure is a different evaluation on different
data; do not conflate them).

**Read this honestly: the hybrid (E3, F1 0.493) does not beat IF alone
(E1, F1 0.560) on this data.** Mamba alone is the weakest detector
(E2, F1 0.457). The hybrid sits between the two engines. Per-scenario
recall (E3): temperature spikes 1.00, missing 1.00, regional events
1.00, sea-breeze 1.00, pressure abnormality 0.47, stuck sensor 0.40,
gradual degradation 0.39, humidity drift 0.34. Slow drift is hard for
both engines; Mamba helps most on pressure abnormalities (IF 0.28 →
hybrid 0.47) and hurts most on gradual degradation (IF 0.66 → hybrid
0.39).

Calibrated weights (validation F1 at the chosen α, clean FPR 0.050):

| District | α (IF) | β (Mamba) | τ_hybrid | val F1 |
| --- | --- | --- | --- | --- |
| Udupi | 0.00 | 1.00 | 0.950 | 0.321 |
| Dakshina Kannada | 0.15 | 0.85 | 0.930 | 0.364 |
| Belagavi | 0.15 | 0.85 | 0.926 | 0.417 |
| Bengaluru Urban | 0.50 | 0.50 | 0.910 | 0.187 |
| Mysore | 0.75 | 0.25 | 0.905 | 0.213 |

Note: the FPR constraint is measured on **clean** validation points.
The injected copy's "normal" points are contaminated by neighbouring
faults through the windows, so their FPR (0.07–0.17) is inflated and
cannot be constrained to 0.05 — an earlier version of the search
constrained it there, found no feasible α, and silently fell back to
α=0.50 with F1=0. That degenerate fallback is fixed: the constraint now
uses the clean FPR (which τ holds at ~0.05 by construction), and the
injected-copy FPR is reported as a diagnostic only.

## Mamba next-step prediction vs baselines (clean test points, MAE)

| District | Variable | Mamba | Persistence | AR(1) | Mamba vs persistence |
| --- | --- | --- | --- | --- | --- |
| Udupi | T (°C) | 0.246 | 0.253 | 0.253 | −3% |
| Udupi | RH (%RH) | 1.161 | 1.101 | 1.108 | +5% (worse) |
| Udupi | P (hPa) | 0.430 | 0.468 | 0.462 | −8% |
| Dakshina Kannada | T | 0.230 | 0.248 | 0.250 | −7% |
| Dakshina Kannada | RH | 1.119 | 1.108 | 1.110 | +1% (worse) |
| Dakshina Kannada | P | 0.426 | 0.477 | 0.471 | −11% |
| Belagavi | T | 0.457 | 0.365 | 0.370 | **+25% (worse)** |
| Belagavi | RH | 2.416 | 1.583 | 1.575 | **+53% (worse)** |
| Belagavi | P | 0.450 | 0.474 | 0.469 | −5% |
| Bengaluru Urban | T | 0.466 | 0.555 | 0.562 | −16% |
| Bengaluru Urban | RH | 1.918 | 2.398 | 2.405 | −20% |
| Bengaluru Urban | P | 0.388 | 0.550 | 0.541 | −29% |
| Mysore | T | 0.416 | 0.544 | 0.541 | −24% |
| Mysore | RH | 1.868 | 2.409 | 2.372 | −22% |
| Mysore | P | 0.402 | 0.552 | 0.542 | −27% |

Mamba beats persistence on temperature and pressure in 4 of 5 districts
and on humidity in 3 of 5 — but **Belagavi is a clear failure**: Mamba
is 25% worse than persistence on temperature and 53% worse on humidity
there (its σ_e is also the largest: T 0.666 °C, RH 2.767 %RH). The
1-hour headroom is small everywhere (lag-1 autocorrelation 0.93–0.97),
so a 1-hour predictor's edge over persistence is necessarily modest.

An earlier version of the harness compared Mamba on the *injected*
test series (fault-contaminated windows) against persistence on *clean*
values, which made Mamba look far worse than it is (Udupi T MAE 0.389
vs 0.247). That comparison was not a comparison; the harness now scores
Mamba on the clean test series for the MAE table.

## Live pipeline verification (AWS-02, Dakshina Kannada, 2026-11-20)

Backfilled the full DK series (2,617 h), trained the IF model through
the real training job (it activated — holdout anomaly rate 0.095 ≤ 0.15),
registered the Mamba model (15 epochs, α=0.05, τ_hybrid=0.934), then
ingested live frames through `POST /api/v1/telemetry/esp32/ingest`:

| Frame | cloud_ml | fusion | mamba_score | hybrid_score | mamba_errors [T, RH, P] |
| --- | --- | --- | --- | --- | --- |
| Normal (24.64 °C, 91.5 %RH) | NORMAL | 0.047 | 0.577 | 0.143 | [0.16, 0.57, 0.05] |
| Spike (+6 °C) | LOCALIZED_ANOMALY_UNCONFIRMED | 0.999 | 21.94 | 0.998 | [6.16, 0.57, 0.05] |

The spike raised incident `INC-49b4…` (temp:THERMAL_SPIKE, severity
high, status open, fault_risk 0.999). Its `evidence_data` carries
`final_assessment` (mamba_score 21.94, hybrid_score 0.998), `hybrid`
(S_IF 0.962, S_M 1.0, S_hybrid 0.998, α 0.05, β 0.95, τ 0.934) and
`mamba_model` (model_id `AWS-02_MAMBA_v1_0`, prediction
[24.48, 92.09, 1009.86], is_anomaly true). See `DEMO_RUNBOOK.md`
("Mamba scoring" section) for the exact commands.

## Known limitations (do not present these as working)

- **Udupi cannot get an ACTIVE IF model through the normal training
  flow.** Its holdout anomaly rate is **0.153**, just above the
  repo's promotion gate (≤ 0.15) — the only district of the five that
  fails (DK 0.095, Belagavi 0.059, Bengaluru 0.015, Mysore 0.092).
  So on AWS-01 the pipeline shows a Mamba score but `hybrid_score`
  is None (no ACTIVE IF model to combine with). The Mamba model
  still registers there via the fresh-forest fallback in
  `register_mamba.py`. Do not bypass the gate to make the demo look
  better — the gate is doing its job.
- **E3/E4 do not beat E1 on this data** (see the table above). The
  hybrid is not yet a win; it is a measured baseline to improve on.
- **Belagavi Mamba underperforms persistence substantially.** Do not
  quote the good districts without the bad one.
- **No wind data** in the supplied files, so front consistency `c` is
  derived from the delayed-neighbour pattern, not a wind shift.
  **No device-health telemetry**, so `h = 0` throughout.
- **NumPy is not in `requirements.txt`.** The ML code needs it
  (installed in the demo venv only). Adding it to requirements.txt
  needs a decision — the repo's style is zero ML dependencies.
- **Mamba does not run on the ESP32.** It runs in the backend only.
- The E1–E4 numbers are on gridded reanalysis-style data, not raw
  ESP32 sensor readings, and the injected faults are synthetic. They
  are a harness measurement, not field performance.

## Tests

```bash
python3 ml/test_mamba_engine.py        # 5 checks, all pass
python3 ml/test_hybrid_engine.py       # 6 checks, all pass
python3 ml/test_station_isolation.py   # existing suite, passes
python3 ml/test_enhanced_engines.py    # existing suite, passes
# test_spatial_intelligence.py FAILS on main too (pre-existing,
# unrelated to this work: residual_temp assertion).
../venv/bin/python -m pytest backend/tests/   # 84 tests, all pass
```

`ml/test_mamba_engine.py` covers discretisation (closed form + small-z
limit), gradient vs finite differences (worst abs 2.7e-10), one training
step reducing loss (0.3874 → 0.0505), save/load round-trip, prediction
shape and S_M range. `backend/tests/test_mamba_pipeline.py` covers the
live wiring: the full hybrid on a DK-valued station, Mamba-only on a
Udupi-valued station (no ACTIVE IF model), and the hourly-window
selection (rapid and off-grid live frames).
