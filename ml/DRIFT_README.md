# Model drift tracking (spec §21, §28.5)

Implemented 2026-11-20 (build-list item 8). Code: `ml/drift_engine.py`
(pure-Python metrics), `backend/app/services/drift_service.py`
(measurement, storage, actions), `backend/app/api/v1/drift.py`
(endpoints), `src/components/DriftPanel.jsx` (UI) plus a drift chip on
Command Center station rows.

## The one idea that matters: drift is measured in residual space

The climatology already removes seasonal and diurnal structure, so a
stable sensor with a correct model produces residuals that are
approximately zero-mean and unit-variance **in every season**. A shift
in the residual distribution therefore means the sensor or the model
changed — not that the season did. A naive PSI on raw T/H/P would fire
on every monsoon onset and withdrawal (two guaranteed false alarms a
year, exactly when the model is most needed). This module never
compares raw values.

## Metrics (14-day rolling window vs the training reference)

| Metric | What it means | Thresholds |
|---|---|---|
| PSI on residuals (T/H/P) | Distribution shape change | < 0.10 stable · 0.10–0.25 moderate · > 0.25 significant |
| Residual mean shift | Bias — the classic calibration-drift signature | < 0.3 ok · 0.3–0.7 warn · > 0.7 alert |
| Residual variance ratio | Noise degradation (>1) or flatlining (<1) | 0.5–2.0 ok · beyond 4× alert |
| Anomaly rate | Fraction flagged vs training contamination | > 3× contamination alerts (and > 5% absolute) |
| KS statistic | Score distribution shift vs training deciles | D > 0.2 warn · > 0.35 alert |
| Feature availability | Fraction of recent observations complete | < 0.8 warn · < 0.5 alert |
| Mamba error ratio | Mean \|prediction error\| / σ_e vs the calibrated 0.80 | < 1.25× ok · 1.5–2× warn · > 2× alert |

The LOW band (PSI 0.05–0.10, mean shift 0.2–0.3, …) is this
implementation's early-warning extension; the MODERATE/HIGH bands
follow §21.3 exactly. Severity is the max over metrics (§21.4).

The window is anchored at the station's **newest observation**, not
wall-clock "now": demo history is backfilled (timestamps shifted to
the present) and a live station's newest observation is "now" anyway.
Fewer than 7 days or 100 rows in the window → `INSUFFICIENT_DATA`,
never a false NONE.

## Training reference

A model trained after this change carries `drift_reference` in its
model card: residual deciles on the cleaned fit partition, IF score
deciles on the holdout, residual variances, contamination. Older
models rebuild the same reference from the training window still in
the station's telemetry; when that window is gone the computation is
marked PARTIAL — PSI/KS are skipped and the rest uses the theoretical
residual reference (mean 0, variance 1). The Mamba reference is the
stored validation scores (`mamba_val_raw`) plus the calibrated σ_e.

## What happens on MODERATE / HIGH

- MODERATE → a retraining **candidate** is queued (priority LOW).
- HIGH → a candidate is queued (priority HIGH) **and** a DATA_QUALITY
  incident is opened, so an operator sees it.
- A candidate is a queue entry only. Training it still needs a human
  to start the job, and activation still needs the promotion gate.
  Candidates and incidents are de-duplicated (one queued candidate
  per station+model per 24 h; one open drift incident per station).

Deliberate deviation from spec §21.4: the spec also reduces the
station's confidence multiplier to 0.8 on HIGH drift. That would
silently change detection behaviour, so it is **not** implemented.
Drift observes; it never retunes scoring, thresholds, or gates.

## Known limitation: regional weather moves residuals too

A sustained regional pattern (e.g. the monsoon-break dry spells that
make Udupi fail its gate) shifts humidity residuals for every station
in the region, and drift will flag it. That is correct behaviour — the
recent weather genuinely differs from what the model learned — but it
means drift is a prompt to **investigate**, not proof of a sensor
fault. The panel shows the per-parameter breakdown (a humidity-driven
shift with temperature and pressure steady points at weather; a
single-parameter offset points at the sensor) and the runbook says so
on stage.

## API

- `GET /stations/{id}/drift?days=90` — latest per model type + history + candidates + thresholds
- `GET /stations/{id}/drift/latest` — latest per model type
- `POST /stations/{id}/drift/compute` — admin-only on-demand compute (performs the actions above)
- `POST /stations/{id}/drift/queue-retraining` — admin-only manual queue

Reads follow the station rule (admin any station, operator only their
own). A nightly job computes drift for every station (first run 5 min
after startup, then every 24 h); in DEMO_MODE it only observes —
automated triggers are disabled in demo mode (§22.2) — while the
admin's on-demand compute still performs the actions, because a human
explicitly asked.

## UI

- **Model Governance**: drift panel under the model card — severity
  badge in plain words, per-metric table with thresholds, 90-day
  sparklines of PSI and residual mean, "Check Drift Now" and "Queue
  Retraining" buttons (admin only).
- **Command Center**: a DRIFT chip on each station row (worst severity
  across its models, refreshed every 60 s).

## Demo

See DEMO_RUNBOOK.md §M6. The recipe: AWS-03 is a demo station
backfilled with the real Dakshina Kannada series plus a sustained
+2 °C offset on the last 21 days, so its drift comes back HIGH with a
real incident and a real queued candidate. AWS-01/AWS-02 show the
genuine dry-spell signal in the real data.
