#!/usr/bin/env python3
"""SkyGuard-AI — fault-scenario injection for validation (paper Table II).

The fixed-offset scenario schedule used by the E1–E4 evaluation harness
(scripts/evaluate_e1_e4.py), the Mamba registration CLI
(scripts/register_mamba.py) and the in-product Mamba training step
(backend/app/services/mamba_training.py). The schedule is deliberately
identical everywhere it is used: validation regions are injected the
same way, so the hybrid-weight (alpha) calibration means the same thing
in the paper evaluation and in the product.

Instances that do not fit in the region are skipped, so the same
schedule works for a 261 h validation region and a 786 h test region.
Labels: 0 normal, 1 fault, 2 regional event. The pair (regional /
sea-breeze) events are only placed when pair=True.
"""

import numpy as np

LABEL_NORMAL, LABEL_FAULT, LABEL_EVENT = 0, 1, 2


def _add(values, i, dt=0.0, dh=0.0, dp=0.0):
    if 0 <= i < len(values):
        values[i, 0] += dt
        values[i, 1] += dh
        values[i, 2] += dp


def inject(values, pair=False):
    """Inject scenarios into a copy of a region's values.

    values: (n, 3) raw [T, RH, P]. Returns (injected, labels, instances).
    labels: 0 normal, 1 fault, 2 regional event. instances: list of dicts
    with type/start/end/label. Instances that do not fit in the region
    are skipped (so the same schedule works for the 261 h validation
    region and the 786 h test region).
    """
    v = values.copy()
    n = len(v)
    labels = np.zeros(n, dtype=int)
    instances = []

    def place(kind, start, dur, label, apply_fn):
        if start < 0 or start + dur > n:
            return
        apply_fn(start, dur)
        labels[start:start + dur] = label
        instances.append({"type": kind, "start": int(start), "end": int(start + dur),
                          "label": int(label)})

    # Temperature spikes: +5 C for 1 h.
    for s in (50, 200, 400):
        place("temp_spike", s, 1, LABEL_FAULT,
              lambda a, d, s=s: _add(v, s, dt=5.0))
    # Humidity drift: +0.15 %RH/h for 48 h (up to +7.2 %RH).
    for s in (100, 350):
        def drift(a, d, s=s):
            for k in range(d):
                _add(v, a + k, dh=0.15 * (k + 1))
        place("humidity_drift", s, 48, LABEL_FAULT, drift)
    # Pressure abnormality: -4 hPa step for 6 h.
    for s in (150, 500):
        def press(a, d, s=s):
            for k in range(d):
                _add(v, a + k, dp=-4.0)
        place("pressure_abnormality", s, 6, LABEL_FAULT, press)
    # Stuck sensor: temperature frozen for 12 h.
    for s in (250, 550):
        def stuck(a, d, s=s):
            frozen = v[a - 1, 0] if a > 0 else v[a, 0]
            for k in range(d):
                v[a + k, 0] = frozen
        place("stuck_sensor", s, 12, LABEL_FAULT, stuck)
    # Missing observations: 6 h gap (all channels NaN).
    for s in (300, 600):
        def missing(a, d, s=s):
            v[a:a + d, :] = np.nan
        place("missing", s, 6, LABEL_FAULT, missing)
    # Gradual sensor degradation: +0.05 C/h for 72 h (up to +3.6 C).
    def degr(a, d):
        for k in range(d):
            _add(v, a + k, dt=0.05 * (k + 1))
    place("gradual_degradation", 650, 72, LABEL_FAULT, degr)

    if pair:
        # Regional weather change: T -3 C, RH +8 %RH over 6 h, ramped.
        ramp = [0.3, 0.7, 1.0, 1.0, 0.7, 0.3]
        def regional(a, d):
            for k in range(d):
                _add(v, a + k, dt=-3.0 * ramp[k], dh=8.0 * ramp[k])
        place("regional_event", 180, 6, LABEL_EVENT, regional)
        # Sea-breeze hard negatives: T -4 C, RH +10 %RH over 4 h.
        # The pair partner applies the same change 3 h later (front
        # propagation); the caller staggers by shifting `values` per
        # district, so here we only place this district's part.
        breeze = [0.5, 1.0, 1.0, 0.5]
        def breeze_fn(a, d):
            for k in range(d):
                _add(v, a + k, dt=-4.0 * breeze[k], dh=10.0 * breeze[k])
        place("sea_breeze", 450, 4, LABEL_EVENT, breeze_fn)
        place("sea_breeze", 730, 4, LABEL_EVENT, breeze_fn)
    return v, labels, instances


def inject_pair(values_u, values_k):
    """Inject the coordinated pair events (Udupi first, DK 2-3 h later)."""
    vu, lu, iu = inject(values_u, pair=True)
    vk, lk, ik = inject(values_k, pair=True)
    # Shift DK's regional event +2 h and sea-breeze +3 h relative to Udupi.
    # Re-inject cleanly: rebuild DK from its clean values with shifted starts.
    vk = values_k.copy()
    lk = np.zeros(len(vk), dtype=int)
    ik = []
    # Re-run the fault-only part for DK by copying from the first inject.
    vk, lk, ik = inject(values_k, pair=False)
    # Now add the pair events with DK's delays.
    ramp = [0.3, 0.7, 1.0, 1.0, 0.7, 0.3]
    for k in range(6):
        _add(vk, 182 + k, dt=-3.0 * ramp[k], dh=8.0 * ramp[k])
    lk[182:188] = LABEL_EVENT
    ik.append({"type": "regional_event", "start": 182, "end": 188, "label": LABEL_EVENT})
    breeze = [0.5, 1.0, 1.0, 0.5]
    for start in (453, 733):
        for k in range(4):
            _add(vk, start + k, dt=-4.0 * breeze[k], dh=10.0 * breeze[k])
        lk[start:start + 4] = LABEL_EVENT
        ik.append({"type": "sea_breeze", "start": start, "end": start + 4,
                   "label": LABEL_EVENT})
    return (vu, lu, iu), (vk, lk, ik)
