#!/usr/bin/env python3
"""SkyGuard-AI — E1-E4 evaluation harness (paper Section X).

Four configurations, evaluated on identical test conditions:
  E1  Isolation Forest only
  E2  Mamba temporal detector only
  E3  IF + Mamba hybrid score (weights calibrated on validation)
  E4  E3 + physical QC + multi-station evidence fusion (three-way)

Protocol (paper Section X.B): chronological splits — train = earliest 60%,
validation = next 10% (threshold calibration, sigma_e, alpha search),
test = latest 30%. Fault scenarios are injected into held-out data only
(validation gets its own injected copy for the alpha search). Metrics are
raw point-wise precision/recall/F1 (no point-adjusted F1), false-positive
rate, and detection delay in hours. Root-cause classification accuracy is
reported separately for E4.

Mamba next-step MAE/RMSE is compared on the SAME test points against
persistence and a refitted AR(1) (both fitted/scored on the same split).

Writes REAL measured results to results/e1_e4_results.json. Nothing here
is invented: every number comes from running the code on the supplied
five-district files.

Fault scenarios (paper Table II), injected at fixed offsets in the test
region (786 h). Regional / sea-breeze events exist only for the prototype
pair Udupi <-> Dakshina Kannada (the only two districts within 60 km in
this data); the other three districts get fault scenarios only.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.mamba_engine import (MambaEncoder, load_district_series,
                             standardise, DISTRICT_ELEVATION_M,
                             DISTRICT_COORDS, consecutive_windows)
from ml.hybrid_engine import (HybridScorer, train_if_forest,
                              if_raw_score, rows_from_series,
                              hard_qc_violation, classify_three_way,
                              nearby_anomaly, PROPAGATION_WINDOW_H)

DATA_DIR = Path("workspace/skyguard-ai/data")
DISTRICTS = ["Udupi", "Dakshina Kannada", "Belagavi", "Bengaluru Urban", "Mysore"]
WINDOW = 24
# Fault injection lives in ml/fault_injection.py (shared with the Mamba
# registration CLI and the in-product Mamba training step); the schedule
# is identical everywhere so the alpha calibration means the same thing.
from ml.fault_injection import (inject, inject_pair, LABEL_NORMAL,
                                LABEL_FAULT, LABEL_EVENT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def region_windows(stamps, z, window, lo, hi):
    """Windows whose L+1 points lie entirely within point range [lo, hi)."""
    sub_stamps = stamps[lo:hi]
    sub_z = z[lo:hi]
    X, Y = consecutive_windows(sub_stamps, sub_z, window)
    return X, Y


def metrics_from_flags(flag, labels):
    """Raw point-wise detection metrics. Positives = fault OR event."""
    pos = labels > 0
    tp = int(np.sum(flag & pos))
    fp = int(np.sum(flag & ~pos))
    fn = int(np.sum(~flag & pos))
    tn = int(np.sum(~flag & ~pos))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def detection_delays(flag, instances):
    """Mean delay (hours) from instance onset to first flag; None if missed."""
    delays = []
    missed = 0
    for inst in instances:
        seg = flag[inst["start"]:inst["end"]]
        hit = np.nonzero(seg)[0]
        if len(hit):
            delays.append(int(hit[0]))
        else:
            missed += 1
    return {"mean_delay_h": float(np.mean(delays)) if delays else None,
            "detected": len(delays), "missed": missed,
            "n_instances": len(instances)}


def per_scenario_recall(flag, instances):
    out = {}
    for inst in instances:
        seg = flag[inst["start"]:inst["end"]]
        out.setdefault(inst["type"], []).append(float(np.mean(seg)) if len(seg) else 0.0)
    return {k: float(np.mean(v)) for k, v in out.items()}


def fit_ar1_predict(train_pairs_x, train_pairs_y, test_x):
    """Least-squares AR(1) x(t+1) = a + b*x(t) per variable; predict test."""
    A = np.column_stack([np.ones(len(train_pairs_x)), train_pairs_x])
    coef, *_ = np.linalg.lstsq(A, train_pairs_y, rcond=None)
    return coef[0] + coef[1] * test_x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, out_path, epochs=15, verbose=True):
    t0 = time.time()
    results = {"method": {
        "splits": "chronological: train earliest 60%, validation next 10%, test latest 30% (point indices)",
        "window": WINDOW,
        "mamba": "d_model=32, state_dim=8, lr=0.003, Adam, epochs=%d, best-val checkpoint" % epochs,
        "if": "repo IsolationForest, n_trees=40, 12-feature contract (feature_engine), climatology fit on train",
        "alpha_search": "grid 0..1 step 0.05 on validation with injected faults, F1 max subject to FPR<=0.05 measured on CLEAN validation points (tau is the 95th pct of clean hybrid scores, so clean FPR is ~0.05 by construction; the injected copy's normal-point FPR is inflated by fault-contaminated windows and is reported separately, not constrained)",
        "fusion": "60 km radius, propagation window 6 h, paper Eqs. 18-23",
        "metrics": "raw point-wise precision/recall/F1 (no point adjustment), FPR, detection delay (h)",
        "notes": [
            "Regional/sea-breeze events are injected only for the prototype pair Udupi/Dakshina Kannada.",
            "No wind data exist in the supplied files, so front consistency c is derived from the delayed-neighbour pattern, not a wind shift.",
            "No device-health telemetry exists in the supplied files, so h = 0 throughout.",
            "Mamba/IF scores are unavailable for points whose window/history contains injected missing values; those points rely on ingest QC.",
        ]},
        "districts": {}, "configs": {}, "classification_E4": {}}

    # Load all districts.
    series = {}
    for d in DISTRICTS:
        stamps, values = load_district_series(data_dir, d)
        series[d] = (stamps, values)
        if verbose:
            print(f"loaded {d}: {len(values)} hourly points")

    # Per-district train/val/test and models.
    per_district = {}
    for d in DISTRICTS:
        stamps, values = series[d]
        n = len(values)
        n_train, n_val = int(0.6 * n), int(0.1 * n)
        lo_tr, hi_tr = 0, n_train
        lo_va, hi_va = n_train, n_train + n_val
        lo_te, hi_te = n_train + n_val, n
        mean, std = standardise(values[lo_tr:hi_tr])
        z = (values - mean) / std

        # --- Mamba ---
        Xtr, Ytr = region_windows(stamps, z, WINDOW, lo_tr, hi_tr)
        Xva, Yva = region_windows(stamps, z, WINDOW, lo_va, hi_va)
        mamba = MambaEncoder(d_model=32, state_dim=8, seed=42)
        mamba.mean, mamba.std = mean, std
        mamba.train(Xtr, Ytr, Xva, Yva, epochs=epochs, batch=64, lr=0.003)
        # sigma_e from signed final-step validation errors (original units)
        va_out, _ = mamba.forward(Xva)
        signed_va = (va_out[:, -1, :] - Yva[:, -1, :]) * std
        mamba.calibrate_sigma_e_signed(signed_va)

        # --- Isolation Forest (12-feature contract) ---
        rows = rows_from_series(stamps, values)
        forest, clim, stats = train_if_forest(
            rows[lo_tr:hi_tr], elevation_m=DISTRICT_ELEVATION_M[d])

        per_district[d] = {
            "stamps": stamps, "values": values, "rows": rows,
            "n_train": n_train, "n_val": n_val, "n_test": n - lo_te,
            "lo_te": lo_te, "mean": mean, "std": std, "z": z,
            "mamba": mamba, "forest": forest, "clim": clim, "stats": stats,
        }
        if verbose:
            print(f"  {d}: mamba trained (sigma_e T={mamba.sigma_e[0]:.3f}, "
                  f"RH={mamba.sigma_e[1]:.3f}, P={mamba.sigma_e[2]:.3f}); IF trained")

    # --- Validation scoring (clean) for ECDF calibration + alpha search ---
    def score_points(d, values_region, rows_region, region_lo):
        """Score every point of a region. Returns dict of arrays.

        values_region: (m, 3) raw values (possibly injected). rows_region:
        matching repo rows. region_lo: global index of region start.
        The Mamba window and IF history are taken from the (possibly
        injected) series itself, exactly as the deployed system would
        see it; clean data before the region supplies the early history.
        """
        info = per_district[d]
        mamba, forest = info["mamba"], info["forest"]
        clim, stats = info["clim"], info["stats"]
        mean, std = info["mean"], info["std"]
        m = len(values_region)
        if_raw = np.full(m, np.nan)
        mamba_raw = np.full(m, np.nan)
        mamba_err = np.full((m, 3), np.nan)
        mamba_err_signed = np.full((m, 3), np.nan)
        q = np.zeros(m, dtype=int)
        # Full series as the deployed system sees it: clean before the
        # region, (possibly injected) values inside it.
        full_vals = np.concatenate([info["values"][:region_lo], values_region])
        full_rows = info["rows"][:region_lo] + rows_region
        for j in range(m):
            gj = region_lo + j
            t, h, p = values_region[j]
            if not (np.isfinite(t) and np.isfinite(h) and np.isfinite(p)):
                q[j] = 1
                continue
            q[j] = hard_qc_violation(t, h, p)
            hist = full_rows[max(0, gj - 12):gj]
            try:
                if_raw[j] = if_raw_score(forest, clim, stats, rows_region[j], hist)
            except Exception:
                if_raw[j] = np.nan
            # Mamba window: previous 24 h of the series as seen.
            if gj >= WINDOW:
                wvals = full_vals[gj - WINDOW:gj]
                if np.isfinite(wvals).all():
                    wz = (wvals - mean) / std
                    pred_std = mamba.predict_next(wz)
                    pred = pred_std * std + mean
                    signed = values_region[j] - pred
                    mamba_err_signed[j] = signed
                    mamba_err[j] = np.abs(signed)
                    mamba_raw[j] = float(np.max(np.abs(signed) / mamba.sigma_e))
        return {"if_raw": if_raw, "mamba_raw": mamba_raw,
                "mamba_err": mamba_err, "mamba_err_signed": mamba_err_signed,
                "q": q}

    # Validation: clean scores for ECDFs, injected copy for alpha search.
    scorers = {}
    for d in DISTRICTS:
        info = per_district[d]
        lo_va = info["n_train"]
        hi_va = lo_va + info["n_val"]
        clean = score_points(d, info["values"][lo_va:hi_va],
                             info["rows"][lo_va:hi_va], lo_va)
        # Injected validation copy (same recipe, fits the 261 h region).
        if d in ("Udupi", "Dakshina Kannada"):
            inj = None  # pair handled below
        else:
            inj_vals, inj_labels, _ = inject(info["values"][lo_va:hi_va].copy())
            inj_rows = rows_from_series(info["stamps"][lo_va:hi_va], inj_vals)
            inj = score_points(d, inj_vals, inj_rows, lo_va)
            inj["labels"] = inj_labels
        scorers[d] = {"clean": clean, "inj": inj}
    # Pair validation injection (coordinated regional/sea-breeze).
    (vu, lu, _), (vk, lk, _) = inject_pair(
        series["Udupi"][1][per_district["Udupi"]["n_train"]:
                            per_district["Udupi"]["n_train"] + per_district["Udupi"]["n_val"]].copy(),
        series["Dakshina Kannada"][1][per_district["Dakshina Kannada"]["n_train"]:
                                       per_district["Dakshina Kannada"]["n_train"] + per_district["Dakshina Kannada"]["n_val"]].copy())
    for d, vals, labels in (("Udupi", vu, lu), ("Dakshina Kannada", vk, lk)):
        info = per_district[d]
        lo_va = info["n_train"]
        rows_inj = rows_from_series(info["stamps"][lo_va:lo_va + info["n_val"]], vals)
        inj = score_points(d, vals, rows_inj, lo_va)
        inj["labels"] = labels
        scorers[d]["inj"] = inj

    # Calibrate ECDFs + alpha per district.
    for d in DISTRICTS:
        clean = scorers[d]["clean"]
        inj = scorers[d]["inj"]
        sc = HybridScorer(d)
        ok_if = clean["if_raw"][np.isfinite(clean["if_raw"])]
        ok_m = clean["mamba_raw"][np.isfinite(clean["mamba_raw"])]
        sc.calibrate(ok_if, ok_m)
        # Alpha search on the injected validation copy.
        vi = np.isfinite(inj["if_raw"]) & np.isfinite(inj["mamba_raw"])
        res = sc.calibrate_alpha(inj["if_raw"][vi], inj["mamba_raw"][vi],
                                 inj["labels"][vi],
                                 clean_if_raw=clean["if_raw"],
                                 clean_mamba_raw=clean["mamba_raw"])
        scorers[d]["scorer"] = sc
        per_district[d]["alpha_result"] = res
        if verbose:
            print(f"  {d}: alpha={res['alpha']:.2f} beta={res['beta']:.2f} "
                  f"tau_hybrid={res['tau_hybrid']:.3f} val F1={res['f1']:.3f} "
                  f"FPR={res['fpr']:.3f}")
            # Diagnostic: why is the validation F1 what it is?
            _s_if = sc.if_calibrator.transform(inj["if_raw"][vi])
            _s_m = sc.mamba_calibrator.transform(inj["mamba_raw"][vi])
            _lab = inj["labels"][vi]
            _flt = _lab == 1
            _hyb = res["alpha"] * _s_if + res["beta"] * _s_m
            print(f"    [diag] val finite={int(vi.sum())}/{len(vi)} "
                  f"fault_pts={int(_flt.sum())} "
                  f"max S_IF on fault={float(np.nanmax(_s_if[_flt])) if _flt.any() else float('nan'):.3f} "
                  f"max S_M on fault={float(np.nanmax(_s_m[_flt])) if _flt.any() else float('nan'):.3f} "
                  f"max hybrid on fault={float(np.nanmax(_hyb[_flt])) if _flt.any() else float('nan'):.3f} "
                  f"tau={res['tau_hybrid']:.3f}")

    # --- Test scoring ---
    test_injected = {}
    (vu, lu, iu), (vk, lk, ik) = inject_pair(
        series["Udupi"][1][per_district["Udupi"]["lo_te"]:].copy(),
        series["Dakshina Kannada"][1][per_district["Dakshina Kannada"]["lo_te"]:].copy())
    test_injected["Udupi"] = (vu, lu, iu)
    test_injected["Dakshina Kannada"] = (vk, lk, ik)
    for d in ("Belagavi", "Bengaluru Urban", "Mysore"):
        info = per_district[d]
        vals, labels, inst = inject(info["values"][info["lo_te"]:].copy())
        test_injected[d] = (vals, labels, inst)

    test_scores = {}
    for d in DISTRICTS:
        info = per_district[d]
        vals, labels, inst = test_injected[d]
        rows_inj = rows_from_series(info["stamps"][info["lo_te"]:], vals)
        sc = score_points(d, vals, rows_inj, info["lo_te"])
        sc["labels"] = labels
        sc["instances"] = inst
        test_scores[d] = sc

    # Clean test scoring (same points, no faults). The Mamba MAE
    # comparison must use it: scoring Mamba on the injected series
    # penalises it for fault-contaminated windows (a point labelled
    # NORMAL can still have an injected fault inside its 24 h window),
    # while persistence/AR(1) are computed on clean values. Comparing
    # the two on different inputs is not a comparison.
    clean_test = {}
    for d in DISTRICTS:
        info = per_district[d]
        clean_test[d] = score_points(d, info["values"][info["lo_te"]:],
                                     info["rows"][info["lo_te"]:],
                                     info["lo_te"])

    # Flags per configuration.
    def hybrid_flag(sc, scorer):
        """E3 flag; falls back to the available engine if one is missing."""
        s_if = scorer.if_calibrator.transform(
            np.where(np.isfinite(sc["if_raw"]), sc["if_raw"], np.nan))
        s_m = scorer.mamba_calibrator.transform(
            np.where(np.isfinite(sc["mamba_raw"]), sc["mamba_raw"], np.nan))
        both = np.isfinite(s_if) & np.isfinite(s_m)
        hybrid = np.where(both, scorer.alpha * s_if + scorer.beta * s_m, np.nan)
        only_if = np.isfinite(s_if) & ~np.isfinite(s_m)
        only_m = np.isfinite(s_m) & ~np.isfinite(s_if)
        hybrid = np.where(only_if, s_if, hybrid)
        hybrid = np.where(only_m, s_m, hybrid)
        return hybrid > scorer.tau_hybrid, hybrid

    flags = {}
    for d in DISTRICTS:
        sc = test_scores[d]
        scorer = scorers[d]["scorer"]
        s_if = scorer.if_calibrator.transform(
            np.where(np.isfinite(sc["if_raw"]), sc["if_raw"], np.nan))
        s_m = scorer.mamba_calibrator.transform(
            np.where(np.isfinite(sc["mamba_raw"]), sc["mamba_raw"], np.nan))
        e1 = (s_if > 0.95) | (sc["q"] == 1)
        e2 = (s_m > 0.95) | (sc["q"] == 1)
        e3, _ = hybrid_flag(sc, scorer)
        e3 = e3 | (sc["q"] == 1)
        flags[d] = {"E1": e1, "E2": e2, "E3": e3,
                    "s_if": s_if, "s_m": s_m, "q": sc["q"]}
    # E4 needs neighbour flags first (prototype pair only).
    for d in DISTRICTS:
        sc = test_scores[d]
        scorer = scorers[d]["scorer"]
        e3 = flags[d]["E3"]
        e4_flag = e3.copy()
        e4_label = np.array(["NORMAL"] * len(e3), dtype=object)
        e4_reason = np.array([""] * len(e3), dtype=object)
        partner = {"Udupi": "Dakshina Kannada",
                   "Dakshina Kannada": "Udupi"}.get(d)
        for j in range(len(e3)):
            q = int(sc["q"][j])
            if q == 1:
                e4_flag[j] = True
                e4_label[j], e4_reason[j] = classify_three_way(1, True, 0, 0, 0)
                continue
            if not e3[j]:
                e4_label[j] = "NORMAL"
                continue
            n, delay = 0, None
            c = 0
            if partner:
                pf = flags[partner]["E3"]
                seg = list(pf[j:j + PROPAGATION_WINDOW_H + 1])
                # Same-sign check on the SIGNED temperature departure
                # (observed - predicted). If either side's Mamba score is
                # unavailable, fall back to assuming the signs agree.
                def _sign(v):
                    return 0 if not np.isfinite(v) else (1 if v > 0 else -1)
                s_sign = _sign(sc["mamba_err_signed"][j, 0])
                p_signed = test_scores[partner]["mamba_err_signed"][
                    j:j + PROPAGATION_WINDOW_H + 1, 0]
                n_signs = []
                for k in range(len(seg)):
                    ns = _sign(p_signed[k]) if k < len(p_signed) else 0
                    n_signs.append(ns if ns != 0 else s_sign if s_sign != 0 else 1)
                n, delay = nearby_anomaly(
                    DISTRICT_COORDS[d], DISTRICT_COORDS[partner],
                    seg, n_signs, s_sign if s_sign != 0 else 1)
                if delay is not None and delay >= 1:
                    c = 1
            e4_label[j], e4_reason[j] = classify_three_way(q, True, n, c, 0)
        flags[d]["E4"] = e4_flag
        flags[d]["E4_label"] = e4_label
        flags[d]["E4_reason"] = e4_reason

    # --- Aggregate metrics ---
    all_labels = np.concatenate([test_scores[d]["labels"] for d in DISTRICTS])
    for cfg in ("E1", "E2", "E3", "E4"):
        all_flag = np.concatenate([flags[d][cfg] for d in DISTRICTS])
        m = metrics_from_flags(all_flag, all_labels)
        # delays per instance (all districts)
        delays, missed, n_inst = [], 0, 0
        per_scen = {}
        for d in DISTRICTS:
            dd = detection_delays(flags[d][cfg], test_scores[d]["instances"])
            delays.append(dd)
            per_scen_d = per_scenario_recall(flags[d][cfg], test_scores[d]["instances"])
            for k, v in per_scen_d.items():
                per_scen.setdefault(k, []).append(v)
        det = [x["mean_delay_h"] for x in delays if x["mean_delay_h"] is not None]
        results["configs"][cfg] = {
            **m,
            "mean_delay_h": float(np.mean(det)) if det else None,
            "instances_detected": sum(x["detected"] for x in delays),
            "instances_missed": sum(x["missed"] for x in delays),
            "per_scenario_recall": {k: float(np.mean(v)) for k, v in per_scen.items()},
        }
        if verbose:
            print(f"{cfg}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"FPR={m['fpr']:.3f} delay={results['configs'][cfg]['mean_delay_h']}h")

    # --- E4 classification accuracy (on detected anomaly points) ---
    det_mask = np.concatenate([flags[d]["E4"] for d in DISTRICTS])
    det_labels = np.concatenate([test_scores[d]["labels"] for d in DISTRICTS])
    det_pred = np.concatenate([flags[d]["E4_label"] for d in DISTRICTS])
    correct, total = 0, 0
    for lab, pred in zip(det_labels[det_mask], det_pred[det_mask]):
        expected = "REGIONAL_WEATHER_EVENT" if lab == LABEL_EVENT else "LOCAL_SENSOR_FAULT"
        total += 1
        correct += int(pred == expected)
    sb_total, sb_blame = 0, 0
    for d in ("Udupi", "Dakshina Kannada"):
        sc = test_scores[d]
        for inst in sc["instances"]:
            if inst["type"] != "sea_breeze":
                continue
            for j in range(inst["start"], inst["end"]):
                if flags[d]["E4"][j]:
                    sb_total += 1
                    if flags[d]["E4_label"][j] == "LOCAL_SENSOR_FAULT":
                        sb_blame += 1
    results["classification_E4"] = {
        "accuracy_on_detected": correct / total if total else 0.0,
        "n_detected_anomaly_points": total,
        "sea_breeze_points_detected": sb_total,
        "sea_breeze_points_blamed_as_fault": sb_blame,
    }

    # --- Mamba vs baselines on the same normal test points ---
    # All three predictors are compared on CLEAN test data at the same
    # points (points labelled NORMAL in the injected set). Mamba's
    # errors come from the clean scoring pass, not the injected one.
    for d in DISTRICTS:
        info = per_district[d]
        sc = test_scores[d]
        normal = sc["labels"] == LABEL_NORMAL
        scored = np.isfinite(clean_test[d]["mamba_err"][:, 0]) & normal
        vals = test_injected[d][0]
        # persistence and AR(1) on the same points (x_i vs x_{i-1})
        idx = np.nonzero(scored)[0] + info["lo_te"]
        pers_err = np.abs(info["values"][idx] - info["values"][idx - 1])
        # AR(1) fit on train points
        train_vals = info["values"][:info["n_train"]]
        ar_err = np.zeros_like(pers_err)
        for ci in range(3):
            a_b = fit_ar1_predict(train_vals[:-1, ci], train_vals[1:, ci],
                                  info["values"][idx - 1, ci])
            ar_err[:, ci] = np.abs(info["values"][idx, ci] - a_b)
        mamba_err = clean_test[d]["mamba_err"][scored]
        out_d = {"sigma_e": info["mamba"].sigma_e.tolist(),
                 "alpha": per_district[d]["alpha_result"]["alpha"],
                 "tau_hybrid": per_district[d]["alpha_result"]["tau_hybrid"],
                 "n_test_points": int(scored.sum())}
        for ci, name in enumerate(["temperature_C", "humidity_pct", "pressure_hPa"]):
            out_d[name] = {
                "mamba_MAE": float(np.mean(mamba_err[:, ci])),
                "mamba_RMSE": float(np.sqrt(np.mean(mamba_err[:, ci] ** 2))),
                "persistence_MAE": float(np.mean(pers_err[:, ci])),
                "persistence_RMSE": float(np.sqrt(np.mean(pers_err[:, ci] ** 2))),
                "ar1_MAE": float(np.mean(ar_err[:, ci])),
                "ar1_RMSE": float(np.sqrt(np.mean(ar_err[:, ci] ** 2))),
            }
        results["districts"][d] = out_d

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"wrote {out_path} in {results['elapsed_seconds']}s")
    return results


def main():
    ap = argparse.ArgumentParser(description="E1-E4 evaluation harness")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="results/e1_e4_results.json")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    run(args.data_dir, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
