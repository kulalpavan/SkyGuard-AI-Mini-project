// Pure helpers for the Fault Injection Lab (findings A/B/C/F).
// No React imports, so this logic is unit-testable with plain
// `node --test` (see faultBaseline.test.js).
//
// (A) The clean baseline: the station's live sensor values BEFORE the
// first inject of the current fault episode. Every fault frame is built
// from that same baseline, so injecting SPIKE twice re-applies the same
// +8.5 C offset instead of stacking onto the already-spiked live
// temperature (which once walked a station to 104.1 C). The baseline
// lives in localStorage per station, so it survives re-renders and
// navigation, and is cleared only when the fault is recovered or reset.
// (B) Recovery frames are built from the same baseline, so they read
// NORMAL instead of re-sending the spiked live temperature. The
// station the lab is working on is remembered in localStorage too, so
// navigating away and back does not silently reset the panel to the
// first station (AWS-01) — the episode banner and its Recover button
// would otherwise act on the wrong station's (open, seeded) incident.
// (C) After injecting, the pipeline may UPDATE an already-open incident
// instead of creating a fresh one; classifyInjectionResult tells the
// two apart from the incident list.

export const FAULT_BASELINE_KEY = 'skyguard_fault_baseline_v1';
export const FAULT_STATION_KEY = 'skyguard_faultlab_station_v1';

// How far back an incident's created_at / updated_at may be and still
// count as caused by the inject that just ran (clock skew + pipeline lag).
export const INJECT_MATCH_GRACE_MS = 5000;

function readBaselines(storage) {
  try {
    const parsed = JSON.parse(storage.getItem(FAULT_BASELINE_KEY) || '{}');
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeBaselines(storage, map) {
  try {
    storage.setItem(FAULT_BASELINE_KEY, JSON.stringify(map));
  } catch {
    // Storage full or blocked (private mode): the baseline is best-effort.
    // Callers fall back to the live sensors, i.e. the old behaviour.
  }
}

/** The stored clean baseline for a station, or null if none was captured. */
export function getFaultBaseline(storage, stationId) {
  if (!storage || !stationId) return null;
  return readBaselines(storage)[stationId] || null;
}

/** Remember the clean baseline for a station (overwrites any previous one). */
export function saveFaultBaseline(storage, stationId, base) {
  if (!storage || !stationId || !base) return;
  const all = readBaselines(storage);
  all[stationId] = {
    temp: base.temp,
    hum: base.hum,
    pres: base.pres,
    battery: base.battery,
    capturedAt: new Date().toISOString(),
  };
  writeBaselines(storage, all);
}

/** Forget the baseline for one station (called on recovery / reset). */
export function clearFaultBaseline(storage, stationId) {
  if (!storage || !stationId) return;
  const all = readBaselines(storage);
  delete all[stationId];
  writeBaselines(storage, all);
}

/** Forget every stored baseline (Reset Fleet To Normal). */
export function clearAllFaultBaselines(storage) {
  if (!storage) return;
  try {
    storage.removeItem(FAULT_BASELINE_KEY);
  } catch {
    // Ignore: nothing stored is also a clean state.
  }
}

/** The station the lab was last working on, or null if never chosen. */
export function getFaultLabStation(storage) {
  if (!storage) return null;
  try {
    const value = storage.getItem(FAULT_STATION_KEY);
    return typeof value === 'string' && value.trim() ? value : null;
  } catch {
    return null;
  }
}

/** Remember the station the lab is working on (survives navigation). */
export function saveFaultLabStation(storage, stationId) {
  if (!storage || !stationId) return;
  try {
    storage.setItem(FAULT_STATION_KEY, stationId);
  } catch {
    // Storage full or blocked: the selection is best-effort.
  }
}

/**
 * Classify what the pipeline did with the faulted readings, from the
 * station's open incidents after the inject.
 *   'decided' — a NEW incident was created by this inject
 *               (created_at at/after the inject started).
 *   'merged'  — no new incident, but an already-open one was UPDATED by
 *               this inject (updated_at at/after the inject started).
 *   'none'    — no open incident was created or touched: the readings
 *               genuinely scored within the station's normal bounds.
 */
export function classifyInjectionResult(openIncidents, startedAtMs, graceMs = INJECT_MATCH_GRACE_MS, knownIds = []) {
  const list = Array.isArray(openIncidents) ? openIncidents : [];
  const known = new Set(Array.isArray(knownIds) ? knownIds : []);
  // A "fresh" incident only counts as decided if it was NOT already
  // open before this inject started: two rapid injects land inside the
  // grace window, so without this the first inject's incident would
  // read as "decided" again instead of "merged".
  const fresh = list.find(
    (i) => !known.has(i.id) && new Date(i.created_at || 0).getTime() >= startedAtMs - graceMs
  );
  if (fresh) return { kind: 'decided', incident: fresh };
  const merged = list.find(
    (i) => new Date(i.updated_at || 0).getTime() >= startedAtMs - graceMs
  );
  if (merged) return { kind: 'merged', incident: merged };
  return { kind: 'none', incident: null };
}
