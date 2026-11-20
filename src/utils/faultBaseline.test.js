// Unit tests for the Fault Lab pure helpers (findings A/B/C).
// Run: node --test src/utils/faultBaseline.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  FAULT_BASELINE_KEY,
  FAULT_STATION_KEY,
  getFaultBaseline,
  saveFaultBaseline,
  clearFaultBaseline,
  clearAllFaultBaselines,
  getFaultLabStation,
  saveFaultLabStation,
  classifyInjectionResult,
} from './faultBaseline.js';

function memoryStorage() {
  const map = {};
  return {
    getItem: (k) => (k in map ? map[k] : null),
    setItem: (k, v) => { map[k] = String(v); },
    removeItem: (k) => { delete map[k]; },
    _raw: map,
  };
}

const BASE = { temp: 25.0, hum: 60, pres: 1010, battery: 12.6 };

describe('fault baseline (finding A/B)', () => {
  it('returns null when nothing was captured', () => {
    assert.equal(getFaultBaseline(memoryStorage(), 'AWS-02'), null);
  });

  it('round-trips a baseline per station, stations isolated', () => {
    const storage = memoryStorage();
    saveFaultBaseline(storage, 'AWS-02', BASE);
    assert.deepEqual(
      { ...getFaultBaseline(storage, 'AWS-02'), capturedAt: undefined },
      { ...BASE, capturedAt: undefined },
    );
    assert.equal(getFaultBaseline(storage, 'AWS-01'), null);
    // A second inject reuses the stored baseline instead of the
    // (now spiked) live values: the stored values do not move.
    saveFaultBaseline(storage, 'AWS-02', { ...BASE, temp: 33.5 });
    assert.equal(getFaultBaseline(storage, 'AWS-02').temp, 33.5);
  });

  it('clearing one station leaves the other alone', () => {
    const storage = memoryStorage();
    saveFaultBaseline(storage, 'AWS-01', BASE);
    saveFaultBaseline(storage, 'AWS-02', BASE);
    clearFaultBaseline(storage, 'AWS-02');
    assert.equal(getFaultBaseline(storage, 'AWS-02'), null);
    assert.notEqual(getFaultBaseline(storage, 'AWS-01'), null);
  });

  it('clear-all removes every stored baseline', () => {
    const storage = memoryStorage();
    saveFaultBaseline(storage, 'AWS-01', BASE);
    saveFaultBaseline(storage, 'AWS-02', BASE);
    clearAllFaultBaselines(storage);
    assert.equal(storage.getItem(FAULT_BASELINE_KEY), null);
    assert.equal(getFaultBaseline(storage, 'AWS-01'), null);
  });

  it('tolerates corrupt stored JSON', () => {
    const storage = memoryStorage();
    storage.setItem(FAULT_BASELINE_KEY, '{not json');
    assert.equal(getFaultBaseline(storage, 'AWS-02'), null);
    // And a later save still works over the corrupt entry.
    saveFaultBaseline(storage, 'AWS-02', BASE);
    assert.equal(getFaultBaseline(storage, 'AWS-02').temp, 25.0);
  });
});

describe('classifyInjectionResult (finding C)', () => {
  const startedAt = Date.parse('2026-11-20T16:44:00.000Z');
  const freshIncident = {
    id: 'INC-abc123',
    created_at: '2026-11-20T16:44:09.000Z',
    updated_at: '2026-11-20T16:44:09.000Z',
  };
  const oldIncidentUntouched = {
    id: 'INC-old999',
    created_at: '2026-11-20T15:00:00.000Z',
    updated_at: '2026-11-20T15:30:00.000Z',
  };
  const oldIncidentMerged = {
    id: 'INC-merged1',
    created_at: '2026-11-20T15:00:00.000Z',
    updated_at: '2026-11-20T16:44:09.000Z',
  };

  it('a freshly created incident is decided', () => {
    const res = classifyInjectionResult([oldIncidentUntouched, freshIncident], startedAt);
    assert.equal(res.kind, 'decided');
    assert.equal(res.incident.id, 'INC-abc123');
  });

  it('an old incident updated by this inject is merged, not decided', () => {
    const res = classifyInjectionResult([oldIncidentMerged], startedAt);
    assert.equal(res.kind, 'merged');
    assert.equal(res.incident.id, 'INC-merged1');
  });

  it('an old untouched incident is none — the readings scored normal', () => {
    const res = classifyInjectionResult([oldIncidentUntouched], startedAt);
    assert.equal(res.kind, 'none');
    assert.equal(res.incident, null);
  });

  it('no open incidents at all is none', () => {
    assert.equal(classifyInjectionResult([], startedAt).kind, 'none');
    assert.equal(classifyInjectionResult(null, startedAt).kind, 'none');
  });

  it('decided wins when both a fresh and a merged incident exist', () => {
    const res = classifyInjectionResult([oldIncidentMerged, freshIncident], startedAt);
    assert.equal(res.kind, 'decided');
  });

  it('an incident already open before this inject is not decided again', () => {
    // Two rapid injects land inside the grace window: the first
    // inject's incident was created ~4 s before the second started,
    // so timestamps alone would call it "decided" a second time.
    const res = classifyInjectionResult([freshIncident], startedAt, undefined, ['INC-abc123']);
    assert.equal(res.kind, 'merged');
    assert.equal(res.incident.id, 'INC-abc123');
  });

  it('a known incident that was not touched is none', () => {
    const res = classifyInjectionResult([oldIncidentUntouched], startedAt, undefined, ['INC-old1']);
    assert.equal(res.kind, 'none');
  });
});

describe('fault-lab station memory (finding B)', () => {
  it('returns null when no station was remembered', () => {
    assert.equal(getFaultLabStation(memoryStorage()), null);
    assert.equal(getFaultLabStation(null), null);
  });

  it('round-trips the last-selected station and overwrites it', () => {
    const storage = memoryStorage();
    saveFaultLabStation(storage, 'AWS-02');
    assert.equal(getFaultLabStation(storage), 'AWS-02');
    assert.equal(storage._raw[FAULT_STATION_KEY], 'AWS-02');
    saveFaultLabStation(storage, 'AWS-01');
    assert.equal(getFaultLabStation(storage), 'AWS-01');
  });

  it('ignores blank or missing values', () => {
    const storage = memoryStorage();
    saveFaultLabStation(storage, '');
    saveFaultLabStation(storage, null);
    assert.equal(getFaultLabStation(storage), null);
    storage.setItem(FAULT_STATION_KEY, '   ');
    assert.equal(getFaultLabStation(storage), null);
  });

  it('tolerates a throwing storage (private mode)', () => {
    const broken = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
    assert.equal(getFaultLabStation(broken), null);
    assert.doesNotThrow(() => saveFaultLabStation(broken, 'AWS-02'));
  });
});
