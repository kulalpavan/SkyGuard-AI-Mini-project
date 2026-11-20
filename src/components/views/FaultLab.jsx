import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useWeather } from '../../context/WeatherContext';
import { tacticalAudio } from '../../utils/audio';
import { apiClient } from '../../utils/apiClient';
import {
  getFaultBaseline,
  saveFaultBaseline,
  clearFaultBaseline,
  clearAllFaultBaselines,
  getFaultLabStation,
  saveFaultLabStation,
  classifyInjectionResult,
} from '../../utils/faultBaseline';

// Fault patterns mirror the backend's reevaluate() mappings, so the readings
// Fault Lab sends are the same faults the pipeline knows how to score.
// The inject endpoint only records an active_fault row — the faulted
// readings sent here are what the pipeline actually scores, which is why
// Fault Lab sends them (see DEMO_RUNBOOK.md §5b note).
const buildFaultFrames = (faultId, base) => {
  const at = (minutesAgo, patch) => ({ minutesAgo, ...patch });
  switch (faultId) {
    case 'SPIKE':
      return [at(0, { temp: base.temp + 8.5, hum: base.hum, pres: base.pres, battery: base.battery })];
    case 'DRIFT':
      return [0.4, 0.8, 1.2].map((off, i) => at(2 - i, { temp: base.temp + off, hum: base.hum, pres: base.pres, battery: base.battery }));
    case 'FLATLINE':
      // Five identical frames: the pipeline flags a stuck sensor once the
      // last four history rows are exactly identical.
      return [4, 3, 2, 1, 0].map(minutesAgo => at(minutesAgo, { temp: base.temp, hum: base.hum, pres: base.pres, battery: base.battery }));
    case 'POWER':
      return [at(0, { temp: base.temp + 8, hum: base.hum, pres: base.pres, battery: 10.8 })];
    case 'STORM':
      return [at(0, { temp: base.temp - 6, hum: 98, pres: base.pres - 8, battery: base.battery })];
    default:
      return [at(0, { temp: base.temp + 8.5, hum: base.hum, pres: base.pres, battery: base.battery })];
  }
};

const FAULT_TYPES = [
  {
    id: 'SPIKE',
    name: 'Sensor Spike',
    icon: 'fa-bolt',
    color: 'var(--neon-crimson)',
    badge: 'RATE_FAIL',
    badgeClass: 'badge-critical',
    desc: 'Instant +8.5°C thermal excursion to test rate-of-change QC.',
    param: '+8.5°C instant offset'
  },
  {
    id: 'DRIFT',
    name: 'Thermal Drift',
    icon: 'fa-chart-line',
    color: 'var(--neon-amber)',
    badge: 'DRIFT',
    badgeClass: 'badge-suspect',
    desc: 'Progressive +0.4°C per cycle bias to evaluate ML anomaly creep.',
    param: '+0.4°C / cycle creep'
  },
  {
    id: 'FLATLINE',
    name: 'Sensor Flatline',
    icon: 'fa-snowflake',
    color: 'var(--neon-cyan)',
    badge: 'FLATLINE',
    badgeClass: 'badge-normal',
    desc: 'Locks sensor output invariant to simulate frozen ADC buffer.',
    param: 'Variance = 0.00'
  },
  {
    id: 'POWER',
    name: 'Low Battery',
    icon: 'fa-car-battery',
    color: 'var(--neon-crimson)',
    badge: 'POWER_LOW',
    badgeClass: 'badge-critical',
    desc: 'Drops supply voltage to 10.8V to trigger hardware risk gates.',
    param: '10.8V supply fault'
  },
  {
    id: 'STORM',
    name: 'Regional Event',
    icon: 'fa-cloud-bolt',
    color: 'var(--neon-purple)',
    badge: 'GENUINE EXTREME',
    badgeClass: 'badge-extreme',
    desc: 'Synchronized rainfall (>30mm) & wind (>35km/h) for coherence testing.',
    param: '>30mm rain, >35km/h wind'
  }
];

export const FaultLab = () => {
  const { role, assignedStationId } = useAuth();
  const { injectFault, clearFaults, stations, setCurrentView } = useWeather();
  // The station the lab works on. Remembered in localStorage so it
  // survives navigating away and back: without this the panel remounts
  // on the first station (AWS-01) and the episode banner + Recover
  // button then act on AWS-01's (open, seeded) incident instead of the
  // station that was actually injected (finding B).
  const [selectedStation, setSelectedStation] = useState(() => {
    if (role === 'station_operator' && assignedStationId) return assignedStationId;
    return getFaultLabStation(window.localStorage) || (stations[0]?.id || '');
  });
  const [selectedFault, setSelectedFault] = useState('SPIKE');
  const [lastInjected, setLastInjected] = useState(null);
  const [sendState, setSendState] = useState(null);
  const [isInjecting, setIsInjecting] = useState(false);
  // The fault episode that is live for the selected station right now,
  // from the backend (open incident and/or active fault row) — NOT from
  // local component state, so the Recover button survives navigating
  // away from this view and back (finding B).
  const [activeEpisode, setActiveEpisode] = useState(null);

  // The station's live sensor values as the fault-frame base. Used only
  // to capture a fresh clean baseline; fault frames themselves are
  // always built from the stored baseline (finding A).
  //
  // A station with no processed observation yet has no fleet entry, so
  // reading the live sensors there silently falls back to 25.0 C / 60 %
  // / 1010 hPa — plausible for temperature but ~30 %RH away from this
  // station's real humidity, which the trained model then flags on
  // every recover frame (residual ~-16 %RH), so the incident never
  // auto-resolves. Use the latest stored telemetry row instead; the
  // hardcoded values are only a last resort for a station with no
  // history at all.
  const fetchLiveBase = async (stationId) => {
    const station = stations.find(s => s.id === stationId) || {};
    const live = {
      temp: station.sensors?.temperature?.value ?? null,
      hum: station.sensors?.humidity?.value ?? null,
      pres: station.sensors?.pressure?.value ?? null,
      battery: station.battery ?? 12.6,
    };
    if (live.temp != null && live.hum != null && live.pres != null) return live;
    try {
      const stats = await apiClient.getStationTelemetryStats(stationId);
      const latest = stats?.latest_observation;
      if (latest && latest.temperature != null) {
        return {
          temp: latest.temperature,
          hum: latest.humidity ?? live.hum ?? 60,
          pres: latest.pressure ?? live.pres ?? 1010,
          battery: latest.battery ?? live.battery ?? 12.6,
        };
      }
    } catch (e) { /* fall through to the hardcoded last resort */ }
    return {
      temp: live.temp ?? 25.0,
      hum: live.hum ?? 60,
      pres: live.pres ?? 1010,
      battery: live.battery ?? 12.6,
    };
  };

  const refreshEpisode = async (stationId) => {
    if (!stationId) return;
    try {
      const [incRes, faultRes] = await Promise.all([
        apiClient.getIncidents(stationId, 'open'),
        apiClient.getFaultStatus(stationId),
      ]);
      const open = (incRes?.incidents || [])[0] || null;
      const fault = faultRes?.has_fault ? faultRes.fault : null;
      // The episode carries its own station: the banner and Recover must
      // target the station the episode belongs to, never whichever
      // station the selector happens to show.
      setActiveEpisode(open || fault ? { incident: open, fault, stationId } : null);
    } catch (e) {
      // Leave the previous episode state: a failed refresh must not
      // hide a Recover button that was visible a moment ago.
    }
  };

  React.useEffect(() => {
    if (selectedStation) refreshEpisode(selectedStation);
  }, [selectedStation]);

  React.useEffect(() => {
    if (selectedStation) saveFaultLabStation(window.localStorage, selectedStation);
  }, [selectedStation]);

  const isOperator = role === 'station_operator' || role === 'STATION_OPERATOR';

  React.useEffect(() => {
    if (stations && stations.length > 0) {
      if (isOperator && assignedStationId) {
        setSelectedStation(assignedStationId);
      } else if (!selectedStation || !stations.some(s => s.id === selectedStation)) {
        setSelectedStation(stations[0].id);
      }
    }
  }, [stations, isOperator, assignedStationId, selectedStation]);


  if (!stations || stations.length === 0) {
    return (
      <div className="cyber-card" style={{ textAlign: 'center', padding: '60px 20px' }}>
        <i className="fa-solid fa-vial-virus" style={{ fontSize: '3rem', color: 'var(--neon-amber)', marginBottom: '16px', opacity: 0.8 }}></i>
        <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.2rem', color: 'var(--neon-amber)', fontWeight: 800 }}>
          NO REGISTERED WEATHER STATIONS
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', maxWidth: '500px', margin: '12px auto 20px auto' }}>
          Provision a weather station first via Station Credentials to inject synthetic stress faults.
        </p>
        <button className="cyber-btn btn-sm btn-primary" onClick={() => setCurrentView('credentials')}>
          <i className="fa-solid fa-key"></i> Provision Weather Station
        </button>
      </div>
    );
  }

  const handleInject = async () => {
    if (isInjecting) return;
    setIsInjecting(true);
    tacticalAudio.playAlarm();
    const startedAt = Date.now();
    setSendState({ phase: 'injecting' });
    try {
      await injectFault(selectedStation, selectedFault);
    } catch (e) {
      setSendState({ phase: 'error', message: `Could not record the fault: ${e.message}` });
      setIsInjecting(false);
      return;
    }
    setLastInjected({ station: selectedStation, fault: selectedFault, time: new Date().toLocaleTimeString() });

    // The inject endpoint only records an active_fault row; the pipeline
    // scores readings, so send the faulted readings through the telemetry
    // batch endpoint. That is what produces a visible model decision.
    // Finding A: build every fault frame from the CLEAN BASELINE captured
    // before the first inject of this episode — never from the live
    // sensors, which already carry the previous inject's offset (the
    // spiked frame becomes the live temperature, so a second SPIKE used
    // to stack +8.5 C onto +8.5 C, walking the station to 104.1 C).
    // The baseline is stored per station in localStorage and is only
    // cleared when the fault is recovered or reset.
    let base = getFaultBaseline(window.localStorage, selectedStation);
    if (!base) {
      base = await fetchLiveBase(selectedStation);
      saveFaultBaseline(window.localStorage, selectedStation, base);
    }
    const frames = buildFaultFrames(selectedFault, base);
    const rows = frames.map(f => ({
      timestamp: new Date(Date.now() - f.minutesAgo * 60000).toISOString(),
      temp: +Number(f.temp).toFixed(2),
      hum: +Number(f.hum).toFixed(2),
      pres: +Number(f.pres).toFixed(2),
      battery: f.battery,
      signal: -72,
      wind: 0,
      rain: 0,
    }));
    setSendState({ phase: 'sending', frames: rows.length });
    // Incidents already open before this inject: without them, two rapid
    // injects land inside the match grace and the first inject's
    // incident reads as "decided" a second time instead of "merged".
    let knownOpenIds = [];
    try {
      const before = await apiClient.getIncidents(selectedStation, 'open');
      knownOpenIds = (before?.incidents || []).map(i => i.id);
    } catch (e) { /* classify falls back to timestamps alone */ }
    let sent;
    try {
      sent = await apiClient.replayTelemetryBatch(selectedStation, rows);
    } catch (e) {
      setSendState({ phase: 'error', message: `Faulted readings were rejected: ${e.message}` });
      setIsInjecting(false);
      return;
    }
    const accepted = sent?.accepted ?? 0;
    setSendState({ phase: 'waiting', frames: rows.length, accepted });
    // Poll the incident queue for what the pipeline did with the
    // faulted readings. Finding C: the pipeline UPDATES an already-open
    // incident for the same station+variable instead of creating a new
    // one, so "no fresh incident" does not mean "scored normal" — check
    // whether an open incident was just updated (merged) before saying
    // the readings scored within normal bounds.
    for (let attempt = 0; attempt < 6; attempt += 1) {
      await new Promise(r => setTimeout(r, 2000));
      try {
        const res = await apiClient.getIncidents(selectedStation, 'open');
        const result = classifyInjectionResult(res?.incidents, startedAt, undefined, knownOpenIds);
        if (result.kind === 'decided') {
          setSendState({ phase: 'decided', incident: result.incident, frames: rows.length, accepted });
          tacticalAudio.playSuccess();
          setIsInjecting(false);
          await refreshEpisode(selectedStation);
          return;
        }
        if (result.kind === 'merged') {
          setSendState({ phase: 'merged', incident: result.incident, frames: rows.length, accepted });
          tacticalAudio.playSuccess();
          setIsInjecting(false);
          await refreshEpisode(selectedStation);
          return;
        }
      } catch (e) { /* keep polling */ }
    }
    setSendState({ phase: 'no-incident', frames: rows.length, accepted });
    setIsInjecting(false);
    await refreshEpisode(selectedStation);
  };

  // Recovery: three jittered normal frames — three consecutive normal
  // readings auto-resolve an open incident (runbook §5c). Finding B:
  // the frames are built from the CLEAN BASELINE captured at inject
  // time, not from the live sensors: after a spike the live temperature
  // IS the spiked temperature, so "live ± 0.2 C" frames stayed hot and
  // SUSPECT and the incident never resolved.
  //
  // Two more things the live run caught:
  // - The frames go to the EPISODE's station, not the selector's: after
  //   navigating away and back the selector used to reset to AWS-01 and
  //   the recover then "fixed" AWS-01 (whose seeded incident was open)
  //   while the AWS-02 incident stayed open.
  // - All three frames are timestamped now / now+1s / now+2s. The
  //   pipeline only counts a normal reading toward the auto-resolve
  //   streak when its timestamp is newer than the last processed
  //   reading, so the old now-2m / now-1m / now stamps left the first
  //   frame older than the spike and the streak stalled at 2 of 3.
  // - The fault clear is station-scoped: the old no-argument call ran
  //   the fleet reset, which DELETED every open incident in the fleet
  //   instead of letting this station's incident resolve on the frames.
  const handleRecover = async () => {
    const targetStation = activeEpisode?.stationId || selectedStation;
    const base = getFaultBaseline(window.localStorage, targetStation) || await fetchLiveBase(targetStation);
    const jitters = [[0.1, 0.5, 0.1], [-0.2, -1, -0.2], [0.2, 1, 0.15]];
    const rows = jitters.map(([dt, dh, dp], i) => ({
      timestamp: new Date(Date.now() + i * 1000).toISOString(),
      temp: +(base.temp + dt).toFixed(2),
      hum: +(base.hum + dh).toFixed(2),
      pres: +(base.pres + dp).toFixed(2),
      battery: base.battery,
      signal: -72,
      wind: 0,
      rain: 0,
    }));
    setSendState({ phase: 'recovering' });
    try {
      await apiClient.replayTelemetryBatch(targetStation, rows);
      await clearFaults(targetStation);
      // The episode is over: forget the baseline so the next inject
      // captures a fresh clean one from the (now normal) live sensors.
      clearFaultBaseline(window.localStorage, targetStation);
      setLastInjected(null);
      setSendState({ phase: 'recovered' });
      tacticalAudio.playSuccess();
      await refreshEpisode(targetStation);
    } catch (e) {
      setSendState({ phase: 'error', message: `Recovery frames were rejected: ${e.message}` });
    }
  };

  const handleReset = () => {
    tacticalAudio.playClick();
    clearFaults();
    clearAllFaultBaselines(window.localStorage);
    setLastInjected(null);
    setSendState(null);
    setActiveEpisode(null);
    tacticalAudio.playSuccess();
  };

  return (
    <div className="cyber-card">
      <div className="cyber-card-header">
        <div className="cyber-card-title">
          <i className="fa-solid fa-vial-virus text-amber"></i> SYNTHETIC FAULT INJECTION LAB
        </div>
        <button className="cyber-btn btn-sm btn-danger" onClick={handleReset}>
          <i className="fa-solid fa-rotate-left"></i> Reset Fleet To Normal
        </button>
      </div>
      <div className="cyber-card-body">
        {/* Step 1: Select Station */}
        <div style={{ background: 'rgba(10,15,29,0.8)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '12px 16px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              1. TARGET WEATHER STATION:
            </span>
            <select
              className="cyber-input"
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
              disabled={isOperator}
              style={{ background: '#050811', padding: '4px 8px', fontSize: '0.8rem', minWidth: '180px' }}
            >
              {(isOperator && assignedStationId ? stations.filter(s => s.id === assignedStationId) : stations).map(st => (
                <option key={st.id} value={st.id}>{st.id} — {st.name}</option>
              ))}
            </select>
          </div>

          {lastInjected && (
            <div style={{ fontSize: '0.72rem', color: 'var(--neon-crimson)', fontFamily: 'var(--font-mono)' }}>
              <i className="fa-solid fa-triangle-exclamation"></i> Active: <strong>{lastInjected.fault}</strong> on <strong>{lastInjected.station}</strong> at {lastInjected.time}
            </div>
          )}
        </div>

        {/* Active fault episode for the selected station, from the
            backend — shown whenever an incident is open or a fault is
            recorded, even after navigating away and back (finding B).
            The Recover button lives here as well as in the injection
            result, so it never disappears with local component state. */}
        {activeEpisode && (
          <div style={{ background: 'rgba(169,32,38,0.12)', border: '1px solid var(--neon-crimson)', borderRadius: '6px', padding: '10px 14px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ fontSize: '0.76rem', color: 'var(--text-primary)', lineHeight: 1.45 }}>
              <i className="fa-solid fa-triangle-exclamation" style={{ color: 'var(--neon-crimson)' }}></i>{' '}
              Active episode on <strong>{activeEpisode.stationId}</strong>
              {activeEpisode.incident ? (
                <> — open incident <span style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}>{activeEpisode.incident.id}</span></>
              ) : null}
              {activeEpisode.fault ? (
                <> — fault <strong>{activeEpisode.fault.fault_type}</strong> recorded</>
              ) : null}
              . Send normal frames to recover.
            </div>
            <button className="cyber-btn btn-sm" onClick={handleRecover} style={{ fontSize: '0.74rem' }}>
              <i className="fa-solid fa-heart-pulse"></i> Send 3 Normal Frames to Recover
            </button>
          </div>
        )}

        {/* Step 2: Choose Fault */}
        <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
          2. SELECT SYNTHETIC FAULT PATTERN:
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          {FAULT_TYPES.map(f => {
            const isSelected = selectedFault === f.id;
            return (
              <div
                key={f.id}
                onClick={() => { setSelectedFault(f.id); tacticalAudio.playClick(); }}
                style={{
                  background: isSelected ? 'rgba(0,240,255,0.06)' : 'rgba(10,15,29,0.7)',
                  border: `1px solid ${isSelected ? 'var(--neon-cyan)' : 'var(--border-subtle)'}`,
                  borderRadius: '6px',
                  padding: '14px',
                  cursor: 'pointer',
                  transition: 'border-color 0.15s, background 0.15s'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <i className={`fa-solid ${f.icon}`} style={{ color: f.color, fontSize: '1rem' }}></i>
                    <span style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.82rem', fontWeight: 700, color: isSelected ? 'var(--neon-cyan)' : 'var(--text-primary)' }}>
                      {f.name}
                    </span>
                  </div>
                  <span className={`cyber-badge ${f.badgeClass}`} style={{ fontSize: '0.62rem' }}>
                    {f.badge}
                  </span>
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: '8px', minHeight: '34px' }}>
                  {f.desc}
                </div>
                <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  Param: <strong style={{ color: 'var(--text-primary)' }}>{f.param}</strong>
                </div>
              </div>
            );
          })}
        </div>

        {/* Step 3: Action */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px', background: 'rgba(5,8,17,0.7)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '12px 16px' }}>
          <div>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.78rem', color: 'var(--neon-cyan)' }}>
              READY TO INJECT: {FAULT_TYPES.find(f => f.id === selectedFault)?.name} → {selectedStation}
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Records the fault, then sends faulted readings through the pipeline so the model decision actually appears below.
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="cyber-btn btn-sm btn-danger" onClick={handleInject} disabled={isInjecting} style={{ padding: '7px 16px', fontSize: '0.78rem' }}>
              <i className="fa-solid fa-play"></i> {isInjecting ? ' Injecting…' : ` Inject Fault on ${selectedStation}`}
            </button>
            <button className="cyber-btn btn-sm" onClick={() => setCurrentView('incidents')} style={{ fontSize: '0.78rem' }}>
              <i className="fa-solid fa-triangle-exclamation"></i> View Incident Queue
            </button>
          </div>
        </div>

        {/* Injection result: the model decision the fault produced */}
        {sendState && (
          <div style={{ marginTop: '16px', background: 'rgba(10,15,29,0.8)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '14px 16px' }}>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.74rem', color: 'var(--neon-cyan)', marginBottom: '8px' }}>
              <i className="fa-solid fa-flask-vial"></i> INJECTION RESULT
            </div>
            {sendState.phase === 'injecting' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Recording the fault on {selectedStation}…</div>
            )}
            {sendState.phase === 'sending' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Sending {sendState.frames} faulted reading{sendState.frames > 1 ? 's' : ''} through the pipeline…</div>
            )}
            {sendState.phase === 'waiting' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                {sendState.accepted} of {sendState.frames} faulted reading{sendState.frames > 1 ? 's' : ''} accepted — waiting for the pipeline decision…
              </div>
            )}
            {sendState.phase === 'decided' && sendState.incident && (
              <div>
                <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.9rem', fontWeight: 800, color: sendState.incident.quality_state === 'REGIONAL_EVENT' ? '#a78bfa' : 'var(--neon-crimson)' }}>
                  DECISION: {String(sendState.incident.quality_state || '').replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', lineHeight: 1.45, marginTop: '4px' }}>
                  {sendState.incident.reasoning?.headline || sendState.incident.explanation || 'An incident was raised.'}
                </div>
                {sendState.incident.reasoning?.suggested_action && (
                  <div style={{ fontSize: '0.76rem', color: 'var(--neon-cyan)', marginTop: '6px' }}>
                    <i className="fa-solid fa-arrow-right"></i> What to do next: {sendState.incident.reasoning.suggested_action} — {sendState.incident.reasoning.suggested_action_reason}
                  </div>
                )}
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap' }}>
                  <button className="cyber-btn btn-sm btn-primary" onClick={() => setCurrentView('incidents')}>
                    {/* Finding F: the id must render exactly as stored
                        (lower-case hex). .cyber-btn upper-cases its text,
                        so the id rides in a span that opts out — the
                        table renders it lower-case, and a direct GET with
                        the displayed id must not 404. */}
                    <i className="fa-solid fa-microscope"></i> Review Incident <span style={{ textTransform: 'none' }}>{sendState.incident.id}</span>
                  </button>
                  <button className="cyber-btn btn-sm" onClick={handleRecover}>
                    <i className="fa-solid fa-heart-pulse"></i> Send 3 Normal Frames to Recover
                  </button>
                </div>
              </div>
            )}
            {sendState.phase === 'merged' && sendState.incident && (
              <div>
                <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.9rem', fontWeight: 800, color: 'var(--neon-amber)' }}>
                  MERGED INTO OPEN INCIDENT
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', lineHeight: 1.45, marginTop: '4px' }}>
                  The faulted readings were scored by the pipeline and folded into the already-open incident{' '}
                  <span style={{ textTransform: 'none', fontFamily: 'var(--font-mono)' }}>{sendState.incident.id}</span>{' '}
                  instead of opening a duplicate — that incident was just updated.
                </div>
                {sendState.incident.reasoning?.headline && (
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45, marginTop: '4px' }}>
                    {sendState.incident.reasoning.headline}
                  </div>
                )}
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap' }}>
                  <button className="cyber-btn btn-sm btn-primary" onClick={() => setCurrentView('incidents')}>
                    <i className="fa-solid fa-microscope"></i> Review Incident <span style={{ textTransform: 'none' }}>{sendState.incident.id}</span>
                  </button>
                  <button className="cyber-btn btn-sm" onClick={handleRecover}>
                    <i className="fa-solid fa-heart-pulse"></i> Send 3 Normal Frames to Recover
                  </button>
                </div>
              </div>
            )}
            {sendState.phase === 'no-incident' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                No incident was raised — the {sendState.frames} faulted reading{sendState.frames > 1 ? 's' : ''} scored within this station&apos;s normal bounds.
                The fault may be too small for this station&apos;s thresholds; try <strong style={{ color: 'var(--text-primary)' }}>Sensor Spike</strong>, which is the reliable demo fault.
              </div>
            )}
            {sendState.phase === 'recovering' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Sending 3 normal frames to recover…</div>
            )}
            {sendState.phase === 'recovered' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--neon-green)' }}>
                Recovery frames sent. The incident auto-resolves after 3 consecutive normal readings — check the Incident Queue.
              </div>
            )}
            {sendState.phase === 'error' && (
              <div style={{ fontSize: '0.78rem', color: 'var(--neon-crimson)' }}>{sendState.message}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
