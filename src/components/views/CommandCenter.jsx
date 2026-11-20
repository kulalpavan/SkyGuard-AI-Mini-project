import React from 'react';
import { useWeather } from '../../context/WeatherContext';
import { tacticalAudio } from '../../utils/audio';
import { displayCode } from '../../utils/display';
import { apiClient } from '../../utils/apiClient';
import { decisionFor } from '../DecisionBanner';

const DRIFT_BADGE = {
  NONE: 'badge-normal',
  LOW: 'badge-offline',
  MODERATE: 'badge-suspect',
  HIGH: 'badge-critical',
  INSUFFICIENT_DATA: 'badge-offline',
  NO_MODEL: 'badge-offline',
};

export const CommandCenter = () => {
  const {
    stations = [],
    incidents = [],
    setCurrentView,
    setActiveStationId,
    activeStationModels = {},
    syncLiveOpenMeteoData
  } = useWeather();

  const safeStations = Array.isArray(stations) ? stations : [];
  const safeIncidents = Array.isArray(incidents) ? incidents : [];

  const normalCount = safeStations.filter(s => s?.status === 'NORMAL').length;
  const suspectCount = safeStations.filter(s => s?.status === 'SUSPECT' || s?.status === 'CRITICAL' || s?.status === 'LOCALIZED_ANOMALY').length;
  const extremeCount = safeStations.filter(s => s?.status === 'REGIONAL_EVENT' || s?.status === 'EXTREME').length;
  const openIncidents = safeIncidents.filter(i => i?.status === 'open').length;

  const handleViewModel = (stationId) => {
    setActiveStationId(stationId);
    setCurrentView('station-hud');
  };

  const getWmoFlagColor = (flag) => {
    if (flag === 0) return 'var(--neon-green)';
    if (flag === 1) return 'var(--neon-amber)';
    if (flag === 2) return 'var(--neon-red)';
    if (flag === 3) return 'var(--neon-cyan)';
    return 'var(--text-muted)';
  };

  const [filterStatus, setFilterStatus] = React.useState('ALL');
  const [filterRootCause, setFilterRootCause] = React.useState('ALL');

  // Model drift chip per station (§21.5): the worst severity across
  // the station's models, refreshed every 60 s. A failed fetch leaves
  // the chip hidden rather than showing a stale verdict.
  const [driftByStation, setDriftByStation] = React.useState({});
  React.useEffect(() => {
    if (safeStations.length === 0) return undefined;
    let cancelled = false;
    const rank = { NONE: 0, LOW: 1, MODERATE: 2, HIGH: 3 };
    const load = async () => {
      const next = {};
      for (const st of safeStations) {
        const id = st.id || st.station_id;
        if (!id) continue;
        try {
          const res = await apiClient.getLatestDrift(id);
          const rows = Object.values(res.latest || {});
          if (rows.length > 0) {
            const worst = rows.reduce((a, b) => ((rank[b.severity] ?? -1) > (rank[a.severity] ?? -1) ? b : a));
            next[id] = worst.severity;
          }
        } catch (e) {
          // Leave the chip hidden for this station; the drift panel
          // in Model Governance shows the full state.
        }
      }
      if (!cancelled) setDriftByStation(next);
    };
    load();
    const interval = setInterval(load, 60000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [safeStations.length]);

  const filteredStations = safeStations.filter(st => {
    const statusMatch = filterStatus === 'ALL' || st.status === filterStatus;
    const rcSeverity = st.root_cause_diagnosis?.severity || 'INFO';
    const rcMatch = filterRootCause === 'ALL' || rcSeverity === filterRootCause;
    return statusMatch && rcMatch;
  });

  return (
    <>
      {/* Fleet Command Operational Header */}
      <div className="cyber-card" style={{ marginBottom: '14px', background: 'rgba(10, 15, 29, 0.85)', padding: '12px 18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '6px', background: 'rgba(0, 240, 255, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid rgba(0, 240, 255, 0.3)' }}>
              <i className="fa-solid fa-gauge-high text-cyan"></i>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.92rem', fontWeight: 800, color: 'var(--neon-cyan)' }}>
                SKYGUARD AI — FLEET INTELLIGENCE & SENSOR HEALTH MATRIX
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                {stations.length === 0
                  ? 'No stations provisioned. Add a station to begin monitoring.'
                  : `Monitoring ${stations.length} Automatic Weather Stations | Thermodynamic Verification & Real-Time Imputation Active`}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              className="cyber-btn btn-sm"
              onClick={() => setCurrentView('fleet-map')}
              style={{ fontSize: '0.74rem' }}
            >
              <i className="fa-solid fa-map-location-dot"></i> Geospatial Radar
            </button>
            <button
              className="cyber-btn btn-sm btn-primary"
              onClick={() => setCurrentView('credentials')}
              style={{ fontSize: '0.74rem' }}
            >
              <i className="fa-solid fa-key"></i> Provision Station
            </button>
          </div>
        </div>
      </div>

      <div className="metrics-grid-4">
        <div className="cyber-card stat-card green-card">
          <div className="stat-header">
            <span className="stat-label">HEALTHY STATIONS</span>
            <div className="stat-icon"><i className="fa-solid fa-tower-broadcast text-green"></i></div>
          </div>
          <div className="stat-value text-green" id="stat-norm-val">
            {normalCount} <span className="stat-unit">/ {stations.length}</span>
          </div>
          <div className="stat-footer"><span>WMO Class 1 Nominal</span><span className="pulse-dot pulse-green"></span></div>
        </div>

        <div className="cyber-card stat-card amber-card">
          <div className="stat-header">
            <span className="stat-label">LOCALIZED ANOMALIES</span>
            <div className="stat-icon"><i className="fa-solid fa-triangle-exclamation text-amber"></i></div>
          </div>
          <div className="stat-value text-amber" id="stat-susp-val">
            {suspectCount} <span className="stat-unit">STATIONS</span>
          </div>
          <div className="stat-footer"><span>Sensor Drift & Spikes</span><span className="pulse-dot pulse-amber"></span></div>
        </div>

        <div className="cyber-card stat-card purple-card">
          <div className="stat-header">
            <span className="stat-label">REGIONAL WEATHER FRONTS</span>
            <div className="stat-icon"><i className="fa-solid fa-cloud-bolt text-purple"></i></div>
          </div>
          <div className="stat-value text-purple" id="stat-ext-val">
            {extremeCount} <span className="stat-unit">EVENTS</span>
          </div>
          <div className="stat-footer"><span>Peer-Corroborated</span><span className="pulse-dot pulse-green"></span></div>
        </div>

        <div className="cyber-card stat-card threat-card">
          <div className="stat-header">
            <span className="stat-label">OPEN INCIDENTS</span>
            <div className="stat-icon"><i className="fa-solid fa-bell text-crimson"></i></div>
          </div>
          <div className="stat-value text-crimson" id="stat-inc-val">
            {openIncidents} <span className="stat-unit">ACTIVE</span>
          </div>
          <div className="stat-footer"><span>Requires Triage</span><span className="pulse-dot pulse-crimson"></span></div>
        </div>
      </div>

      <div className="cyber-card" style={{ marginTop: '14px' }}>
        <div className="cyber-card-header">
          <div className="cyber-card-title"><i className="fa-solid fa-network-wired"></i> FLEET TELEMETRY, ROOT-CAUSE DIAGNOSIS & SENSOR HEALTH (SHI)</div>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Status:</span>
            <select className="cyber-input" style={{ padding: '4px 8px', width: 'auto', fontSize: '0.74rem' }} value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
              <option value="ALL">ALL</option>
              <option value="NORMAL">NORMAL</option>
              <option value="AWAITING_DATA">AWAITING DATA</option>
              <option value="INACTIVE">INACTIVE</option>
              <option value="LOCALIZED_ANOMALY">ANOMALOUS</option>
              <option value="SUSPECT">SUSPECT</option>
              <option value="REGIONAL_EVENT">REGIONAL EVENT</option>
            </select>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginLeft: '10px' }}>Severity:</span>
            <select className="cyber-input" style={{ padding: '4px 8px', width: 'auto', fontSize: '0.74rem' }} value={filterRootCause} onChange={e => setFilterRootCause(e.target.value)}>
              <option value="ALL">ALL</option>
              <option value="CRITICAL">CRITICAL</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="ALERT">ALERT</option>
              <option value="INFO">INFO</option>
            </select>
          </div>
        </div>
        <div className="cyber-card-body" style={{ padding: 0 }}>
          {filteredStations.length === 0 ? (
            <div style={{ padding: '36px 20px', textAlign: 'center', background: 'rgba(5,8,17,0.7)', borderRadius: '4px' }}>
              <i className="fa-solid fa-tower-broadcast" style={{ fontSize: '2.2rem', color: 'var(--neon-cyan)', marginBottom: '12px', opacity: 0.8 }}></i>
              <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1rem', color: 'var(--neon-cyan)', fontWeight: 800 }}>
                {safeStations.length === 0 ? 'NO STATIONS PROVISIONED' : 'NO STATIONS MATCH FILTER'}
              </div>
            </div>
          ) : (
            <div className="tactical-table-wrapper">
              <table className="tactical-table">
                <thead>
                  <tr>
                    <th>STATION ID</th>
                    <th>LOCATION & ELEVATION</th>
                    <th>ASSESSMENT & ROOT CAUSE</th>
                    <th>SENSOR HEALTH (SHI)</th>
                    <th>WMO FLAGS (T/H/P)</th>
                    <th>AIR TEMP (°C)</th>
                    <th>HUMIDITY (%)</th>
                    <th>PRESSURE (hPa)</th>
                    <th>ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStations.map(st => {
                    const hasObservation = Boolean(st.source_timestamp);
                    const badge = !hasObservation || st.status === 'INACTIVE' ? 'badge-offline' : st.status === 'NORMAL' ? 'badge-normal' : (st.status === 'REGIONAL_EVENT' ? 'badge-extreme' : (st.status === 'SUSPECT' ? 'badge-suspect' : 'badge-critical'));
                    const rootCause = hasObservation ? (st.root_cause_diagnosis?.root_cause || st.final_assessment?.root_cause || 'NOMINAL') : 'No assessed observation yet';
                    const shi = hasObservation ? st.sensor_health?.overall_health_score : null;
                    const decision = hasObservation ? decisionFor(st.final_assessment?.classification, st.final_assessment?.quality_state) : null;
                    const nextStep = hasObservation ? (st.root_cause_diagnosis?.recommended_action || null) : null;
                    const hybridScore = hasObservation ? st.final_assessment?.hybrid_score : null;
                    const openIncident = safeIncidents.find(i => String(i.station_id || '').toUpperCase() === String(st.id || st.station_id || '').toUpperCase() && String(i.status || '').toLowerCase() === 'open');
                    const tFlag = hasObservation ? st.sensors?.temperature?.wmo_flag : null;
                    const hFlag = hasObservation ? st.sensors?.humidity?.wmo_flag : null;
                    const pFlag = hasObservation ? st.sensors?.pressure?.wmo_flag : null;
                    const reading = sensor => hasObservation && sensor?.value != null ? `${sensor.value} ${sensor.unit || ''}` : '—';

                    return (
                      <tr key={st.id || st.station_id}>
                        <td style={{ fontWeight: 'bold', color: 'var(--neon-cyan)' }}>{st.id || st.station_id}</td>
                        <td>
                          <div style={{ fontWeight: 600 }}>{st.name || st.station_name}</div>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                            {st.region} • {Number(st.lat ?? st.latitude ?? 0).toFixed(2)}°N, {Number(st.lon ?? st.longitude ?? 0).toFixed(2)}°E ({st.elevation ?? 0}m)
                          </div>
                        </td>
                        <td>
                          <span className={`cyber-badge ${badge}`} style={{ fontSize: '0.68rem', marginRight: '6px' }}>{displayCode(st.status)}</span>
                          <div style={{ fontSize: '0.68rem', color: rootCause === 'NOMINAL' ? 'var(--text-muted)' : 'var(--neon-amber)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                            {displayCode(rootCause)}
                          </div>
                          {driftByStation[st.id || st.station_id] && (
                            <div style={{ marginTop: '4px' }} title="Model drift over the last 14 days vs training — see Model Governance for the numbers">
                              <span className={`cyber-badge ${DRIFT_BADGE[driftByStation[st.id || st.station_id]] || 'badge-offline'}`} style={{ fontSize: '0.65rem' }}>
                                DRIFT: {driftByStation[st.id || st.station_id]}
                              </span>
                            </div>
                          )}
                          {decision && st.status !== 'NORMAL' && (
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.35, maxWidth: '280px' }}>
                              <strong style={{ color: 'var(--text-primary)' }}>{decision.title}:</strong> {decision.plain}
                              {hybridScore != null && (
                                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--neon-cyan)' }}> Hybrid {Number(hybridScore).toFixed(2)}.</span>
                              )}
                              {nextStep && (
                                <div style={{ color: 'var(--neon-cyan)', marginTop: '2px' }}>
                                  <i className="fa-solid fa-arrow-right"></i> {nextStep}
                                </div>
                              )}
                            </div>
                          )}
                        </td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: shi == null ? 'var(--text-muted)' : shi >= 80 ? 'var(--neon-green)' : (shi >= 50 ? 'var(--neon-amber)' : 'var(--neon-red)') }}>
                              {shi == null ? '—' : `${shi}%`}
                            </span>
                            {shi != null && st.sensor_health?.predictive_maintenance?.degradation_projection_days != null && <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                              ({st.sensor_health.predictive_maintenance.degradation_projection_days}d projection)
                            </span>}
                          </div>
                        </td>
                        <td>
                          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', display: 'flex', gap: '4px' }}>
                            <span style={{ color: getWmoFlagColor(tFlag) }}>T:{tFlag ?? '—'}</span>
                            <span style={{ color: getWmoFlagColor(hFlag) }}>H:{hFlag ?? '—'}</span>
                            <span style={{ color: getWmoFlagColor(pFlag) }}>P:{pFlag ?? '—'}</span>
                          </div>
                        </td>
                        <td style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                          {reading(st.sensors?.temperature)}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{reading(st.sensors?.humidity)}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{reading(st.sensors?.pressure)}</td>
                        <td>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px', alignItems: 'flex-start' }}>
                            <button className="cyber-btn btn-sm" onClick={() => handleViewModel(st.id || st.station_id)}>
                              <i className="fa-solid fa-radar"></i> Inspect Station
                            </button>
                            {openIncident && (
                              <button className="cyber-btn btn-sm btn-danger" onClick={() => setCurrentView('incidents')} title={`Open incident ${openIncident.id}: ${openIncident.reasoning?.headline || openIncident.explanation || ''}`}>
                                <i className="fa-solid fa-triangle-exclamation"></i> Triage Incident
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
};
