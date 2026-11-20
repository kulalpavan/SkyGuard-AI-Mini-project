import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useWeather } from '../../context/WeatherContext';
import { tacticalAudio } from '../../utils/audio';
import { apiClient } from '../../utils/apiClient';

// QC rules that the backend actually enforces, and the one rule an admin
// can actually change from this screen.
//
// What the pipeline enforces on every observation
// (backend/app/services/observation_pipeline.py, ml/thermo_engine.py):
//   * Hard physics (same for every station, set in the backend, not here):
//     temperature -80..+65 C, pressure 500..1100 hPa, humidity 0..105%
//     (super-saturation above 100.5%), step change 6 C/h, 30 %RH/h,
//     10 hPa/h, flatline = 4 identical readings, QC residual 3 sigma
//     (severe 6 sigma).
//   * Station normal envelope (station_qc_config): a reading outside the
//     envelope sets the z_qc evidence flag. This is the editable rule:
//     PUT /admin/stations/{id}/qc, audited in config_audit.
//   * Fusion weights: backend defaults (FUSION_DEFAULT_WEIGHTS); fitted
//     coefficients from model calibration override them where present.
//     They are not hand-editable per station.
//
// The previous version of this screen had sliders for the hard limits and
// the fusion weights and a "Commit Hard Limits" button that only played a
// sound and showed an alert — nothing ever reached the backend. Those
// controls are gone: the constants are shown read-only, the envelope is
// the only thing this screen saves, and the save reports the backend's
// real answer.
const HARD_LIMITS = [
  { label: 'Temperature physical bounds', value: '-80 … +65 °C', note: 'thermo_engine' },
  { label: 'Pressure physical bounds', value: '500 … 1100 hPa', note: 'thermo_engine' },
  { label: 'Humidity physical bounds', value: '0 … 105 %', note: 'super-saturation > 100.5%' },
  { label: 'Max step change', value: '6 °C/h · 30 %RH/h · 10 hPa/h', note: 'per hour vs previous reading' },
  { label: 'Flatline', value: '4 identical readings', note: 'temp + humidity + pressure' },
  { label: 'QC residual', value: '3σ (severe 6σ)', note: 'vs station climatology' },
];

// Backend fusion defaults (observation_pipeline.FUSION_DEFAULT_WEIGHTS).
// Fitted coefficients override these where a calibration exists.
const FUSION_DEFAULTS = [
  { label: 'Intercept', value: '-3.0' },
  { label: 'QC evidence (z_qc)', value: '1.5' },
  { label: 'Physics violations (z_phys)', value: '3.0' },
  { label: 'Temporal / spike (z_temp)', value: '2.5' },
  { label: 'Isolation Forest (z_ml)', value: '3.0' },
  { label: 'Mamba temporal (z_mamba)', value: '2.5' },
  { label: 'Multi-evidence (z_multi)', value: '1.0' },
  { label: 'Sensor health (z_health)', value: '1.0' },
];

const ENVELOPE_FIELDS = [
  { key: 'temperature', label: 'Temperature', unit: '°C', step: '0.1' },
  { key: 'humidity', label: 'Humidity', unit: '%', step: '0.1' },
  { key: 'pressure', label: 'Pressure', unit: 'hPa', step: '0.1' },
];

export const QCRules = () => {
  const { role, assignedStationId } = useAuth();
  const { activeStationId } = useWeather();
  const [stationQC, setStationQC] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [draft, setDraft] = useState({});
  const [saveState, setSaveState] = useState(null); // { ok, message }
  const [isSaving, setIsSaving] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);

  const isAdmin = role === 'admin' || role === 'CENTRAL_ADMIN';
  const hasAccess = isAdmin || (role === 'station_operator' && assignedStationId === activeStationId);

  const fetchQC = async () => {
    if (!hasAccess) return;
    setIsLoading(true);
    try {
      const res = await apiClient.getStationQC(activeStationId);
      if (res.has_config) {
        setStationQC(res.config);
        setDraft({
          temperature_normal_min: res.config.temperature_normal_min ?? '',
          temperature_normal_max: res.config.temperature_normal_max ?? '',
          humidity_normal_min: res.config.humidity_normal_min ?? '',
          humidity_normal_max: res.config.humidity_normal_max ?? '',
          pressure_normal_min: res.config.pressure_normal_min ?? '',
          pressure_normal_max: res.config.pressure_normal_max ?? '',
        });
      } else {
        setStationQC(null);
        setDraft({});
      }
    } catch (err) {
      console.warn("Failed to fetch station QC:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => { fetchQC(); }, [activeStationId, hasAccess]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!hasAccess) {
    return (
      <div className="cyber-card" style={{ textAlign: 'center', padding: '60px 20px' }}>
        <i className="fa-solid fa-lock" style={{ fontSize: '3rem', color: 'var(--neon-crimson)', marginBottom: '16px', opacity: 0.8 }}></i>
        <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.2rem', color: 'var(--neon-crimson)', fontWeight: 800 }}>
          ACCESS DENIED — UNAUTHORIZED STATION
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', maxWidth: '500px', margin: '12px auto 20px auto' }}>
          Station Operators can only view the Quality Control Physics Matrix for their assigned station.
        </p>
      </div>
    );
  }

  const handleSave = async () => {
    if (!isAdmin) return;
    // Every filled pair must be a valid min < max; an empty pair leaves
    // that variable unset (the backend keeps its current value, if any).
    const payload = {};
    for (const { key } of ENVELOPE_FIELDS) {
      const lo = draft[`${key}_normal_min`];
      const hi = draft[`${key}_normal_max`];
      if (lo === '' && hi === '') continue;
      const loNum = lo === '' ? null : parseFloat(lo);
      const hiNum = hi === '' ? null : parseFloat(hi);
      if ((loNum !== null && Number.isNaN(loNum)) || (hiNum !== null && Number.isNaN(hiNum))) {
        setSaveState({ ok: false, message: `Enter numbers for ${key} (min and max).` });
        return;
      }
      if (loNum !== null && hiNum !== null && loNum >= hiNum) {
        setSaveState({ ok: false, message: `${key}: min must be below max.` });
        return;
      }
      if (loNum !== null) payload[`${key}_normal_min`] = loNum;
      if (hiNum !== null) payload[`${key}_normal_max`] = hiNum;
    }
    if (Object.keys(payload).length === 0) {
      setSaveState({ ok: false, message: 'Nothing to save — fill in at least one min/max pair.' });
      return;
    }
    setIsSaving(true);
    setSaveState(null);
    try {
      const res = await apiClient.updateStationQC(activeStationId, {
        ...payload,
        justification: 'Envelope edited from the QC Physics Matrix screen',
      });
      if (res?.success) {
        tacticalAudio.playSuccess();
        setSaveState({ ok: true, message: `Envelope saved for ${activeStationId} and written to the audit ledger. New observations outside it will be flagged.` });
        await fetchQC();
      } else {
        setSaveState({ ok: false, message: res?.message || 'The backend did not confirm the save.' });
      }
    } catch (err) {
      tacticalAudio.playAlarm();
      setSaveState({ ok: false, message: `Save failed: ${err.message}` });
    } finally {
      setIsSaving(false);
    }
  };

  // Derive the envelope from the station's own valid history
  // (1st-99th percentiles, min 720 usable readings). The backend
  // refuses with a plain message when there is not enough history
  // instead of inventing values.
  const handleCalibrate = async () => {
    if (!isAdmin) return;
    setIsCalibrating(true);
    setSaveState(null);
    try {
      const res = await apiClient.calibrateStationQC(activeStationId);
      if (res?.success) {
        tacticalAudio.playSuccess();
        setSaveState({
          ok: true,
          message: `Envelope calibrated from ${res.config?.calibration_record_count ?? res.valid_records} valid historical readings for ${activeStationId} (1st–99th percentiles, method ${res.config?.calibration_method}). New observations outside it will be flagged.`,
        });
        await fetchQC();
      } else {
        setSaveState({ ok: false, message: res?.message || 'The backend did not confirm the calibration.' });
      }
    } catch (err) {
      tacticalAudio.playAlarm();
      setSaveState({ ok: false, message: `Calibration failed: ${err.message}` });
    } finally {
      setIsCalibrating(false);
    }
  };

  return (
    <div className="cyber-card">
      <div className="cyber-card-header">
        <div className="cyber-card-title">
          <i className="fa-solid fa-sliders text-cyan"></i> QC PHYSICS MATRIX: {activeStationId}
        </div>
        {isAdmin && (
          <div style={{ display: 'flex', gap: '8px' }}>
            <button className="cyber-btn btn-sm" onClick={handleCalibrate} disabled={isCalibrating || isSaving}>
              <i className="fa-solid fa-chart-line"></i> {isCalibrating ? 'Calibrating…' : 'Calibrate from history'}
            </button>
            <button className="cyber-btn btn-sm btn-primary" onClick={handleSave} disabled={isSaving || isCalibrating}>
              <i className="fa-solid fa-floppy-disk"></i> {isSaving ? 'Saving…' : 'Save Envelope'}
            </button>
          </div>
        )}
      </div>
      <div className="cyber-card-body">
        <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Immutable hard physical limits execute first. Station-calibrated normal envelopes execute second.
          Only the envelope below is editable here — the hard limits and fusion weights are pipeline-wide
          backend values, shown for reference. “Calibrate from history” derives the envelope from this
          station’s own valid readings (1st–99th percentiles, needs at least 720); a hand edit after that
          overwrites it and is recorded as a manual override.
        </p>

        {saveState && (
          <div role="status" style={{
            padding: '10px 14px', marginBottom: '16px', borderRadius: '4px', fontSize: '0.78rem',
            border: `1px solid ${saveState.ok ? 'var(--neon-green)' : 'var(--neon-crimson)'}`,
            color: saveState.ok ? 'var(--neon-green)' : 'var(--neon-crimson)',
            background: 'rgba(10,15,29,0.85)',
          }}>
            {saveState.message}
          </div>
        )}

        <div className="sim-control-panel">
          {/* Box 1: Hard Physics Bounds — read-only backend constants */}
          <div className="sim-box" style={{ gridColumn: 'span 2' }}>
            <div className="sim-box-title">
              <span><i className="fa-solid fa-globe text-crimson"></i> IMMUTABLE HARD PHYSICS</span>
              <span className="cyber-badge badge-extreme">GLOBAL · READ-ONLY</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px 14px' }}>
              {HARD_LIMITS.map(item => (
                <div key={item.label} style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                  <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{item.label}</div>
                  <div style={{ color: 'var(--neon-cyan)', fontFamily: 'var(--font-mono)' }}>{item.value}</div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{item.note}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Box 2: Station Normal Envelope — the editable QC rule */}
          <div className="sim-box" style={{ gridColumn: 'span 2' }}>
            <div className="sim-box-title">
              <span><i className="fa-solid fa-chart-area text-cyan"></i> STATION NORMAL ENVELOPE</span>
              <span className="cyber-badge badge-normal">
                {!stationQC
                  ? 'NOT SET'
                  : stationQC.calibration_method === 'HISTORICAL_P01_P99'
                    ? 'CALIBRATED FROM HISTORY'
                    : 'MANUAL OVERRIDE'}
              </span>
            </div>

            {isLoading ? (
              <div style={{ textAlign: 'center', padding: '20px', color: 'var(--neon-cyan)' }}>
                <i className="fa-solid fa-circle-notch fa-spin"></i> Loading Station Configuration...
              </div>
            ) : (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '14px' }}>
                  {ENVELOPE_FIELDS.map(({ key, label, unit, step }) => (
                    <div key={key} className="stat-card" style={{ padding: '10px' }}>
                      <div className="stat-label">{label}</div>
                      {isAdmin ? (
                        <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginTop: '4px' }}>
                          <input
                            type="number" step={step} className="cyber-input"
                            aria-label={`${label} minimum`}
                            placeholder="min"
                            value={draft[`${key}_normal_min`] ?? ''}
                            onChange={(e) => setDraft(prev => ({ ...prev, [`${key}_normal_min`]: e.target.value }))}
                            style={{ width: '70px', padding: '4px 6px', fontSize: '0.8rem' }}
                          />
                          <span style={{ color: 'var(--text-muted)' }}>–</span>
                          <input
                            type="number" step={step} className="cyber-input"
                            aria-label={`${label} maximum`}
                            placeholder="max"
                            value={draft[`${key}_normal_max`] ?? ''}
                            onChange={(e) => setDraft(prev => ({ ...prev, [`${key}_normal_max`]: e.target.value }))}
                            style={{ width: '70px', padding: '4px 6px', fontSize: '0.8rem' }}
                          />
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{unit}</span>
                        </div>
                      ) : (
                        <div className="stat-value" style={{ fontSize: '1rem', color: 'var(--neon-cyan)' }}>
                          {stationQC && stationQC[`${key}_normal_min`] != null && stationQC[`${key}_normal_max`] != null
                            ? `${stationQC[`${key}_normal_min`]}${unit} – ${stationQC[`${key}_normal_max`]}${unit}`
                            : '—'}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {stationQC ? (
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(0, 255, 255, 0.1)', paddingTop: '10px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    <div><i className="fa-solid fa-database"></i> Records: <span style={{ color: 'var(--text-primary)' }}>{stationQC.calibration_record_count}</span></div>
                    <div><i className="fa-solid fa-code-branch"></i> Method: <span style={{ color: 'var(--text-primary)' }}>{stationQC.calibration_method}</span></div>
                    <div><i className="fa-solid fa-clock"></i> Calibrated: <span style={{ color: 'var(--text-primary)' }}>{new Date(stationQC.calibrated_at).toLocaleString()}</span></div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '12px', color: 'var(--neon-amber)', fontSize: '0.78rem' }}>
                    <i className="fa-solid fa-triangle-exclamation"></i> No envelope set for this station yet.
                    {isAdmin
                      ? ' Fill in the min/max pairs above and save — the pipeline will flag readings outside them.'
                      : ' The system falls back to the global hard physics limits above.'}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Box 3: Fusion weights — read-only backend defaults */}
          <div className="sim-box" style={{ gridColumn: 'span 2' }}>
            <div className="sim-box-title">
              <span><i className="fa-solid fa-atom text-purple"></i> EVIDENCE FUSION WEIGHTS</span>
              <span className="cyber-badge badge-extreme">BACKEND DEFAULTS · READ-ONLY</span>
            </div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0 0 10px 0' }}>
              Weights from the observation pipeline. Fitted coefficients from model calibration
              override them where present — they are set by calibration, not by hand.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px 14px' }}>
              {FUSION_DEFAULTS.map(item => (
                <div key={item.label} style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                  <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{item.label}</div>
                  <div style={{ color: 'var(--neon-purple)', fontFamily: 'var(--font-mono)' }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
