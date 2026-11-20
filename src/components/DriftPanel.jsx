import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { apiClient } from '../utils/apiClient';
import { tacticalAudio } from '../utils/audio';

// Model drift panel (spec §21.5): current severity badge in plain
// words, per-metric table with thresholds, 90-day sparklines of PSI
// and residual mean, and a "Queue Retraining" button that creates a
// candidate — never trains inline. Drift observes; it does not retune
// detection, so nothing here changes scoring or gates.

const SEVERITY_BADGE = {
  NONE: 'badge-normal',
  LOW: 'badge-offline',
  MODERATE: 'badge-suspect',
  HIGH: 'badge-critical',
  INSUFFICIENT_DATA: 'badge-offline',
  NO_MODEL: 'badge-offline',
};

const SEVERITY_WORDS = {
  NONE: 'Stable — the model still matches recent weather.',
  LOW: 'Slight shift — worth watching, nothing to do yet.',
  MODERATE: 'Drifting — the model no longer matches recent weather well; queue a retraining.',
  HIGH: 'Needs retraining — the model has moved well away from recent weather; retrain before trusting its verdicts.',
  INSUFFICIENT_DATA: 'Not enough recent data to judge drift yet.',
  NO_MODEL: 'No model registered yet, so there is nothing to compare against.',
};

const SEVERITY_COLOR = {
  NONE: 'var(--neon-green)',
  LOW: 'var(--neon-cyan)',
  MODERATE: 'var(--neon-amber)',
  HIGH: 'var(--neon-red)',
  INSUFFICIENT_DATA: 'var(--text-muted)',
  NO_MODEL: 'var(--text-muted)',
};

const fmt = (v, digits = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(digits));
const fmtPct = (v) => (v === null || v === undefined ? '—' : `${(Number(v) * 100).toFixed(1)}%`);

function Sparkline({ values, color, height = 44, label }) {
  const pts = (values || []).filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  if (pts.length < 2) {
    return <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '8px 0' }}>Not enough history yet — compute drift again tomorrow.</div>;
  }
  const w = 260;
  const h = height;
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const span = max - min || 1;
  const step = w / (pts.length - 1);
  const path = pts.map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(h - 4 - ((v - min) / span) * (h - 8)).toFixed(1)}`).join(' ');
  return (
    <div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: 'block' }}>
        <path d={path} fill="none" stroke={color} strokeWidth="1.5" />
        <circle cx={((pts.length - 1) * step).toFixed(1)} cy={(h - 4 - ((pts[pts.length - 1] - min) / span) * (h - 8)).toFixed(1)} r="2.5" fill={color} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        <span>{label} · {pts.length} checks</span>
        <span>latest {fmt(pts[pts.length - 1])} · range {fmt(min)}–{fmt(max)}</span>
      </div>
    </div>
  );
}

function MetricTable({ entries }) {
  if (!entries || entries.length === 0) return null;
  return (
    <div className="tactical-table-wrapper" style={{ marginTop: '10px' }}>
      <table className="tactical-table">
        <thead>
          <tr>
            <th>METRIC</th>
            <th>VALUE</th>
            <th>THRESHOLD</th>
            <th>STATE</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr key={e.key}>
              <td style={{ fontSize: '0.72rem' }}>{e.label}</td>
              <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: SEVERITY_COLOR[e.severity] || 'var(--text-primary)' }}>
                {e.key === 'feature_availability' ? fmtPct(e.value) : fmt(e.value)}
              </td>
              <td style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{e.threshold}</td>
              <td>
                <span className={`cyber-badge ${SEVERITY_BADGE[e.severity] || 'badge-offline'}`} style={{ fontSize: '0.65rem' }}>
                  {e.severity}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelDriftBlock({ row, history, isAdmin, onQueue, queueing, queuedVersions }) {
  const modelType = row.model_type || 'IsolationForest';
  const severity = row.severity || 'NO_MODEL';
  const metrics = row.metrics || {};
  const entries = row.metric_entries || [];
  const detail = row.detail || {};
  const perParam = detail.per_parameter || {};
  const modelHistory = (history || []).filter((h) => (h.model_type || 'IsolationForest') === modelType);
  const psiSeries = modelHistory.map((h) => {
    const vals = [h.psi_temperature, h.psi_humidity, h.psi_pressure].filter((v) => v !== null && v !== undefined);
    if (vals.length > 0) return Math.max(...vals);
    const m = h.metrics || {};
    return m.psi_mamba_score ?? m.psi ?? null;
  });
  const meanSeries = modelHistory.map((h) => {
    const m = h.metrics || {};
    return h.residual_mean_shift ?? m.mamba_error_ratio ?? null;
  });
  const isQueued = queuedVersions.has(`${modelType}:${row.model_version}`);
  const queuedHere = (row.triggered_candidate && severity !== 'NONE' && severity !== 'LOW');
  return (
    <div style={{ background: 'rgba(10,15,29,0.6)', border: '1px solid var(--border-subtle)', padding: '14px', borderRadius: '4px', marginTop: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.82rem', fontWeight: 800, color: 'var(--neon-cyan)' }}>
          <i className="fa-solid fa-wave-square"></i> {modelType === 'Mamba' ? 'MAMBA TEMPORAL MODEL' : 'ISOLATION FOREST'} — {row.model_version}
          <span className="cyber-badge badge-offline" style={{ marginLeft: '8px', fontSize: '0.65rem' }}>{row.model_status || ''}</span>
        </div>
        <span className={`cyber-badge ${SEVERITY_BADGE[severity]}`} style={{ fontSize: '0.72rem' }}>{severity}</span>
      </div>

      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '8px 0 0 0', lineHeight: 1.5 }}>
        {row.plain_words || SEVERITY_WORDS[severity] || SEVERITY_WORDS.NO_MODEL}
        {row.is_partial && (
          <span style={{ color: 'var(--neon-amber)' }}> Partial reference — PSI/KS skipped{row.partial_reason ? `: ${row.partial_reason}` : ''}.</span>
        )}
        {row.reason && severity === 'INSUFFICIENT_DATA' && (
          <span style={{ color: 'var(--text-muted)' }}> {row.reason}</span>
        )}
      </p>

      {severity === 'HIGH' && (
        <div style={{ marginTop: '10px', padding: '10px 12px', border: '1px solid var(--neon-red)', background: 'rgba(255,60,60,0.08)', borderRadius: '4px', fontSize: '0.76rem', color: 'var(--neon-red)' }}>
          <i className="fa-solid fa-triangle-exclamation"></i> A DATA_QUALITY incident was opened for this drift and a retraining candidate was queued. The model keeps scoring — drift never switches detection off — but treat its verdicts with caution until it is retrained.
        </div>
      )}
      {severity === 'MODERATE' && (
        <div style={{ marginTop: '10px', padding: '10px 12px', border: '1px solid var(--neon-amber)', background: 'rgba(255,170,0,0.07)', borderRadius: '4px', fontSize: '0.76rem', color: 'var(--neon-amber)' }}>
          <i className="fa-solid fa-circle-exclamation"></i> A retraining candidate was queued. Nothing retrains itself — an admin starts the training job when ready.
        </div>
      )}

      <MetricTable entries={entries} />

      {modelType === 'IsolationForest' && Object.keys(perParam).length > 0 && (
        <div style={{ marginTop: '10px', fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', lineHeight: 1.7 }}>
          Per parameter — residual mean / PSI:{' '}
          {Object.entries(perParam).map(([p, v]) => (
            <span key={p} style={{ marginRight: '12px' }}>
              <strong style={{ color: 'var(--text-secondary)' }}>{p.slice(0, 4).toUpperCase()}</strong> {fmt(v.residual_mean, 2)} / {fmt(v.psi)}
            </span>
          ))}
          <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>
            Anomaly rate {fmtPct(metrics.anomaly_rate)} vs {fmtPct(metrics.anomaly_rate_reference)} in training
            {metrics.score_mean !== undefined && metrics.score_mean !== null ? ` · mean score ${fmt(metrics.score_mean)}` : ''}
          </div>
        </div>
      )}
      {modelType === 'Mamba' && Object.keys(perParam).length > 0 && (
        <div style={{ marginTop: '10px', fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', lineHeight: 1.7 }}>
          Per parameter — mean |error| / sigma_e:{' '}
          {Object.entries(perParam).map(([p, v]) => (
            <span key={p} style={{ marginRight: '12px' }}>
              <strong style={{ color: 'var(--text-secondary)' }}>{p.slice(0, 4).toUpperCase()}</strong> {fmt(v.mean_abs_normalised_error)}× ({fmt(v.mean_abs_error, 2)} / σ {fmt(v.sigma_e, 2)})
            </span>
          ))}
          <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>
            Anomaly rate {fmtPct(metrics.anomaly_rate)} vs {fmtPct(metrics.anomaly_rate_reference)} in training · {metrics.windows_scored ?? detail.windows_scored ?? '—'} windows scored
          </div>
        </div>
      )}
      {modelType === 'Mamba' && metrics.mamba_error_ratio !== undefined && metrics.mamba_error_ratio !== null && (
        <div style={{ marginTop: '6px', fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', lineHeight: 1.7 }}>
          Mean prediction error {fmt(metrics.mean_abs_normalised_error)}× sigma_e (calibrated expectation 0.80)
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px', marginTop: '12px' }}>
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>PSI — DISTRIBUTION SHIFT (90 DAYS)</div>
          <Sparkline values={psiSeries} color="var(--neon-cyan)" label="PSI" />
        </div>
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>{modelType === 'Mamba' ? 'MAMBA ERROR RATIO (90 DAYS)' : 'RESIDUAL MEAN SHIFT (90 DAYS)'}</div>
          <Sparkline values={meanSeries} color={severity === 'HIGH' ? 'var(--neon-red)' : severity === 'MODERATE' ? 'var(--neon-amber)' : 'var(--neon-green)'} label={modelType === 'Mamba' ? 'error ratio' : '|mean z|'} />
        </div>
      </div>

      {isAdmin && severity !== 'NO_MODEL' && severity !== 'INSUFFICIENT_DATA' && (
        <div style={{ marginTop: '12px', display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            className="cyber-btn btn-sm btn-primary"
            disabled={queueing || isQueued || queuedHere}
            onClick={() => onQueue(row)}
          >
            <i className="fa-solid fa-rotate"></i> {queueing ? 'Queueing…' : isQueued || queuedHere ? 'Retraining Queued' : 'Queue Retraining'}
          </button>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
            Queues a candidate only — training starts manually and activation still needs the promotion gate.
          </span>
        </div>
      )}
      <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: '10px', fontFamily: 'var(--font-mono)' }}>
        Checked {row.computed_at ? new Date(row.computed_at).toLocaleString() : '—'} · {row.window_days || 14}-day window vs training reference
      </div>
    </div>
  );
}

export const DriftPanel = ({ stationId }) => {
  const { role } = useAuth();
  const isAdmin = role === 'admin' || role === 'CENTRAL_ADMIN';
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [statusMsg, setStatusMsg] = useState('');
  const [computing, setComputing] = useState(false);
  const [queueing, setQueueing] = useState(false);

  const fetchDrift = useCallback(async (stId) => {
    if (!stId) return;
    setIsLoading(true);
    setLoadError('');
    try {
      const res = await apiClient.getStationDrift(stId, 90);
      if (res.success) setData(res);
    } catch (e) {
      setLoadError(e.message || 'Could not load drift status.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDrift(stationId);
  }, [stationId, fetchDrift]);

  const queuedVersions = new Set(
    (data?.retraining_candidates || [])
      .filter((c) => c.status === 'QUEUED')
      .map((c) => {
        const type = String(c.model_version || '').includes('mamba') ? 'Mamba' : 'IsolationForest';
        return `${type}:${c.model_version}`;
      }),
  );

  const handleCompute = async () => {
    setComputing(true);
    setStatusMsg('');
    try {
      const res = await apiClient.computeStationDrift(stationId);
      if (res.success) {
        setStatusMsg(`Drift checked: ${res.overall_severity} — ${res.plain_words || ''}`);
        tacticalAudio.playSuccess();
        fetchDrift(stationId);
      }
    } catch (e) {
      setStatusMsg(`Drift check failed: ${e.message}`);
      tacticalAudio.playAlarm();
    } finally {
      setComputing(false);
    }
  };

  const handleQueue = async (row) => {
    setQueueing(true);
    setStatusMsg('');
    try {
      const res = await apiClient.queueDriftRetraining(stationId, row.model_version, row.severity === 'HIGH' ? 'HIGH' : 'LOW');
      if (res.success) {
        setStatusMsg(res.created
          ? `Retraining queued for ${row.model_version}. Training still starts manually.`
          : `A retraining candidate for ${row.model_version} is already queued.`);
        tacticalAudio.playSuccess();
        fetchDrift(stationId);
      }
    } catch (e) {
      setStatusMsg(`Queue failed: ${e.message}`);
      tacticalAudio.playAlarm();
    } finally {
      setQueueing(false);
    }
  };

  if (!stationId) return null;
  const latest = data?.latest || {};
  const rows = Object.values(latest);
  const worst = rows.length > 0
    ? rows.reduce((a, b) => {
      const rank = { NONE: 0, LOW: 1, MODERATE: 2, HIGH: 3, INSUFFICIENT_DATA: -1, NO_MODEL: -1 };
      return (rank[b.severity] ?? -1) > (rank[a.severity] ?? -1) ? b : a;
    })
    : null;

  return (
    <div className="cyber-card" style={{ marginTop: '16px' }}>
      <div className="cyber-card-header">
        <div className="cyber-card-title">
          <i className="fa-solid fa-wave-square text-cyan"></i> MODEL DRIFT — {stationId} (LAST 14 DAYS VS TRAINING)
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {worst && (
            <span className={`cyber-badge ${SEVERITY_BADGE[worst.severity] || 'badge-offline'}`} style={{ fontSize: '0.72rem' }}>
              {worst.severity}
            </span>
          )}
          {isAdmin && (
            <button className="cyber-btn btn-sm" onClick={handleCompute} disabled={computing}>
              <i className="fa-solid fa-arrows-rotate"></i> {computing ? 'Checking…' : 'Check Drift Now'}
            </button>
          )}
        </div>
      </div>
      <div className="cyber-card-body">
        {isLoading && <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Loading drift status…</div>}
        {loadError && (
          <div style={{ fontSize: '0.78rem', color: 'var(--neon-amber)' }}>
            <i className="fa-solid fa-triangle-exclamation"></i> {loadError}
          </div>
        )}
        {statusMsg && (
          <div style={{ fontSize: '0.76rem', color: 'var(--neon-cyan)', background: 'rgba(0,255,255,0.06)', border: '1px solid var(--border-subtle)', padding: '8px 12px', borderRadius: '4px', marginBottom: '10px' }}>
            {statusMsg}
          </div>
        )}
        {!isLoading && !loadError && rows.length === 0 && (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            No drift check has run for this station yet.
            {isAdmin
              ? ' Use “Check Drift Now” to compute it from the station’s own recent telemetry — it takes a few seconds.'
              : ' An admin can run the first check from here.'}
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '6px' }}>
              Drift is measured in residual space (how far recent readings sit from what the model learned), not raw values — so a change of season alone does not trigger it. A sustained regional weather pattern can still move residuals; drift means “investigate”, not “the sensor is broken”.
            </div>
          </div>
        )}
        {rows.map((row) => (
          <ModelDriftBlock
            key={row.model_type}
            row={row}
            history={data?.history || []}
            isAdmin={isAdmin}
            onQueue={handleQueue}
            queueing={queueing}
            queuedVersions={queuedVersions}
          />
        ))}
        {rows.length > 0 && (
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '12px', lineHeight: 1.6 }}>
            Thresholds: PSI &lt; 0.10 stable · 0.10–0.25 moderate · &gt; 0.25 significant; residual mean shift &lt; 0.3 ok · 0.3–0.7 warn · &gt; 0.7 alert; anomaly rate &gt; 3× training contamination alerts. Drift never retrains, activates or deactivates a model by itself.
          </div>
        )}
      </div>
    </div>
  );
};
