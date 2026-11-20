import React, { useState, useEffect } from 'react';

// Live, phase-by-phase view of a station training job: the 8 Isolation
// Forest phases followed by the 3 Mamba phases. The backend records one
// stage at a time (see training_service.execute_training_job and GET
// /stations/{id}/training-jobs/{job_id}/status); this component only
// renders that state — done / in progress / pending — it never invents progress.
// Shared by StationUpload (where training is triggered) and ModelGovernance
// (where the job is watched), so the two views can never disagree.

export const TRAINING_STEPS = [
  { label: 'Data Ingested', desc: 'Historical telemetry loaded from the station' },
  { label: 'Data Validated', desc: 'Physical plausibility & sanity checks' },
  { label: 'Data Preprocessed', desc: 'Climatology fitted, contaminated fit rows scrubbed' },
  { label: 'Features Generated', desc: '12-feature causal station vectors' },
  { label: 'Training Isolation Forest', desc: 'Fitting 100 isolation trees from scratch' },
  { label: 'Model Evaluation', desc: 'Temporal holdout scoring & promotion gates' },
  { label: 'Model Registered', desc: 'Candidate model saved to the registry' },
  { label: 'Model Activated', desc: 'Ready for live real-time scoring' },
  { label: 'Training Mamba', desc: 'Fitting the selective state-space temporal predictor on station history' },
  { label: 'Calibrating Mamba', desc: 'Error scale, ECDFs and hybrid weights from validation' },
  { label: 'Mamba Registered', desc: 'Temporal model saved to the registry for live hybrid scoring' },
];

export function parseCompletedStages(value) {
  if (Array.isArray(value)) return value;
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }
  return [];
}

function formatDuration(ms) {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return '—';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) return `${seconds}s`;
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

export const TrainingProgress = ({ job, stationId, startedAt }) => {
  const [now, setNow] = useState(() => Date.now());

  const status = job?.status;
  const isRunning = status === 'RUNNING';
  const isTerminal = status === 'COMPLETED' || status === 'FAILED';

  useEffect(() => {
    if (!isRunning) return undefined;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, [isRunning]);

  if (!job) return null;

  const completed = parseCompletedStages(job.completed_stages);
  const completedSet = new Set(completed);
  const progress = job.progress != null
    ? job.progress
    : Math.round((completed.length / TRAINING_STEPS.length) * 100);

  const startMs = job.started_at ? Date.parse(job.started_at)
    : (startedAt ? (typeof startedAt === 'number' ? startedAt : Date.parse(startedAt)) : null);
  const endMs = isTerminal && job.completed_at ? Date.parse(job.completed_at) : now;
  const elapsedMs = startMs != null ? endMs - startMs : null;

  // The step in progress is the first one the backend has not recorded as
  // done yet. The backend records a stage when its work finishes, so while
  // RUNNING exactly one step is "in progress" and the rest are pending.
  const activeIndex = TRAINING_STEPS.findIndex(s => !completedSet.has(s.label));
  const failedIndex = status === 'FAILED' ? activeIndex : -1;

  const tone = status === 'COMPLETED'
    ? { color: 'var(--neon-green)', border: 'var(--neon-green)', bg: 'rgba(0,255,102,0.06)' }
    : status === 'FAILED'
    ? { color: 'var(--neon-crimson)', border: 'var(--neon-crimson)', bg: 'rgba(255,0,85,0.08)' }
    : { color: 'var(--neon-cyan)', border: 'var(--neon-cyan)', bg: 'rgba(5,8,17,0.9)' };

  // Per-engine outcome, derived from the completed stages alone — no new
  // backend field needed. A job can fail on one engine while the other
  // finished (Udupi: IF activation fails the gate, Mamba still registers).
  const ifActivated = completedSet.has('Model Activated');
  const ifRegistered = completedSet.has('Model Registered');
  const mambaRegistered = completedSet.has('Mamba Registered');

  const statusLabel = status === 'COMPLETED'
    ? (mambaRegistered ? 'MODELS TRAINED' : 'MODEL ACTIVATED')
    : status === 'FAILED' ? 'TRAINING FAILED'
    : status === 'RUNNING' ? 'TRAINING IN PROGRESS'
    : String(status || 'UNKNOWN');

  return (
    <div
      data-testid="training-progress"
      data-job-status={status}
      style={{
        background: tone.bg,
        border: `1px solid ${tone.border}`,
        borderRadius: '6px',
        padding: '16px',
        marginBottom: '16px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.9rem', fontWeight: 800, color: tone.color }}>
            <i className={`fa-solid ${isRunning ? 'fa-circle-notch fa-spin' : status === 'COMPLETED' ? 'fa-shield-check' : status === 'FAILED' ? 'fa-triangle-exclamation' : 'fa-brain'}`}></i>{' '}
            TRAINING PIPELINE {stationId ? `— ${stationId}` : ''} {job.model_version ? `(${job.model_version})` : ''}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '3px' }}>
            {statusLabel}
            {job.job_id != null ? ` · Job #${job.job_id}` : ''}
            {elapsedMs != null ? ` · Elapsed ${formatDuration(elapsedMs)}` : ''}
            {isTerminal && startMs != null && endMs != null ? ` · Total ${formatDuration(endMs - startMs)}` : ''}
          </div>
        </div>
        <span className="cyber-badge" style={{ color: tone.color, borderColor: tone.border, fontSize: '0.68rem' }}>
          {completed.length}/{TRAINING_STEPS.length} STEPS · {progress}%
        </span>
      </div>

      {/* Progress bar */}
      <div style={{ height: '6px', background: 'rgba(5,8,17,0.9)', border: '1px solid var(--border-subtle)', borderRadius: '3px', marginTop: '12px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${Math.min(100, Math.max(0, progress))}%`,
          background: tone.color,
          transition: 'width 0.4s ease',
        }} />
      </div>

      {/* Phase checklist */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '8px', marginTop: '14px' }}>
        {TRAINING_STEPS.map((step, idx) => {
          const done = completedSet.has(step.label);
          const active = isRunning && idx === activeIndex;
          const failed = idx === failedIndex;
          const stateColor = done ? 'var(--neon-green)' : failed ? 'var(--neon-crimson)' : active ? 'var(--neon-cyan)' : 'var(--text-muted)';
          const stateLabel = done ? 'DONE' : failed ? 'FAILED HERE' : active ? 'IN PROGRESS' : 'PENDING';
          return (
            <div
              key={step.label}
              data-testid={`training-step-${idx}`}
              data-step-state={done ? 'done' : failed ? 'failed' : active ? 'in-progress' : 'pending'}
              style={{
                display: 'flex', gap: '10px', alignItems: 'flex-start',
                background: 'rgba(5,8,17,0.7)',
                border: `1px solid ${active || failed ? stateColor : 'var(--border-subtle)'}`,
                borderRadius: '4px', padding: '9px 11px',
              }}
            >
              <span style={{ color: stateColor, fontSize: '0.95rem', marginTop: '1px', minWidth: '18px', textAlign: 'center' }}>
                {done
                  ? <i className="fa-solid fa-circle-check"></i>
                  : failed
                  ? <i className="fa-solid fa-circle-xmark"></i>
                  : active
                  ? <i className="fa-solid fa-circle-notch fa-spin"></i>
                  : <i className="fa-regular fa-circle"></i>}
              </span>
              <span style={{ minWidth: 0 }}>
                <span style={{ display: 'block', fontFamily: 'var(--font-tactical)', fontSize: '0.76rem', fontWeight: 700, color: done || active || failed ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {idx + 1}. {step.label}
                </span>
                <span style={{ display: 'block', fontSize: '0.68rem', color: 'var(--text-secondary)', marginTop: '1px' }}>
                  {step.desc}
                </span>
                <span style={{ display: 'block', fontSize: '0.64rem', fontFamily: 'var(--font-mono)', color: stateColor, marginTop: '3px', letterSpacing: '0.5px' }}>
                  {stateLabel}
                </span>
              </span>
            </div>
          );
        })}
      </div>

      {/* Final outcome */}
      {status === 'COMPLETED' && (
        <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)', fontSize: '0.78rem', color: 'var(--neon-green)', fontFamily: 'var(--font-mono)' }}>
          <i className="fa-solid fa-circle-check"></i> {mambaRegistered
            ? 'Isolation Forest activated and Mamba registered — ready for live hybrid scoring'
            : 'Model activated and ready for live scoring'}
          {elapsedMs != null ? ` — finished in ${formatDuration(elapsedMs)}` : ''}.
        </div>
      )}
      {status === 'FAILED' && (
        <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)', fontSize: '0.78rem', color: 'var(--neon-crimson)', lineHeight: 1.5 }}>
          <i className="fa-solid fa-triangle-exclamation"></i>{' '}
          <strong>Training stopped{activeIndex >= 0 ? ` at step ${activeIndex + 1} of ${TRAINING_STEPS.length} (${TRAINING_STEPS[activeIndex]?.label})` : ''}.</strong>
          <div style={{ fontFamily: 'var(--font-mono)', marginTop: '6px', fontSize: '0.72rem', lineHeight: 1.6 }}>
            <div>Isolation Forest: {ifActivated ? 'activated' : ifRegistered ? 'trained, activation failed' : 'did not finish'}</div>
            <div>Mamba: {mambaRegistered ? 'registered' : 'did not finish'}</div>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', marginTop: '4px', wordBreak: 'break-word' }}>
            {job.error_message || 'No reason recorded by the backend.'}
          </div>
          <div style={{ color: 'var(--text-secondary)', marginTop: '6px', fontSize: '0.72rem' }}>
            {ifActivated ? 'The Isolation Forest is active. ' : 'The previous active model (if any) is still in place. '}
            {mambaRegistered ? 'The Mamba model was registered and is scoring. ' : 'No Mamba model was registered. '}
            Fix the cause and train again.
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingProgress;
