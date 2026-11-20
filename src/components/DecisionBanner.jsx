import React from 'react';

// Plain-language reading of the pipeline's model decision, shared by every
// surface where an operator must understand what the model decided and what
// to do about it (incident modal, station HUD, command centre, live monitor).
// The wording comes from the backend's own classification / reasoning — this
// component only translates codes into plain words; it never invents an
// action the backend did not produce.

const TONES = {
  normal: { color: 'var(--neon-green)', border: 'var(--neon-green)', bg: 'rgba(0,255,102,0.07)' },
  critical: { color: 'var(--neon-crimson)', border: 'var(--neon-crimson)', bg: 'rgba(255,0,85,0.08)' },
  suspect: { color: 'var(--neon-amber)', border: 'var(--neon-amber)', bg: 'rgba(255,170,0,0.07)' },
  extreme: { color: '#a78bfa', border: '#8b5cf6', bg: 'rgba(139,92,246,0.08)' },
  offline: { color: 'var(--text-muted)', border: 'var(--border-subtle)', bg: 'rgba(5,8,17,0.6)' },
};

export function decisionFor(classification, qualityState) {
  const cls = String(classification || '').toUpperCase();
  if (qualityState === 'INVALID' || cls === 'INVALID') {
    return {
      key: 'INVALID', tone: 'critical',
      title: 'INVALID READING',
      plain: 'This reading failed a hard data or physics check — it cannot be trusted, whatever the weather is doing.',
    };
  }
  switch (cls) {
    case 'REGIONAL_EVENT':
      return {
        key: 'REGIONAL_EVENT', tone: 'extreme',
        title: 'REGIONAL WEATHER EVENT',
        plain: 'Nearby stations show the same change — this is real weather, not a sensor fault. Do not blame the sensor.',
      };
    case 'LOCALIZED_ANOMALY':
      return {
        key: 'LOCALIZED_ANOMALY', tone: 'critical',
        title: 'LOCAL SENSOR FAULT',
        plain: 'Only this station shows the departure while neighbours do not — the sensor is the likely cause.',
      };
    case 'LOCALIZED_ANOMALY_UNCONFIRMED':
      return {
        key: 'LOCALIZED_ANOMALY_UNCONFIRMED', tone: 'suspect',
        title: 'LIKELY LOCAL SENSOR FAULT — UNCONFIRMED',
        plain: 'Only this station shows it, but neighbours partly agree — a local weather front is possible. Do not invalidate the reading yet.',
      };
    case 'NORMAL':
      return {
        key: 'NORMAL', tone: 'normal',
        title: 'NORMAL',
        plain: 'Readings are within this station\u2019s normal range. No action needed.',
      };
    default:
      return {
        key: cls || 'UNKNOWN', tone: 'offline',
        title: cls ? cls.replace(/_/g, ' ') : 'NO DECISION YET',
        plain: cls ? 'The pipeline has not classified this observation.' : 'No assessment recorded for this station yet.',
      };
  }
}

const fmtScore = (v, digits = 2) => (v == null || !Number.isFinite(Number(v)) ? null : Number(v).toFixed(digits));

function topMambaError(errors) {
  if (!Array.isArray(errors) || errors.length !== 3) return null;
  const labels = ['temperature', 'humidity', 'pressure'];
  const units = ['°C', '%RH', 'hPa'];
  let best = 0;
  for (let i = 1; i < 3; i += 1) {
    if (Math.abs(errors[i]) > Math.abs(errors[best])) best = i;
  }
  if (!Number.isFinite(Number(errors[best]))) return null;
  return `largest prediction error: ${labels[best]} ${Math.abs(Number(errors[best])).toFixed(2)}${units[best]}`;
}

export const DecisionBanner = ({
  assessment,
  mamba,
  hybrid,
  rootCause,
  reasoning,
  incident,
  onReviewIncident,
  ifScore,
  ifThreshold,
}) => {
  const classification = assessment?.classification;
  const qualityState = assessment?.quality_state;
  if (!classification && !qualityState) return null;

  const decision = decisionFor(classification, qualityState);
  const tone = TONES[decision.tone] || TONES.offline;

  const hybridScore = assessment?.hybrid_score ?? hybrid?.S_hybrid ?? null;
  const contrib = assessment?.hybrid_contributions || (hybrid ? { IF: hybrid.contrib_IF, Mamba: hybrid.contrib_Mamba } : null);
  const mambaScore = assessment?.mamba_score ?? mamba?.score ?? null;
  const mambaErrors = assessment?.mamba_errors || mamba?.errors || null;
  const mambaThreshold = mamba?.threshold ?? null;
  const fusionScore = assessment?.fusion?.score ?? null;
  const action = rootCause?.recommended_action || reasoning?.handling_steps?.[0] || null;
  const rootCauseLabel = rootCause?.root_cause || assessment?.root_cause || null;

  const scoreBits = [];
  if (hybridScore != null) {
    const cIf = contrib?.IF != null ? fmtScore(contrib.IF) : null;
    const cM = contrib?.Mamba != null ? fmtScore(contrib.Mamba) : null;
    scoreBits.push(`Hybrid ${fmtScore(hybridScore)}${cIf != null && cM != null ? ` (Isolation Forest ${cIf} + Mamba ${cM})` : ''}`);
  }
  if (ifScore != null) {
    scoreBits.push(`Isolation Forest ${fmtScore(ifScore)}${ifThreshold != null ? ` (threshold ${fmtScore(ifThreshold)})` : ''}`);
  }
  if (mambaScore != null) {
    const err = topMambaError(mambaErrors);
    scoreBits.push(`Mamba ${fmtScore(mambaScore)}${mambaThreshold != null ? ` (anomaly above ${fmtScore(mambaThreshold)})` : ''}${err ? ` — ${err}` : ''}`);
  }
  if (fusionScore != null) {
    scoreBits.push(`Evidence fusion ${fmtScore(fusionScore)}`);
  }

  return (
    <div
      data-testid="decision-banner"
      style={{
        background: tone.bg,
        border: `1px solid ${tone.border}`,
        borderLeft: `4px solid ${tone.border}`,
        borderRadius: '6px',
        padding: '14px 16px',
        marginBottom: '14px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
        <div style={{ minWidth: '220px', flex: 1 }}>
          <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-tactical)', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>
            MODEL DECISION
          </div>
          <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.05rem', fontWeight: 800, color: tone.color, marginTop: '2px' }}>
            {decision.title}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', lineHeight: 1.45, marginTop: '4px' }}>
            {decision.plain}
          </div>
          {rootCauseLabel && rootCauseLabel !== 'NOMINAL' && (
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
              Likely cause: <strong style={{ color: 'var(--text-primary)' }}>{rootCauseLabel.replace(/_/g, ' ')}</strong>
            </div>
          )}
        </div>
        {incident && onReviewIncident && (
          <button
            className="cyber-btn btn-sm btn-primary"
            onClick={onReviewIncident}
            style={{ fontSize: '0.74rem', whiteSpace: 'nowrap' }}
          >
            <i className="fa-solid fa-triangle-exclamation"></i> Review open incident {incident.id ? `(${incident.id})` : ''}
          </button>
        )}
      </div>

      {scoreBits.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 18px', marginTop: '10px', fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
          {scoreBits.map((bit, idx) => (
            <span key={idx}>{bit}</span>
          ))}
        </div>
      )}
      {scoreBits.length === 0 && (
        <div style={{ marginTop: '8px', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          No model scores recorded for this observation — the decision rests on physics and QC checks only.
        </div>
      )}

      {action && (
        <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-subtle)', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
          <i className="fa-solid fa-arrow-right" style={{ color: tone.color, marginTop: '2px' }}></i>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-primary)', lineHeight: 1.4 }}>
            <strong style={{ fontFamily: 'var(--font-tactical)', fontSize: '0.7rem', color: tone.color }}>WHAT TO DO NEXT: </strong>
            {action}
          </div>
        </div>
      )}
    </div>
  );
};

export default DecisionBanner;
