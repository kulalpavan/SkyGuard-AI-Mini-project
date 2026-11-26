import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useWeather } from '../../context/WeatherContext';
import { tacticalAudio } from '../../utils/audio';
import { apiClient } from '../../utils/apiClient';

export const Sidebar = ({ collapsed, setCollapsed }) => {
  const { role, logout, assignedStationId } = useAuth();
  const { currentView, setCurrentView, incidents, offlineBuffer, activeStationId, stations, setActiveStationId, activeStationModels } = useWeather();
  const [audioActive, setAudioActive] = useState(true);
  const [modelRegistry, setModelRegistry] = useState({});

  const toggleSidebar = (e) => {
    e?.stopPropagation();
    setCollapsed(prev => !prev);
    tacticalAudio.playClick();
  };

  const toggleAudio = () => {
    const active = tacticalAudio.toggle();
    setAudioActive(active);
  };

  const handleLogout = () => {
    tacticalAudio.playAlarm();
    logout();
  };

  const handleNavClick = (view) => {
    setCurrentView(view);
  };

  const openIncidentsCount = incidents.filter(i => i.status === 'open').length;

  // Stations needing attention: anything the model did not call NORMAL.
  const attentionStations = (stations || []).filter(s =>
    s?.status && !['NORMAL', 'AWAITING_DATA', 'INACTIVE'].includes(String(s.status).toUpperCase())
  );

  // Model registry status per station (ACTIVE / CANDIDATE), from the real
  // model registry API. Refreshed when the station roster changes or a
  // training event updates the active model.
  // Finding D: operators may only see their own station — fetching
  // /stations/{id}/models for every station in the roster 403s for the
  // stations they cannot access ("Station identity violation") and
  // litters the console on every login. Scope the fetch (and the rows
  // below) to the stations the logged-in user can actually see; admin
  // behaviour (all stations) is unchanged.
  const isOperator = role !== 'admin';
  const visibleStations = isOperator
    ? (stations || []).filter(s => s.id === assignedStationId)
    : (stations || []);
  const stationIdsKey = visibleStations.map(s => s.id).sort().join(',');
  useEffect(() => {
    let cancelled = false;
    const ids = visibleStations.map(s => s.id).filter(Boolean);
    if (ids.length === 0) return;
    Promise.all(ids.map(async (id) => {
      try {
        const res = await apiClient.getStationModels(id);
        return [id, Array.isArray(res?.models) ? res.models : []];
      } catch (e) {
        return [id, null];
      }
    })).then(entries => {
      if (cancelled) return;
      const next = {};
      for (const [id, models] of entries) {
        if (models == null) { next[id] = null; continue; }
        const pick = (test) => {
          const found = models.filter(test);
          if (found.some(m => String(m.status || '').toUpperCase() === 'ACTIVE')) return 'ACTIVE';
          if (found.some(m => String(m.status || '').toUpperCase() === 'CANDIDATE')) return 'CANDIDATE';
          return found.length > 0 ? String(found[0].status || 'REGISTERED').toUpperCase() : null;
        };
        next[id] = {
          if: pick(m => /isolation|_IF_/i.test(`${m.model_type || ''} ${m.model_id || ''}`)),
          mamba: pick(m => /mamba/i.test(`${m.model_type || ''} ${m.model_id || ''}`)),
        };
      }
      setModelRegistry(next);
    });
    return () => { cancelled = true; };
  }, [stationIdsKey, activeStationModels]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleAttentionClick = (stationId) => {
    tacticalAudio.playClick();
    if (role === 'admin') {
      setCurrentView('incidents');
    } else {
      if (setActiveStationId) setActiveStationId(stationId);
      setCurrentView('station-hud');
    }
  };

  return (
    <aside className={`cyber-sidebar ${collapsed ? 'collapsed' : ''}`} id="cyber-sidebar">
      {/* Sidebar Header & Brand */}
      <div className="sidebar-header" onClick={collapsed ? toggleSidebar : undefined}>
        <a className="brand-logo" href="#" onClick={(e) => { e.preventDefault(); if (collapsed) toggleSidebar(); }}>
          <div className="brand-icon">
            <i className="fa-solid fa-satellite-dish"></i>
          </div>
          <div className="brand-title-group">
            <span className="brand-text">SKYGUARD</span>
            <span className="brand-sub">TACTICAL MET-AI HUD</span>
          </div>
        </a>
        <button
          className="cyber-btn btn-sm"
          id="sidebar-toggle-btn"
          title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}
          onClick={toggleSidebar}
        >
          <i className={`fa-solid fa-angles-${collapsed ? 'right' : 'left'}`}></i>
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="sidebar-nav" id="sidebar-nav">
        {role === 'admin' ? (
          <>
            <div className="nav-section-title">Central Command</div>
            <a
              className={`nav-item ${currentView === 'command-center' ? 'active' : ''}`}
              data-view="command-center"
              onClick={() => handleNavClick('command-center')}
              title="Command Center"
            >
              <i className="fa-solid fa-gauge-high"></i>
              <span>Command Center</span>
            </a>
            <a
              className={`nav-item ${currentView === 'fleet-map' ? 'active' : ''}`}
              data-view="fleet-map"
              onClick={() => handleNavClick('fleet-map')}
              title="Fleet Radar Map"
            >
              <i className="fa-solid fa-map-location-dot"></i>
              <span>Fleet Radar Map</span>
            </a>
            <a
              className={`nav-item ${currentView === 'incidents' ? 'active' : ''}`}
              data-view="incidents"
              onClick={() => handleNavClick('incidents')}
              title="Incident Triage"
            >
              <i className="fa-solid fa-triangle-exclamation"></i>
              <span>Incident Triage</span>
              {openIncidentsCount > 0 && <span className="nav-badge">{openIncidentsCount}</span>}
            </a>

            <div className="nav-section-title">Administration & Calibration</div>
            <a
              className={`nav-item ${currentView === 'credentials' ? 'active' : ''}`}
              data-view="credentials"
              onClick={() => handleNavClick('credentials')}
              title="Station Credentials"
            >
              <i className="fa-solid fa-key"></i>
              <span>Station Credentials</span>
            </a>
            <a
              className={`nav-item ${currentView === 'qc-rules' ? 'active' : ''}`}
              data-view="qc-rules"
              onClick={() => handleNavClick('qc-rules')}
              title="QC Physics Matrix"
            >
              <i className="fa-solid fa-sliders"></i>
              <span>QC Physics Matrix</span>
            </a>
            <a
              className={`nav-item ${currentView === 'fault-lab' ? 'active' : ''}`}
              data-view="fault-lab"
              onClick={() => handleNavClick('fault-lab')}
              title="Fault Injection Lab"
            >
              <i className="fa-solid fa-vial-virus"></i>
              <span>Fault Injection Lab</span>
            </a>
            <a
              className={`nav-item ${currentView === 'model-governance' ? 'active' : ''}`}
              data-view="model-governance"
              onClick={() => handleNavClick('model-governance')}
              title="Model Governance"
            >
              <i className="fa-solid fa-brain"></i>
              <span>Model Governance</span>
            </a>
            <a
              className={`nav-item ${currentView === 'station-upload' ? 'active' : ''}`}
              data-view="station-upload"
              onClick={() => handleNavClick('station-upload')}
              title="Station Training Studio"
            >
              <i className="fa-solid fa-microchip"></i>
              <span>Station Training Studio</span>
            </a>
            <a
              className={`nav-item ${currentView === 'export' ? 'active' : ''}`}
              data-view="export"
              onClick={() => handleNavClick('export')}
              title="Quality Data Export"
            >
              <i className="fa-solid fa-file-export"></i>
              <span>Quality Data Export</span>
            </a>
          </>
        ) : (
          <>
            <div className="nav-section-title">{activeStationId} Operator Terminal</div>
            <a
              className={`nav-item ${currentView === 'station-hud' ? 'active' : ''}`}
              data-view="station-hud"
              onClick={() => handleNavClick('station-hud')}
              title="Live Cockpit HUD"
            >
              <i className="fa-solid fa-desktop"></i>
              <span>Live Cockpit HUD</span>
            </a>
            <a
              className={`nav-item ${currentView === 'station-upload' ? 'active' : ''}`}
              data-view="station-upload"
              onClick={() => handleNavClick('station-upload')}
              title="Historical Data Ingest & ML Training"
            >
              <i className="fa-solid fa-file-arrow-up"></i>
              <span>Historical Ingest & ML</span>
            </a>
            <a
              className={`nav-item ${currentView === 'station-diagnostics' ? 'active' : ''}`}
              data-view="station-diagnostics"
              onClick={() => handleNavClick('station-diagnostics')}
              title="Hardware Telemetry"
            >
              <i className="fa-solid fa-microchip"></i>
              <span>Hardware Telemetry</span>
            </a>
            <a
              className={`nav-item ${currentView === 'station-checklist' ? 'active' : ''}`}
              data-view="station-checklist"
              onClick={() => handleNavClick('station-checklist')}
              title="Maintenance Checklist"
            >
              <i className="fa-solid fa-list-check"></i>
              <span>Maintenance Checklist</span>
            </a>
            <a
              className={`nav-item ${currentView === 'edge-sync' ? 'active' : ''}`}
              data-view="edge-sync"
              onClick={() => handleNavClick('edge-sync')}
              title="Edge Buffer & Sync"
            >
              <i className="fa-solid fa-cloud-arrow-up"></i>
              <span>Edge Buffer & Sync</span>
              {offlineBuffer.length > 0 && <span className="nav-badge">{offlineBuffer.length}</span>}
            </a>
            <a
              className={`nav-item ${currentView === 'esp32-live-monitor' ? 'active' : ''}`}
              data-view="esp32-live-monitor"
              onClick={() => handleNavClick('esp32-live-monitor')}
              title="ESP32 Live Monitor"
            >
              <i className="fa-solid fa-microchip"></i>
              <span>ESP32 Live Monitor</span>
              <span className="nav-badge" style={{ background: 'rgba(0,255,102,0.15)', color: 'var(--neon-green)', borderColor: 'var(--neon-green)' }}>LIVE</span>
            </a>

            <div className="nav-section-title">Station Triage & Health</div>
            <a
              className={`nav-item ${currentView === 'incidents' ? 'active' : ''}`}
              data-view="incidents"
              onClick={() => handleNavClick('incidents')}
              title="Incident Triage"
            >
              <i className="fa-solid fa-triangle-exclamation"></i>
              <span>Incident Triage</span>
              {openIncidentsCount > 0 && <span className="nav-badge">{openIncidentsCount}</span>}
            </a>
            <a
              className={`nav-item ${currentView === 'fault-lab' ? 'active' : ''}`}
              data-view="fault-lab"
              onClick={() => handleNavClick('fault-lab')}
              title="Fault Lab"
            >
              <i className="fa-solid fa-vial-virus"></i>
              <span>Fault Lab</span>
            </a>
            <a
              className={`nav-item ${currentView === 'model-governance' ? 'active' : ''}`}
              data-view="model-governance"
              onClick={() => handleNavClick('model-governance')}
              title="Model Governance"
            >
              <i className="fa-solid fa-brain"></i>
              <span>Model Governance</span>
            </a>
            <a
              className={`nav-item ${currentView === 'fleet-map' ? 'active' : ''}`}
              data-view="fleet-map"
              onClick={() => handleNavClick('fleet-map')}
              title="Fleet Radar Map"
            >
              <i className="fa-solid fa-map-location-dot"></i>
              <span>Fleet Radar Map</span>
            </a>
          </>
        )}
      </nav>

      {/* Fleet Status — live operational state from the 5 s fleet sync */}
      {!collapsed && (
        <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border-subtle)' }}>
          <div className="nav-section-title" style={{ marginTop: 0 }}>Fleet Status</div>
          <div
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-secondary)', cursor: 'pointer', padding: '3px 0' }}
            onClick={() => { tacticalAudio.playClick(); setCurrentView('incidents'); }}
            title="Open the incident triage queue"
          >
            <span><i className="fa-solid fa-bell" style={{ marginRight: '6px', color: openIncidentsCount > 0 ? 'var(--neon-crimson)' : 'var(--text-muted)' }}></i>Open incidents</span>
            <span className={`cyber-badge ${openIncidentsCount > 0 ? 'badge-critical' : 'badge-normal'}`} style={{ fontSize: '0.66rem' }}>
              {openIncidentsCount}
            </span>
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', padding: '3px 0' }}>
            <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: '6px', color: attentionStations.length > 0 ? 'var(--neon-amber)' : 'var(--text-muted)' }}></i>
            Needs attention: <strong style={{ color: attentionStations.length > 0 ? 'var(--neon-amber)' : 'var(--text-primary)' }}>{attentionStations.length}</strong>
          </div>
          {attentionStations.slice(0, 4).map(st => (
            <div
              key={st.id}
              onClick={() => handleAttentionClick(st.id)}
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.7rem', color: 'var(--text-primary)', cursor: 'pointer', padding: '3px 0 3px 22px' }}
              title={`${st.id}: ${st.status} — open to triage`}
            >
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{st.id}</span>
              <span className={`cyber-badge ${st.status === 'REGIONAL_EVENT' ? 'badge-extreme' : 'badge-critical'}`} style={{ fontSize: '0.6rem' }}>
                {String(st.status).replace(/_/g, ' ')}
              </span>
            </div>
          ))}
          <div style={{ marginTop: '6px', paddingTop: '6px', borderTop: '1px solid var(--border-subtle)' }}>
            {visibleStations.map(st => {
              const reg = modelRegistry[st.id];
              // The IF status comes from the registry API this sidebar
              // already fetches (reg.if). The fleet station's
              // model_status only reflects the latest live assessment,
              // so it said PENDING for AWS-02 even though its IF model
              // is ACTIVE in the registry. While the registry fetch is
              // still loading (reg undefined) or failed (reg null),
              // fall back to the fleet value.
              const ifStatus = reg
                ? (reg.if || 'NOT REGISTERED')
                : (st.model_status === 'ACTIVE_PRODUCTION' ? 'ACTIVE' : 'PENDING');
              const ifColor = ifStatus === 'ACTIVE'
                ? 'var(--neon-green)'
                : ifStatus === 'NOT REGISTERED'
                  ? 'var(--text-muted)'
                  : 'var(--neon-amber)';
              const mambaStatus = reg ? (reg.mamba || 'NOT REGISTERED') : (st.mamba_model?.has_model ? 'REGISTERED' : 'NOT REGISTERED');
              return (
                <div key={st.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', padding: '2px 0' }}>
                  <span style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{st.id}</span>
                  <span>
                    IF <strong style={{ color: ifColor }}>{ifStatus}</strong>
                    {' · '}Mamba <strong style={{ color: mambaStatus === 'ACTIVE' ? 'var(--neon-green)' : mambaStatus === 'CANDIDATE' ? 'var(--neon-amber)' : 'var(--text-muted)' }}>{mambaStatus}</strong>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        <button
          className={`audio-toggle-btn ${audioActive ? 'active' : ''}`}
          id="audio-toggle-btn"
          onClick={toggleAudio}
        >
          <i className={`fa-solid fa-volume-${audioActive ? 'high' : 'xmark'}`}></i>
          <span>AUDIO: {audioActive ? 'ON' : 'MUTED'}</span>
        </button>

        <button
          className="cyber-btn btn-sm btn-danger"
          id="logout-btn"
          style={{ width: '100%', padding: '6px 10px', fontSize: '0.7rem' }}
          title="Terminate Tactical Session"
          onClick={handleLogout}
        >
          <i className="fa-solid fa-arrow-right-from-bracket"></i> <span>LOGOUT</span>
        </button>
      </div>
    </aside>
  );
};
