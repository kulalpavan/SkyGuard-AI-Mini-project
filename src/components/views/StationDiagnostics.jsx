import React from 'react';
import { useWeather } from '../../context/WeatherContext';
import { displayCode } from '../../utils/display';

export const StationDiagnostics = () => {
  const { stations, activeStationId, setCurrentView } = useWeather();
  const station = stations.find(s => s.id?.toUpperCase() === activeStationId?.toUpperCase()) || stations[0] || {};

  if (!station || !station.id) {
    return (
      <div className="cyber-card" style={{ textAlign: 'center', padding: '60px 20px' }}>
        <i className="fa-solid fa-microchip" style={{ fontSize: '3rem', color: 'var(--neon-cyan)', marginBottom: '16px', opacity: 0.8 }}></i>
        <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.2rem', color: 'var(--neon-cyan)', fontWeight: 800 }}>
          NO REGISTERED WEATHER STATION
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', maxWidth: '500px', margin: '12px auto 20px auto' }}>
          No weather station is active. Provision a station in Station Credentials to inspect hardware telemetry.
        </p>
        <button className="cyber-btn btn-sm btn-primary" onClick={() => setCurrentView('credentials')}>
          <i className="fa-solid fa-key"></i> Provision Weather Station
        </button>
      </div>
    );
  }

  const temp = station.sensors?.temperature?.value ?? null;
  const hum = station.sensors?.humidity?.value ?? null;
  const pres = station.sensors?.pressure?.value ?? null;
  // Wind, battery, signal, firmware and uptime only exist when the station
  // actually reports them. This screen used to invent them (wind defaulted
  // to 14 km/h, battery to 91%, and solar / network / gateway internals
  // were hardcoded constants) — now anything unreported shows as "—".
  const wind = station.sensors?.wind_speed?.value ?? null;
  const battery = typeof station.battery === 'number' ? station.battery : null;
  const batteryPct = battery === null ? null
    : Math.round(Math.min(100, Math.max(0, ((battery - 11.0) / 2.0) * 100)));
  const batteryLow = battery !== null && battery < 11.8;
  const signal = typeof station.signal === 'number' ? station.signal : null;
  const firmware = station.firmware ?? null;
  const uptimeS = typeof station.uptime_s === 'number' ? station.uptime_s : null;

  return (
    <div className="cyber-card">
      <div className="cyber-card-header">
        <div className="cyber-card-title">
          <i className="fa-solid fa-microchip text-cyan"></i> HARDWARE TELEMETRY & SENSOR STATUS
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>{station.id} — {station.name}</span>
          <span className={`cyber-badge ${station.status === 'NORMAL' ? 'badge-normal' : 'badge-suspect'}`}>
            {displayCode(station.status)}
          </span>
        </div>
      </div>
      <div className="cyber-card-body">
        {/* Dominant Live Sensor Values Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          <div style={{ background: 'rgba(10,15,29,0.85)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
              <i className="fa-solid fa-temperature-half text-cyan"></i> Temperature
            </div>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.75rem', fontWeight: 900, color: 'var(--neon-cyan)' }}>
              {temp ?? '—'} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>°C</span>
            </div>
            <div style={{ fontSize: '0.66rem', color: 'var(--neon-green)', marginTop: '4px' }}>{temp === null ? 'No reading' : 'Nominal Range'}</div>
          </div>

          <div style={{ background: 'rgba(10,15,29,0.85)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
              <i className="fa-solid fa-droplet text-green"></i> Humidity
            </div>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.75rem', fontWeight: 900, color: 'var(--neon-green)' }}>
              {hum ?? '—'} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>%</span>
            </div>
            <div style={{ fontSize: '0.66rem', color: 'var(--neon-green)', marginTop: '4px' }}>{hum === null ? 'No reading' : 'RH Valid'}</div>
          </div>

          <div style={{ background: 'rgba(10,15,29,0.85)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
              <i className="fa-solid fa-gauge text-cyan"></i> Pressure
            </div>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.75rem', fontWeight: 900, color: 'var(--text-primary)' }}>
              {pres ?? '—'} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>hPa</span>
            </div>
            <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: '4px' }}>{pres === null ? 'No reading' : 'Barometric Normal'}</div>
          </div>

          <div style={{ background: 'rgba(10,15,29,0.85)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
              <i className="fa-solid fa-wind text-purple"></i> Wind Speed
            </div>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.75rem', fontWeight: 900, color: 'var(--neon-purple)' }}>
              {wind ?? '—'} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>km/h</span>
            </div>
            <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: '4px' }}>{wind === null ? 'Not reported' : 'Sonic Anemometer'}</div>
          </div>

          <div style={{ background: 'rgba(10,15,29,0.85)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '14px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
              <i className="fa-solid fa-car-battery text-green"></i> Battery
            </div>
            <div style={{ fontFamily: 'var(--font-tactical)', fontSize: '1.75rem', fontWeight: 900, color: batteryLow ? 'var(--neon-crimson)' : 'var(--neon-green)' }}>
              {batteryPct === null ? '—' : `${batteryPct}%`} <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{battery === null ? '' : `(${battery}V)`}</span>
            </div>
            <div style={{ fontSize: '0.66rem', color: batteryLow ? 'var(--neon-crimson)' : 'var(--neon-green)', marginTop: '4px' }}>
              {battery === null ? 'Not reported' : batteryLow ? 'Low Voltage' : 'Normal Capacity'}
            </div>
          </div>
        </div>

        {/* Subsystem Telemetry Boxes */}
        <div className="sim-control-panel">
          {/* Card 1: Power & Battery */}
          <div className="sim-box">
            <div className="sim-box-title">
              <span><i className="fa-solid fa-solar-panel text-green"></i> SOLAR & POWER</span>
              <span className={`cyber-badge ${batteryLow ? 'badge-critical' : 'badge-normal'}`}>
                {battery === null ? '—' : `${battery}V`}
              </span>
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
              <div>Battery voltage: <strong style={{ color: batteryLow ? 'var(--neon-crimson)' : 'var(--neon-green)' }}>{battery === null ? 'Not reported' : `${battery}V`}</strong></div>
              <div>Solar PV output / charging current: <strong>Not reported by this station</strong></div>
              <div>Subsystem Status: <strong style={{ color: batteryLow ? 'var(--neon-crimson)' : 'var(--neon-green)' }}>
                {battery === null ? 'UNKNOWN — NO BATTERY TELEMETRY' : batteryLow ? 'LOW VOLTAGE' : 'NOMINAL'}
              </strong></div>
            </div>
          </div>

          {/* Card 2: Cellular & Modbus */}
          <div className="sim-box">
            <div className="sim-box-title">
              <span><i className="fa-solid fa-tower-cell text-cyan"></i> NETWORK & BUS</span>
              <span className="cyber-badge badge-normal">{signal === null ? '—' : `${signal} dBm`}</span>
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
              <div>Signal strength: <strong>{signal === null ? 'Not reported' : `${signal} dBm`}</strong></div>
              <div>Bearer / CRC errors / latency: <strong>Not reported by this station</strong></div>
            </div>
          </div>

          {/* Card 3: Gateway Compute */}
          <div className="sim-box">
            <div className="sim-box-title">
              <span><i className="fa-solid fa-memory text-purple"></i> GATEWAY MCU</span>
              <span className="cyber-badge badge-offline">{firmware ?? '—'}</span>
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
              <div>Uptime: <strong>{uptimeS === null ? 'Not reported' : `${Math.floor(uptimeS / 3600)}h ${Math.floor((uptimeS % 3600) / 60)}m`}</strong></div>
              <div>Firmware: <strong>{firmware ?? 'Not reported'}</strong></div>
              <div>Flash storage / enclosure humidity: <strong>Not reported by this station</strong></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
