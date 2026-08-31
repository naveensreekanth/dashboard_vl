import React from 'react';

export const KpiCard = ({ label, value, sub, color = 'var(--text-title)', contextLabel, onInspect }) => (
  <div className="dark-card" style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%', marginBottom: 0 }}>
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
        <div>
          {contextLabel && (
            <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-dark)', marginBottom: '2px' }}>
              {contextLabel}
            </div>
          )}
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>
            {label}
          </div>
        </div>
        {onInspect && (
          <button
            className="btn-secondary"
            style={{ padding: '3px 8px', fontSize: '11px', flexShrink: 0 }}
            onClick={onInspect}
          >
            Inspect
          </button>
        )}
      </div>
      <div className="mono" style={{ fontSize: '30px', fontWeight: 700, color, lineHeight: 1.1, margin: '6px 0' }}>
        {value}
      </div>
    </div>
    {sub && (
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
        {sub}
      </div>
    )}
  </div>
);

export const KpiEventsDevicesCard = ({ label, events, devices, color = 'var(--text-title)', contextLabel, onInspect }) => (
  <div className="dark-card" style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%', marginBottom: 0 }}>
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
        <div>
          {contextLabel && (
            <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-dark)', marginBottom: '2px' }}>
              {contextLabel}
            </div>
          )}
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>
            {label}
          </div>
        </div>
        {onInspect && (
          <button
            className="btn-secondary"
            style={{ padding: '3px 8px', fontSize: '11px', flexShrink: 0 }}
            onClick={onInspect}
          >
            Inspect
          </button>
        )}
      </div>
      <div className="flex items-end gap-6" style={{ margin: '6px 0' }}>
        <div className="flex-col">
          <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-dark)', marginBottom: '2px' }}>Events</div>
          <div className="mono" style={{ fontSize: '30px', fontWeight: 700, color, lineHeight: 1.1 }}>{events}</div>
        </div>
        <div className="flex-col">
          <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-dark)', marginBottom: '2px' }}>Devices</div>
          <div className="mono" style={{ fontSize: '30px', fontWeight: 700, color, lineHeight: 1.1 }}>{devices}</div>
        </div>
      </div>
    </div>
  </div>
);

export const KpiCostCard = ({ label, value, sub, color = 'var(--text-title)', contextLabel, onInspect }) => (
  <div className="dark-card" style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%', marginBottom: 0 }}>
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
        <div>
          {contextLabel && (
            <div style={{ fontSize: '10px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-dark)', marginBottom: '2px' }}>
              {contextLabel}
            </div>
          )}
          <div style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)' }}>
            {label}
          </div>
        </div>
        {onInspect && (
          <button
            className="btn-secondary"
            style={{ padding: '3px 8px', fontSize: '11px', flexShrink: 0 }}
            onClick={onInspect}
          >
            Inspect
          </button>
        )}
      </div>
      <div className="mono" style={{ fontSize: '26px', fontWeight: 700, color, lineHeight: 1.1, margin: '6px 0' }}>
        {value}
      </div>
    </div>
    {sub && (
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px' }}>
        {sub}
      </div>
    )}
  </div>
);
