import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { getSingleEventOptions, predictSingleWithShap } from '../services/api';
import { formatMoney } from '../utils/formatters';

export const SingleEventTab = ({ costPerHour }) => {
  const [options, setOptions] = useState({ devices: [], events: [], features: [] });
  const [selectedDev, setSelectedDev] = useState('');
  const [selectedEventIndex, setSelectedEventIndex] = useState(0);
  const [predResult, setPredResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getSingleEventOptions().then(data => {
      setOptions(data);
      if (data.devices && data.devices.length > 0) {
        const defaultDev = data.devices.includes('DEV004') ? 'DEV004' : data.devices[0];
        setSelectedDev(defaultDev);
      }
    }).catch(console.error);
  }, []);

  const devEvents = options.events.filter(e => e.Device_ID === selectedDev);
  const currentEvent = devEvents[selectedEventIndex] || devEvents[0];

  useEffect(() => {
    if (!currentEvent) return;
    setLoading(true);
    predictSingleWithShap(currentEvent)
      .then(res => setPredResult(res))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedDev, selectedEventIndex, options.events]);

  const pVal = predResult?.prediction?.probability_retest_beneficial ?? currentEvent?.['P(RETEST_BENEFICIAL)'] ?? 0;
  const pPct = predResult?.prediction?.probability_percent ?? pVal * 100;
  const rec = predResult?.prediction?.recommendation ?? (pVal >= 0.3 ? 'RETEST' : "DON'T RETEST");
  const baseP = predResult?.prediction?.probability_base ?? pVal;
  const adaptedP = predResult?.prediction?.probability_adapted ?? pVal;
  const olActive = Boolean(predResult?.prediction?.online_adaptation_active);

  const estSec = Number(predResult?.prediction?.estimated_retest_time_sec ?? currentEvent?.Estimated_Retest_Time_sec ?? 0);
  const predSec = rec === 'RETEST' ? estSec : 0;
  const rate = Number(costPerHour || 350.0);
  const eventCost = (predSec * (rate / 3600.0)).toFixed(2);
  const ifRunCost = (estSec * (rate / 3600.0)).toFixed(2);

  const shapFeatures = predResult?.explanation?.top_features ? [...predResult.explanation.top_features].reverse() : [];
  const explanations = predResult?.explanation?.engineering_explanations || [];

  return (
    <div>
      <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '4px' }}>Single Event Analysis</h3>
      <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '20px' }}>
        Should this failed event be sent for retest? Probability and recommendation are separate outputs.
      </div>

      {/* Selectors */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div>
          <label className="text-xs uppercase font-semibold text-muted block mb-1">Device Identifier</label>
          <select
            className="input-base"
            value={selectedDev}
            onChange={e => { setSelectedDev(e.target.value); setSelectedEventIndex(0); }}
          >
            {options.devices.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs uppercase font-semibold text-muted block mb-1">Failure Event Index</label>
          <select
            className="input-base"
            value={selectedEventIndex}
            onChange={e => setSelectedEventIndex(Number(e.target.value))}
          >
            {devEvents.map((ev, idx) => (
              <option key={idx} value={idx}>
                Event #{ev.Failure_Event} ({ev.Fail_Test})
              </option>
            ))}
          </select>
        </div>
        <div style={{ paddingTop: '22px', fontSize: '13px', color: '#94a3b8' }}>
          Dataset: <b style={{ color: '#ffffff' }}>Month 12 (unseen inference)</b> — outcomes not used.
        </div>
      </div>

      {currentEvent && (
        <div className="grid grid-cols-2 gap-6 mb-6">
          {/* Event Information */}
          <div className="dark-card" style={{ padding: '20px' }}>
            <div className="dark-card-header">Event Information</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Device ID</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8', marginTop: '3px' }}>{currentEvent.Device_ID}</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Failure Event</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>{currentEvent.Failure_Event}</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Wafer ID</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>{currentEvent.Wafer_ID}</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>ATE Site</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>Site {currentEvent.ATE_Site}</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Fail Test</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#a855f7', marginTop: '3px' }}>{currentEvent.Fail_Test}</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Fail Bin</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>Bin {currentEvent.Fail_Bin}</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Voltage</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>{Number(currentEvent.Voltage_V || 0).toFixed(2)} V</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Temperature</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>{currentEvent.Temperature_C} °C</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>First Test Time</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>{Number(currentEvent.First_Test_Time_sec || 0).toFixed(1)} s</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Est. Retest Time</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9', marginTop: '3px' }}>{estSec.toFixed(1)} s</div>
              </div>
              <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px', gridColumn: 'span 2' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Initial Result</div>
                <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: '#ef4444', marginTop: '3px' }}>{currentEvent.First_Result}</div>
              </div>
            </div>
          </div>

          {/* AI Prediction & Recommendation */}
          <div className="dark-card" style={{ padding: '20px' }}>
            <div className="dark-card-header">AI Prediction and Recommendation</div>

            <div style={{ background: '#111a2d', border: '1px solid #233554', borderRadius: '12px', padding: '24px', textAlign: 'center' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1.2px', color: '#a855f7', marginBottom: '6px' }}>
                RETEST-BENEFICIAL PROBABILITY
              </div>
              <div className="mono" style={{ fontSize: '52px', fontWeight: 800, color: '#ffffff' }}>
                {pPct.toFixed(1)}%
              </div>
              <div style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>
                P(RETEST_BENEFICIAL) = <code className="mono" style={{ color: '#a855f7' }}>{pVal.toFixed(4)}</code>
              </div>
              <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '10px', textAlign: 'left', lineHeight: 1.6 }}>
                Base Probability: <code className="mono" style={{ color: '#38bdf8' }}>{(baseP * 100).toFixed(1)}%</code><br />
                Adapted Probability: <code className="mono" style={{ color: '#a855f7' }}>{(adaptedP * 100).toFixed(1)}%</code><br />
                Final Recommendation: <b>{rec}</b><br />
                Online adaptation: {olActive ? 'Active' : 'Not active'}
              </div>
            </div>

            {rec === 'RETEST' ? (
              <div className="retest-banner" style={{ marginTop: '14px' }}>
                <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1.5px', color: '#10b981', marginBottom: '6px' }}>AI RECOMMENDATION</div>
                <div className="mono" style={{ fontSize: '38px', fontWeight: 800, color: '#10b981' }}>RETEST</div>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '8px' }}>
                  Reference / DOCX decision policy — subject to validation<br />
                  If P ≥ 0.30 → RETEST. Probability is not modified by this policy.
                </div>
              </div>
            ) : (
              <div className="skip-banner" style={{ marginTop: '14px' }}>
                <div style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1.5px', color: '#ef4444', marginBottom: '6px' }}>AI RECOMMENDATION</div>
                <div className="mono" style={{ fontSize: '38px', fontWeight: 800, color: '#ef4444' }}>DON'T RETEST</div>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '8px' }}>
                  Reference / DOCX decision policy — subject to validation<br />
                  If P &lt; 0.30 → DON'T RETEST. Probability is not modified by this policy.
                </div>
              </div>
            )}

            <div style={{ background: '#0d1526', border: '1px solid #1a263d', borderRadius: '8px', padding: '12px 14px', marginTop: '14px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Predicted retest cost</div>
              <div className="mono" style={{ fontSize: '16px', fontWeight: 700, color: rec === 'RETEST' ? '#10b981' : '#ef4444', marginTop: '3px' }}>
                {formatMoney(Number(eventCost), 'USD')}
              </div>
              <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '8px' }}>
                Estimated duration {estSec.toFixed(1)} s · rate {formatMoney(rate, 'USD')}/h. {rec === 'RETEST' ? 'Charged because AI recommends RETEST.' : `AI skip → $0. If retested anyway, about ${formatMoney(Number(ifRunCost), 'USD')}.`}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SHAP Features & Engineering Explanation */}
      <div className="dark-card" style={{ padding: '20px' }}>
        <div className="dark-card-header">Which input features contributed to this prediction?</div>
        <div className="grid grid-cols-2 gap-6">
          <div>
            {shapFeatures.length > 0 ? (
              <Plot
                data={[
                  {
                    type: 'bar',
                    orientation: 'h',
                    x: shapFeatures.map(f => f.shap_value),
                    y: shapFeatures.map(f => f.feature),
                    marker: {
                      color: shapFeatures.map(f => (f.shap_value >= 0 ? '#10b981' : '#ef4444')),
                    },
                  }
                ]}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { family: 'Inter', color: '#f1f5f9' },
                  xaxis: { gridcolor: '#1e293b', title: 'Contribution to model prediction' },
                  yaxis: { gridcolor: '#1e293b', autorange: 'reversed' },
                  margin: { l: 150, r: 20, t: 10, b: 30 },
                  height: 260,
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            ) : (
              <div style={{ color: '#64748b', fontSize: '13px', padding: '40px 0' }}>Computing SHAP explanations...</div>
            )}
          </div>

          <div>
            <div style={{ fontSize: '13px', lineHeight: 1.8, marginBottom: '12px' }}>
              {explanations.slice(0, 4).map((bullet, idx) => (
                <div key={idx} style={{ marginBottom: '8px' }}>
                  • {bullet}
                </div>
              ))}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              SHAP shows association with the model prediction, not physical causation.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
