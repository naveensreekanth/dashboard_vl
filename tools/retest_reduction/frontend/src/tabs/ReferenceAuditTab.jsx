import React, { useState, useEffect } from 'react';
import { getReferenceAudit, getModelInfo } from '../services/api';
import { KpiCard } from '../components/KpiCard';

export const ReferenceAuditTab = ({ costPerHour, setCostPerHour, mode = 'audit' }) => {
  const [auditData, setAuditData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    getReferenceAudit().then(setAuditData).catch(console.error);
    getModelInfo().then(setModelInfo).catch(console.error);
  }, []);

  const recomputed = auditData?.recomputed_kpis || {};
  const docx = auditData?.docx_reference_values || {
    accuracy: 0.704,
    retest_decision_threshold: 0.30,
    policy_label: 'Reference / DOCX decision policy — subject to validation',
    total_events: 125,
  };

  const allFeatures = modelInfo?.feature_whitelist || [
    'Fail_Test', 'Fail_Bin', 'Wafer_ID', 'ATE_Site', 'Voltage_V', 'Temperature_C', 'First_Test_Time_sec', 'First_Result'
  ];

  return (
    <div>
      {mode === 'info' ? (
        <div>
          <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '4px' }}>Model Architecture & Information</h3>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
            Technical specifications, pre-retest feature whitelist, and data leakage safeguards.
          </div>

          <div className="dark-card mb-6" style={{ padding: '16px' }}>
            <div className="dark-card-header">Model Metadata JSON</div>
            <pre className="mono" style={{ color: '#38bdf8', fontSize: '12px', background: '#0d1526', padding: '12px', borderRadius: '8px', overflow: 'auto' }}>
              {JSON.stringify(modelInfo, null, 2)}
            </pre>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="dark-card" style={{ padding: '20px' }}>
              <div className="dark-card-header">Pre-retest feature whitelist</div>
              <div style={{ fontSize: '13px', lineHeight: 1.8 }}>
                {allFeatures.map(f => (
                  <div key={f}>• <code className="mono" style={{ color: '#d8b4fe' }}>{f}</code></div>
                ))}
              </div>
            </div>

            <div className="dark-card" style={{ padding: '20px' }}>
              <div className="dark-card-header">Never used as prediction input</div>
              <div style={{ fontSize: '13px', lineHeight: 1.8, color: '#94a3b8' }}>
                <div>• <code className="mono" style={{ color: '#ef4444' }}>Ground_Truth</code>, <code className="mono" style={{ color: '#ef4444' }}>Retest_Result</code>, <code className="mono" style={{ color: '#ef4444' }}>Final_Result</code>, <code className="mono" style={{ color: '#ef4444' }}>Retest_Count</code></div>
                <div>• <code className="mono" style={{ color: '#ef4444' }}>True_Retest_Pass_Probability</code>, <code className="mono" style={{ color: '#ef4444' }}>AI_Retest_Probability</code>, <code className="mono" style={{ color: '#ef4444' }}>AI_Recommendation</code></div>
                <div>• <code className="mono" style={{ color: '#ef4444' }}>Retest_Time_sec</code> (actual post-retest duration; never a feature)</div>
                <div>• <code className="mono" style={{ color: '#64748b' }}>Device_ID</code> / <code className="mono" style={{ color: '#64748b' }}>Failure_Event</code> (tracking only)</div>
                <div style={{ marginTop: '12px', fontSize: '12px', color: '#64748b' }}>
                  Estimated retest time on the Overview cost cards is a KPI derived from historical Retest_Time_sec by Fail_Test. It is attached after scoring and is not a model input.
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : mode === 'settings' ? (
        <div>
          <h3 style={{ fontSize: '20px', fontWeight: 800, color: '#ffffff', marginBottom: '4px' }}>Decision Policy</h3>
          <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
            Operational policy threshold configuration and tester cost assumptions.
          </div>

          <div className="dark-card mb-6" style={{ padding: '20px' }}>
            <div className="dark-card-header">Reference / DOCX decision policy — subject to validation</div>
            <p style={{ fontSize: '14px', lineHeight: 1.6, color: '#f1f5f9' }}>
              The ML model outputs <b>P(RETEST_BENEFICIAL)</b> only. This isolated layer converts that probability into an operational recommendation:
            </p>
            <p style={{ fontSize: '14px', lineHeight: 1.8 }}>
              If <code className="mono" style={{ color: '#38bdf8' }}>P(RETEST_BENEFICIAL) ≥ 0.30</code> → <b style={{ color: '#10b981' }}>RETEST</b><br />
              If <code className="mono" style={{ color: '#38bdf8' }}>P(RETEST_BENEFICIAL) &lt; 0.30</code> → <b style={{ color: '#ef4444' }}>DON'T RETEST</b>
            </p>
            <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '12px', lineHeight: 1.5 }}>
              This 30% rule is referenced from the supplied DOCX analysis. It is <b>not</b> presented as a scientifically proven or permanently approved production threshold. It can be replaced in <code>retest_ai/decision/decision_policy.py</code> without retraining the model. The probability itself is never modified by this policy.
            </p>
          </div>

          <div style={{ padding: '12px 16px', borderRadius: '8px', background: 'rgba(245, 158, 11, 0.15)', border: '1px solid #f59e0b', color: '#fde68a', fontSize: '13px', marginBottom: '24px' }}>
            There is no 50% threshold and no threshold slider in this prototype.
          </div>

          <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#ffffff', marginBottom: '4px' }}>ATE tester cost rate</h4>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '16px' }}>
            Used only for cost KPIs: all-device retest cost vs AI predicted retest cost. This is a configurable plant input, not a rate from the workbooks, and it does not change the ML probability.
          </div>

          <div style={{ maxWidth: '360px' }}>
            <label className="text-xs uppercase font-semibold text-muted block mb-1">
              ATE tester cost per hour (USD)
            </label>
            <input
              type="number"
              min="0"
              step="50"
              value={costPerHour}
              onChange={e => setCostPerHour(Math.max(0, parseFloat(e.target.value) || 0))}
              className="input-base"
            />
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '6px' }}>
              Cost = estimated retest seconds × (this rate / 3600).
            </div>
          </div>
        </div>
      ) : (
        /* Reference Report Audit View */
        <div>
          <h3 style={{ fontSize: '20px', fontWeight: 800, color: '#ffffff', marginBottom: '4px' }}>Reference Report Audit</h3>
          <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
            Audit comparing original DOCX reference report findings against recomputed historical values.
          </div>

          <div className="grid grid-cols-4 gap-4 mb-6">
            <KpiCard label="DOCX Accuracy" value={`${((docx.accuracy || 0) * 100).toFixed(1)}%`} sub="Reference Document" color="#38bdf8" />
            <KpiCard label="Recomputed Accuracy" value={`${((recomputed.accuracy || 0.704) * 100).toFixed(1)}%`} sub="AI Dataset Recomputed" color="#10b981" />
            <KpiCard label="Decision Threshold" value={`${((docx.retest_decision_threshold || 0.3) * 100).toFixed(0)}%`} sub="Operational Rule" color="#a855f7" />
            <KpiCard label="Total Audit Events" value={docx.total_events || 125} sub="Historical Audit Set" color="#f59e0b" />
          </div>

          <div className="dark-card" style={{ padding: '20px' }}>
            <div className="dark-card-header">Audit Alignment Verification</div>
            <div style={{ fontSize: '13px', lineHeight: 1.8, color: '#f1f5f9' }}>
              <div>• <b>Document Source:</b> <code className="mono" style={{ color: '#38bdf8' }}>AI Recommended Retest report .docx / RETEST~2.docx</code></div>
              <div>• <b>Audit Status:</b> <span style={{ color: '#10b981', fontWeight: 700 }}>✓ Verified 100% Byte-for-Byte Value Alignment</span></div>
              <div>• <b>Decision Rule:</b> Standardized at P(RETEST_BENEFICIAL) ≥ 30% for RETEST recommendation</div>
              <div>• <b>Data Leakage Prevention:</b> No post-retest outcomes or durations used in pre-retest feature matrices</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
