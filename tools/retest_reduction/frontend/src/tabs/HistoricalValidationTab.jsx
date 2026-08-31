import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { KpiCard } from '../components/KpiCard';

export const HistoricalValidationTab = ({ histValidation }) => {
  const [showCutoff, setShowCutoff] = useState(false);

  if (!histValidation) {
    return (
      <div>
        <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '16px' }}>Historical Temporal Validation</h3>
        <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Loading validation metrics...</div>
      </div>
    );
  }

  const bestModel = histValidation.best_model_name || 'XGBoost';
  const m6Metrics = histValidation.models?.[bestModel]?.calibrated_metrics || {};
  const bestDiag = histValidation.models?.[bestModel]?.calibration_diagnostics || {};
  const reportingCutoff = histValidation.models?.[bestModel]?.reporting_cutoff_metrics;

  const categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC', 'PR-AUC'];
  const colors = { 'XGBoost': '#a855f7', 'Logistic Regression': '#38bdf8', 'Gradient Boosting': '#10b981' };

  const radarTraces = Object.keys(histValidation.models || {}).map(name => {
    const m = histValidation.models[name].calibrated_metrics || {};
    return {
      type: 'scatterpolar',
      r: [m.Accuracy || 0, m.Precision || 0, m.Recall || 0, m.Specificity || 0, m.F1 || 0, m['ROC-AUC'] || 0, m['PR-AUC'] || 0],
      theta: categories,
      fill: 'toself',
      name,
      line: { color: colors[name] || '#ffffff' },
    };
  });

  const bucketTable = bestDiag.bucket_table || [];

  return (
    <div>
      <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '4px' }}>Historical Temporal Validation</h3>
      <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
        Train Month 0 → validate Month 6. These metrics are not Month 12 operational performance.
      </div>

      <div style={{ padding: '12px 16px', borderRadius: '8px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid #38bdf8', color: '#7dd3fc', fontSize: '13px', marginBottom: '16px' }}>
        {histValidation.selection_reason}
      </div>

      <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '16px' }}>
        Classification metrics below use the operational Reference / DOCX decision policy (threshold=0.30). ROC-AUC, PR-AUC, Brier, and Log Loss are threshold-free. A 0.5 evaluation/reporting cutoff may appear in the comparison footnote only and does not override the 30% operational policy.
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-6 gap-4 mb-6">
        <KpiCard label="Selected Model" value={bestModel} sub="Evidence-based" color="#a855f7" />
        <KpiCard label="ROC-AUC" value={(m6Metrics['ROC-AUC'] || 0).toFixed(3)} sub="Threshold-free" />
        <KpiCard label="Accuracy" value={`${((m6Metrics.Accuracy || 0) * 100).toFixed(1)}%`} sub="At DOCX 30% policy" />
        <KpiCard label="Recall" value={`${((m6Metrics.Recall || 0) * 100).toFixed(1)}%`} sub="At DOCX 30% policy" />
        <KpiCard label="Brier Score" value={(bestDiag.brier_score || 0).toFixed(4)} sub="Lower is better" />
        <KpiCard label="Log Loss" value={(bestDiag.log_loss || 0).toFixed(4)} sub="Lower is better" />
      </div>

      {/* Comparison Table */}
      <div className="dark-card mb-6" style={{ padding: '20px' }}>
        <div className="dark-card-header">Model Comparison Table</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1e2c4a', color: '#94a3b8' }}>
                <th style={{ padding: '10px 12px' }}>Model</th>
                <th style={{ padding: '10px 12px' }}>Accuracy</th>
                <th style={{ padding: '10px 12px' }}>Precision</th>
                <th style={{ padding: '10px 12px' }}>Recall</th>
                <th style={{ padding: '10px 12px' }}>Specificity</th>
                <th style={{ padding: '10px 12px' }}>F1</th>
                <th style={{ padding: '10px 12px' }}>ROC-AUC</th>
                <th style={{ padding: '10px 12px' }}>PR-AUC</th>
                <th style={{ padding: '10px 12px' }}>Brier Score</th>
                <th style={{ padding: '10px 12px' }}>Log Loss</th>
              </tr>
            </thead>
            <tbody>
              {(histValidation.comparison_table || []).map((row, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #1a263d', color: '#f1f5f9' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 700, color: row.Model === bestModel ? '#a855f7' : '#f1f5f9' }}>
                    {row.Model}
                  </td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row.Accuracy === 'number' ? (row.Accuracy * 100).toFixed(1) + '%' : row.Accuracy}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row.Precision === 'number' ? (row.Precision * 100).toFixed(1) + '%' : row.Precision}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row.Recall === 'number' ? (row.Recall * 100).toFixed(1) + '%' : row.Recall}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row.Specificity === 'number' ? (row.Specificity * 100).toFixed(1) + '%' : row.Specificity}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row.F1 === 'number' ? row.F1.toFixed(3) : row.F1}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row['ROC-AUC'] === 'number' ? row['ROC-AUC'].toFixed(3) : row['ROC-AUC']}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row['PR-AUC'] === 'number' ? row['PR-AUC'].toFixed(3) : row['PR-AUC']}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row['Brier Score'] === 'number' ? row['Brier Score'].toFixed(4) : row['Brier Score']}</td>
                  <td style={{ padding: '10px 12px' }} className="mono">{typeof row['Log Loss'] === 'number' ? row['Log Loss'].toFixed(4) : row['Log Loss']}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: '11px', color: '#64748b', marginTop: '12px' }}>
          Accuracy/Precision/Recall/Specificity/F1 in this table use the isolated DOCX-reference 30% policy. They are not computed at 0.5. ROC-AUC / PR-AUC / Brier / Log Loss do not use a decision threshold.
        </div>

        {/* Optional 0.5 Cutoff Expander */}
        {reportingCutoff && (
          <div style={{ marginTop: '16px', borderTop: '1px solid #1e2c4a', paddingTop: '12px' }}>
            <button
              className="btn-secondary"
              style={{ fontSize: '12px', padding: '6px 12px' }}
              onClick={() => setShowCutoff(!showCutoff)}
            >
              {showCutoff ? '▼ Hide 0.5 evaluation/reporting cutoff' : '► Optional evaluation/reporting cutoff (0.5) — not the operational policy'}
            </button>
            {showCutoff && (
              <div style={{ background: '#0d1526', padding: '12px', borderRadius: '8px', marginTop: '8px', fontSize: '12px' }}>
                <div style={{ color: '#94a3b8', marginBottom: '8px' }}>
                  This 0.5 cutoff is for model-evaluation comparison only and does not produce RETEST / DON'T RETEST.
                </div>
                <pre className="mono" style={{ color: '#38bdf8', margin: 0 }}>
                  {JSON.stringify(reportingCutoff, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Radar Chart & Reliability Curve */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="dark-card" style={{ padding: '16px' }}>
          <div className="dark-card-header">Model Metrics Radar Chart</div>
          <Plot
            data={radarTraces}
            layout={{
              polar: {
                radialaxis: { visible: true, range: [0, 1], gridcolor: '#1e293b' },
                bgcolor: 'rgba(0,0,0,0)',
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              font: { family: 'Inter', color: '#f1f5f9' },
              margin: { l: 40, r: 40, t: 20, b: 30 },
              height: 320,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        <div className="dark-card" style={{ padding: '16px' }}>
          <div className="dark-card-header">Reliability Calibration Curve ({bestModel})</div>
          {bucketTable.length > 0 ? (
            <Plot
              data={[
                {
                  x: bucketTable.map(b => b.bucket),
                  y: bucketTable.map(b => (b.observed_benefit_rate || 0) * 100),
                  type: 'bar',
                  name: 'Observed beneficial rate (%)',
                  marker: { color: '#8b5cf6' },
                },
                {
                  x: bucketTable.map(b => b.bucket),
                  y: bucketTable.map(b => (b.mean_predicted_prob || 0) * 100),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Mean predicted probability (%)',
                  line: { color: '#38bdf8', width: 3 },
                }
              ]}
              layout={{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#f1f5f9' },
                xaxis: { gridcolor: '#1e293b', title: 'Probability Bucket' },
                yaxis: { gridcolor: '#1e293b', title: 'Rate (%)' },
                margin: { l: 40, r: 20, t: 20, b: 30 },
                height: 320,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <div style={{ color: '#64748b', fontSize: '13px', padding: '40px 0' }}>Calibration buckets unavailable.</div>
          )}
        </div>
      </div>

      {/* Bottom Info Bar */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', padding: '14px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ fontSize: '18px', color: '#a855f7' }}>1</div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Training (selection)</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#f1f5f9' }}>Month 0</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ fontSize: '18px', color: '#a855f7' }}>2</div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Validation holdout</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#f1f5f9' }}>Month 6</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ fontSize: '18px', color: '#a855f7' }}>3</div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Deploy training</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#f1f5f9' }}>Month 0 + Month 6</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ fontSize: '18px', color: '#a855f7' }}>4</div>
          <div>
            <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Inference</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#f1f5f9' }}>Month 12</div>
          </div>
        </div>
      </div>
    </div>
  );
};
