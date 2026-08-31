import React, { useState } from 'react';
import { EventTable } from '../components/EventTable';
import { exportToCSV, exportToExcel } from '../utils/exportHelper';

export const BatchInferenceTab = ({ dfM12, costPerHour }) => {
  const [recFilter, setRecFilter] = useState([]);
  const [testFilter, setTestFilter] = useState([]);
  const [waferFilter, setWaferFilter] = useState([]);
  const [siteFilter, setSiteFilter] = useState([]);
  const [probRange, setProbRange] = useState([0.0, 1.0]);

  if (!dfM12 || dfM12.length === 0) {
    return (
      <div>
        <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '16px' }}>Month 12 Batch Inference</h3>
        <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
          No batch inference dataset loaded. Please upload or load Month 12 data on the Overview tab.
        </div>
      </div>
    );
  }

  const uniqueTests = Array.from(new Set(dfM12.map(r => r.Fail_Test).filter(Boolean))).sort();
  const uniqueWafers = Array.from(new Set(dfM12.map(r => r.Wafer_ID).filter(Boolean))).sort();
  const uniqueSites = Array.from(new Set(dfM12.map(r => r.ATE_Site).filter(Boolean))).sort();

  const filtered = dfM12.filter(r => {
    if (recFilter.length > 0 && !recFilter.includes(String(r.AI_Recommendation || '').trim())) return false;
    if (testFilter.length > 0 && !testFilter.includes(r.Fail_Test)) return false;
    if (waferFilter.length > 0 && !waferFilter.includes(r.Wafer_ID)) return false;
    if (siteFilter.length > 0 && !siteFilter.includes(r.ATE_Site)) return false;
    const p = Number(r['P(RETEST_BENEFICIAL)'] ?? 0);
    if (p < probRange[0] || p > probRange[1]) return false;
    return true;
  });

  return (
    <div>
      <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '4px' }}>Month 12 Batch Inference</h3>
      <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
        Complete batch table of unscored pre-retest events scored with supervised ML P(RETEST_BENEFICIAL).
      </div>

      {/* Filter Row */}
      <div className="dark-card mb-4" style={{ padding: '16px' }}>
        <div className="dark-card-header">Filter Events</div>
        <div className="grid grid-cols-5 gap-4">
          <div>
            <label className="text-xs uppercase font-semibold text-muted block mb-1">Recommendation</label>
            <select
              className="input-base"
              multiple
              value={recFilter}
              onChange={e => setRecFilter(Array.from(e.target.selectedOptions, o => o.value))}
              style={{ height: '70px' }}
            >
              <option value="RETEST">RETEST</option>
              <option value="DON'T RETEST">DON'T RETEST</option>
            </select>
          </div>
          <div>
            <label className="text-xs uppercase font-semibold text-muted block mb-1">Fail Test</label>
            <select
              className="input-base"
              multiple
              value={testFilter}
              onChange={e => setTestFilter(Array.from(e.target.selectedOptions, o => o.value))}
              style={{ height: '70px' }}
            >
              {uniqueTests.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs uppercase font-semibold text-muted block mb-1">Wafer ID</label>
            <select
              className="input-base"
              multiple
              value={waferFilter}
              onChange={e => setWaferFilter(Array.from(e.target.selectedOptions, o => o.value))}
              style={{ height: '70px' }}
            >
              {uniqueWafers.map(w => <option key={w} value={w}>{w}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs uppercase font-semibold text-muted block mb-1">ATE Site</label>
            <select
              className="input-base"
              multiple
              value={siteFilter}
              onChange={e => setSiteFilter(Array.from(e.target.selectedOptions, o => o.value))}
              style={{ height: '70px' }}
            >
              {uniqueSites.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs uppercase font-semibold text-muted block mb-1">
              Prob Range: {probRange[0].toFixed(2)} - {probRange[1].toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={probRange[1]}
              onChange={e => setProbRange([probRange[0], parseFloat(e.target.value)])}
              className="w-full"
            />
            <div className="flex gap-2 mt-2">
              <button className="btn-secondary" style={{ padding: '4px 8px', fontSize: '11px' }} onClick={() => { setRecFilter([]); setTestFilter([]); setWaferFilter([]); setSiteFilter([]); setProbRange([0, 1]); }}>
                Reset Filters
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center mb-2">
        <div style={{ fontSize: '13px', color: '#94a3b8' }}>
          Showing {filtered.length} of {dfM12.length} events
        </div>
        <div className="flex gap-2">
          <button className="btn-secondary" onClick={() => exportToCSV(filtered, 'Month12_Batch_Inference.csv')}>
            Export CSV
          </button>
          <button className="btn-secondary" onClick={() => exportToExcel(filtered, 'Month12_Batch_Inference.xlsx', 'Batch Inference')}>
            Export Excel
          </button>
        </div>
      </div>

      <EventTable data={filtered} height="400px" />
    </div>
  );
};
