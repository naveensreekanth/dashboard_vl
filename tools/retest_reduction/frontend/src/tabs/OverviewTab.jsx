import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { KpiCard, KpiEventsDevicesCard, KpiCostCard } from '../components/KpiCard';
import { WorkflowPanel } from '../components/WorkflowPanel';
import { EventTable } from '../components/EventTable';
import { OnlineLearningPanel } from '../components/OnlineLearningPanel';
import { formatMoney, formatSeconds } from '../utils/formatters';
import { exportToCSV, exportToExcel } from '../utils/exportHelper';
import { uploadPreRetest, validateOutcomes } from '../services/api';

const REC_COLOR_MAP = { 'RETEST': '#10b981', "DON'T RETEST": '#ef4444' };
const OUTCOME_COLOR_MAP = { 'RETEST_BENEFICIAL': '#a855f7', 'PERSISTENT_FAILURE': '#f59e0b' };

export const OverviewTab = ({
  dfM12,
  setDfM12,
  costImpact,
  setCostImpact,
  costPerHour,
  predictionSourceLabel,
  setPredictionSourceLabel,
  activeOutcomes,
  setActiveOutcomes,
  outcomesLoaded,
  setOutcomesLoaded,
  validationData,
  setValidationData,
  histValidation,
}) => {
  const [view, setView] = useState('overview');
  const [showUpload, setShowUpload] = useState(false);
  const [preRetestFile, setPreRetestFile] = useState(null);
  const [outcomeFile, setOutcomeFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Filters for Month12 inspect view
  const [recFilter, setRecFilter] = useState([]);
  const [testFilter, setTestFilter] = useState([]);
  const [waferFilter, setWaferFilter] = useState([]);
  const [siteFilter, setSiteFilter] = useState([]);
  const [probRange, setProbRange] = useState([0.0, 1.0]);

  const hasActiveAnalysis = dfM12 && dfM12.length > 0;

  // Compute counts
  const counts = useMemo(() => {
    if (!hasActiveAnalysis) return { total: 0, retest: 0, dontRetest: 0, devices: 0, retestDevices: 0, dontRetestDevices: 0 };
    const total = dfM12.length;
    const retest = dfM12.filter(r => String(r.AI_Recommendation || '').trim() === 'RETEST').length;
    const dontRetest = total - retest;
    const allDevs = new Set(dfM12.map(r => r.Device_ID).filter(Boolean));
    const retestDevs = new Set(dfM12.filter(r => String(r.AI_Recommendation || '').trim() === 'RETEST').map(r => r.Device_ID).filter(Boolean));
    return {
      total,
      retest,
      dontRetest,
      devices: allDevs.size,
      retestDevices: retestDevs.size,
      dontRetestDevices: allDevs.size - retestDevs.size,
    };
  }, [dfM12, hasActiveAnalysis]);

  // Stage for workflow panel
  const workflowStage = useMemo(() => {
    if (outcomesLoaded && validationData?.learned) return 'learned';
    if (outcomesLoaded) return 'validate';
    if (hasActiveAnalysis) return 'recommend';
    return 'empty';
  }, [outcomesLoaded, validationData, hasActiveAnalysis]);

  // Helper for test family
  const getTestFamily = (val) => {
    const s = String(val || '').toLowerCase();
    if (s.includes('scan')) return 'Scan';
    if (s.includes('mbist')) return 'MBIST';
    if (s.includes('iddq')) return 'IDDQ';
    if (s.includes('func')) return 'Func';
    if (s.includes('atspeed') || s.includes('at_speed')) return 'AtSpeed';
    return String(val);
  };

  const handleAnalyzeUpload = async () => {
    if (!preRetestFile) {
      setErrorMsg('Upload a pre-retest XLSX before analyzing.');
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await uploadPreRetest(preRetestFile, costPerHour);
      setDfM12(res.records || []);
      setCostImpact(res.cost_impact);
      setPredictionSourceLabel('Uploaded pre-retest data');
      setOutcomesLoaded(false);
      setValidationData(null);
      setActiveOutcomes(null);
    } catch (e) {
      setErrorMsg(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClearAnalysis = () => {
    setDfM12([]);
    setCostImpact(null);
    setPredictionSourceLabel(null);
    setOutcomesLoaded(false);
    setValidationData(null);
    setActiveOutcomes(null);
    setView('overview');
  };

  const handleLoadOutcomes = async (useLocal = false) => {
    if (!useLocal && !outcomeFile) {
      setErrorMsg('Choose an outcomes XLSX before loading.');
      return;
    }
    if (!dfM12 || dfM12.length === 0) {
      setErrorMsg('Analyze or load pre-retest predictions first before validating outcomes.');
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    try {
      const predsJson = JSON.stringify(dfM12);
      const res = await validateOutcomes(useLocal ? null : outcomeFile, useLocal, predsJson);
      setValidationData(res);
      setOutcomesLoaded(true);
    } catch (e) {
      setErrorMsg(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const BackButton = () => (
    <button
      className="btn-secondary mb-4"
      onClick={() => setView('overview')}
      style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
    >
      ← Back to Overview
    </button>
  );

  // -------------------------------------------------------------
  // SUBVIEWS
  // -------------------------------------------------------------
  if (view === 'month12') {
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

    const probs = dfM12.map(r => Number(r['P(RETEST_BENEFICIAL)'] || 0));
    const meanP = probs.length ? (probs.reduce((a, b) => a + b, 0) / probs.length) * 100 : 0;
    const maxP = probs.length ? Math.max(...probs) * 100 : 0;
    const minP = probs.length ? Math.min(...probs) * 100 : 0;

    return (
      <div>
        <BackButton />
        <h3 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '6px' }}>Pre-Retest Analysis</h3>
        <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '20px' }}>
          Active data: {predictionSourceLabel}. Predictions use only pre-retest features. These recommendations are not claimed correct unless outcomes are loaded separately.
        </div>

        <div className="grid grid-cols-4 gap-4 mb-4">
          <KpiCard label="Total Events" value={dfM12.length} sub={predictionSourceLabel} />
          <KpiCard label="Average Probability" value={`${meanP.toFixed(2)}%`} sub="Lot mean P" color="#38bdf8" />
          <KpiCard label="Highest Probability" value={`${maxP.toFixed(2)}%`} sub="Max P" color="#10b981" />
          <KpiCard label="Lowest Probability" value={`${minP.toFixed(2)}%`} sub="Min P" color="#ef4444" />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="dark-card" style={{ padding: '16px' }}>
            <Plot
              data={[
                {
                  x: ['RETEST', "DON'T RETEST"],
                  y: [counts.retest, counts.dontRetest],
                  type: 'bar',
                  marker: { color: ['#10b981', '#ef4444'] },
                }
              ]}
              layout={{
                title: { text: 'AI Recommendation Distribution', font: { color: '#f1f5f9', size: 14 } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#f1f5f9' },
                xaxis: { gridcolor: '#1e293b' },
                yaxis: { gridcolor: '#1e293b' },
                margin: { l: 30, r: 20, t: 40, b: 30 },
                height: 240,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>

          <div className="dark-card" style={{ padding: '16px' }}>
            <Plot
              data={[
                {
                  x: dfM12.filter(r => r.AI_Recommendation === 'RETEST').map(r => r['P(RETEST_BENEFICIAL)']),
                  type: 'histogram',
                  name: 'RETEST',
                  marker: { color: '#10b981' },
                  nbinsx: 20,
                },
                {
                  x: dfM12.filter(r => r.AI_Recommendation !== 'RETEST').map(r => r['P(RETEST_BENEFICIAL)']),
                  type: 'histogram',
                  name: "DON'T RETEST",
                  marker: { color: '#ef4444' },
                  nbinsx: 20,
                }
              ]}
              layout={{
                title: { text: 'Probability Distribution', font: { color: '#f1f5f9', size: 14 } },
                barmode: 'stack',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#f1f5f9' },
                xaxis: { gridcolor: '#1e293b', title: 'P(RETEST_BENEFICIAL)' },
                yaxis: { gridcolor: '#1e293b', title: 'Count' },
                margin: { l: 30, r: 20, t: 40, b: 30 },
                height: 240,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        </div>

        {/* Filters */}
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

        {/* Table & Exports */}
        <div className="flex justify-between items-center mb-2">
          <div style={{ fontSize: '13px', color: '#94a3b8' }}>Showing {filtered.length} of {dfM12.length} events</div>
          <div className="flex gap-2">
            <button className="btn-secondary" onClick={() => exportToCSV(filtered, 'Retest_Recommendations.csv')}>
              Export CSV
            </button>
            <button className="btn-secondary" onClick={() => exportToExcel(filtered, 'Retest_Recommendations.xlsx', 'Recommendations')}>
              Export Excel
            </button>
          </div>
        </div>
        <EventTable data={filtered} height="360px" />
      </div>
    );
  }

  if (view === 'retest_recommendations' || view === 'dont_retest_recommendations') {
    const recLabel = view === 'retest_recommendations' ? 'RETEST' : "DON'T RETEST";
    const filtered = dfM12.filter(r => String(r.AI_Recommendation || '').trim() === recLabel);
    const color = REC_COLOR_MAP[recLabel];

    return (
      <div>
        <BackButton />
        <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#f1f5f9', marginBottom: '4px' }}>
          AI Recommended: {recLabel}
        </h3>
        <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
          {filtered.length} events recommended for {recLabel.toLowerCase()}
        </div>

        <div className="dark-card mb-4" style={{ padding: '16px' }}>
          <Plot
            data={[
              {
                x: filtered.map(r => r['P(RETEST_BENEFICIAL)']),
                type: 'histogram',
                marker: { color },
                nbinsx: 20,
              }
            ]}
            layout={{
              title: { text: `Probability Distribution of ${recLabel} Recommendations`, font: { color: '#f1f5f9', size: 14 } },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { family: 'Inter', color: '#f1f5f9' },
              xaxis: { gridcolor: '#1e293b', title: 'P(RETEST_BENEFICIAL)' },
              yaxis: { gridcolor: '#1e293b', title: 'Count' },
              margin: { l: 30, r: 20, t: 40, b: 30 },
              height: 240,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        <div className="flex gap-2 mb-4">
          <button className="btn-secondary" onClick={() => exportToCSV(filtered, `ai_recommended_${recLabel.toLowerCase().replace(/[^a-z]/g, '_')}_events.csv`)}>
            Export CSV
          </button>
          <button className="btn-secondary" onClick={() => exportToExcel(filtered, `ai_recommended_${recLabel.toLowerCase().replace(/[^a-z]/g, '_')}_events.xlsx`, `${recLabel} Events`)}>
            Export Excel
          </button>
        </div>

        <EventTable data={filtered} title={`${recLabel} events`} height="320px" />
      </div>
    );
  }

  if (view === 'all_device_cost' || view === 'ai_retest_cost') {
    if (!costImpact) {
      return (
        <div>
          <BackButton />
          <div style={{ color: '#94a3b8' }}>Upload a pre-retest workbook and analyze with AI to estimate cost.</div>
        </div>
      );
    }
    const isAiCost = view === 'ai_retest_cost';
    const shown = isAiCost
      ? dfM12.filter(r => String(r.AI_Recommendation || '').trim() === 'RETEST')
      : dfM12;

    const title = isAiCost ? 'AI predicted retest cost' : 'Actual cost of all devices';
    const valCol = isAiCost ? 'AI_Predicted_Retest_Cost' : 'Estimated_Retest_Cost';

    // Group by Fail_Test
    const byTestMap = {};
    shown.forEach(r => {
      const t = r.Fail_Test || 'Unknown';
      byTestMap[t] = (byTestMap[t] || 0) + Number(r[valCol] || 0);
    });
    const byTestX = Object.keys(byTestMap);
    const byTestY = byTestX.map(k => byTestMap[k]);

    return (
      <div>
        <BackButton />
        <h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '4px' }}>{title}</h3>
        <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '8px' }}>
          {isAiCost ? (
            `${formatMoney(costImpact.ai_predicted_retest_cost, 'USD')} · ${costImpact.retest_recommendations_count} RETEST events · ${formatSeconds(costImpact.ai_predicted_retest_time_sec)} · ${formatMoney(costImpact.cost_per_hour, 'USD')}/h`
          ) : (
            `${formatMoney(costImpact.all_device_retest_cost, 'USD')} · If every failure event is retested · ${costImpact.total_events} events · ${formatSeconds(costImpact.all_device_retest_time_sec)} · ${formatMoney(costImpact.cost_per_hour, 'USD')}/h`
          )}
        </div>
        <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '16px' }}>
          Duration is estimated from historical Retest_Time_sec by Fail_Test (Month 0 + Month 6). It is not actual Month 12 tester time and not used as a model feature.
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="dark-card" style={{ padding: '16px' }}>
            <Plot
              data={[
                {
                  x: ['All devices retested', 'AI recommended RETEST', 'Estimated savings'],
                  y: [costImpact.all_device_retest_cost, costImpact.ai_predicted_retest_cost, costImpact.estimated_savings],
                  type: 'bar',
                  marker: { color: ['#38bdf8', '#10b981', '#a855f7'] },
                }
              ]}
              layout={{
                title: { text: 'Tester-time cost comparison', font: { color: '#f1f5f9', size: 14 } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#f1f5f9' },
                xaxis: { gridcolor: '#1e293b' },
                yaxis: { gridcolor: '#1e293b' },
                margin: { l: 40, r: 20, t: 40, b: 30 },
                height: 240,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>

          <div className="dark-card" style={{ padding: '16px' }}>
            <Plot
              data={[
                {
                  x: byTestX,
                  y: byTestY,
                  type: 'bar',
                  marker: { color: '#38bdf8' },
                }
              ]}
              layout={{
                title: { text: `Estimated cost of ${isAiCost ? 'AI RETEST' : 'all-device'} events by Fail_Test`, font: { color: '#f1f5f9', size: 14 } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#f1f5f9' },
                xaxis: { gridcolor: '#1e293b' },
                yaxis: { gridcolor: '#1e293b' },
                margin: { l: 40, r: 20, t: 40, b: 30 },
                height: 240,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%' }}
            />
          </div>
        </div>

        <EventTable data={shown} title={title} height="320px" />
      </div>
    );
  }

  if (view === 'benefit_rate' || view === 'unnecessary_retests') {
    const kpis = validationData?.kpis;
    const isBenefit = view === 'benefit_rate';
    const events = isBenefit ? (validationData?.beneficial_events || []) : (validationData?.unnecessary_events || []);

    return (
      <div>
        <BackButton />
        <h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '4px' }}>
          {isBenefit ? 'Retest Benefit Rate' : 'Unnecessary Retests'}
        </h3>
        <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
          {isBenefit ? (
            `${kpis?.benefit_events_count || 0} / ${counts.retest} = ${kpis?.benefit_rate_pct ?? 0}% — Definition: share of AI RETEST recommendations whose actual outcome was RETEST_BENEFICIAL.`
          ) : (
            `${kpis?.fp || 0} / ${kpis?.total_events || 0} = ${kpis?.unnecessary_retests_pct ?? 0}% — Definition: events where AI recommended RETEST but the actual outcome was PERSISTENT_FAILURE.`
          )}
        </div>

        <div className="dark-card mb-4" style={{ padding: '16px' }}>
          <Plot
            data={[
              {
                x: [isBenefit ? 'RETEST_BENEFICIAL' : 'PERSISTENT_FAILURE'],
                y: [events.length],
                type: 'bar',
                marker: { color: isBenefit ? '#a855f7' : '#f59e0b' },
              }
            ]}
            layout={{
              title: { text: isBenefit ? 'Outcome of AI Recommended RETEST Events' : 'Unnecessary Retests: Outcome Was Persistent Failure', font: { color: '#f1f5f9', size: 14 } },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { family: 'Inter', color: '#f1f5f9' },
              xaxis: { gridcolor: '#1e293b' },
              yaxis: { gridcolor: '#1e293b' },
              margin: { l: 40, r: 20, t: 40, b: 30 },
              height: 240,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        <EventTable data={events} title={isBenefit ? 'Beneficial events' : 'Unnecessary Retest events'} height="320px" />
      </div>
    );
  }

  // -------------------------------------------------------------
  // DEFAULT OVERVIEW VIEW
  // -------------------------------------------------------------
  return (
    <div>
      <h3 style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-title)', marginBottom: '16px' }}>Overview</h3>

      {errorMsg && (
        <div style={{ padding: '10px 14px', borderRadius: '8px', background: 'var(--semantic-red-bg)', border: '1px solid var(--semantic-red)', color: 'var(--semantic-red)', marginBottom: '16px', fontSize: '13px' }}>
          {errorMsg}
        </div>
      )}

      {/* Top workflow row + upload toggle */}
      <div style={{ display: 'grid', gridTemplateColumns: '5fr 2fr', gap: '16px', marginBottom: '16px' }}>
        <div>
          <WorkflowPanel stage={workflowStage} />
        </div>
        <div>
          <button
            className="btn-primary w-full mb-2"
            onClick={() => setShowUpload(!showUpload)}
          >
            {showUpload ? '− Upload Pre-Retest Data' : '+ Upload Pre-Retest Data'}
          </button>

          {hasActiveAnalysis && (
            <button
              className="btn-secondary w-full mb-2"
              onClick={handleClearAnalysis}
            >
              Clear Analysis
            </button>
          )}

          {showUpload && (
            <div className="dark-card" style={{ padding: '16px' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
                Pre-retest events workbook (.xlsx)
              </div>
              <input
                type="file"
                accept=".xlsx"
                className="input-base mb-3"
                onChange={e => setPreRetestFile(e.target.files?.[0] || null)}
              />
              <button
                className="btn-primary w-full"
                onClick={handleAnalyzeUpload}
                disabled={loading || !preRetestFile}
              >
                {loading ? 'Analyzing...' : 'Analyze with AI'}
              </button>
            </div>
          )}
        </div>
      </div>

      {!hasActiveAnalysis ? (
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', textAlign: 'center', padding: '40px 0' }}>
          Upload a pre-retest workbook and select Analyze with AI to start analysis.
        </div>
      ) : (
        <>
          {/* AI Recommended Retest Cards */}
          <div className="dark-card-header">AI Recommended Retest</div>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <KpiEventsDevicesCard
              label="Total Events"
              events={counts.total}
              devices={counts.devices}
              onInspect={() => setView('month12')}
            />
            <KpiEventsDevicesCard
              label="RETEST"
              events={counts.retest}
              devices={counts.retestDevices}
              color="var(--semantic-green)"
              contextLabel="AI Recommended"
              onInspect={() => setView('retest_recommendations')}
            />
            <KpiEventsDevicesCard
              label="DON'T RETEST"
              events={counts.dontRetest}
              devices={counts.dontRetestDevices}
              color="var(--semantic-red)"
              contextLabel="AI Recommended"
              onInspect={() => setView('dont_retest_recommendations')}
            />
          </div>

          {/* Retest Cost Estimate Cards */}
          <div className="dark-card-header mt-6">Retest Cost Estimate</div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
            Cost = estimated retest time × {formatMoney(costPerHour, 'USD')}/h tester rate (configurable on Decision Policy). Time is estimated from historical Retest_Time_sec by Fail_Test, not from actual Month 12 retest duration.
          </div>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <KpiCostCard
              label="All-device retest cost"
              value={formatMoney(costImpact?.all_device_retest_cost || 0, 'USD')}
              sub={`${counts.total} events · ${formatSeconds(costImpact?.all_device_retest_time_sec || 0)}`}
              color="var(--accent-primary)"
              contextLabel="If every fail is retested"
              onInspect={() => setView('all_device_cost')}
            />
            <KpiCostCard
              label="AI predicted retest cost"
              value={formatMoney(costImpact?.ai_predicted_retest_cost || 0, 'USD')}
              sub={`${counts.retest} RETEST events · ${formatSeconds(costImpact?.ai_predicted_retest_time_sec || 0)}`}
              color="var(--semantic-green)"
              contextLabel="AI recommended RETEST"
              onInspect={() => setView('ai_retest_cost')}
            />
            <KpiCostCard
              label="Estimated savings"
              value={formatMoney(costImpact?.estimated_savings || 0, 'USD')}
              sub={`${counts.dontRetest} skipped events · ${formatSeconds(costImpact?.skipped_retest_time_sec || 0)}`}
              color="var(--semantic-purple)"
              contextLabel="All-device minus AI retest"
            />
          </div>

          {/* Validation of AI Recommended Retest */}
          <div className="dark-card-header mt-6">Validation of AI Recommended Retest</div>
          <div className="grid grid-cols-3 gap-4 mb-6">
            {outcomesLoaded && validationData ? (
              <KpiCard
                label="Retest Benefit Rate"
                value={`${validationData.kpis?.benefit_rate_pct ?? 0}%`}
                sub={`${validationData.kpis?.benefit_events_count || 0} events · ${validationData.kpis?.benefit_devices_count || 0} devices`}
                color="var(--semantic-purple)"
                onInspect={() => setView('benefit_rate')}
              />
            ) : (
              <div className="dark-card h-full flex items-center justify-center text-center p-4" style={{ marginBottom: 0 }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Outcomes are not loaded for the current analysis. Retest Benefit Rate is not shown.
                </span>
              </div>
            )}

            {outcomesLoaded && validationData ? (
              <KpiCard
                label="Unnecessary Retests"
                value={`${validationData.kpis?.unnecessary_retests_pct ?? 0}%`}
                sub={`${validationData.kpis?.fp || 0} events · ${validationData.kpis?.unnecessary_devices_count || 0} devices`}
                color="var(--semantic-amber)"
                onInspect={() => setView('unnecessary_retests')}
              />
            ) : (
              <div className="dark-card h-full flex items-center justify-center text-center p-4" style={{ marginBottom: 0 }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Outcomes are not loaded for the current analysis. Unnecessary Retests is not shown.
                </span>
              </div>
            )}

            <div className="dark-card" style={{ padding: '16px', marginBottom: 0 }}>
              <div className="dark-card-header" style={{ marginBottom: '6px' }}>UPLOAD OUTCOME DATA</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                Upload actual post-retest outcomes for validation only. Never used as AI prediction input.
              </div>
              <input
                type="file"
                accept=".xlsx"
                className="input-base mb-2"
                onChange={e => setOutcomeFile(e.target.files?.[0] || null)}
              />
              <button
                className="btn-primary w-full mb-2"
                style={{ padding: '6px 12px', fontSize: '12px' }}
                onClick={() => handleLoadOutcomes(false)}
                disabled={loading || !outcomeFile}
              >
                Load uploaded outcomes
              </button>
              <button
                className="btn-secondary w-full"
                style={{ padding: '6px 12px', fontSize: '12px' }}
                onClick={() => handleLoadOutcomes(true)}
                disabled={loading}
              >
                Load local private outcomes file if present
              </button>
            </div>
          </div>

          {/* Online Learning Panel */}
          <OnlineLearningPanel
            m12Val={validationData?.joined_records}
            m12HasOutcomes={outcomesLoaded}
            onLearned={res => {
              if (res && res.learned > 0) {
                setValidationData(prev => ({ ...prev, learned: true }));
              }
            }}
          />

          {/* Model Quality & Test Family Breakdown */}
          <div className="dark-card mt-6" style={{ padding: '20px' }}>
            <div className="dark-card-header">Model Quality</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '16px' }}>
              Historical Temporal Validation — Month 0 train / Month 6 holdout. These metrics are not the current upload's performance.
            </div>

            {histValidation && (
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <KpiCard label="Active Model" value={histValidation.best_model_name || 'XGBoost'} sub="Selected from Month 6 holdout" color="var(--semantic-purple)" />
                  <div style={{ fontSize: '13px', lineHeight: 1.8, marginTop: '16px' }}>
                    {(() => {
                      const m = histValidation.models?.[histValidation.best_model_name]?.calibrated_metrics || {};
                      return (
                        <>
                          <div>• <b>Precision:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{((m.Precision || 0) * 100).toFixed(1)}%</code></div>
                          <div>• <b>Recall:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{((m.Recall || 0) * 100).toFixed(1)}%</code></div>
                          <div>• <b>Specificity:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{((m.Specificity || 0) * 100).toFixed(1)}%</code></div>
                          <div>• <b>ROC-AUC:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{(m['ROC-AUC'] || 0).toFixed(3)}</code></div>
                          <div>• <b>PR-AUC:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{(m['PR-AUC'] || 0).toFixed(3)}</code></div>
                          <div>• <b>Brier Score:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{(m['Brier Score'] || 0).toFixed(4)}</code></div>
                          <div>• <b>Log Loss:</b> <code className="mono" style={{ color: 'var(--semantic-purple)', fontWeight: 600 }}>{(m['Log Loss'] || 0).toFixed(4)}</code></div>
                        </>
                      );
                    })()}
                  </div>
                </div>

                <div>
                  <div className="dark-card-header" style={{ marginBottom: '6px' }}>Test Type Breakdown</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>
                    {predictionSourceLabel}: failure events by test family
                  </div>
                  {(() => {
                    const famOrder = ['Scan', 'Func', 'MBIST', 'IDDQ', 'AtSpeed'];
                    const famMap = { Scan: 0, Func: 0, MBIST: 0, IDDQ: 0, AtSpeed: 0 };
                    dfM12.forEach(r => {
                      const fam = getTestFamily(r.Fail_Test);
                      if (famMap[fam] !== undefined) famMap[fam]++;
                    });
                    return (
                      <Plot
                        data={[
                          {
                            x: famOrder,
                            y: famOrder.map(f => famMap[f]),
                            type: 'bar',
                            marker: { color: ['#8b5cf6', '#38bdf8', '#10b981', '#f59e0b', '#ec4899'] },
                          }
                        ]}
                        layout={{
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                          font: { family: 'Inter', color: '#f1f5f9' },
                          xaxis: { gridcolor: '#1e293b' },
                          yaxis: { gridcolor: '#1e293b' },
                          margin: { l: 30, r: 20, t: 20, b: 30 },
                          height: 220,
                        }}
                        config={{ responsive: true, displayModeBar: false }}
                        style={{ width: '100%' }}
                      />
                    );
                  })()}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
