import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import TopHeader from './components/TopHeader';
import { OverviewTab } from './tabs/OverviewTab';
import { SingleEventTab } from './tabs/SingleEventTab';
import { BatchInferenceTab } from './tabs/BatchInferenceTab';
import { HistoricalValidationTab } from './tabs/HistoricalValidationTab';
import { ReferenceAuditTab } from './tabs/ReferenceAuditTab';
import { getMonth12Batch, getHistoricalValidation } from './services/api';

export default function App() {
  const [currentPage, setCurrentPage] = useState('overview');
  const [costPerHour, setCostPerHour] = useState(350.0);
  
  // Theme state
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('retest_ai_theme') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('retest_ai_theme', theme);
  }, [theme]);

  // Data state
  const [dfM12, setDfM12] = useState([]);
  const [costImpact, setCostImpact] = useState(null);
  const [predictionSourceLabel, setPredictionSourceLabel] = useState('Month 12 (unseen inference)');
  const [activeOutcomes, setActiveOutcomes] = useState(null);
  const [outcomesLoaded, setOutcomesLoaded] = useState(false);
  const [validationData, setValidationData] = useState(null);
  const [histValidation, setHistValidation] = useState(null);
  const [loading, setLoading] = useState(true);

  // Initial load
  useEffect(() => {
    Promise.all([
      getMonth12Batch(costPerHour).then(res => {
        setDfM12(res.records || []);
        setCostImpact(res.cost_impact);
      }),
      getHistoricalValidation().then(res => {
        setHistValidation(res);
      }),
    ])
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Recalculate cost impact when costPerHour changes
  useEffect(() => {
    if (dfM12 && dfM12.length > 0) {
      getMonth12Batch(costPerHour).then(res => {
        setCostImpact(res.cost_impact);
      }).catch(console.error);
    }
  }, [costPerHour]);

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: 'var(--bg-main)' }}>
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />

      <main style={{ flex: 1, height: '100vh', overflowY: 'auto', padding: '1.5rem 2rem 2rem 2rem' }}>
        <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
          <TopHeader theme={theme} setTheme={setTheme} />

          {currentPage === 'overview' && (
            <OverviewTab
              dfM12={dfM12}
              setDfM12={setDfM12}
              costImpact={costImpact}
              setCostImpact={setCostImpact}
              costPerHour={costPerHour}
              predictionSourceLabel={predictionSourceLabel}
              setPredictionSourceLabel={setPredictionSourceLabel}
              activeOutcomes={activeOutcomes}
              setActiveOutcomes={setActiveOutcomes}
              outcomesLoaded={outcomesLoaded}
              setOutcomesLoaded={setOutcomesLoaded}
              validationData={validationData}
              setValidationData={setValidationData}
              histValidation={histValidation}
            />
          )}

          {currentPage === 'single' && (
            <SingleEventTab costPerHour={costPerHour} />
          )}

          {currentPage === 'batch' && (
            <BatchInferenceTab dfM12={dfM12} costPerHour={costPerHour} />
          )}

          {currentPage === 'models' && (
            <HistoricalValidationTab histValidation={histValidation} />
          )}

          {currentPage === 'info' && (
            <ReferenceAuditTab costPerHour={costPerHour} setCostPerHour={setCostPerHour} mode="info" />
          )}

          {currentPage === 'settings' && (
            <ReferenceAuditTab costPerHour={costPerHour} setCostPerHour={setCostPerHour} mode="settings" />
          )}

          {currentPage === 'reference' && (
            <ReferenceAuditTab costPerHour={costPerHour} setCostPerHour={setCostPerHour} mode="audit" />
          )}
        </div>
      </main>
    </div>
  );
}
