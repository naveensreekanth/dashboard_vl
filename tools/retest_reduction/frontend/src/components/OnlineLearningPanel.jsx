import React, { useState, useEffect } from 'react';
import { getOnlineLearningStatus, learnFromOutcomes, resetOnlineLearning } from '../services/api';

export const OnlineLearningPanel = ({ m12Val, m12HasOutcomes, onLearned, onReset }) => {
  const [status, setStatus] = useState(null);
  const [flash, setFlash] = useState(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchStatus = async () => {
    try {
      const s = await getOnlineLearningStatus();
      setStatus(s);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleLearn = async () => {
    if (!m12Val || m12Val.length === 0) return;
    setLoading(true);
    try {
      const res = await learnFromOutcomes(m12Val);
      if (res.active) {
        setFlash({ type: 'success', msg: `Online learning updated with ${res.learned || 0} validated events. The adaptation layer is active.` });
      } else if (res.already_learned) {
        setFlash({ type: 'info', msg: 'These validated outcomes have already been used for online learning.' });
      } else {
        setFlash({ type: 'success', msg: `Online learning has collected ${res.update_count || res.learned || 0} validated events. It will begin adapting predictions after ${res.activation_threshold || 20} events.` });
      }
      fetchStatus();
      if (onLearned) onLearned(res);
    } catch (e) {
      setFlash({ type: 'warning', msg: e.response?.data?.detail || e.message });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      const res = await resetOnlineLearning();
      setFlash({ type: 'info', msg: 'Online learning was reset. Future predictions use the base model until new validated outcomes are learned.' });
      setConfirmReset(false);
      fetchStatus();
      if (onReset) onReset(res);
    } catch (e) {
      setFlash({ type: 'warning', msg: e.response?.data?.detail || e.message });
    } finally {
      setLoading(false);
    }
  };

  const updateCount = status ? Number(status.update_count || 0) : 0;
  const threshold = status ? Number(status.activation_threshold || 20) : 20;
  const active = status ? Boolean(status.active) : false;
  const baseModel = status?.model_name || 'XGBoost';
  const forgettingFactor = status?.forgetting_factor ?? 0.95;

  const learnedDisplay = active ? `${updateCount}` : `${updateCount} / ${threshold}`;
  const adaptationLabel = active ? 'Active' : 'Warming Up';
  const adaptColor = active ? 'var(--semantic-green)' : 'var(--semantic-amber)';
  const behavior = active
    ? 'Future probabilities are adjusted using recent approved post-retest outcomes.'
    : 'Predictions currently use the base model until enough validated outcomes are learned.';

  return (
    <div className="dark-card-compact" style={{ marginTop: '16px' }}>
      <div className="dark-card-header" style={{ marginBottom: '8px' }}>ONLINE LEARNING</div>
      
      {m12HasOutcomes && m12Val && m12Val.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <button
            className="btn-primary"
            style={{ width: '100%', padding: '8px 12px' }}
            onClick={handleLearn}
            disabled={loading}
          >
            Learn from These Validated Outcomes
          </button>
        </div>
      )}

      {flash && (
        <div
          style={{
            padding: '10px 14px',
            borderRadius: '6px',
            marginBottom: '12px',
            fontSize: '12px',
            border: `1px solid ${flash.type === 'success' ? 'var(--semantic-green)' : flash.type === 'warning' ? 'var(--semantic-amber)' : 'var(--border-accent)'}`,
            backgroundColor: flash.type === 'success' ? 'var(--semantic-green-bg)' : flash.type === 'warning' ? 'var(--semantic-amber-bg)' : 'var(--accent-primary-subtle)',
            color: flash.type === 'success' ? 'var(--semantic-green)' : flash.type === 'warning' ? 'var(--semantic-amber)' : 'var(--accent-primary)',
          }}
        >
          {flash.msg}
        </div>
      )}

      <div className="stat-row">
        <span className="stat-label">Base Model</span>
        <span className="stat-value mono">{baseModel}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Validated Events Learned</span>
        <span className="stat-value mono">{learnedDisplay}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Adaptation</span>
        <span className="stat-value font-bold" style={{ color: adaptColor }}>{adaptationLabel}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Forgetting Factor</span>
        <span className="stat-value mono">{forgettingFactor}</span>
      </div>
      
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '10px', lineHeight: 1.4 }}>
        {behavior}
      </div>

      <div style={{ marginTop: '14px' }}>
        {confirmReset ? (
          <div>
            <div style={{ fontSize: '12px', color: 'var(--semantic-amber)', marginBottom: '8px', lineHeight: 1.3 }}>
              Resetting online learning returns future predictions to the base model until new validated outcomes are learned. This does not delete the trained model, uploaded predictions, outcomes, or validation results.
            </div>
            <div className="flex gap-2">
              <button
                className="btn-primary"
                style={{ background: 'var(--semantic-red)', borderColor: 'var(--semantic-red)', flex: 1, padding: '6px 12px', fontSize: '12px' }}
                onClick={handleReset}
                disabled={loading}
              >
                Confirm Reset
              </button>
              <button
                className="btn-secondary"
                style={{ flex: 1, padding: '6px 12px', fontSize: '12px' }}
                onClick={() => setConfirmReset(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            className="btn-secondary"
            style={{ width: '100%', padding: '6px 12px', fontSize: '12px' }}
            onClick={() => setConfirmReset(true)}
          >
            Reset Online Learning
          </button>
        )}
      </div>
    </div>
  );
};
