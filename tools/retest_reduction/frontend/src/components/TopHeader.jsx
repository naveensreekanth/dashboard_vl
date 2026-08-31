import React from 'react';
import { Sun, Moon } from 'lucide-react';

const TopHeader = ({ theme = 'dark', setTheme }) => {
  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    if (setTheme) {
      setTheme(nextTheme);
    }
  };

  return (
    <header
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingBottom: '16px',
        marginBottom: '20px',
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <div>
        <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-title)', letterSpacing: '-0.02em' }}>
          ATE Retest-Benefit Prediction AI
        </div>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div
          style={{
            background: 'var(--accent-primary-subtle)',
            border: '1px solid var(--border-accent)',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '11px',
            fontWeight: 600,
            color: 'var(--accent-primary)',
          }}
        >
          Policy: P ≥ 0.30 → RETEST
        </div>

        <div
          style={{
            background: 'var(--semantic-green-bg)',
            border: '1px solid var(--semantic-green)',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '11px',
            fontWeight: 600,
            color: 'var(--semantic-green)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--semantic-green)' }} />
          Inference Ready
        </div>

        {/* Light / Dark Mode Toggle */}
        <button
          onClick={toggleTheme}
          className="btn-secondary"
          style={{
            padding: '4px 10px',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            borderRadius: '6px',
          }}
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} theme`}
        >
          {theme === 'dark' ? (
            <>
              <Sun size={14} style={{ color: '#F59E0B' }} />
              <span>Light</span>
            </>
          ) : (
            <>
              <Moon size={14} style={{ color: '#0284C7' }} />
              <span>Dark</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};

export default TopHeader;
