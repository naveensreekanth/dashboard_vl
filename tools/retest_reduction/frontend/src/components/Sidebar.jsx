import React from 'react';
import clsx from 'clsx';
import { Activity, Cpu, Database, FileCheck, Layers, Settings } from 'lucide-react';

const Sidebar = ({ currentPage, setCurrentPage }) => {
  const navItems = [
    { section: 'Dashboard' },
    { id: 'overview', label: 'Overview Dashboard', icon: Activity },
    
    { section: 'Analysis' },
    { id: 'single', label: 'Single Event Analysis', icon: Cpu },
    
    { section: 'System' },
    { id: 'settings', label: 'Decision Policy & Audit', icon: Settings },
  ];

  return (
    <aside
      style={{
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-subtle)',
        width: '260px',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
      {/* Brand Header */}
      <div
        style={{
          padding: '16px 20px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}
      >
        <div
          style={{
            width: '28px',
            height: '28px',
            borderRadius: '6px',
            backgroundColor: 'var(--accent-primary-subtle)',
            border: '1px solid var(--border-card)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-primary)',
          }}
        >
          <Activity size={16} />
        </div>
        <div>
          <div style={{ fontSize: '13px', fontWeight: 700, letterSpacing: '0.3px', color: '#F1F5F9', lineHeight: 1.1 }}>
            ATE RETEST AI
          </div>
        </div>
      </div>

      {/* Nav List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 8px' }}>
        {navItems.map((item, idx) => {
          if (item.section) {
            return (
              <div
                key={idx}
                style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                  color: '#64748B',
                  marginTop: idx === 0 ? '4px' : '16px',
                  marginBottom: '6px',
                  paddingLeft: '12px',
                }}
              >
                {item.section}
              </div>
            );
          }

          const Icon = item.icon;
          const isActive = currentPage === item.id;

          return (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={clsx('nav-item', isActive && 'active')}
            >
              <Icon size={15} style={{ opacity: isActive ? 1 : 0.7 }} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* System Status Footer - Clean dot, no glow */}
      <div
        style={{
          padding: '12px 16px',
          borderTop: '1px solid var(--border-subtle)',
          background: 'var(--bg-input)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: 'var(--semantic-green)',
            }}
          />
          <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)' }}>Model Active</span>
        </div>
        <span className="mono" style={{ fontSize: '11px', color: 'var(--text-dark)' }}>XGBoost</span>
      </div>
    </aside>
  );
};

export default Sidebar;
