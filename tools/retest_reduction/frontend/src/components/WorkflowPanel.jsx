import React from 'react';
import { ArrowRight, CheckCircle2, Circle, CircleDot } from 'lucide-react';

export const WorkflowPanel = ({ stage }) => {
  let marks = ["now", "todo", "todo", "todo", "todo", "todo", "todo", "todo"];
  if (stage === "learned") {
    marks = ["done", "done", "done", "done", "done", "done", "done", "now"];
  } else if (stage === "validate") {
    marks = ["done", "done", "done", "done", "done", "done", "now", "todo"];
  } else if (stage === "recommend") {
    marks = ["done", "done", "now", "now", "todo", "todo", "todo", "todo"];
  }

  const labels = [
    <>Upload Pre-Retest Data</>,
    <>Analyze with AI</>,
    <>AI Recommendation &mdash; <span style={{ color: 'var(--semantic-green)', fontWeight: 600 }}>RETEST</span> or <span style={{ color: 'var(--semantic-red)', fontWeight: 600 }}>DON'T RETEST</span></>,
    <>Estimate Retest Cost &mdash; all-device vs AI</>,
    <>Perform Actual Retest</>,
    <>Upload Actual Outcomes</>,
    <>Validate AI Recommendation</>,
    <>Click <b style={{ color: 'var(--text-title)' }}>Learn from These Validated Outcomes</b> &mdash; RLS</>,
  ];

  return (
    <div className="dark-card h-full" style={{ display: 'flex', flexDirection: 'column', minHeight: '320px', padding: '18px 20px' }}>
      <div className="dark-card-header">
        <span>How AI Analysis Works</span>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'none' }}>
          8-stage operational pipeline
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 16px', alignItems: 'center', flex: 1, margin: '8px 0' }}>
        {labels.map((label, idx) => {
          const mark = marks[idx];
          const i = idx + 1;
          const arrow = i > 1 ? <ArrowRight size={14} style={{ color: 'var(--text-dark)', marginRight: '6px', flexShrink: 0 }} /> : null;
          
          let Icon;
          let containerStyle = {
            fontSize: '13px',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            lineHeight: 1.3,
          };
          
          if (mark === "done") {
            containerStyle.color = 'var(--semantic-green)';
            containerStyle.fontWeight = 600;
            Icon = <CheckCircle2 size={15} style={{ color: 'var(--semantic-green)', flexShrink: 0 }} />;
          } else if (mark === "now") {
            containerStyle.color = 'var(--accent-primary)';
            containerStyle.fontWeight = 600;
            containerStyle.background = 'var(--accent-primary-subtle)';
            containerStyle.border = '1px solid var(--border-accent)';
            containerStyle.borderRadius = '6px';
            containerStyle.padding = '6px 10px';
            Icon = <CircleDot size={15} style={{ color: 'var(--accent-primary)', flexShrink: 0 }} />;
          } else {
            containerStyle.color = 'var(--text-muted)';
            Icon = <Circle size={15} style={{ color: 'var(--text-dark)', flexShrink: 0 }} />;
          }

          return (
            <div key={idx} style={{ display: 'flex', alignItems: 'center' }}>
              {arrow}
              <div style={containerStyle}>
                {Icon}
                <span>{i}. {label}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '12px', lineHeight: 1.4, borderTop: '1px solid var(--border-subtle)', paddingTop: '10px' }}>
        The AI recommends whether a retest may be beneficial and estimates tester-time cost. After testing, upload outcomes to validate. Online learning (RLS) activates only when you explicitly trigger it.
      </div>
    </div>
  );
};
