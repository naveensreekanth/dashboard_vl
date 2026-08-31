import React from 'react';

const EVENT_DETAIL_COLS = [
  "Device_ID", "Failure_Event", "Fail_Test", "Fail_Bin", "Wafer_ID", "ATE_Site",
  "Voltage_V", "Temperature_C", "First_Test_Time_sec", "Estimated_Retest_Time_sec",
  "Estimated_Retest_Cost", "AI_Predicted_Retest_Cost",
  "P(RETEST_BENEFICIAL)", "AI_Recommendation", "Ground_Truth"
];

export const EventTable = ({ data, title, height = '280px' }) => {
  if (!data || data.length === 0) {
    return (
      <div>
        {title && <div style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '8px' }}>{title}</div>}
        <div style={{ padding: '12px', background: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', borderRadius: '8px', border: '1px solid #38bdf8' }}>
          No events in this cell.
        </div>
      </div>
    );
  }

  const columns = EVENT_DETAIL_COLS.filter(c => Object.keys(data[0]).includes(c));
  
  return (
    <div className="flex-col w-full">
      {title && <div style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '8px' }}>{title}</div>}
      <div style={{ height, overflow: 'auto', background: '#0d1526', border: '1px solid #1e2c4a', borderRadius: '10px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
          <thead style={{ position: 'sticky', top: 0, background: '#111a2d', zIndex: 1, borderBottom: '1px solid #1e2c4a' }}>
            <tr>
              <th style={{ padding: '10px 12px', color: '#94a3b8', fontWeight: 600, borderRight: '1px solid #1e2c4a' }}>S.No</th>
              {columns.map(col => (
                <th key={col} style={{ padding: '10px 12px', color: '#94a3b8', fontWeight: 600, borderRight: '1px solid #1e2c4a', whiteSpace: 'nowrap' }}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #1a263d', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.02)' }}>
                <td style={{ padding: '8px 12px', color: '#64748b', borderRight: '1px solid #1a263d' }}>{i + 1}</td>
                {columns.map(col => (
                  <td key={col} style={{ padding: '8px 12px', color: '#f1f5f9', borderRight: '1px solid #1a263d', whiteSpace: 'nowrap' }}>
                    {typeof row[col] === 'number' && !Number.isInteger(row[col]) ? row[col].toFixed(4) : row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
