"use client";

import { useEffect, useState } from "react";
import {
  RETEST_DEMO_EVENTS,
  type RetestDemoEvent,
} from "@/lib/retestDemoEvents";

const ROW_H = 34;
const ROW_H_COMPACT = 18;
const COMPACT_ROWS = 8;
const COMPACT_COL_HEADER_H = 22;
const COLS =
  "grid-cols-[36px_68px_52px_78px_56px_58px_52px_62px_58px_88px_92px]";

/**
 * Flush RETEST events table with vertical live conveyor.
 * Presentation dataset matching Retest AI Agent table layout.
 */
export function RetestLiveFeed({
  compact = false,
}: {
  /** Smaller height for embedding inside the Optimization KPI card. */
  compact?: boolean;
}) {
  const rows = RETEST_DEMO_EVENTS;
  const [paused, setPaused] = useState(false);
  const [offset, setOffset] = useState(0);
  const rowH = compact ? ROW_H_COMPACT : ROW_H;
  /** Compact: 8 dense rows (+ sticky column header) — ~half prior height. */
  const feedH = compact
    ? COMPACT_COL_HEADER_H + COMPACT_ROWS * ROW_H_COMPACT
    : 260;

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced || paused || rows.length < 2) return;

    let raf = 0;
    let last = performance.now();
    const speed = compact ? 10 : 14;

    const loop = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setOffset((o) => {
        const next = o + speed * dt;
        const cycle = rows.length * rowH;
        return cycle > 0 ? next % cycle : 0;
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [paused, rows.length, compact, rowH]);

  const doubled = [...rows, ...rows];

  return (
    <div
      className={`overflow-hidden border border-[rgba(107,193,242,0.18)] bg-[#070b12] ${
        compact ? "shrink-0 rounded-[5px]" : "rounded-[8px]"
      }`}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
    >
      <div
        className={`flex shrink-0 items-center justify-between gap-2 border-b border-[rgba(107,193,242,0.16)] ${
          compact ? "px-1.5 py-1" : "px-3 py-2"
        }`}
      >
        <span
          className={`font-semibold tracking-[0.04em] text-[var(--text-bright)] ${
            compact ? "text-[9px]" : "text-[11px]"
          }`}
        >
          RETEST events
        </span>
        <span className="vl-live-badge">
          <span className="vl-live-dot" />
          LIVE
        </span>
      </div>

      <div
        className="vl-feed-mask relative overflow-x-auto overflow-y-hidden"
        style={{ height: feedH }}
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        <div className="min-w-[920px]">
          <div
            className={`sticky top-0 z-[1] grid ${COLS} gap-x-1 border-b border-[rgba(107,193,242,0.2)] bg-[#0c121c] px-2 font-semibold uppercase tracking-[0.04em] text-[#9ec9ef] ${
              compact
                ? "py-0.5 text-[8px]"
                : "py-1.5 text-[9px]"
            }`}
            style={compact ? { height: COMPACT_COL_HEADER_H } : undefined}
          >
            <span>S.No</span>
            <span>Device_ID</span>
            <span>Failure_Event</span>
            <span>Fail_Test</span>
            <span>Fail_Bin</span>
            <span>Wafer_ID</span>
            <span>ATE_Site</span>
            <span>Voltage_V</span>
            <span>Temperature_C</span>
            <span>P(RETEST_BENEFICIAL)</span>
            <span>AI_Recommendation</span>
          </div>

          <div
            className="will-change-transform"
            style={{ transform: `translateY(-${offset}px)` }}
          >
            {doubled.map((row, i) => (
              <TableRow
                key={`${row.deviceId}-${row.sno}-${i}`}
                row={row}
                height={rowH}
                compact={compact}
              />
            ))}
          </div>
        </div>
      </div>

      {!compact ? (
        <p className="border-t border-[rgba(107,193,242,0.12)] px-3 py-1.5 text-[10px] text-[var(--muted-2)]">
          Hover to pause · Open Retest AI Agent for full analysis
        </p>
      ) : null}
    </div>
  );
}

function TableRow({
  row,
  height,
  compact,
}: {
  row: RetestDemoEvent;
  height: number;
  compact: boolean;
}) {
  const retest = row.aiRecommendation === "RETEST";
  return (
    <div
      className={`vl-enter-row grid ${COLS} items-center gap-x-1 border-t border-[rgba(107,193,242,0.08)] px-2 text-[#c8d6e6] first:border-t-0 ${
        compact ? "text-[9px]" : "text-[10.5px]"
      }`}
      style={{ height }}
    >
      <span className="font-mono text-[var(--muted-2)]">{row.sno}</span>
      <span className="font-mono text-[var(--text)]">{row.deviceId}</span>
      <span className="font-mono">{row.failureEvent}</span>
      <span className="truncate font-mono text-[var(--text)]">{row.failTest}</span>
      <span className="font-mono">{row.failBin}</span>
      <span className="font-mono">{row.waferId}</span>
      <span className="font-mono">{row.ateSite}</span>
      <span className="font-mono">{row.voltageV.toFixed(2)}</span>
      <span className="font-mono">{row.temperatureC}</span>
      <span className="font-mono text-[var(--text-bright)]">
        {row.pRetestBeneficial.toFixed(4)}
      </span>
      <span
        className={`font-semibold uppercase tracking-[0.04em] ${
          retest ? "text-[var(--green)]" : "text-[var(--red)]"
        }`}
      >
        {row.aiRecommendation}
      </span>
    </div>
  );
}
