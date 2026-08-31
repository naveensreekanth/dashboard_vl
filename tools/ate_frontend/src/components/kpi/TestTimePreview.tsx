"use client";

import { useEffect, useId, useMemo, useState } from "react";
import {
  TEST_TIME_COMPARE_METRICS,
  TEST_TIME_LIVE_MEMORY,
  TEST_TIME_LIVE_MEMORY_COMPACT,
  type LiveMemoryPoint,
  type TestTimeCompareMetric,
} from "@/lib/testTimeDemo";

const W = 640;
const H = 220;
const PAD = { top: 18, right: 14, bottom: 36, left: 36 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;
const Y_MAX = 16;

/**
 * Test Time Optimization preview — comparison bars + live vector memory chart.
 * Bars/values flash in; chart draws with a cycle-reset flash.
 */
export function TestTimePreview({
  compact = false,
}: {
  compact?: boolean;
}) {
  const series = compact ? TEST_TIME_LIVE_MEMORY_COMPACT : TEST_TIME_LIVE_MEMORY;
  const [reveal, setReveal] = useState(0);
  const [resetFlash, setResetFlash] = useState(false);
  const uid = useId().replace(/:/g, "");

  const resetAt = useMemo(() => {
    const mid = series.findIndex(
      (p, idx) => idx > 2 && p.tick === 1 && p.without === 0 && p.withAgent === 0,
    );
    return mid > 0 ? mid / (series.length - 1) : 0.48;
  }, [series]);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setReveal(1);
      return;
    }
    setReveal(0);
    setResetFlash(false);
    let raf = 0;
    let flashed = false;
    const start = performance.now();
    const dur = compact ? 1100 : 1600;
    const loop = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      setReveal(t);
      if (!flashed && t >= resetAt) {
        flashed = true;
        setResetFlash(true);
      }
      if (t < 1) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [compact, resetAt]);

  const visibleCount = Math.max(
    2,
    Math.floor(1 + reveal * (series.length - 1)),
  );
  const visible = series.slice(0, visibleCount);

  return (
    <div
      className={`flex flex-col ${compact ? "gap-1.5" : "gap-3"}`}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
    >
      <div className={`grid grid-cols-2 ${compact ? "gap-1.5" : "gap-2.5"}`}>
        {TEST_TIME_COMPARE_METRICS.map((m, i) => (
          <CompareCard
            key={m.id}
            metric={m}
            compact={compact}
            delayMs={i * 90}
          />
        ))}
      </div>

      <LiveMemoryChart
        points={visible}
        all={series}
        compact={compact}
        uid={uid}
        resetFlash={resetFlash}
        resetAt={resetAt}
      />
    </div>
  );
}

function CompareCard({
  metric,
  compact,
  delayMs,
}: {
  metric: TestTimeCompareMetric;
  compact: boolean;
  delayMs: number;
}) {
  const optimized = metric.tone === "optimized";
  return (
    <div
      className={`vl-tto-compare-flash overflow-hidden border border-[rgba(107,193,242,0.2)] bg-[#070b12] ${
        compact ? "rounded-[5px] px-2 py-1.5" : "rounded-[8px] px-3 py-2.5"
      }`}
      style={{ animationDelay: `${delayMs}ms` }}
    >
      <div
        className={`leading-snug text-[var(--text-soft)] ${
          compact ? "text-[8px]" : "text-[10px]"
        }`}
      >
        {metric.label}
      </div>
      <div
        className={`vl-num-flash mt-0.5 font-mono font-semibold tabular-nums ${
          compact ? "text-[12px]" : "text-[16px]"
        } ${optimized ? "text-[var(--green)]" : "text-[var(--cyan)]"}`}
        style={{ animationDelay: `${delayMs + 120}ms` }}
      >
        {metric.value.toFixed(2)} {metric.unit}
      </div>
      <div
        className={`mt-1.5 overflow-hidden rounded-sm bg-[rgba(107,193,242,0.12)] ${
          compact ? "h-1.5" : "h-2"
        }`}
      >
        <div
          className={`vl-tto-bar-fill h-full rounded-sm ${
            optimized ? "bg-[var(--green)]" : "bg-[var(--cyan)]"
          }`}
          style={{
            width: `${Math.round(metric.bar * 100)}%`,
            animationDelay: `${delayMs + 80}ms`,
          }}
        />
      </div>
    </div>
  );
}

function LiveMemoryChart({
  points,
  all,
  compact,
  uid,
  resetFlash,
  resetAt,
}: {
  points: LiveMemoryPoint[];
  all: LiveMemoryPoint[];
  compact: boolean;
  uid: string;
  resetFlash: boolean;
  resetAt: number;
}) {
  const blueId = `tto-blue-${uid}`;
  const greenId = `tto-green-${uid}`;

  const paths = useMemo(() => {
    const n = Math.max(1, all.length - 1);
    const x = (i: number) => PAD.left + (i / n) * PLOT_W;
    const y = (v: number) =>
      PAD.top + PLOT_H * (1 - Math.min(Y_MAX, Math.max(0, v)) / Y_MAX);

    const line = (key: "without" | "withAgent") =>
      points
        .map(
          (p, idx) =>
            `${idx === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p[key]).toFixed(1)}`,
        )
        .join(" ");

    const area = (key: "without" | "withAgent") => {
      if (points.length < 2) return "";
      const top = points
        .map(
          (p, idx) =>
            `${idx === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p[key]).toFixed(1)}`,
        )
        .join(" ");
      const last = points[points.length - 1];
      const first = points[0];
      return `${top} L${x(last.i).toFixed(1)},${y(0).toFixed(1)} L${x(first.i).toFixed(1)},${y(0).toFixed(1)} Z`;
    };

    return {
      withoutLine: line("without"),
      withLine: line("withAgent"),
      withoutArea: area("without"),
      withArea: area("withAgent"),
    };
  }, [points, all.length]);

  const xTicks = useMemo(() => {
    const ticks: { i: number; label: string }[] = [];
    const mid = all.findIndex(
      (p, idx) => idx > 0 && p.tick === 1 && p.without === 0,
    );
    const pushEvery = (from: number, to: number, step: number) => {
      for (let i = from; i < to; i += step) {
        const p = all[i];
        if (p) ticks.push({ i: p.i, label: String(p.tick) });
      }
    };
    if (mid > 0) {
      pushEvery(0, mid, Math.max(1, Math.floor(mid / 4)));
      pushEvery(mid, all.length, Math.max(1, Math.floor((all.length - mid) / 4)));
    } else {
      pushEvery(0, all.length, Math.max(1, Math.floor(all.length / 6)));
    }
    return ticks;
  }, [all]);

  const yTicks = [0, 4, 8, 12, 16];
  const flashX = PAD.left + resetAt * PLOT_W;

  return (
    <div
      className={`overflow-hidden border border-[rgba(107,193,242,0.22)] bg-[#060a10] ${
        compact ? "rounded-[5px]" : "rounded-[8px]"
      }`}
    >
      <div
        className={`border-b border-[rgba(107,193,242,0.14)] text-[var(--text-soft)] ${
          compact ? "px-1.5 py-1 text-[8px]" : "px-3 py-2 text-[11px]"
        }`}
      >
        Live vector memory (MB) —{" "}
        <span className="text-[var(--cyan)]">blue = without</span>
        {", "}
        <span className="text-[var(--green)]">green = with verilumen agent</span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={`block h-auto w-full ${compact ? "max-h-[118px]" : "max-h-[260px]"}`}
        role="img"
        aria-label="Live vector memory comparison chart"
      >
        <defs>
          <linearGradient id={blueId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(107,193,242,0.35)" />
            <stop offset="100%" stopColor="rgba(107,193,242,0.02)" />
          </linearGradient>
          <linearGradient id={greenId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(46,204,113,0.35)" />
            <stop offset="100%" stopColor="rgba(46,204,113,0.02)" />
          </linearGradient>
        </defs>

        {yTicks.map((v) => {
          const yy = PAD.top + PLOT_H * (1 - v / Y_MAX);
          return (
            <g key={`y-${v}`}>
              <line
                x1={PAD.left}
                y1={yy}
                x2={PAD.left + PLOT_W}
                y2={yy}
                stroke="rgba(107,193,242,0.12)"
                strokeDasharray="3 3"
              />
              <text
                x={PAD.left - 6}
                y={yy + 3}
                textAnchor="end"
                fill="#8aa4bc"
                fontSize={compact ? 8 : 10}
                fontFamily="ui-monospace, monospace"
              >
                {v}
              </text>
            </g>
          );
        })}

        {xTicks.map((t) => {
          const xx = PAD.left + (t.i / Math.max(1, all.length - 1)) * PLOT_W;
          return (
            <text
              key={`x-${t.i}-${t.label}`}
              x={xx}
              y={H - 10}
              textAnchor="middle"
              fill="#8aa4bc"
              fontSize={compact ? 7 : 9}
              fontFamily="ui-monospace, monospace"
            >
              {t.label}
            </text>
          );
        })}

        {paths.withoutArea ? (
          <path d={paths.withoutArea} fill={`url(#${blueId})`} />
        ) : null}
        {paths.withArea ? (
          <path d={paths.withArea} fill={`url(#${greenId})`} />
        ) : null}

        <path
          d={paths.withoutLine}
          fill="none"
          stroke="#6bc1f2"
          strokeWidth={compact ? 1.4 : 1.8}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <path
          d={paths.withLine}
          fill="none"
          stroke="#2ecc71"
          strokeWidth={compact ? 1.4 : 1.8}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {resetFlash ? (
          <line
            x1={flashX}
            y1={PAD.top}
            x2={flashX}
            y2={PAD.top + PLOT_H}
            stroke="#e8f4ff"
            strokeWidth={2}
            className="vl-tto-reset-flash"
          />
        ) : null}

        <text
          x={PAD.left}
          y={12}
          fill="#9ec9ef"
          fontSize={compact ? 8 : 10}
          fontFamily="ui-sans-serif, system-ui"
        >
          Live vector memory (MB)
        </text>
      </svg>

      {!compact ? (
        <div className="flex items-center justify-center gap-4 border-t border-[rgba(107,193,242,0.12)] px-3 py-1.5 text-[10px] text-[var(--text-soft)]">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[var(--cyan)]" />
            without verilumen agent
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[var(--green)]" />
            with verilumen agent
          </span>
        </div>
      ) : null}
    </div>
  );
}
