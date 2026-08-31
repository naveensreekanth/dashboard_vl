"use client";

import { useEffect, useState } from "react";
import {
  SHMOO_DEMO_BOUNDARY,
  SHMOO_DEMO_FREQ_RANGE,
  SHMOO_DEMO_OP,
  SHMOO_DEMO_POINTS,
  SHMOO_DEMO_VDD_RANGE,
  type ShmooDemoPoint,
  type ShmooPointKind,
} from "@/lib/shmooDemoScatter";
import { useShmooStore } from "@/stores/shmooStore";

const W = 640;
const H = 380;
const PAD = { top: 28, right: 24, bottom: 42, left: 52 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

const KIND_COLOR: Record<ShmooPointKind, string> = {
  pass: "#2ecc71",
  freq_margin: "#e74c3c",
  timing: "#f39c12",
};

/**
 * Staged ensemble SHMOO scatter matching the ML tool visualization.
 * Uses demo points; OP from shmooStore when a session exists.
 */
export function ShmooPlotPreview({
  compact = false,
}: {
  compact?: boolean;
}) {
  const results = useShmooStore((s) => s.results);
  const [stage, setStage] = useState(0);
  const [showOpTip, setShowOpTip] = useState(false);

  const vddRange = SHMOO_DEMO_VDD_RANGE;
  const freqRange = SHMOO_DEMO_FREQ_RANGE;

  // Star + crosshairs locked to geometric plot center.
  const op = SHMOO_DEMO_OP;

  const boundary = SHMOO_DEMO_BOUNDARY;
  const points = SHMOO_DEMO_POINTS;

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setStage(4);
      return;
    }
    setStage(0);
    const t1 = window.setTimeout(() => setStage(1), 200);
    const t2 = window.setTimeout(() => setStage(2), 700);
    const t3 = window.setTimeout(() => setStage(3), 1400);
    const t4 = window.setTimeout(() => setStage(4), 2000);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
      window.clearTimeout(t4);
    };
  }, []);

  const xScale = (v: number) =>
    PAD.left +
    ((v - vddRange[0]) / Math.max(0.001, vddRange[1] - vddRange[0])) * PLOT_W;
  const yScale = (f: number) =>
    PAD.top +
    (1 - (f - freqRange[0]) / Math.max(0.001, freqRange[1] - freqRange[0])) *
      PLOT_H;

  const b1 = {
    x: xScale(vddRange[0]),
    y: yScale(boundary.slope * vddRange[0] + boundary.intercept),
  };
  const b2 = {
    x: xScale(vddRange[1]),
    y: yScale(boundary.slope * vddRange[1] + boundary.intercept),
  };
  const boundaryLen = Math.hypot(b2.x - b1.x, b2.y - b1.y);

  const opX = PAD.left + PLOT_W / 2;
  const opY = PAD.top + PLOT_H / 2;

  const gridXs = 6;
  const gridYs = 6;

  return (
    <div
      className={`overflow-hidden border border-[rgba(167,139,250,0.28)] bg-[#080c14] ${
        compact ? "shrink-0 rounded-[5px]" : "rounded-[8px]"
      }`}
      onClick={(e) => e.stopPropagation()}
    >
      {compact ? (
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[rgba(107,193,242,0.16)] px-1.5 py-1">
          <span className="text-[8px] font-semibold uppercase tracking-[0.08em] text-[var(--violet)]">
            Ensemble scatter
          </span>
          <Legend compact />
        </div>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[rgba(107,193,242,0.16)] px-3 py-2">
          <div className="rounded-[6px] border border-[rgba(107,193,242,0.25)] bg-[rgba(8,16,26,0.85)] px-2.5 py-1 text-[11px] text-[var(--text-soft)]">
            View Mode:{" "}
            <span className="font-semibold text-[var(--text-bright)]">
              Show All Devices (Ensemble)
            </span>
          </div>
          <Legend />
        </div>
      )}

      <div className={`relative ${compact ? "px-0.5 pt-0.5" : "px-1 pt-1"}`}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className={`block h-auto w-full ${compact ? "max-h-[128px]" : "max-h-[400px]"}`}
          role="img"
          aria-label="SHMOO ensemble scatter plot"
        >
          {/* Axes / grid — stage 1 */}
          <g
            style={{
              opacity: stage >= 1 ? 1 : 0,
              transition: "opacity 400ms ease",
            }}
          >
            {Array.from({ length: gridXs + 1 }, (_, i) => {
              const x = PAD.left + (PLOT_W * i) / gridXs;
              return (
                <line
                  key={`gx-${i}`}
                  x1={x}
                  y1={PAD.top}
                  x2={x}
                  y2={PAD.top + PLOT_H}
                  stroke="rgba(107,193,242,0.12)"
                  strokeDasharray="2 3"
                />
              );
            })}
            {Array.from({ length: gridYs + 1 }, (_, i) => {
              const y = PAD.top + (PLOT_H * i) / gridYs;
              return (
                <line
                  key={`gy-${i}`}
                  x1={PAD.left}
                  y1={y}
                  x2={PAD.left + PLOT_W}
                  y2={y}
                  stroke="rgba(107,193,242,0.12)"
                  strokeDasharray="2 3"
                />
              );
            })}
            <line
              x1={PAD.left}
              y1={PAD.top + PLOT_H}
              x2={PAD.left + PLOT_W}
              y2={PAD.top + PLOT_H}
              stroke="rgba(156,180,204,0.45)"
            />
            <line
              x1={PAD.left}
              y1={PAD.top}
              x2={PAD.left}
              y2={PAD.top + PLOT_H}
              stroke="rgba(156,180,204,0.45)"
            />
            {Array.from({ length: gridXs + 1 }, (_, i) => {
              const v =
                vddRange[0] + ((vddRange[1] - vddRange[0]) * i) / gridXs;
              const x = PAD.left + (PLOT_W * i) / gridXs;
              return (
                <text
                  key={`tx-${i}`}
                  x={x}
                  y={PAD.top + PLOT_H + 18}
                  textAnchor="middle"
                  fill="#8aa4bc"
                  fontSize={10}
                  fontFamily="ui-monospace, monospace"
                >
                  {v.toFixed(2)}
                </text>
              );
            })}
            {Array.from({ length: gridYs + 1 }, (_, i) => {
              const f =
                freqRange[1] - ((freqRange[1] - freqRange[0]) * i) / gridYs;
              const y = PAD.top + (PLOT_H * i) / gridYs;
              return (
                <text
                  key={`ty-${i}`}
                  x={PAD.left - 8}
                  y={y + 3}
                  textAnchor="end"
                  fill="#8aa4bc"
                  fontSize={10}
                  fontFamily="ui-monospace, monospace"
                >
                  {f.toFixed(2)}
                </text>
              );
            })}
            <text
              x={PAD.left + PLOT_W / 2}
              y={H - 6}
              textAnchor="middle"
              fill="#9ec9ef"
              fontSize={11}
            >
              Supply Voltage (VDD_V)
            </text>
            <text
              x={14}
              y={PAD.top + PLOT_H / 2}
              textAnchor="middle"
              fill="#9ec9ef"
              fontSize={11}
              transform={`rotate(-90 14 ${PAD.top + PLOT_H / 2})`}
            >
              Frequency (GHz)
            </text>
          </g>

          {/* Points — stage 2 */}
          <g>
            {points.map((p, i) => (
              <PointRect
                key={i}
                point={p}
                x={xScale(p.vdd)}
                y={yScale(p.freq)}
                visible={stage >= 2}
                delayMs={Math.min(720, Math.floor(i / 26) * 28 + (i % 26) * 8)}
              />
            ))}
          </g>

          {/* Boundary — stage 3 */}
          <line
            x1={b1.x}
            y1={b1.y}
            x2={b2.x}
            y2={b2.y}
            stroke="#7dd3fc"
            strokeWidth={1.6}
            strokeDasharray={stage >= 3 ? "6 4" : `${boundaryLen}`}
            strokeDashoffset={stage >= 3 ? 0 : boundaryLen}
            style={{
              transition:
                "stroke-dashoffset 700ms cubic-bezier(0.22, 1, 0.36, 1), opacity 400ms ease",
              opacity: stage >= 3 ? 1 : 0,
            }}
          />

          {/* Full-span center crosshairs (not scaled — must stay edge-to-edge) */}
          <g
            style={{
              opacity: stage >= 4 ? 1 : 0,
              transition: "opacity 350ms ease",
            }}
          >
            <line
              x1={PAD.left}
              y1={opY}
              x2={PAD.left + PLOT_W}
              y2={opY}
              stroke="#c4b5fd"
              strokeWidth={1.5}
              strokeLinecap="square"
              vectorEffect="non-scaling-stroke"
            />
            <line
              x1={opX}
              y1={PAD.top}
              x2={opX}
              y2={PAD.top + PLOT_H}
              stroke="#c4b5fd"
              strokeWidth={1.5}
              strokeLinecap="square"
              vectorEffect="non-scaling-stroke"
            />
          </g>

          {/* Recommended OP star — stage 4 (animated separately so crosshairs stay full length) */}
          {stage >= 4 ? (
            <g
              style={{
                transformOrigin: `${opX}px ${opY}px`,
                animation: "vl-op-star-in 450ms cubic-bezier(0.22, 1, 0.36, 1) both",
              }}
            >
              <circle
                cx={opX}
                cy={opY}
                r={14}
                fill="rgba(168,85,247,0.18)"
                className="vl-op-glow"
              />
              <Star cx={opX} cy={opY} size={11} />
              <circle
                cx={opX}
                cy={opY}
                r={16}
                fill="transparent"
                className="cursor-pointer"
                onMouseEnter={() => setShowOpTip(true)}
                onMouseLeave={() => setShowOpTip(false)}
              />
            </g>
          ) : null}
        </svg>

        {showOpTip && stage >= 4 && !compact ? (
          <div
            className="pointer-events-none absolute z-[2] rounded-[8px] border border-[rgba(167,139,250,0.4)] bg-[rgba(8,16,26,0.95)] px-3 py-2 text-[11px] shadow-lg"
            style={{
              left: `min(70%, max(8%, ${(opX / W) * 100}%))`,
              top: `min(72%, max(10%, ${(opY / H) * 100 + 4}%))`,
            }}
          >
            <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.08em] text-[var(--violet)]">
              Recommended OP
            </div>
            <div className="font-mono text-[var(--text-bright)]">
              {op.vdd.toFixed(3)} V · {op.freq.toFixed(3)} GHz
            </div>
            {results ? (
              <div className="mt-0.5 font-mono text-[10px] text-[var(--muted)]">
                PASS {results.n_pass} · FAIL {results.n_fail}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {compact ? (
        <div className="shrink-0 border-t border-[rgba(107,193,242,0.14)] px-1.5 py-1 text-[8px] font-semibold tracking-[0.04em] text-[#9ec9ef]">
          PATTERN:{" "}
          <span className="font-semibold text-[var(--text-bright)]">
            Normal Shmoo (Linear Speedpath)
          </span>
        </div>
      ) : (
        <div className="space-y-1.5 border-t border-[rgba(107,193,242,0.14)] px-3 py-2.5">
          <div className="inline-flex items-center rounded-full border border-[rgba(107,193,242,0.35)] bg-[rgba(107,193,242,0.1)] px-2.5 py-0.5 text-[10px] font-semibold tracking-[0.04em] text-[#9ec9ef]">
            IDENTIFIED SHMOO PATTERN:{" "}
            <span className="ml-1 text-[var(--text-bright)]">
              Normal Shmoo (Well-Behaved Linear Speedpath)
            </span>
          </div>
          <p className="text-[11px] leading-relaxed text-[var(--text-soft)]">
            The plot exhibits a standard, well-behaved linear pass/fail boundary.
            Maximum operating frequency (Fmax) scales smoothly with supply voltage
            (VDD).
          </p>
        </div>
      )}
    </div>
  );
}

function PointRect({
  point,
  x,
  y,
  visible,
  delayMs,
}: {
  point: ShmooDemoPoint;
  x: number;
  y: number;
  visible: boolean;
  delayMs: number;
}) {
  return (
    <rect
      x={x - 3}
      y={y - 3}
      width={6}
      height={6}
      fill={KIND_COLOR[point.kind]}
      rx={0.5}
      style={{
        opacity: visible ? 0.92 : 0,
        transition: `opacity 320ms ease ${delayMs}ms`,
      }}
    />
  );
}

function Star({ cx, cy, size }: { cx: number; cy: number; size: number }) {
  const pts: string[] = [];
  for (let i = 0; i < 5; i++) {
    const a = (-Math.PI / 2) + (i * 2 * Math.PI) / 5;
    const b = a + Math.PI / 5;
    pts.push(
      `${cx + Math.cos(a) * size},${cy + Math.sin(a) * size}`,
      `${cx + Math.cos(b) * (size * 0.42)},${cy + Math.sin(b) * (size * 0.42)}`,
    );
  }
  return (
    <polygon
      points={pts.join(" ")}
      fill="#a855f7"
      stroke="#c4b5fd"
      strokeWidth={0.8}
      style={{ filter: "drop-shadow(0 0 6px rgba(168,85,247,0.85))" }}
    />
  );
}

function Legend({ compact = false }: { compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[8px] text-[var(--text-soft)]">
        <span className="inline-flex items-center gap-0.5">
          <span className="inline-block h-1.5 w-1.5 bg-[#2ecc71]" /> PASS
        </span>
        <span className="inline-flex items-center gap-0.5">
          <span className="inline-block h-1.5 w-1.5 bg-[#e74c3c]" /> FAIL
        </span>
        <span className="inline-flex items-center gap-0.5">
          <span className="inline-block h-1.5 w-1.5 bg-[#f39c12]" /> TIMING
        </span>
        <span className="inline-flex items-center gap-0.5">
          <span className="text-[#a855f7]">★</span> OP
        </span>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[var(--text-soft)]">
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2.5 w-2.5 bg-[#2ecc71]" /> PASS
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2.5 w-2.5 bg-[#e74c3c]" /> FAIL (FREQ_MARGIN)
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2.5 w-2.5 bg-[#f39c12]" /> FAIL (TIMING/DEFECT)
      </span>
      <span className="inline-flex items-center gap-1">
        <span
          className="inline-block h-0 w-4 border-t border-dashed border-[#7dd3fc]"
          style={{ borderTopWidth: 2 }}
        />{" "}
        Population Boundary
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="text-[#a855f7]">★</span> Recommended OP
      </span>
    </div>
  );
}
