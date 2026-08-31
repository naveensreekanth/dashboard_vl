"use client";

import { useMemo, useState } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { LoadingState } from "@/components/common/LoadingState";
import { DieDetailPanel } from "@/components/wafer/DieDetailPanel";
import { DieTooltip } from "@/components/wafer/DieTooltip";
import { useWafer } from "@/hooks/useWafer";
import { useWaferDies } from "@/hooks/useWaferDies";
import { formatNumber } from "@/lib/utils";
import { useWaferStore } from "@/stores/waferStore";
import type { Die } from "@/types/wafer";
import { DIE_RESULT_COLORS } from "@/types/wafer";

const R = 118;
const CX = 130;
const CY = 130;
const VIEW = 260;

export interface WaferMapProps {
  waferId: string | null | undefined;
}

/**
 * Production wafer map — renders backend die coordinates only.
 * Live die_* events patch a single die via Zustand; aggregates from REST / yield_updated.
 */
export function WaferMap({ waferId }: WaferMapProps) {
  const { wafer, lifecycle, isLoading, isError, refetch } = useWafer(waferId);
  const { dies, isLoading: diesLoading, isError: diesError, refetch: refetchDies } = useWaferDies(waferId);
  // Realtime subscription is owned by DashboardShell (single WS). Hooks still read live store.

  const selectedDieId = useWaferStore((s) => s.selectedDieId);
  const hoveredDieId = useWaferStore((s) => s.hoveredDieId);
  const selectDie = useWaferStore((s) => s.selectDie);
  const setHoveredDieId = useWaferStore((s) => s.setHoveredDieId);

  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  const layout = useMemo(() => computeLayout(dies), [dies]);
  const selectedDie = selectedDieId ? dies.find((d) => d.die_id === selectedDieId) ?? null : null;
  const hoveredDie = hoveredDieId ? dies.find((d) => d.die_id === hoveredDieId) ?? null : null;

  const status = resolveStatus(lifecycle, isLoading || diesLoading, isError || diesError, dies.length);

  if (status === "loading") {
    return <LoadingState label="Loading wafer map…" />;
  }

  if (status === "error") {
    return (
      <ErrorState
        message="Unable to load wafer state from the API."
        onRetry={() => {
          void refetch();
          void refetchDies();
        }}
      />
    );
  }

  if (status === "empty") {
    return <EmptyState message="No dies reported for this wafer." />;
  }

  return (
    <div className="flex w-full flex-col items-center gap-3">
      <div className="flex w-full items-center justify-between">
        <span className="rounded border border-[rgba(110,231,168,0.3)] bg-[rgba(110,231,168,0.1)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--green)]">
          {statusLabel(status)}
        </span>
        <span className="font-mono text-[11px] text-[#c5d8ec]">{wafer?.wafer_id ?? "—"}</span>
      </div>

      <svg
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        width={260}
        height={260}
        aria-label="Live wafer map"
        className="select-none"
      >
        {/* Wafer boundary */}
        <circle cx={CX} cy={CY} r={R + 3} fill="#0A0F17" stroke="#2A3648" strokeWidth={1.5} />
        {/* Notch */}
        <circle cx={CX} cy={CY + R + 3} r={3} fill="#06090F" stroke="#2A3648" />

        {layout.cells.map((cell) => {
          const die = cell.die;
          const fill = DIE_RESULT_COLORS[die.result] ?? DIE_RESULT_COLORS.untested;
          const isSelected = die.die_id === selectedDieId;
          const isHovered = die.die_id === hoveredDieId;
          return (
            <rect
              key={die.die_id}
              x={cell.px - layout.cell / 2 + 0.6}
              y={cell.py - layout.cell / 2 + 0.6}
              width={layout.cell - 1.2}
              height={layout.cell - 1.2}
              rx={0.6}
              fill={fill}
              opacity={die.result === "pass" ? 0.85 : 1}
              stroke={isSelected || isHovered ? "#EAF0F6" : "transparent"}
              strokeWidth={isSelected ? 1.2 : isHovered ? 0.8 : 0}
              className="cursor-pointer"
              onMouseEnter={(e) => {
                setHoveredDieId(die.die_id);
                setTooltipPos({ x: e.clientX, y: e.clientY });
              }}
              onMouseMove={(e) => setTooltipPos({ x: e.clientX, y: e.clientY })}
              onMouseLeave={() => {
                setHoveredDieId(null);
                setTooltipPos(null);
              }}
              onClick={() => selectDie(die.die_id === selectedDieId ? null : die.die_id)}
            >
              <title>
                {die.die_id} · {die.result}
              </title>
            </rect>
          );
        })}
      </svg>

      <div className="text-[10.5px] uppercase tracking-[0.08em] text-[var(--muted-2)]">
        {wafer?.caption ?? "Live wafer map"}
      </div>

      {/* Server-provided aggregates — flex boxes sized to fit full labels */}
      <div className="flex w-full flex-wrap gap-2">
        <Stat label="Total Dies" value={wafer ? String(wafer.total_dies) : "—"} />
        <Stat label="Tested" value={wafer ? String(wafer.tested_dies) : "—"} />
        <Stat label="Pass" value={wafer ? String(wafer.pass_count) : "—"} color="var(--green)" />
        <Stat label="Retest" value={wafer ? String(wafer.retest_count) : "—"} color="var(--amber)" />
        <Stat label="Fail" value={wafer ? String(wafer.fail_count) : "—"} color="var(--red)" />
        <Stat label="Reclass" value={wafer ? String(wafer.reclass_count) : "—"} color="var(--cyan)" />
        <Stat
          label="Yield"
          value={wafer ? `${formatNumber(wafer.yield_pct)}%` : "—"}
          color="var(--cyan)"
        />
      </div>

      {hoveredDie && tooltipPos ? (
        <DieTooltip die={hoveredDie} x={tooltipPos.x} y={tooltipPos.y} />
      ) : null}

      {selectedDie ? (
        <DieDetailPanel die={selectedDie} onClose={() => selectDie(null)} />
      ) : null}
    </div>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="min-w-[5.75rem] flex-1 rounded border border-[var(--line)] bg-[var(--panel)] px-2.5 py-2 text-center">
      <div className="whitespace-nowrap text-[9px] uppercase tracking-[0.05em] text-[var(--muted-2)]">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-[13px] font-semibold" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function statusLabel(status: string): string {
  switch (status) {
    case "live":
      return "Live";
    case "completed":
      return "Completed";
    case "offline":
      return "Offline";
    default:
      return status;
  }
}

function resolveStatus(
  lifecycle: string,
  loading: boolean,
  error: boolean,
  dieCount: number,
): "loading" | "empty" | "error" | "offline" | "live" | "completed" {
  if (loading && dieCount === 0) return "loading";
  if (error && dieCount === 0) return "error";
  if (dieCount === 0) return "empty";
  if (lifecycle === "offline") return "offline";
  if (lifecycle === "completed") return "completed";
  if (lifecycle === "error") return "error";
  return "live";
}

/** Layout dies using backend coordinates (column=x, row=y). */
function computeLayout(dies: Die[]) {
  if (dies.length === 0) {
    return { cell: 0, cells: [] as { die: Die; px: number; py: number }[] };
  }
  const maxCol = Math.max(...dies.map((d) => d.column));
  const maxRow = Math.max(...dies.map((d) => d.row));
  const cols = maxCol + 1;
  const rows = maxRow + 1;
  const cell = (2 * R) / Math.max(cols, rows, 1);

  const cells = dies.map((die) => {
    const px = CX - R + die.column * cell + cell / 2;
    const py = CY - R + die.row * cell + cell / 2;
    return { die, px, py };
  });

  return { cell, cells };
}
