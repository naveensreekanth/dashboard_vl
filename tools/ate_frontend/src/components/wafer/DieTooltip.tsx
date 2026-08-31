"use client";

import { formatTime } from "@/lib/utils";
import type { Die } from "@/types/wafer";

export function DieTooltip({
  die,
  x,
  y,
}: {
  die: Die;
  x: number;
  y: number;
}) {
  return (
    <div
      className="pointer-events-none fixed z-50 min-w-[180px] rounded border border-[var(--line-bright)] bg-[var(--panel)] px-3 py-2 text-[11px] shadow-lg"
      style={{ left: x + 12, top: y + 12 }}
    >
      <div className="mb-1.5 font-semibold text-[var(--text)]">Die {die.die_id.replace(/^[^:]+:/, "")}</div>
      <Row label="Die ID" value={die.die_id} />
      <Row label="Row" value={String(die.row)} />
      <Row label="Column" value={String(die.column)} />
      <Row label="Result" value={die.result} />
      <Row label="Fail Code" value={die.fail_code ?? "—"} />
      <Row
        label="Test Time"
        value={die.test_time_ms != null ? `${Math.round(die.test_time_ms)} ms` : "—"}
      />
      <Row
        label="Confidence"
        value={die.confidence != null ? `${(die.confidence * 100).toFixed(1)}%` : "—"}
      />
      <Row label="Timestamp" value={die.timestamp ? formatTime(die.timestamp) : "—"} />
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-0.5">
      <span className="text-[var(--muted-2)]">{label}</span>
      <span className="font-mono text-[var(--text)]">{value}</span>
    </div>
  );
}
