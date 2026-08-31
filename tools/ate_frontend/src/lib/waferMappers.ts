import type { DieOut, WaferDetail } from "@/types/api";
import type { Die, DieResult, Wafer, WaferTelemetryEvent } from "@/types/wafer";
import { eventTypeToResult } from "@/types/wafer";

export function mapWaferDetail(detail: WaferDetail): Wafer {
  return {
    wafer_id: detail.wafer_id,
    lot_id: detail.lot_id,
    status: detail.status,
    caption: detail.caption,
    total_dies: detail.total_dies,
    tested_dies: detail.tested_dies,
    pass_count: detail.pass_count ?? detail.bin_counts.pass,
    retest_count: detail.retest_count ?? detail.bin_counts.retest,
    fail_count: detail.fail_count ?? detail.bin_counts.fail,
    reclass_count: detail.reclass_count ?? detail.bin_counts.reclass,
    yield_pct: detail.yield_pct,
  };
}

export function mapDieOut(d: DieOut): Die {
  const result = (d.result ?? d.bin ?? "untested") as DieResult;
  const column = d.column ?? d.x;
  const row = d.row ?? d.y;
  return {
    die_id: d.die_id,
    wafer_id: d.wafer_id,
    column,
    row,
    x: d.x,
    y: d.y,
    result,
    fail_code: d.fail_code ?? null,
    test_time_ms: d.test_time_ms ?? null,
    confidence: d.confidence ?? null,
    timestamp: d.timestamp ?? null,
  };
}

/** Apply a single backend die event onto one Die — no full-state rebuild. */
export function applyDieTelemetryEvent(
  existing: Die | undefined,
  event: WaferTelemetryEvent,
): Die | null {
  if (!event.wafer_id || event.die_id == null) return null;

  const x = typeof event.payload.x === "number" ? event.payload.x : existing?.x;
  const y = typeof event.payload.y === "number" ? event.payload.y : existing?.y;

  let column = x;
  let row = y;
  if (column == null || row == null) {
    const raw = event.die_id.replace(/^\w+:/, "").replace(/[()]/g, "");
    const parts = raw.split(",");
    if (parts.length === 2) {
      column = Number(parts[0]);
      row = Number(parts[1]);
    }
  }
  if (column == null || row == null || Number.isNaN(column) || Number.isNaN(row)) {
    return null;
  }

  const result = eventTypeToResult(event.event_type, event.payload.bin);
  const dieId =
    existing?.die_id ??
    (event.die_id.includes(":") ? event.die_id : `${event.wafer_id}:${column},${row}`);

  return {
    die_id: dieId,
    wafer_id: event.wafer_id,
    column,
    row,
    x: column,
    y: row,
    result,
    fail_code:
      event.payload.fail_code !== undefined
        ? (event.payload.fail_code as string | null)
        : (existing?.fail_code ?? null),
    test_time_ms:
      typeof event.payload.test_time_ms === "number"
        ? event.payload.test_time_ms
        : (existing?.test_time_ms ?? null),
    confidence:
      typeof event.payload.confidence === "number"
        ? event.payload.confidence
        : (existing?.confidence ?? null),
    timestamp: event.timestamp ?? existing?.timestamp ?? null,
  };
}
