/** Strongly typed wafer-domain models. Backend is authoritative. */

export type DieResult = "pass" | "retest" | "fail" | "reclass" | "untested";

export type WaferLifecycleStatus =
  | "loading"
  | "empty"
  | "error"
  | "offline"
  | "live"
  | "completed";

export interface DieTestResult {
  result: DieResult;
  fail_code: string | null;
  test_time_ms: number | null;
  confidence: number | null;
  timestamp: string | null;
}

export interface Die {
  die_id: string;
  wafer_id: string;
  /** Column index from backend */
  column: number;
  /** Row index from backend */
  row: number;
  /** Raw backend coordinates (column = x, row = y) */
  x: number;
  y: number;
  result: DieResult;
  fail_code: string | null;
  test_time_ms: number | null;
  confidence: number | null;
  timestamp: string | null;
}

export interface Wafer {
  wafer_id: string;
  lot_id: string;
  status: string;
  caption: string;
  total_dies: number;
  tested_dies: number;
  pass_count: number;
  retest_count: number;
  fail_count: number;
  reclass_count: number;
  yield_pct: number;
}

export type WaferDieEventType =
  | "die_tested"
  | "die_pass"
  | "die_fail"
  | "die_retest"
  | "die_reclassified";

export interface WaferTelemetryEvent {
  event_id: string;
  event_type: WaferDieEventType | string;
  timestamp: string;
  source: string;
  tester_id: string | null;
  site_id: string | null;
  lot_id: string | null;
  wafer_id: string | null;
  die_id: string | null;
  sequence_number: number;
  payload: {
    x?: number;
    y?: number;
    bin?: DieResult;
    fail_code?: string | null;
    test_time_ms?: number | null;
    confidence?: number | null;
    pattern_group?: number | string;
    yield_pct?: number;
    pass?: number;
    fail?: number;
    retest?: number;
    reclass?: number;
    total?: number;
    tested_dies?: number;
    total_dies?: number;
    [key: string]: unknown;
  };
}

export const DIE_RESULT_COLORS: Record<DieResult, string> = {
  pass: "#6EE7A8",
  retest: "#F2B155",
  fail: "#F0667A",
  reclass: "#6BC1F2",
  untested: "#1C2532",
};

export const DIE_EVENT_TYPES: WaferDieEventType[] = [
  "die_tested",
  "die_pass",
  "die_fail",
  "die_retest",
  "die_reclassified",
];

export function eventTypeToResult(eventType: string, payloadBin?: string): DieResult {
  switch (eventType) {
    case "die_pass":
      return "pass";
    case "die_fail":
      return "fail";
    case "die_retest":
      return "retest";
    case "die_reclassified":
      return "reclass";
    case "die_tested":
      if (
        payloadBin === "pass" ||
        payloadBin === "fail" ||
        payloadBin === "retest" ||
        payloadBin === "reclass" ||
        payloadBin === "untested"
      ) {
        return payloadBin;
      }
      return "untested";
    default:
      return "untested";
  }
}
