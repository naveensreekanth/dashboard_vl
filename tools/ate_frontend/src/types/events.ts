export type EventSeverity = "INFO" | "PASS" | "WARN" | "ERROR" | "CRITICAL";

export type OpsConnectionStatus =
  | "LIVE"
  | "RECONNECTING"
  | "DEGRADED"
  | "OFFLINE"
  | "STALE";

export interface TestEvent {
  event_id: string;
  timestamp: string;
  severity: EventSeverity;
  event_type: string;
  source: string;
  tester_id: string | null;
  site_id: string | null;
  lot_id: string | null;
  wafer_id: string | null;
  die_id: string | null;
  message: string;
  metadata: Record<string, unknown>;
  acknowledged: boolean;
  acknowledged_by?: string | null;
  acknowledged_at?: string | null;
  sequence_number: number;
}

export interface TestEventsListOut {
  total: number;
  unacknowledged: number;
  items: TestEvent[];
}

export interface EventFilterOptions {
  testers: string[];
  sites: string[];
  lots: string[];
  wafers: string[];
  severities: EventSeverity[];
  event_types: string[];
}

export interface EventFiltersState {
  q: string;
  tester_id: string;
  site_id: string;
  lot_id: string;
  wafer_id: string;
  severity: EventSeverity[];
  since: string;
  until: string;
  acknowledged: "" | "true" | "false";
}

export const EMPTY_EVENT_FILTERS: EventFiltersState = {
  q: "",
  tester_id: "",
  site_id: "",
  lot_id: "",
  wafer_id: "",
  severity: [],
  since: "",
  until: "",
  acknowledged: "",
};

export const SEVERITY_ORDER: Record<EventSeverity, number> = {
  CRITICAL: 0,
  ERROR: 1,
  WARN: 2,
  PASS: 3,
  INFO: 4,
};

export const SEVERITY_STYLES: Record<EventSeverity, { fg: string; bg: string }> = {
  INFO: { fg: "var(--cyan)", bg: "var(--cyan-dim)" },
  PASS: { fg: "var(--green)", bg: "var(--green-dim)" },
  WARN: { fg: "var(--amber)", bg: "var(--amber-dim)" },
  ERROR: { fg: "var(--red)", bg: "var(--red-dim)" },
  CRITICAL: { fg: "#ff8a9a", bg: "rgba(240, 102, 122, 0.22)" },
};

/** Floor event types that belong in the event center (die pass noise excluded). */
export const FLOOR_EVENT_TYPES = new Set([
  "wafer_started",
  "wafer_progress",
  "lot_started",
  "lot_completed",
  "yield_updated",
  "test_time_updated",
  "pattern_optimization",
  "predictive_maintenance",
  "dynamic_limit_updated",
  "escape_risk_detected",
  "tester_status_changed",
  "engineering_hold",
  "optimization_completed",
  "die_fail",
  "die_retest",
  "die_reclassified",
]);
