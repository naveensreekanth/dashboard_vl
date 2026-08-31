export type DieBin = "pass" | "retest" | "fail" | "reclass" | "untested";

export type LimitStatus =
  | "RECOMMENDED"
  | "PENDING_APPROVAL"
  | "ACTIVE"
  | "REJECTED"
  | "ROLLED_BACK";

export type LimitDirection = "tightened" | "widened" | "unchanged";

export interface BinCounts {
  pass: number;
  retest: number;
  fail: number;
  reclass: number;
}

export interface HeaderStats {
  lots_in_test: number;
  test_time_saved_hours: number;
  overall_yield_pct: number;
}

export interface DieOut {
  die_id: string;
  wafer_id: string;
  x: number;
  y: number;
  row?: number;
  column?: number;
  result?: DieBin;
  bin?: DieBin;
  fail_code?: string | null;
  test_time_ms?: number | null;
  confidence?: number | null;
  timestamp?: string | null;
  tested_at?: string | null;
}

export interface WaferDetail {
  wafer_id: string;
  lot_id: string;
  status: string;
  yield_pct: number;
  total_dies: number;
  tested_dies: number;
  caption: string;
  bin_counts: BinCounts;
  pass_count?: number;
  fail_count?: number;
  retest_count?: number;
  reclass_count?: number;
  updated_at?: string;
}

export interface WaferListItem {
  wafer_id: string;
  lot_id: string;
  status: string;
  yield_pct: number;
  tested_dies: number;
  total_dies: number;
}

export interface KpiCard {
  id: string;
  name: string;
  value: number;
  unit: string;
  baseline?: number;
  target?: number;
  previous_value?: number;
  improvement?: number;
  trend?: "up" | "down" | "flat";
  status?: string;
  timestamp?: string;
  history?: { t: string; v: number }[];
  description?: string;
  accent?: string;
  title?: string;
  spark?: number[];
  series?: number[];
}

export interface MaintenanceAssetOut {
  asset_id: string;
  name: string;
  health_pct: number | null;
  status: string;
  rul_days: number | null;
  tester_id?: string;
  component?: string;
  failure_probability?: number | null;
  confidence?: number | null;
  severity?: string;
  recommended_action?: string | null;
  model_available?: boolean;
  message?: string | null;
  updated_at?: string;
}

export interface MaintenanceOut {
  flagged_count: number;
  model_available?: boolean;
  assets: MaintenanceAssetOut[];
}

export interface TestLimitOut {
  limit_id: string;
  parameter: string;
  test_name: string;
  name: string;
  site_id: string | null;
  tester_id: string | null;
  lot_id: string | null;
  previous_limit: number;
  current_limit: number;
  delta: number;
  change_percentage: number;
  change_pct: number;
  change_label: string;
  direction: LimitDirection;
  cpk: number | null;
  target_cpk: number;
  confidence: number | null;
  reason: string | null;
  status: LimitStatus;
  created_at: string;
  updated_at: string;
}

export interface TestLimitsOut {
  adjustments_today: number;
  items: TestLimitOut[];
}

export interface EventLogItem {
  event_id: string;
  event_type: string;
  timestamp: string;
  tag: "pass" | "warn" | "info";
  text: string;
  lot_id: string | null;
  wafer_id: string | null;
  tester_id: string | null;
}

export interface DashboardSummary {
  header: HeaderStats;
  active_wafer: WaferDetail | null;
  kpis: KpiCard[];
  maintenance: MaintenanceOut;
  test_limits: TestLimitsOut;
  recent_events: EventLogItem[];
  connection_hint: string;
}

export interface TelemetryEvent {
  event_id: string;
  event_type: string;
  timestamp: string;
  source: string;
  tester_id: string | null;
  site_id: string | null;
  lot_id: string | null;
  wafer_id: string | null;
  die_id: string | null;
  sequence_number: number;
  payload: Record<string, unknown>;
}

export interface WsMessage {
  kind: "telemetry_event" | "projection_snapshot";
  event?: TelemetryEvent | null;
  summary?: DashboardSummary | null;
}
