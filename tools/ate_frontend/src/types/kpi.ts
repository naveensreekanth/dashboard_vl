export type KpiTrend = "up" | "down" | "flat";
export type KpiStatus = "on_track" | "at_risk" | "below_target" | "exceeds_target";

export interface KpiHistoryPoint {
  timestamp: string;
  value: number;
  index?: number | null;
}

/** Full KPI contract from GET /api/kpis and GET /api/kpis/{id} */
export interface Kpi {
  id: string;
  name: string;
  value: number;
  unit: string;
  baseline: number;
  target: number;
  previous_value: number;
  improvement: number;
  trend: KpiTrend;
  status: KpiStatus;
  timestamp: string;
  history: KpiHistoryPoint[];
  description?: string;
  accent?: string;
}

export interface KpiDetail extends Kpi {
  lots: number;
  wafers: number;
  testers: number;
  sites: number;
  recent_events: Array<{
    event_id: string;
    event_type: string;
    timestamp: string;
    tag: "pass" | "warn" | "info";
    text: string;
    lot_id: string | null;
    wafer_id: string | null;
    tester_id: string | null;
  }>;
}

export interface KpiHistoryResponse {
  id: string;
  name: string;
  unit: string;
  history: KpiHistoryPoint[];
}

export const KPI_LIVE_EVENTS = [
  "yield_updated",
  "test_time_updated",
  "pattern_optimization",
  "optimization_completed",
] as const;

export type KpiLiveEvent = (typeof KPI_LIVE_EVENTS)[number];
