export type MaintenanceSeverity =
  | "healthy"
  | "watch"
  | "warning"
  | "critical"
  | "offline"
  | "unavailable";

export interface MaintenanceAsset {
  asset_id: string;
  name: string;
  health_pct: number | null;
  status: "ok" | "warn" | "unavailable";
  rul_days: number | null;
  tester_id: string | null;
  component: string | null;
  failure_probability: number | null;
  confidence: number | null;
  severity: MaintenanceSeverity;
  recommended_action: string | null;
  model_available: boolean;
  message: string | null;
  updated_at: string | null;
}

export interface MaintenanceList {
  flagged_count: number;
  model_available: boolean;
  assets: MaintenanceAsset[];
}

export interface MaintenanceHistoryItem {
  history_id: string;
  tester_id: string;
  component: string;
  event_type: string;
  detail: string;
  health_score: number | null;
  severity: string | null;
  created_at: string;
}

export interface MaintenanceHealthPoint {
  timestamp: string;
  health_score: number | null;
  failure_probability: number | null;
  rul_days: number | null;
  severity: string;
  component: string;
}

export interface MaintenanceTesterDetail {
  tester_id: string;
  name: string;
  status: string;
  site_id: string | null;
  overall_severity: MaintenanceSeverity;
  model_available: boolean;
  components: MaintenanceAsset[];
  history: MaintenanceHistoryItem[];
  health_series: MaintenanceHealthPoint[];
}

export const SEVERITY_COLORS: Record<MaintenanceSeverity, string> = {
  healthy: "#6EE7A8",
  watch: "#6BC1F2",
  warning: "#F2B155",
  critical: "#F0667A",
  offline: "#56637A",
  unavailable: "#8291A6",
};
