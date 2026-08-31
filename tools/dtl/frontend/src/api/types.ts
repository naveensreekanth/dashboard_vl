export type Decision = "RECOMMEND" | "KEEP_CURRENT" | "REVIEW_REQUIRED" | "REJECT";

export type EvidenceLevel =
  | "HIGH_EVIDENCE"
  | "MODERATE_EVIDENCE"
  | "LOW_EVIDENCE"
  | "INSUFFICIENT_EVIDENCE";

export interface SimulationEvidence {
  evidence_origin: string;
  population_level_aggregate: boolean;
  parameter: string;
  candidate_limit: number;
  simulated_yield: number | null;
  simulated_fail_rate?: number | null;
  violation_rate: number | null;
  borderline_rate: number | null;
  risky_rate?: number | null;
  false_fail_proxy?: number | null;
  defective_proxy?: number | null;
  objective_score: number | null;
  worst_condition_yield: number | null;
  worst_condition_violation_rate: number | null;
  evaluated_conditions: number | null;
  found: boolean;
  raw?: Record<string, unknown>;
  note?: string;
}

export interface SafetyCheck {
  name: string;
  passed: boolean;
  layer: number;
  message: string;
  severity: string;
}

export interface SafetyResult {
  status: string;
  checks: SafetyCheck[];
}

export interface DTLRecommendation {
  request_id: string;
  lot_id: string;
  die_id: string;
  parameter: string;
  test_id: string;
  unit: string;
  direction: string;
  current_limit: number | null;
  recommended_limit: number | null;
  decision: Decision;
  ml_score: number | null;
  ml_rank: number | null;
  n_candidates: number;
  model_id: string | null;
  source_status: string;
  simulation_evidence: SimulationEvidence;
  safety_result: SafetyResult;
  evidence_level: EvidenceLevel;
  explanation: Record<string, unknown>;
  model_version: string;
  checkpoint_id: string | null;
  dataset_version: string;
  feature_registry_hash: string | null;
  simulation_config_version: string | null;
  policy_config_version: string;
  timestamp: string;
  core_available: boolean;
  parametric_available: boolean;
  cross_domain_available: boolean;
  evidence_origin: string;
  production_month?: string | null;
  model_used?: string | null;
}

export interface CandidateSetEntry {
  parameter: string;
  candidate_limit: number;
  current_limit?: number;
  ml_score?: number | null;
  ml_rank?: number | null;
  tighten_or_loosen?: string;
  unit?: string;
  direction?: string;
  model_id?: string;
  catalog_valid?: boolean;
  delta_absolute?: number;
}

export interface AuditRecord {
  timestamp?: string;
  request_id?: string;
  lot_id?: string;
  die_id?: string;
  parameters_requested?: string[];
  core_available?: boolean;
  parametric_available?: boolean;
  cross_domain_available?: boolean;
  dataset_version?: string;
  dataset_version_parametric?: string;
  feature_registry_hash?: string | null;
  ml_dataset_version?: string;
  model_version?: string;
  checkpoint_id?: Record<string, string | null>;
  simulation_config_version?: string | null;
  policy_config_version?: string;
  TOP_N?: number;
  candidate_set?: CandidateSetEntry[];
  ml_predictions?: Record<string, unknown>[];
  simulation_evidence_rows?: Record<string, unknown>[];
  safety_check_trace?: Record<string, unknown>[];
  policy_trace?: string[];
  final_decisions?: Record<string, unknown>[];
  evidence_origin?: string;
  include_tree_baseline_diagnostic?: boolean;
  joint_enabled?: boolean;
  context_errors?: string[];
}

export interface LotRecommendationResult {
  request_id: string;
  lot_id: string;
  die_id: string;
  core_available: boolean;
  parametric_available: boolean;
  cross_domain_available: boolean;
  production_month?: string | null;
  recommendations: DTLRecommendation[];
  audit: AuditRecord;
}

export interface RecommendationRequest {
  lot_id: string;
  die_id: string;
  parameters?: string[];
  /** Omit or null for legacy path; 2026-01|2026-02|2026-03 for temporal hybrid. */
  production_month?: string | null;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    request_id: string | null;
  };
}

export interface HealthResponse {
  status: string;
}

export interface ReadyResponse {
  status: string;
  reason?: string;
}

export interface LotsResponse {
  lots: string[];
}

export interface LotDiesResponse {
  lot_id: string;
  dies: string[];
}

export interface LotDieParametersResponse {
  lot_id: string;
  die_id: string;
  parameters: string[];
}

export const CORE_PARAMETERS = ["ir_drop", "thermal"] as const;
export const PARAMETRIC_PARAMETERS = [
  "VMIN",
  "VMAX",
  "IDDQ",
  "SUPPLY_CURRENT",
  "CONTACT_RESISTANCE",
  "INTERCONNECT_RESISTANCE",
  "ON_RESISTANCE",
] as const;

export const ALL_PARAMETERS = [...CORE_PARAMETERS, ...PARAMETRIC_PARAMETERS] as const;

/** Phase 10.11 measurement API contracts (read-only). */
export interface DieMeasurementResponse {
  lot_id: string;
  die_id: string;
  parameter: string;
  domain: "core" | "parametric" | string;
  unit: string | null;
  observed_value: number | null;
  observed_value_rule?: string;
  condition_id?: string;
  source_classification: string;
  dataset_version: string;
  found: boolean;
  disclaimer?: string;
}

export interface DieDistributionResponse {
  lot_id: string;
  die_id: string;
  parameter: string;
  domain: "core" | "parametric" | string;
  unit: string | null;
  scope: "die" | "lot" | string;
  n: number;
  min: number | null;
  median: number | null;
  p95: number | null;
  max: number | null;
  source_classification: string;
  dataset_version: string;
  stats_method: string;
  found: boolean;
  condition_id?: string;
  disclaimer?: string;
  stats_source?: string;
}

export interface ConditionMeasurementRow {
  condition_id: string;
  temperature_c: number | null;
  vdd_applied: number | null;
  test_mode: string | null;
  measurement_value: number | null;
  unit: string | null;
  pass_fail_condition: string | null;
}

export interface DieConditionsResponse {
  lot_id: string;
  die_id: string;
  parameter: string;
  domain: "core" | "parametric" | string;
  unit: string | null;
  source_classification: string;
  dataset_version: string;
  found: boolean;
  reason?: string;
  conditions: ConditionMeasurementRow[];
  disclaimer?: string;
}
