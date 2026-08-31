/** Phase 13.0 — types for precomputed Phase 12.9 analysis (read-only). */

export type AnalysisDecision =
  | "RECOMMEND"
  | "KEEP_CURRENT"
  | "REVIEW_REQUIRED"
  | "REJECT";

export interface AnalysisRecommendationRow {
  lot_category?: string;
  lot_id: string;
  die_id: string;
  sequence_id?: string;
  production_month: string;
  month_label?: string;
  parameter: string;
  parameter_display: string;
  unit?: string | null;
  current_limit: number;
  recommended_limit: number;
  recommendation_delta?: number | null;
  recommendation_delta_percent?: number | null;
  max_eligible_simulated_yield?: number | null;
  ml_score?: number | null;
  ml_rank?: number | null;
  model_used?: string | null;
  model_expected?: string | null;
  decision: AnalysisDecision | string;
  policy_reason?: string | null;
  yield_tie?: boolean;
  tie_breaker?: string | null;
  selection_text?: string | null;
  explanation_text?: string | null;
  why_selected?: string | null;
  safety_status?: string | null;
  evidence_origin?: string | null;
  is_primary_die?: boolean;
}

export interface AnalysisCandidateRow {
  production_month: string;
  lot_id: string;
  die_id: string;
  parameter: string;
  parameter_display: string;
  candidate_limit: number;
  simulated_yield?: number | null;
  safety_status?: string | null;
  eligible?: boolean;
  in_policy_gate_set?: boolean;
  ml_score?: number | null;
  ml_rank?: number | null;
  is_current?: boolean;
  is_selected?: boolean;
  model_used?: string | null;
  decision?: string | null;
}

export interface TemporalChangeRow {
  parameter?: string;
  parameter_display?: string;
  jan_recommendation?: number | null;
  feb_recommendation?: number | null;
  mar_recommendation?: number | null;
  recommendation_changed?: boolean;
  jan_yield?: number | null;
  feb_yield?: number | null;
  mar_yield?: number | null;
  jan_ml_rank?: number | null;
  feb_ml_rank?: number | null;
  mar_ml_rank?: number | null;
  jan_decision?: string | null;
  feb_decision?: string | null;
  mar_decision?: string | null;
  current_dtl_changed?: boolean;
  yield_changed?: boolean;
  model_used?: string | null;
  transition?: string | null;
  previous_dtl?: number | null;
  new_dtl?: number | null;
  previous_yield?: number | null;
  new_yield?: number | null;
  previous_ml_rank?: number | null;
  new_ml_rank?: number | null;
  factual_note?: string | null;
}

export interface SameDieRow {
  lot_category?: string;
  lot_id: string;
  die_id: string;
  sequence_id?: string;
  production_month: string;
  parameter: string;
  parameter_display: string;
  current_limit?: number | null;
  recommended_limit?: number | null;
  max_eligible_simulated_yield?: number | null;
  ml_score?: number | null;
  ml_rank?: number | null;
  model_used?: string | null;
  decision?: string | null;
  why_selected?: string | null;
  observed_n?: number | null;
  observed_mean?: number | null;
  observed_min?: number | null;
  observed_max?: number | null;
}

export interface ModelTraceRow {
  parameter: string;
  parameter_display: string;
  model_expected: string;
  models_observed: string;
  routing_ok: boolean;
}

export interface YieldFirstProof {
  kind: string;
  production_month: string;
  parameter: string;
  parameter_display: string;
  lot_id: string;
  die_id: string;
  winner: {
    candidate_limit: number;
    simulated_yield: number;
    ml_score?: number | null;
    ml_rank?: number | null;
  };
  loser_higher_ml: {
    candidate_limit: number;
    simulated_yield: number;
    ml_score?: number | null;
    ml_rank?: number | null;
  };
  statement?: string;
}

export interface MlTieBreakProof {
  kind: string;
  production_month: string;
  parameter: string;
  parameter_display: string;
  lot_id: string;
  die_id: string;
  tied_yield: number;
  winner_limit: number;
  tied_candidates: Array<{
    candidate_limit: number;
    simulated_yield: number;
    ml_score?: number | null;
    ml_rank?: number | null;
    is_selected?: boolean;
  }>;
  statement?: string;
}

export interface ThreeMonthAnalysisBundle {
  source: string;
  disclaimer: string;
  allowed_months: string[];
  scorable_parameters: string[];
  non_scorable_parameters: string[];
  non_scorable_note: string;
  executive_summary: {
    parameters_recommendation_changed?: string[];
    parameters_recommendation_stable?: string[];
    what_changed_summary?: string;
    what_ml_does?: string;
    yield_first_proof_example?: YieldFirstProof | null;
    ml_tie_break_proof_example?: MlTieBreakProof | null;
    yield_first_proof_count?: number;
    ml_tie_break_proof_count?: number;
    primary_die?: { lot_id: string; die_id: string };
    limitations?: string[];
    [key: string]: unknown;
  };
  primary_recommendations: AnalysisRecommendationRow[];
  all_recommendations: AnalysisRecommendationRow[];
  candidate_explanations: AnalysisCandidateRow[];
  temporal_changes: TemporalChangeRow[];
  same_die_analysis: SameDieRow[];
  model_traceability: ModelTraceRow[];
  policy_proofs: {
    yield_first_proofs?: YieldFirstProof[];
    ml_tie_break_proofs?: MlTieBreakProof[];
    month_isolation_checks?: Record<string, unknown>[];
  };
  viz_recommended_dtl_by_month: Array<Record<string, unknown>>;
  doc_reference: string;
  artifact_reference: string;
  die_level_identities?: DieLevelIdentities | null;
  used_uploaded_measurements?: boolean;
  used_static_three_month_measurements?: boolean;
  analysis_session_id?: string | null;
  data_provenance?: string;
}

export interface CostSavingsAggregate {
  records_evaluated: number;
  eligible_records: number;
  records_with_predicted_skip: number;
  records_with_zero_savings: number;
  total_baseline_test_time_s: number;
  total_dtl_test_time_s: number;
  total_estimated_seconds_saved: number;
  estimated_seconds_saved_per_record: number;
  estimated_seconds_saved_per_eligible_record?: number;
  predicted_time_saved_pct: number;
  tester_hours_saved: number;
  tester_cost_per_hour: number;
  total_predicted_cost_saving: number;
  predicted_cost_saved_per_record: number;
  predicted_cost_saved_per_1000_records: number;
  production_volume_supplied: boolean;
  note?: string;
}

export interface CostSavingsEstimatorMeta {
  type: string;
  production_facing?: boolean;
  read_only?: boolean;
  mechanism: string;
  label: string;
  condition_duration_s: number;
  skip_threshold: number;
  tester_cost_per_hour: number;
  n_baseline_conditions: number;
  first_condition_id: string;
  cost_source: string;
  duration_source?: string;
  skip_threshold_source?: string;
  formulas?: Record<string, string>;
  assumptions?: Record<string, number | string>;
}

export interface SelectedScopeMeta {
  category?: string;
  lot_id?: string;
  die_id?: string;
  production_month?: string;
}

export interface CostSavingsPayload {
  status: string;
  is_measured_ate_saving: boolean;
  label: string;
  disclaimer: string;
  estimator: CostSavingsEstimatorMeta;
  aggregate: CostSavingsAggregate;
  source?: Record<string, unknown>;
  selected_scope?: SelectedScopeMeta;
  per_device?: Array<Record<string, unknown>>;
}

export interface DieLevelIdentities {
  months: string[];
  categories: string[];
  lots_by_category: Record<string, string[]>;
  dies_by_lot: Record<string, string[]>;
  counts?: {
    lots: number;
    dies: number;
    dies_per_lot?: Record<string, number>;
    by_category_dies?: Record<string, number>;
  };
  stable_across_months?: boolean;
  note?: string;
  cache_coverage?: { cached_files: number; by_month: Record<string, number> };
}

export interface DieRecommendationPayload {
  recommendation: AnalysisRecommendationRow;
  candidates: Array<Partial<AnalysisCandidateRow> & {
    candidate_limit: number;
    simulated_yield?: number | null;
    safety_status?: string | null;
    eligible?: boolean;
    ml_score?: number | null;
    ml_rank?: number | null;
    is_current?: boolean;
    is_selected?: boolean;
  }>;
  cached?: boolean;
  cache_path?: string;
}

export interface DieHistoryPayload {
  lot_id: string;
  die_id: string;
  parameter: string;
  parameter_display: string;
  history: AnalysisRecommendationRow[];
}

export interface ObservedSummaryPayload {
  lot_id: string;
  die_id: string;
  observed_means: Record<string, Record<string, number | null>>;
}

export interface LotDiesBrowsePayload {
  production_month: string;
  lot_id: string;
  parameter: string;
  parameter_display: string;
  dies: AnalysisRecommendationRow[];
  summary: {
    dies: number;
    decision_counts: Record<string, number>;
    average_recommended_dtl: number | null;
    min_recommended_dtl: number | null;
    max_recommended_dtl: number | null;
  };
}

export const MONTH_OPTIONS = [
  { value: "2026-01", label: "January 2026" },
  { value: "2026-02", label: "February 2026" },
  { value: "2026-03", label: "March 2026" },
] as const;

export const DEFAULT_ANALYSIS_MONTH = "2026-01";
export const DEFAULT_ANALYSIS_PARAMETER = "IR_DROP_MV";
export const DEFAULT_ANALYSIS_LOT = "DTL_NORM_001";
export const DEFAULT_ANALYSIS_DIE = "DTL_NORM_001_D001";
