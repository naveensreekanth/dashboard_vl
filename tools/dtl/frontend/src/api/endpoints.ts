import { apiFetch } from "./client";
import type {
  DieHistoryPayload,
  DieLevelIdentities,
  DieRecommendationPayload,
  ObservedSummaryPayload,
  CostSavingsPayload,
  ThreeMonthAnalysisBundle,
} from "./analysisTypes";
import type {
  DieConditionsResponse,
  DieDistributionResponse,
  DieMeasurementResponse,
  HealthResponse,
  LotDieParametersResponse,
  LotDiesResponse,
  LotRecommendationResult,
  LotsResponse,
  ReadyResponse,
  RecommendationRequest,
} from "./types";

const PREFIX = "/api/v1";

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiFetch<HealthResponse>(`${PREFIX}/health`, { signal });
}

export function getReady(signal?: AbortSignal): Promise<ReadyResponse> {
  return apiFetch<ReadyResponse>(`${PREFIX}/ready`, { signal });
}

export function postRecommendation(
  body: RecommendationRequest,
  signal?: AbortSignal,
): Promise<LotRecommendationResult> {
  return apiFetch<LotRecommendationResult>(`${PREFIX}/recommendations`, {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}

export interface UploadRecommendationResult extends LotRecommendationResult {
  upload?: {
    source_filename: string;
    production_month: string;
    lot_id: string;
    die_id: string;
    parameters: string[];
    used_uploaded_measurements: boolean;
    used_static_three_month_measurements: boolean;
    parametric_uploaded: boolean;
    input_format: string;
  };
}

export async function postRecommendationUpload(
  file: File,
  options?: {
    parametricFile?: File | null;
    parameters?: string[];
    signal?: AbortSignal;
  },
): Promise<UploadRecommendationResult> {
  const form = new FormData();
  form.append("file", file);
  if (options?.parametricFile) {
    form.append("parametric_file", options.parametricFile);
  }
  if (options?.parameters?.length) {
    form.append("parameters", options.parameters.join(","));
  }
  return apiFetch<UploadRecommendationResult>(`${PREFIX}/recommendations/upload`, {
    method: "POST",
    body: form,
    signal: options?.signal,
  });
}

export interface AnalysisUploadResult {
  analysis_session_id: string;
  months: string[];
  status: string;
  stage?: string;
  progress_pct?: number;
  used_uploaded_measurements: boolean;
  used_static_three_month_measurements: boolean;
  source_files?: Record<string, string>;
  primary_die?: { lot_id: string; die_id: string; lot_category?: string };
  scorable_parameters?: string[];
  data_provenance?: string;
}

export interface AnalysisUploadStatus {
  analysis_session_id: string;
  status: "queued" | "processing" | "completed" | "failed" | string;
  stage: string;
  progress_pct: number;
  error?: string | null;
  months?: string[];
  used_uploaded_measurements?: boolean;
  used_static_three_month_measurements?: boolean;
  source_files?: Record<string, string>;
  primary_die?: { lot_id: string; die_id: string; lot_category?: string };
  scorable_parameters?: string[];
  data_provenance?: string;
}

export async function postAnalysisUpload(
  files: { january: File; february: File; march: File },
  signal?: AbortSignal,
): Promise<AnalysisUploadResult> {
  const form = new FormData();
  form.append("january", files.january);
  form.append("february", files.february);
  form.append("march", files.march);
  return apiFetch<AnalysisUploadResult>(`${PREFIX}/analysis/upload`, {
    method: "POST",
    body: form,
    signal,
  });
}

export function getUploadStatus(
  analysisSessionId: string,
  signal?: AbortSignal,
): Promise<AnalysisUploadStatus> {
  return apiFetch<AnalysisUploadStatus>(
    `${PREFIX}/analysis/upload/status/${encodeURIComponent(analysisSessionId)}`,
    { signal },
  );
}

function withSession(path: string, analysisSessionId?: string | null): string {
  if (!analysisSessionId) return path;
  const join = path.includes("?") ? "&" : "?";
  return `${path}${join}analysis_session_id=${encodeURIComponent(analysisSessionId)}`;
}

export function getThreeMonthAnalysis(
  signal?: AbortSignal,
  analysisSessionId?: string | null,
): Promise<ThreeMonthAnalysisBundle> {
  return apiFetch<ThreeMonthAnalysisBundle>(
    withSession(`${PREFIX}/analysis/three-month`, analysisSessionId),
    { signal },
  );
}

export function getCostSavings(
  params?: {
    condition_duration_s?: number;
    skip_threshold?: number;
    tester_cost_per_hour?: number;
    include_per_device?: boolean;
    analysis_session_id?: string | null;
    lot_id?: string | null;
    die_id?: string | null;
    production_month?: string | null;
  },
  signal?: AbortSignal,
): Promise<CostSavingsPayload> {
  const query = new URLSearchParams();
  if (params?.condition_duration_s != null) {
    query.set("condition_duration_s", String(params.condition_duration_s));
  }
  if (params?.skip_threshold != null) {
    query.set("skip_threshold", String(params.skip_threshold));
  }
  if (params?.tester_cost_per_hour != null) {
    query.set("tester_cost_per_hour", String(params.tester_cost_per_hour));
  }
  if (params?.include_per_device != null) {
    query.set("include_per_device", String(params.include_per_device));
  }
  if (params?.analysis_session_id) {
    query.set("analysis_session_id", params.analysis_session_id);
  }
  if (params?.lot_id) {
    query.set("lot_id", params.lot_id);
  }
  if (params?.die_id) {
    query.set("die_id", params.die_id);
  }
  if (params?.production_month) {
    query.set("production_month", params.production_month);
  }
  const qs = query.toString();
  return apiFetch<CostSavingsPayload>(
    `${PREFIX}/analysis/cost-savings${qs ? `?${qs}` : ""}`,
    { signal },
  );
}

export function getThreeMonthIdentities(
  signal?: AbortSignal,
  analysisSessionId?: string | null,
): Promise<DieLevelIdentities> {
  return apiFetch<DieLevelIdentities>(
    withSession(`${PREFIX}/analysis/three-month/identities`, analysisSessionId),
    { signal },
  );
}

export function getThreeMonthDieRecommendation(
  params: {
    production_month: string;
    lot_id: string;
    die_id: string;
    parameter: string;
    force_refresh?: boolean;
    analysis_session_id?: string | null;
  },
  signal?: AbortSignal,
): Promise<DieRecommendationPayload> {
  const query = new URLSearchParams({
    production_month: params.production_month,
    lot_id: params.lot_id,
    die_id: params.die_id,
    parameter: params.parameter,
  });
  if (params.force_refresh) query.set("force_refresh", "true");
  if (params.analysis_session_id) {
    query.set("analysis_session_id", params.analysis_session_id);
  }
  return apiFetch<DieRecommendationPayload>(
    `${PREFIX}/analysis/three-month/dies?${query.toString()}`,
    { signal },
  );
}

export function getThreeMonthDieHistory(
  params: {
    lot_id: string;
    die_id: string;
    parameter: string;
    analysis_session_id?: string | null;
  },
  signal?: AbortSignal,
): Promise<DieHistoryPayload> {
  const query = new URLSearchParams({
    lot_id: params.lot_id,
    die_id: params.die_id,
    parameter: params.parameter,
  });
  if (params.analysis_session_id) {
    query.set("analysis_session_id", params.analysis_session_id);
  }
  return apiFetch<DieHistoryPayload>(
    `${PREFIX}/analysis/three-month/die-history?${query.toString()}`,
    { signal },
  );
}

export function getThreeMonthObserved(
  params: {
    lot_id: string;
    die_id: string;
    analysis_session_id?: string | null;
  },
  signal?: AbortSignal,
): Promise<ObservedSummaryPayload> {
  const query = new URLSearchParams({
    lot_id: params.lot_id,
    die_id: params.die_id,
  });
  if (params.analysis_session_id) {
    query.set("analysis_session_id", params.analysis_session_id);
  }
  return apiFetch<ObservedSummaryPayload>(
    `${PREFIX}/analysis/three-month/observed?${query.toString()}`,
    { signal },
  );
}

export function getLots(signal?: AbortSignal): Promise<LotsResponse> {
  return apiFetch<LotsResponse>(`${PREFIX}/lots`, { signal });
}

export function getLotDies(lotId: string, signal?: AbortSignal): Promise<LotDiesResponse> {
  return apiFetch<LotDiesResponse>(
    `${PREFIX}/lots/${encodeURIComponent(lotId)}/dies`,
    { signal },
  );
}

export function getLotDieParameters(
  lotId: string,
  dieId: string,
  signal?: AbortSignal,
): Promise<LotDieParametersResponse> {
  return apiFetch<LotDieParametersResponse>(
    `${PREFIX}/lots/${encodeURIComponent(lotId)}/dies/${encodeURIComponent(dieId)}/parameters`,
    { signal },
  );
}

export function getDieMeasurements(
  dieId: string,
  params: { lot_id: string; parameter: string; condition_id?: string },
  signal?: AbortSignal,
): Promise<DieMeasurementResponse> {
  const query = new URLSearchParams({
    lot_id: params.lot_id,
    parameter: params.parameter,
  });
  if (params.condition_id) {
    query.set("condition_id", params.condition_id);
  }
  return apiFetch<DieMeasurementResponse>(
    `${PREFIX}/dies/${encodeURIComponent(dieId)}/measurements?${query.toString()}`,
    { signal },
  );
}

export function getDieDistribution(
  dieId: string,
  params: {
    lot_id: string;
    parameter: string;
    scope?: "die" | "lot";
    condition_id?: string;
  },
  signal?: AbortSignal,
): Promise<DieDistributionResponse> {
  const query = new URLSearchParams({
    lot_id: params.lot_id,
    parameter: params.parameter,
    scope: params.scope ?? "die",
  });
  if (params.condition_id) {
    query.set("condition_id", params.condition_id);
  }
  return apiFetch<DieDistributionResponse>(
    `${PREFIX}/dies/${encodeURIComponent(dieId)}/distribution?${query.toString()}`,
    { signal },
  );
}

export function getDieConditions(
  dieId: string,
  params: { lot_id: string; parameter: string },
  signal?: AbortSignal,
): Promise<DieConditionsResponse> {
  const query = new URLSearchParams({
    lot_id: params.lot_id,
    parameter: params.parameter,
  });
  return apiFetch<DieConditionsResponse>(
    `${PREFIX}/dies/${encodeURIComponent(dieId)}/conditions?${query.toString()}`,
    { signal },
  );
}
