export interface ShmooMeta {
  n_points: number;
  n_dies: number;
  is_multi_die: boolean;
  die_id_cols: string[];
  vdd_range: [number, number];
  freq_range: [number, number];
  pass_rate: number;
  n_pass: number;
  n_fail: number;
  failure_codes: Record<string, number>;
  lot_id: string;
  wafer_id: string;
  die_id: string;
  temp_c: number | null;
}

export interface ShmooFaultPattern {
  source?: string;
  pattern?: string;
  fail_count: number;
  fault_type?: string;
}

export interface ShmooResults {
  accuracy: number;
  cv_accuracy: number;
  cv_std: number;
  boundary_slope: number;
  boundary_intercept: number;
  boundary_r2: number;
  recommended_vdd: number;
  recommended_freq: number;
  voltage_margin_v: number;
  freq_margin_ghz: number;
  n_pass: number;
  n_fail: number;
  failure_code_dist: Record<string, number>;
  critical_fault_patterns: ShmooFaultPattern[];
  timing_fail_patterns: ShmooFaultPattern[];
  yield_by_vdd?: Record<string, number>;
  fmax_by_vdd?: Record<string, number>;
}

export interface ShmooUploadResponse {
  status: string;
  session_id: string;
  filename?: string;
  meta: ShmooMeta;
  results: ShmooResults;
  plot_url: string;
  plot_urls?: {
    character?: string;
    yield?: string;
    debug?: string;
  };
}
