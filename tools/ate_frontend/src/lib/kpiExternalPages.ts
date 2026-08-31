/**
 * External pages opened when an Optimization KPI card is clicked.
 * Replace via NEXT_PUBLIC_KPI_* env vars when needed.
 */
export const SHMOO_VL_BASE =
  process.env.NEXT_PUBLIC_KPI_M_BIST_SHMOO_URL?.trim() ||
  "https://shmoo-vl.vercel.app";

/** Capability tabs + embedded metric ids (hidden KPIs powering the parent card). */
export const SHMOO_CAPABILITIES = [
  {
    id: "yield",
    label: "Yield Analysis",
    view: "yield",
    metricKpiId: "shmoo_yield_analysis",
    primaryLabel: "Value",
    secondaryLabel: "Margin",
    secondaryValue: 4.2,
    secondaryUnit: "%",
  },
  {
    id: "debug",
    label: "Debugging",
    view: "debug",
    metricKpiId: "shmoo_debugging",
    primaryLabel: "Value",
    secondaryLabel: "Issues",
    secondaryValue: 3,
    secondaryUnit: "",
  },
  {
    id: "binning",
    label: "Binning",
    view: "binning",
    metricKpiId: "shmoo_binning",
    primaryLabel: "Value",
    secondaryLabel: "Bins",
    secondaryValue: 8,
    secondaryUnit: "",
  },
  {
    id: "character",
    label: "Characterization",
    view: "character",
    metricKpiId: "shmoo_characterization",
    primaryLabel: "Value",
    secondaryLabel: "Rec. OP",
    secondaryValue: 92.5,
    secondaryUnit: "%",
  },
] as const;

export type ShmooCapabilityId = (typeof SHMOO_CAPABILITIES)[number]["id"];

/** Popup + card shared metric shape (primary value + secondary demo fields). */
export type ShmooCapabilityMetric = {
  id: ShmooCapabilityId;
  label: string;
  value: number;
  unit: string;
  primaryLabel?: string;
  secondaryLabel?: string;
  secondaryValue?: number;
  secondaryUnit?: string;
};

export function shmooCapabilityUrl(view: string): string {
  const base = SHMOO_VL_BASE.replace(/\/$/, "");
  return `${base}?view=${encodeURIComponent(view)}`;
}

/** Combined Test Time Optimization parent (Test Time + Vector Memory). */
export const TEST_TIME_OPT_URL =
  process.env.NEXT_PUBLIC_KPI_TEST_TIME_URL?.trim() ||
  "https://test-time-optimization-l69q.onrender.com";

export const TEST_TIME_CAPABILITIES = [
  {
    id: "test_time",
    label: "Test Time",
    metricKpiId: "test_time_reduction",
    primaryLabel: "Value",
  },
  {
    id: "vector_memory",
    label: "Vector Memory",
    metricKpiId: "vector_memory_optimization",
    primaryLabel: "Value",
  },
] as const;

export type TestTimeCapabilityId = (typeof TEST_TIME_CAPABILITIES)[number]["id"];

export type TestTimeCapabilityMetric = {
  id: TestTimeCapabilityId;
  label: string;
  value: number;
  unit: string;
  primaryLabel?: string;
};

export const KPI_EXTERNAL_URLS: Record<string, string | undefined> = {
  false_failure_reduction:
    process.env.NEXT_PUBLIC_KPI_FALSE_FAILURE_URL ??
    "https://placeholder-false-failure.vercel.app",
  test_time_reduction: TEST_TIME_OPT_URL,
  yield_improvement:
    process.env.NEXT_PUBLIC_KPI_YIELD_URL ?? "https://placeholder-yield.vercel.app",
  retest_reduction:
    process.env.NEXT_PUBLIC_KPI_RETEST_URL ??
    "https://ate-retest-benefit-prediction-ai-n9yvp4wajm9yeq4vwfzhmq.streamlit.app",
  escape_prevention:
    process.env.NEXT_PUBLIC_KPI_ESCAPE_URL ?? "https://placeholder-escape.vercel.app",
  vector_memory_optimization:
    process.env.NEXT_PUBLIC_KPI_VECTOR_MEMORY_URL ??
    "https://placeholder-vector-memory.vercel.app",
  pattern_count_reduction:
    process.env.NEXT_PUBLIC_KPI_PATTERN_COUNT_URL ??
    "https://placeholder-pattern-count.vercel.app",
  m_bist_shmoo: SHMOO_VL_BASE,
};

export function getKpiExternalUrl(kpiId: string): string | undefined {
  const url = KPI_EXTERNAL_URLS[kpiId]?.trim();
  return url || undefined;
}

/**
 * AI Recommended Retest / Don't Retest counts shown on the Retest Reduction card.
 * Demo values aligned with the Streamlit overview until a live API exists.
 */
export type RetestAiRecommendation = {
  id: "retest" | "dont_retest";
  title: string;
  events: number;
  devices: number;
  /** Text color for Events / Devices values */
  valueColor: string;
  borderColor: string;
  bgColor: string;
};

export const RETEST_AI_RECOMMENDATIONS: RetestAiRecommendation[] = [
  {
    id: "retest",
    title: "Retest",
    events: 119,
    devices: 71,
    valueColor: "#6EE7A8",
    borderColor: "rgba(110, 231, 168, 0.35)",
    bgColor: "rgba(110, 231, 168, 0.1)",
  },
  {
    id: "dont_retest",
    title: "Don't Retest",
    events: 81,
    devices: 24,
    valueColor: "#F0667A",
    borderColor: "rgba(240, 102, 122, 0.35)",
    bgColor: "rgba(240, 102, 122, 0.1)",
  },
];

export function isPlaceholderKpiUrl(url: string): boolean {
  return /placeholder-/i.test(url) || /example\.com/i.test(url);
}

/** Streamlit Cloud apps break in third-party iframes (cookie redirect loops). */
export function isStreamlitKpiUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === "share.streamlit.io" || host.endsWith(".streamlit.app");
  } catch {
    return /streamlit\.(app|io)/i.test(url);
  }
}

/** External tools that should open in a new tab (not iframe). */
export function isExternalToolOnlyUrl(url: string): boolean {
  if (isStreamlitKpiUrl(url)) return true;
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host.endsWith(".onrender.com");
  } catch {
    return /onrender\.com/i.test(url);
  }
}
