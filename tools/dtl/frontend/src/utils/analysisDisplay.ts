/** Pure display helpers for Phase 13.0 — no recommendation logic. */

import type { AnalysisRecommendationRow } from "@/api/analysisTypes";
import { formatPercent, formatUnit } from "@/utils/formatUnit";

export function monthLabel(month: string): string {
  if (month === "2026-01") return "January 2026";
  if (month === "2026-02") return "February 2026";
  if (month === "2026-03") return "March 2026";
  return month;
}

export function shortMonth(month: string): string {
  if (month === "2026-01") return "Jan 2026";
  if (month === "2026-02") return "Feb 2026";
  if (month === "2026-03") return "Mar 2026";
  return month;
}

export function modelFriendlyName(modelUsed: string | null | undefined): string {
  // Presentation: never surface architecture/checkpoint names in the UI.
  if (!modelUsed) return "—";
  return "AI ranking";
}

/** Strip architecture jargon from backend explanation text for display only. */
export function sanitizeUiText(text: string): string {
  return text
    .replace(/core_gru_temporal_v1/gi, "AI ranking")
    .replace(/unified_parameter_gru_v1/gi, "AI ranking")
    .replace(/Unified\s*Parameter\s*GRU/gi, "AI ranking")
    .replace(/CoreGRU/gi, "AI ranking")
    .replace(/\bGRU\b/gi, "AI ranking")
    .replace(/neural\s*network/gi, "ranking model");
}

export function findRecommendation(
  rows: AnalysisRecommendationRow[],
  opts: {
    month: string;
    parameterDisplay: string;
    lotId: string;
    dieId: string;
  },
): AnalysisRecommendationRow | undefined {
  return rows.find(
    (r) =>
      r.production_month === opts.month &&
      r.parameter_display === opts.parameterDisplay &&
      r.lot_id === opts.lotId &&
      r.die_id === opts.dieId,
  );
}

export function historyForParameter(
  rows: AnalysisRecommendationRow[],
  opts: { parameterDisplay: string; lotId: string; dieId: string },
): AnalysisRecommendationRow[] {
  const order = ["2026-01", "2026-02", "2026-03"];
  return order
    .map((m) =>
      rows.find(
        (r) =>
          r.production_month === m &&
          r.parameter_display === opts.parameterDisplay &&
          r.lot_id === opts.lotId &&
          r.die_id === opts.dieId,
      ),
    )
    .filter((r): r is AnalysisRecommendationRow => Boolean(r));
}

export function formatSimulatedYield(value: number | null | undefined): string {
  return formatPercent(value);
}

export function formatLimit(row: AnalysisRecommendationRow | undefined, which: "current" | "recommended"): string {
  if (!row) return "—";
  const v = which === "current" ? row.current_limit : row.recommended_limit;
  return formatUnit(v, row.unit ?? "");
}

export function displayDelta(row: AnalysisRecommendationRow | undefined): {
  abs: string;
  pct: string;
} {
  if (!row || row.recommendation_delta == null) {
    return { abs: "—", pct: "—" };
  }
  const sign = row.recommendation_delta > 0 ? "+" : "";
  const abs = `${sign}${formatUnit(row.recommendation_delta, row.unit ?? "")}`;
  const pct =
    row.recommendation_delta_percent == null
      ? "—"
      : `${row.recommendation_delta_percent > 0 ? "+" : ""}${row.recommendation_delta_percent.toFixed(1)}%`;
  return { abs, pct };
}

/** Generate clean, human-readable explanation sentences dynamically without backend jargon. */
export function buildHumanReadableExplanation(opts: {
  recommendedVal: string;
  currentVal: string;
  isRecommend: boolean;
  yieldTie?: boolean;
  safetyPass?: boolean;
  hasYield?: boolean;
  whySelectedRaw?: string;
  selectionTextRaw?: string;
}): string {
  const { recommendedVal, currentVal, isRecommend, yieldTie, safetyPass, hasYield, whySelectedRaw } = opts;

  if (whySelectedRaw && whySelectedRaw.trim().length > 0 && !whySelectedRaw.startsWith("policy_reason=")) {
    return sanitizeUiText(whySelectedRaw);
  }

  if (!isRecommend || recommendedVal === currentVal) {
    return "The current limit already satisfies the required checks, so no change was recommended.";
  }

  if (yieldTie) {
    return `Several candidates had the same simulated yield, so ML rank tie-break was used to select ${recommendedVal}.`;
  }

  if (safetyPass && hasYield) {
    return `${recommendedVal} was selected because it passed the required safety checks and provided the highest simulated yield among eligible candidates.`;
  }

  if (safetyPass) {
    return `${recommendedVal} was selected because it was the highest candidate that passed the required safety checks.`;
  }

  return `${recommendedVal} was selected based on the available test-data simulation while satisfying the required safety checks.`;
}

/** Prefer backend why_selected / selection_text — never invent policy. */
export function whySelectedText(row: AnalysisRecommendationRow | undefined): string {
  if (!row) return "Recommendation unavailable for this selection.";
  const raw =
    row.why_selected ||
    row.selection_text ||
    row.explanation_text ||
    (row.policy_reason ? `policy_reason=${row.policy_reason}` : null) ||
    "No explanation text provided by the analysis artifact.";
  return sanitizeUiText(String(raw));
}
