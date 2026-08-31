import type { CandidateSetEntry, Decision, DTLRecommendation } from "@/api/types";
import { formatUnit } from "@/utils/formatUnit";

export interface DecisionContextPanelProps {
  rec: DTLRecommendation;
  candidates: CandidateSetEntry[];
}

function strField(exp: Record<string, unknown>, key: string): string | null {
  const v = exp[key];
  return typeof v === "string" ? v : null;
}

export function DecisionContextPanel({ rec }: DecisionContextPanelProps) {
  const actionText = strField(rec.explanation, "action_text");
  const isRecommend = rec.decision === "RECOMMEND";
  const isKeep = rec.decision === "KEEP_CURRENT";
  const isReview = rec.decision === "REVIEW_REQUIRED";
  const isReject = rec.decision === "REJECT";

  const actionHeading = isRecommend
    ? "Change DTL"
    : isKeep
      ? "Keep current DTL"
      : isReview
        ? "Engineering review required"
        : "Recommendation rejected";

  const actionBody = isRecommend
    ? (actionText ??
      `${formatUnit(rec.current_limit, rec.unit)} → ${formatUnit(rec.recommended_limit, rec.unit)}`)
    : isKeep
      ? formatUnit(rec.current_limit, rec.unit)
      : isReview
        ? "Do not change DTL until required evidence is available."
        : "Do not change DTL.";

  return (
    <section
      className="rounded-lg border border-gray-800 bg-gray-900 p-4"
      aria-label="DTL decision"
    >
      <h2 className="text-sm font-semibold text-gray-200 mb-3">DTL Decision</h2>

      <div className="grid gap-3 sm:grid-cols-2 text-sm">
        {(rec.production_month || rec.model_used) && (
          <>
            <div className="rounded border border-gray-800 bg-gray-950 p-3" data-testid="context-production-month">
              <p className="text-xs text-gray-500">Production Month</p>
              <p className="font-mono text-lg text-gray-200">{rec.production_month ?? "legacy"}</p>
            </div>
            <div className="rounded border border-gray-800 bg-gray-950 p-3" data-testid="context-model-used">
              <p className="text-xs text-gray-500">Model Used</p>
              <p className="font-mono text-lg text-cyan-300">{rec.model_used ?? rec.model_id ?? "—"}</p>
            </div>
          </>
        )}
        <div
          className="rounded border border-violet-800 bg-gray-950 p-3"
          data-testid="context-current-limit"
        >
          <p className="text-xs text-gray-500">Current DTL</p>
          <p className="font-mono text-lg text-violet-300">
            {formatUnit(rec.current_limit, rec.unit)}
          </p>
        </div>

        {isRecommend && (
          <div
            className="rounded border border-cyan-800 bg-gray-950 p-3"
            data-testid="context-final-recommendation"
          >
            <p className="text-xs text-gray-500">Recommended DTL</p>
            <p className="font-mono text-lg text-cyan-300">
              {formatUnit(rec.recommended_limit, rec.unit)}
            </p>
          </div>
        )}

        {isKeep && (
          <div
            className="rounded border border-cyan-800 bg-gray-950 p-3"
            data-testid="context-final-recommendation"
          >
            <p className="text-xs text-gray-500">Recommended DTL</p>
            <p className="font-mono text-lg text-cyan-300">
              {formatUnit(rec.recommended_limit, rec.unit)}
            </p>
          </div>
        )}

        {isReview && (
          <div
            className="rounded border border-amber-800 bg-gray-950 p-3"
            data-testid="context-final-recommendation"
          >
            <p className="text-xs text-gray-500">Recommended DTL</p>
            <p className="text-lg text-amber-300">Engineering review required</p>
          </div>
        )}

        {isReject && (
          <div
            className="rounded border border-red-800 bg-gray-950 p-3"
            data-testid="context-final-recommendation"
          >
            <p className="text-xs text-gray-500">Recommended DTL</p>
            <p className="text-lg text-red-300">Recommendation rejected</p>
          </div>
        )}
      </div>

      <div className="mt-3 rounded border border-gray-800 bg-gray-950 p-3" data-testid="context-action">
        <p className="text-xs uppercase tracking-wide text-gray-500">{actionHeading}</p>
        <p className="text-sm text-gray-100 mt-1 font-mono">{actionBody}</p>
      </div>

      <p className="sr-only" data-testid="context-decision" data-decision={rec.decision as Decision}>
        {rec.decision}
      </p>
    </section>
  );
}
