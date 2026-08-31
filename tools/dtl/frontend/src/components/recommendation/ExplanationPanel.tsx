import type { DTLRecommendation, SafetyCheck } from "@/api/types";
import { buildHumanReadableExplanation } from "@/utils/analysisDisplay";
import { formatScore, formatUnit } from "@/utils/formatUnit";

interface ExplanationPanelProps {
  rec: DTLRecommendation;
  /** When true, show the primary explanation expanded (decision-first layout). */
  primary?: boolean;
}

function namedCheck(rec: DTLRecommendation, name: string): SafetyCheck | undefined {
  return rec.safety_result?.checks?.find((c) => c.name === name);
}

function listHas(exp: Record<string, unknown>, key: string, name: string): boolean {
  const arr = exp[key];
  return Array.isArray(arr) && arr.includes(name);
}

/** Only report a gate when the backend recorded it. Never fabricate. */
function gateLine(rec: DTLRecommendation, name: string, label: string): string | null {
  const check = namedCheck(rec, name);
  if (check) {
    return check.passed ? `✓ ${label}` : null;
  }
  if (listHas(rec.explanation, "safety_checks_passed", name)) {
    return `✓ ${label}`;
  }
  return null;
}

export function ExplanationPanel({ rec, primary = false }: ExplanationPanelProps) {
  const exp = rec.explanation;
  const policyTrace = Array.isArray(exp.policy_trace) ? (exp.policy_trace as string[]) : [];
  const passed = Array.isArray(exp.safety_checks_passed)
    ? (exp.safety_checks_passed as string[])
    : [];
  const failed = Array.isArray(exp.safety_checks_failed)
    ? (exp.safety_checks_failed as string[])
    : [];

  const text = typeof exp.text === "string" ? exp.text : null;
  const policyReason = typeof exp.policy_reason === "string" ? exp.policy_reason : null;

  if (primary) {
    const isRecommend = rec.decision === "RECOMMEND";
    const currentFormatted = formatUnit(rec.current_limit, rec.unit);
    const recFormatted = formatUnit(rec.recommended_limit, rec.unit);
    const hasChange = isRecommend && rec.current_limit !== rec.recommended_limit;

    const yieldVal =
      typeof exp.selected_simulated_yield === "number"
        ? exp.selected_simulated_yield
        : rec.simulation_evidence?.simulated_yield;
    const yieldLabel =
      typeof yieldVal === "number" ? `${(yieldVal * 100).toFixed(2)}%` : null;
    const mlRank =
      typeof exp.ml_rank === "number"
        ? exp.ml_rank
        : rec.ml_rank;
    const yieldTie = exp.yield_tie === true;
    const safetyPass = rec.safety_result?.status === "PASS";

    const eligibility = [
      gateLine(rec, "catalog_membership", "Test data is valid"),
      gateLine(rec, "simulation_evidence", "Simulation supports the selection"),
      gateLine(rec, "condition_coverage", "All required test conditions are covered"),
      safetyPass ? "✓ Passed safety checks" : null,
    ].filter((line): line is string => Boolean(line));

    const explanationText = buildHumanReadableExplanation({
      recommendedVal: recFormatted,
      currentVal: currentFormatted,
      isRecommend,
      yieldTie,
      safetyPass,
      hasYield: yieldLabel != null,
      whySelectedRaw: text ?? undefined,
      selectionTextRaw: typeof exp.selection_text === "string" ? exp.selection_text : undefined,
    });

    return (
      <section
        className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-sm"
        aria-label="Why this decision"
      >
        <h2 className="text-sm font-semibold text-gray-200 mb-3">
          {isRecommend ? "Why was this DTL selected?" : "Why this decision?"}
        </h2>

        {eligibility.length > 0 && (
          <div className="mb-4">
            <ul className="space-y-1.5 text-sm text-green-400 font-medium" data-testid="why-eligibility">
              {eligibility.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Compact Structured Details: DTL Limit, Simulated Yield, ML Rank */}
        <div className="grid gap-3 sm:grid-cols-3 border-t border-b border-gray-800/80 py-3 mb-3">
          <div>
            <dt className="text-xs text-gray-500 uppercase tracking-wide">DTL Limit</dt>
            <dd className="font-mono text-gray-200 mt-0.5" data-testid="why-limit-change">
              {currentFormatted} → {hasChange ? recFormatted : "No change"}
            </dd>
          </div>

          {yieldLabel && (
            <div>
              <dt className="text-xs text-gray-500 uppercase tracking-wide">Simulated Yield</dt>
              <dd className="font-mono text-emerald-400 mt-0.5" data-testid="why-simulated-yield">
                {yieldLabel}
              </dd>
              <p className="text-[11px] text-gray-500 italic mt-0.5">Estimated from test data</p>
            </div>
          )}

          {typeof mlRank === "number" && (
            <div>
              <dt className="text-xs text-gray-500 uppercase tracking-wide">ML Rank</dt>
              <dd className="font-mono text-gray-200 mt-0.5" data-testid="why-ml-rank">
                #{mlRank}
              </dd>
            </div>
          )}
        </div>

        {/* Concise Human Explanation */}
        <div className="mb-3">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Explanation</p>
          <p className="text-sm text-gray-200 leading-relaxed" data-testid="why-decision-text">
            {explanationText}
          </p>
        </div>

        {/* Collapsible Technical Details */}
        <details className="mt-3 rounded border border-gray-800/80 bg-gray-950/60 p-2.5 text-xs text-gray-400" data-testid="technical-details">
          <summary className="cursor-pointer font-medium text-gray-400 hover:text-cyan-300 transition-colors">
            Technical details
          </summary>
          <dl className="mt-2.5 space-y-1.5 border-t border-gray-800/80 pt-2 font-mono text-[11px]">
            <div className="flex justify-between">
              <dt className="text-gray-500">Decision:</dt>
              <dd className="text-gray-300">{rec.decision}</dd>
            </div>
            {policyReason && (
              <div className="flex justify-between">
                <dt className="text-gray-500">Policy reason:</dt>
                <dd className="text-gray-300">{policyReason}</dd>
              </div>
            )}
            {rec.evidence_origin && (
              <div className="flex justify-between">
                <dt className="text-gray-500">Evidence origin:</dt>
                <dd className="text-gray-300 truncate max-w-[220px]">{rec.evidence_origin}</dd>
              </div>
            )}
            {typeof rec.ml_score === "number" && (
              <div className="flex justify-between">
                <dt className="text-gray-500">Raw ML score:</dt>
                <dd className="text-gray-300">{formatScore(rec.ml_score)}</dd>
              </div>
            )}
          </dl>
        </details>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4 text-sm space-y-3">
      <h2 className="text-sm font-semibold text-gray-200">Explanation detail</h2>
      {text && <p className="text-gray-300 leading-relaxed">{text}</p>}
      {policyReason && (
        <p className="text-gray-400">
          <span className="text-gray-500">Policy reason:</span> {policyReason}
        </p>
      )}
      {policyTrace.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Policy trace</p>
          <ul className="list-disc pl-5 text-xs text-gray-400 space-y-1">
            {policyTrace.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}
      {(passed.length > 0 || failed.length > 0) && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          {passed.length > 0 && (
            <div>
              <p className="text-gray-500">Checks passed</p>
              <ul className="text-green-400">
                {passed.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </div>
          )}
          {failed.length > 0 && (
            <div>
              <p className="text-gray-500">Checks failed</p>
              <ul className="text-red-400">
                {failed.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {typeof exp.disclaimer === "string" && (
        <p className="text-xs text-gray-500 border-t border-gray-800 pt-3">{exp.disclaimer}</p>
      )}
      {typeof exp.evidence_origin === "string" && (
        <p className="text-xs text-gray-500">Evidence origin: {exp.evidence_origin}</p>
      )}
    </section>
  );
}
