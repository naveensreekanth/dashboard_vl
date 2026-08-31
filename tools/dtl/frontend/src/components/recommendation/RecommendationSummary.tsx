import type { DTLRecommendation } from "@/api/types";
import { EvidenceLevelBadge } from "@/components/evidence/EvidenceLevelBadge";
import { formatUnit } from "@/utils/formatUnit";
import { modelDisplayName } from "@/utils/formatEvidence";

interface RecommendationSummaryProps {
  rec: DTLRecommendation;
}

export function RecommendationSummary({ rec }: RecommendationSummaryProps) {
  const modelLabel = rec.model_used ?? rec.model_id;
  const policyReason =
    typeof rec.explanation?.policy_reason === "string"
      ? rec.explanation.policy_reason
      : null;
  const maxYield =
    typeof rec.simulation_evidence?.simulated_yield === "number"
      ? rec.simulation_evidence.simulated_yield
      : null;

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Recommendation summary">
      <h2 className="text-sm font-semibold text-gray-200 mb-3">DTL Recommendation</h2>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-xs text-gray-500">Production Month</dt>
          <dd className="font-mono" data-testid="summary-production-month">
            {rec.production_month ?? "legacy"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Model Used</dt>
          <dd data-testid="summary-model-used">{modelDisplayName(modelLabel)}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Current Limit</dt>
          <dd className="font-mono text-violet-300">{formatUnit(rec.current_limit, rec.unit)}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Recommended Limit</dt>
          <dd className="font-mono text-cyan-300">{formatUnit(rec.recommended_limit, rec.unit)}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Max Eligible Yield</dt>
          <dd className="font-mono" data-testid="summary-max-yield">
            {maxYield == null ? "—" : maxYield.toFixed(4)}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">ML Rank</dt>
          <dd className="font-mono" data-testid="summary-ml-rank">
            {rec.ml_rank ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Decision</dt>
          <dd className="font-mono">{rec.decision}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Policy Reason</dt>
          <dd className="font-mono text-xs" data-testid="summary-policy-reason">
            {policyReason ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Evidence Origin</dt>
          <dd className="text-cyan-400">{rec.evidence_origin}</dd>
        </div>
        <div>
          <dt className="text-xs text-gray-500">Evidence Level</dt>
          <dd>
            <EvidenceLevelBadge level={rec.evidence_level} />
          </dd>
        </div>
      </dl>
      <p className="mt-3 text-xs text-gray-500 font-mono">Request: {rec.request_id}</p>
    </section>
  );
}
