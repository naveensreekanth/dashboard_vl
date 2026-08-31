import type { DTLRecommendation } from "@/api/types";
import { formatScore } from "@/utils/formatUnit";
import { isCoreParameter, modelDisplayName } from "@/utils/formatEvidence";

interface ModelPanelProps {
  rec: DTLRecommendation;
  jointEnabled: boolean;
  treeDiagnostic: boolean;
}

export function ModelPanel({ rec, jointEnabled, treeDiagnostic }: ModelPanelProps) {
  const coreParam = isCoreParameter(rec.parameter);

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Ranking details">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">Ranking Details</h2>
      <div className="space-y-3 text-sm">
        <div className="rounded border border-gray-800 bg-gray-950 p-3">
          <p className="text-xs text-gray-500">Active ranking for this parameter</p>
          <p className="font-mono text-cyan-300">{modelDisplayName(rec.model_used ?? rec.model_id)}</p>
          <p className="text-xs text-gray-500 mt-1">
            Domain: {coreParam ? "Core" : "Parametric"}
            {rec.production_month ? ` · month ${rec.production_month}` : ""}
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <dt className="text-gray-500">ML Score</dt>
            <dd className="font-mono">{formatScore(rec.ml_score)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">ML Rank</dt>
            <dd className="font-mono">{rec.ml_rank ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Candidates</dt>
            <dd className="font-mono">{rec.n_candidates}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Source</dt>
            <dd className="font-mono">{rec.source_status}</dd>
          </div>
        </dl>
        <div className="border-t border-gray-800 pt-3 space-y-2 text-xs text-gray-500">
          <p>Core ranking: {rec.core_available ? "available" : "unavailable"} (IR / Thermal)</p>
          <p>Parametric ranking: {rec.parametric_available ? "available" : "unavailable"}</p>
          {jointEnabled ? <p>Joint mode enabled</p> : null}
          {treeDiagnostic ? <p>Tree diagnostic enabled</p> : null}
        </div>
      </div>
    </section>
  );
}
