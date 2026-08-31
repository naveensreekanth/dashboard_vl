import type { DTLRecommendation } from "@/api/types";
import { formatPercent } from "@/utils/formatUnit";

interface SimulationEvidenceProps {
  rec: DTLRecommendation;
}

export function SimulationEvidencePanel({ rec }: SimulationEvidenceProps) {
  const ev = rec.simulation_evidence;

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Simulation evidence">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-200">Simulation Evidence</h2>
        <span className="rounded border border-cyan-800 px-2 py-0.5 text-xs text-cyan-400">
          {ev.evidence_origin || rec.evidence_origin}
        </span>
      </div>
      {!ev.found ? (
        <p className="text-sm text-amber-400">Simulation evidence unavailable — not fabricated.</p>
      ) : (
        <dl className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <dt className="text-xs text-gray-500">Yield</dt>
            <dd className="font-mono">{formatPercent(ev.simulated_yield)}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Violation Rate</dt>
            <dd className="font-mono">{formatPercent(ev.violation_rate)}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Borderline Rate</dt>
            <dd className="font-mono">{formatPercent(ev.borderline_rate)}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Worst Condition Yield</dt>
            <dd className="font-mono">{formatPercent(ev.worst_condition_yield)}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Worst Condition Violation</dt>
            <dd className="font-mono">{formatPercent(ev.worst_condition_violation_rate)}</dd>
          </div>
          <div>
            <dt className="text-xs text-gray-500">Evaluated Conditions</dt>
            <dd className="font-mono">{ev.evaluated_conditions ?? "—"}</dd>
          </div>
        </dl>
      )}
      <p className="mt-3 text-xs text-gray-500 border-t border-gray-800 pt-3">
        {ev.note ??
          "SIMULATOR_DERIVED evidence. objective_score is not production reliability or true optimality."}
      </p>
      {ev.population_level_aggregate && (
        <p className="mt-2 text-xs text-gray-500">Population-level aggregate evidence.</p>
      )}
    </section>
  );
}
