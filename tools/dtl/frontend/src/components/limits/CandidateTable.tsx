import type { CandidateSetEntry, DTLRecommendation } from "@/api/types";
import { formatPercent, formatScore, formatUnit } from "@/utils/formatUnit";

interface CandidateTableProps {
  rec: DTLRecommendation;
  candidates: CandidateSetEntry[];
  simulationRows: Record<string, unknown>[];
}

function simRowFor(
  rows: Record<string, unknown>[],
  candidateLimit: number,
): Record<string, unknown> | undefined {
  return rows.find((r) => Number(r.candidate_limit) === candidateLimit);
}

export function CandidateTable({ rec, candidates, simulationRows }: CandidateTableProps) {
  const rows = candidates
    .filter((c) => c.parameter === rec.parameter)
    .sort((a, b) => {
      const ya = Number(simRowFor(simulationRows, a.candidate_limit)?.simulated_yield);
      const yb = Number(simRowFor(simulationRows, b.candidate_limit)?.simulated_yield);
      const na = Number.isFinite(ya) ? ya : Number.NEGATIVE_INFINITY;
      const nb = Number.isFinite(yb) ? yb : Number.NEGATIVE_INFINITY;
      if (nb !== na) return nb - na;
      return (a.ml_rank ?? 999) - (b.ml_rank ?? 999);
    });

  if (rows.length === 0) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Candidate Comparison</h2>
        <p className="text-sm text-gray-500">No candidates returned in audit.</p>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4 overflow-x-auto">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">Candidate Comparison</h2>
      <p className="text-xs text-gray-500 mb-3">
        Sorted by simulated yield (primary), then ML rank (tie-breaker). Delta does not determine
        the winner. Yield and rank values are simulator-derived.
      </p>
      <table className="min-w-full text-left text-xs">
        <thead className="text-gray-500 border-b border-gray-800">
          <tr>
            <th className="py-2 pr-3">Candidate</th>
            <th className="py-2 pr-3 text-cyan-400" title="Primary selection criterion.">
              Simulated Yield
            </th>
            <th className="py-2 pr-3 text-amber-400" title="Tie-breaker when yields are equal.">
              ML Rank
            </th>
            <th className="py-2 pr-3" title="Score produced by the ML candidate-ranking stage.">
              ML Score
            </th>
            <th className="py-2 pr-3">Delta</th>
            <th className="py-2 pr-3" title="Simulated population violating this candidate.">
              Violation
            </th>
            <th className="py-2 pr-3" title="Simulated population near the configured boundary.">
              Borderline
            </th>
            <th className="py-2 pr-3">Type</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const isCurrent =
              rec.current_limit !== null && c.candidate_limit === rec.current_limit;
            const isRecommended =
              rec.recommended_limit !== null &&
              c.candidate_limit === rec.recommended_limit;
            const sim = simRowFor(simulationRows, c.candidate_limit);
            return (
              <tr
                key={`${c.candidate_limit}-${c.ml_rank}`}
                className={`border-b border-gray-800/50 ${
                  isCurrent
                    ? "bg-violet-950/30"
                    : isRecommended
                      ? "bg-cyan-950/20"
                      : ""
                }`}
              >
                <td className="py-2 pr-3 font-mono">
                  {formatUnit(c.candidate_limit, rec.unit)}
                  {isCurrent && (
                    <span className="ml-2 text-violet-400" aria-label="Current limit">
                      CURRENT
                    </span>
                  )}
                  {isRecommended && !isCurrent && (
                    <span className="ml-2 text-cyan-400" aria-label="Recommended">
                      SELECTED
                    </span>
                  )}
                </td>
                <td className="py-2 pr-3 font-mono text-sm font-semibold text-cyan-300">
                  {formatPercent(sim?.simulated_yield as number | null | undefined)}
                </td>
                <td className="py-2 pr-3 font-mono text-sm font-semibold text-amber-300">
                  {c.ml_rank ?? "—"}
                </td>
                <td className="py-2 pr-3 font-mono">{formatScore(c.ml_score ?? null)}</td>
                <td className="py-2 pr-3 font-mono text-gray-500">
                  {c.delta_absolute !== undefined ? formatUnit(c.delta_absolute, rec.unit) : "—"}
                </td>
                <td className="py-2 pr-3 font-mono">
                  {formatPercent(sim?.violation_rate as number | null | undefined)}
                </td>
                <td className="py-2 pr-3 font-mono">
                  {formatPercent(sim?.borderline_rate as number | null | undefined)}
                </td>
                <td className="py-2 pr-3">{c.tighten_or_loosen ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
