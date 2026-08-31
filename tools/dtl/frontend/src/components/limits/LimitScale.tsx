import type { CandidateSetEntry, DTLRecommendation } from "@/api/types";
import { formatUnit } from "@/utils/formatUnit";

interface LimitScaleProps {
  rec: DTLRecommendation;
  candidates: CandidateSetEntry[];
}

export function LimitScale({ rec, candidates }: LimitScaleProps) {
  const paramCandidates = candidates.filter((c) => c.parameter === rec.parameter);
  const limits = paramCandidates.map((c) => c.candidate_limit);
  const current = rec.current_limit;
  const recommended = rec.recommended_limit;

  if (limits.length === 0) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Limit Scale</h2>
        <p className="text-sm text-gray-500">No candidate limits in audit record.</p>
      </section>
    );
  }

  const allValues = [...limits];
  if (current !== null) allValues.push(current);
  if (recommended !== null) allValues.push(recommended);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const span = max - min || 1;

  const toPct = (v: number) => ((v - min) / span) * 100;

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Limit scale">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-200">Candidate DTL Range</h2>
        <span className="text-xs text-gray-500">
          Direction: <span className="font-mono text-gray-300">{rec.direction}</span>
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-3">
        Candidate DTL values from the recommendation audit — not the observed measurement range.
      </p>
      <div className="relative h-16 rounded bg-gray-950 border border-gray-800">
        <div className="absolute inset-x-4 top-1/2 h-px bg-gray-700" />
        {paramCandidates.map((c) => {
          const isCurrent = current !== null && c.candidate_limit === current;
          const isRecommended =
            recommended !== null && c.candidate_limit === recommended && !isCurrent;
          const isRank1 = c.ml_rank === 1;
          return (
            <div
              key={`${c.candidate_limit}-${c.ml_rank}`}
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2"
              style={{ left: `calc(1rem + ${toPct(c.candidate_limit)}% * (100% - 2rem) / 100)` }}
              title={`${formatUnit(c.candidate_limit, rec.unit)} rank ${c.ml_rank ?? "—"}`}
            >
              <div
                className={`h-3 w-3 rounded-full ${
                  isCurrent
                    ? "bg-violet-400 ring-2 ring-violet-600"
                    : isRecommended
                      ? "bg-cyan-400 ring-2 ring-cyan-600"
                      : isRank1
                        ? "bg-cyan-300"
                        : "bg-gray-500"
                }`}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex justify-between text-xs font-mono text-gray-500">
        <span>{formatUnit(min, rec.unit)}</span>
        <span>{formatUnit(max, rec.unit)}</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-4 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-violet-400" /> Current DTL
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-cyan-400" /> Final / ML Top Candidate
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-gray-500" /> Candidate
        </span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded border border-violet-800 bg-gray-950 p-3">
          <p className="text-xs text-gray-500">CURRENT DTL</p>
          <p className="font-mono text-violet-300">{formatUnit(rec.current_limit, rec.unit)}</p>
        </div>
        <div className="rounded border border-cyan-800 bg-gray-950 p-3">
          <p className="text-xs text-gray-500">FINAL RECOMMENDATION</p>
          <p className="font-mono text-cyan-300">
            {formatUnit(rec.recommended_limit, rec.unit)}
          </p>
        </div>
      </div>
    </section>
  );
}
