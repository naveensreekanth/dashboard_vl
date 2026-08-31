import type { CandidateSetEntry, DieDistributionResponse, DTLRecommendation } from "@/api/types";
import { LoadingState } from "@/components/common/LoadingState";
import { formatUnit } from "@/utils/formatUnit";

export interface ObservedRangeChartProps {
  distribution: DieDistributionResponse | null;
  loading?: boolean;
  error?: string | null;
  rec?: DTLRecommendation | null;
  candidates?: CandidateSetEntry[];
}

function clampPct(v: number): number {
  return Math.max(0, Math.min(100, v));
}

function Marker({
  pct,
  barClass,
  textClass,
  label,
  testId,
}: {
  pct: number;
  barClass: string;
  textClass: string;
  label: string;
  testId: string;
}) {
  return (
    <div
      className="absolute top-0 bottom-0 -translate-x-1/2"
      style={{ left: `${clampPct(pct)}%` }}
      data-testid={testId}
      title={label}
    >
      <div className={`mx-auto h-full w-0.5 ${barClass}`} />
      <div
        className={`absolute -bottom-5 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] font-semibold ${textClass}`}
      >
        {label}
      </div>
    </div>
  );
}

export function ObservedRangeChart({
  distribution,
  loading = false,
  error = null,
  rec = null,
  candidates = [],
}: ObservedRangeChartProps) {
  if (loading) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Distribution range">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Distribution Range</h2>
        <LoadingState message="Loading distribution…" />
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Distribution range">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Distribution Range</h2>
        <p className="text-sm text-red-400">Unable to load distribution data.</p>
      </section>
    );
  }

  if (
    !distribution ||
    !distribution.found ||
    distribution.min === null ||
    distribution.median === null ||
    distribution.p95 === null ||
    distribution.max === null
  ) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Distribution range">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Distribution Range</h2>
        <div className="rounded border border-dashed border-gray-700 bg-gray-950 p-4 text-sm text-gray-400">
          <p className="font-medium text-amber-400">Distribution unavailable</p>
        </div>
      </section>
    );
  }

  const unit = distribution.unit ?? rec?.unit ?? "";
  const min = distribution.min;
  const max = distribution.max;
  const span = max - min || 1;
  const toPct = (v: number) => ((v - min) / span) * 100;

  const current = rec?.current_limit ?? null;
  const recommended = rec?.recommended_limit ?? null;
  const aiCandidate =
    candidates.find((c) => c.parameter === rec?.parameter && c.ml_rank === 1)?.candidate_limit ??
    null;

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Distribution range">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">Distribution Range</h2>
      <p className="text-xs text-gray-500 mb-4">
        MIN — MEDIAN — P95 — MAX with optional DTL markers (not a histogram).
      </p>

      <div className="relative mx-1 h-14 mb-8">
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded bg-gray-800" />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gray-400"
          style={{ left: "0%" }}
          title={`Min ${formatUnit(min, unit)}`}
        />
        <div
          className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-400 ring-2 ring-emerald-700"
          style={{ left: `${toPct(distribution.median)}%` }}
          title={`Median ${formatUnit(distribution.median, unit)}`}
          data-testid="range-median"
        />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-sky-400"
          style={{ left: `${toPct(distribution.p95)}%` }}
          title={`P95 ${formatUnit(distribution.p95, unit)}`}
          data-testid="range-p95"
        />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gray-400"
          style={{ left: "100%" }}
          title={`Max ${formatUnit(max, unit)}`}
        />

        {current !== null && current >= min && current <= max && (
          <Marker
            pct={toPct(current)}
            barClass="bg-violet-400"
            textClass="text-violet-300"
            label="Current"
            testId="range-current-limit"
          />
        )}
        {aiCandidate !== null &&
          aiCandidate >= min &&
          aiCandidate <= max &&
          aiCandidate !== recommended && (
            <Marker
              pct={toPct(aiCandidate)}
              barClass="bg-amber-400"
              textClass="text-amber-300"
              label="ML Top"
              testId="range-ml-top-candidate"
            />
          )}
        {recommended !== null && recommended >= min && recommended <= max && (
          <Marker
            pct={toPct(recommended)}
            barClass="bg-cyan-400"
            textClass="text-cyan-300"
            label="Final"
            testId="range-final-recommendation"
          />
        )}
      </div>

      <div className="flex justify-between text-[10px] font-mono text-gray-500">
        <span>{formatUnit(min, unit)}</span>
        <span>{formatUnit(distribution.median, unit)}</span>
        <span>{formatUnit(distribution.p95, unit)}</span>
        <span>{formatUnit(max, unit)}</span>
      </div>

      <div className="mt-4 flex flex-wrap gap-3 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" /> Median (observed)
        </span>
        <span className="flex items-center gap-1" data-testid="legend-current-limit">
          <span className="inline-block h-2 w-2 rounded-full bg-violet-400" /> Current DTL Limit
        </span>
        <span className="flex items-center gap-1" data-testid="legend-ml-top-candidate">
          <span className="inline-block h-2 w-2 rounded-full bg-amber-400" /> ML Top Candidate
        </span>
        <span className="flex items-center gap-1" data-testid="legend-final-recommendation">
          <span className="inline-block h-2 w-2 rounded-full bg-cyan-400" /> Final Recommendation
        </span>
      </div>
    </section>
  );
}
