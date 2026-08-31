import type { DieDistributionResponse } from "@/api/types";
import { LoadingState } from "@/components/common/LoadingState";
import { isSyntheticSource } from "@/utils/formatMeasurement";
import { formatUnit } from "@/utils/formatUnit";

export interface MeasurementDistributionProps {
  distribution: DieDistributionResponse | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export function MeasurementDistribution({
  distribution,
  loading = false,
  error = null,
  onRetry,
}: MeasurementDistributionProps) {
  if (loading) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Measurement distribution">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Measurement Distribution</h2>
        <LoadingState message="Loading distribution…" />
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Measurement distribution">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Measurement Distribution</h2>
        <div className="rounded border border-red-900 bg-red-950/30 p-4" role="alert">
          <p className="text-sm font-medium text-red-400">Unable to load distribution data.</p>
          <p className="mt-1 text-sm text-gray-300">{error}</p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 rounded border border-gray-700 px-3 py-1 text-xs text-gray-200 hover:bg-gray-800"
            >
              Retry
            </button>
          )}
        </div>
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
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Measurement distribution">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Measurement Distribution</h2>
        <div className="rounded border border-dashed border-gray-700 bg-gray-950 p-4 text-sm text-gray-400">
          <p className="font-medium text-amber-400">Distribution unavailable</p>
          <p className="mt-2 text-xs leading-relaxed">
            No distribution is available for the selected lot, die, and parameter.
          </p>
        </div>
      </section>
    );
  }

  const unit = distribution.unit ?? "";
  const scope = (distribution.scope || "die").toUpperCase();
  const synthetic = isSyntheticSource(distribution.source_classification);

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Measurement distribution">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <h2 className="text-sm font-semibold text-gray-200">Measurement Distribution</h2>
        <div className="flex flex-wrap items-center gap-2 text-[10px]">
          <span className="rounded border border-gray-700 px-2 py-0.5 font-semibold tracking-wide text-gray-300">
            SCOPE {scope}
          </span>
          <span className="rounded border border-gray-700 px-2 py-0.5 font-mono text-gray-400">
            n={distribution.n}
          </span>
          {synthetic && (
            <span className="rounded border border-amber-800/80 bg-amber-950/40 px-2 py-0.5 font-semibold tracking-wide text-amber-300">
              SYNTHETIC DATASET
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-sm" data-testid="distribution-stats">
        <div>
          <p className="text-xs text-gray-500">Min</p>
          <p className="font-mono text-gray-100" data-testid="dist-min">
            {formatUnit(distribution.min, unit)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Median</p>
          <p className="font-mono text-emerald-300" data-testid="dist-median">
            {formatUnit(distribution.median, unit)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">P95</p>
          <p className="font-mono text-gray-100" data-testid="dist-p95">
            {formatUnit(distribution.p95, unit)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Max</p>
          <p className="font-mono text-gray-100" data-testid="dist-max">
            {formatUnit(distribution.max, unit)}
          </p>
        </div>
      </div>

      <p className="mt-3 text-xs text-gray-500">
        Statistical markers only (min / median / P95 / max) — not a fitted probability density.
      </p>
    </section>
  );
}
