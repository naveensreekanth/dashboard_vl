import type { DieMeasurementResponse } from "@/api/types";
import { LoadingState } from "@/components/common/LoadingState";
import { formatObservedRule, isSyntheticSource } from "@/utils/formatMeasurement";
import { formatUnit } from "@/utils/formatUnit";

export interface MeasurementCardProps {
  measurement: DieMeasurementResponse | null;
  loading?: boolean;
  error?: string | null;
  patternCount?: number | null;
  /** Optional observed range from distribution API (min–max). */
  observedRange?: { min: number; max: number; unit?: string | null } | null;
  onRetry?: () => void;
}

export function MeasurementCard({
  measurement,
  loading = false,
  error = null,
  patternCount = null,
  observedRange = null,
  onRetry,
}: MeasurementCardProps) {
  if (loading) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Observed measurement">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Observed Measurement</h2>
        <LoadingState message="Loading measurement…" />
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Observed measurement">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Observed Measurement</h2>
        <div className="rounded border border-red-900 bg-red-950/30 p-4" role="alert">
          <p className="text-sm font-medium text-red-400">Unable to load measurement data.</p>
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

  if (!measurement || !measurement.found || measurement.observed_value === null) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Observed measurement">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Observed Measurement</h2>
        <div className="rounded border border-dashed border-gray-700 bg-gray-950 p-4 text-sm text-gray-400">
          <p className="font-medium text-amber-400">Measurement unavailable</p>
          <p className="mt-2 text-xs leading-relaxed">
            No measurement is available for the selected lot, die, and parameter.
          </p>
        </div>
      </section>
    );
  }

  const unit = measurement.unit ?? "";
  const ruleLabel = formatObservedRule(measurement.observed_value_rule, {
    n: patternCount,
    conditionId: measurement.condition_id,
  });
  const synthetic = isSyntheticSource(measurement.source_classification);

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Observed measurement">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-gray-500">Observed Measurement</p>
          <h2 className="text-sm font-semibold text-gray-100 font-mono mt-0.5">
            {measurement.parameter}
          </h2>
        </div>
        {synthetic && (
          <span
            className="rounded border border-amber-800/80 bg-amber-950/40 px-2 py-0.5 text-[10px] font-semibold tracking-wide text-amber-300"
            data-testid="synthetic-label"
          >
            SYNTHETIC DATASET
          </span>
        )}
      </div>

      <p className="text-3xl font-mono text-emerald-300" data-testid="observed-value">
        {formatUnit(measurement.observed_value, unit)}
      </p>

      {ruleLabel && <p className="mt-2 text-xs text-gray-400">{ruleLabel}</p>}

      {observedRange && (
        <p className="mt-2 text-sm text-gray-400" data-testid="observed-range">
          Observed range:{" "}
          <span className="font-mono text-gray-200">
            {formatUnit(observedRange.min, observedRange.unit ?? unit)} –{" "}
            {formatUnit(observedRange.max, observedRange.unit ?? unit)}
          </span>
        </p>
      )}

      <p className="mt-3 text-xs text-gray-500">
        Observed measurement · Synthetic dataset — not a production chip reading.
      </p>
    </section>
  );
}
