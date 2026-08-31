import type { DieConditionsResponse } from "@/api/types";
import { LoadingState } from "@/components/common/LoadingState";
import { formatUnit } from "@/utils/formatUnit";

export interface ConditionTableProps {
  conditions: DieConditionsResponse | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export function ConditionTable({
  conditions,
  loading = false,
  error = null,
  onRetry,
}: ConditionTableProps) {
  if (loading) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Condition table">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Condition Details</h2>
        <LoadingState message="Loading conditions…" />
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Condition table">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Condition Details</h2>
        <div className="rounded border border-red-900 bg-red-950/30 p-4" role="alert">
          <p className="text-sm font-medium text-red-400">Unable to load condition data.</p>
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

  if (!conditions) {
    return null;
  }

  if (!conditions.found && conditions.reason === "not_condition_aware") {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Condition table">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Condition Details</h2>
        <div className="rounded border border-dashed border-gray-700 bg-gray-950 p-4 text-sm text-gray-400">
          <p className="font-medium text-gray-300" data-testid="not-condition-aware">
            Not condition-aware
          </p>
          <p className="mt-2 text-xs leading-relaxed">
            Core measurements are not condition-aware.
          </p>
        </div>
      </section>
    );
  }

  if (!conditions.found || conditions.conditions.length === 0) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Condition table">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Condition Details</h2>
        <div className="rounded border border-dashed border-gray-700 bg-gray-950 p-4 text-sm text-gray-400">
          <p className="font-medium text-amber-400">Conditions unavailable</p>
          <p className="mt-2 text-xs leading-relaxed">
            No condition-level measurements are available for this selection.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Condition details">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">Condition Details</h2>
      <p className="text-xs text-gray-500 mb-3">
        Observed parameter values under individual test conditions. These readings are not DTL
        candidates.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[32rem] text-left text-sm" data-testid="condition-table">
          <thead className="text-xs uppercase text-gray-500">
            <tr>
              <th className="pb-2 pr-3 font-medium">Condition</th>
              <th className="pb-2 pr-3 font-medium">Reading</th>
              <th className="pb-2 pr-3 font-medium">Temp °C</th>
              <th className="pb-2 pr-3 font-medium">VDD</th>
              <th className="pb-2 pr-3 font-medium">Mode</th>
              <th className="pb-2 font-medium">P/F</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {conditions.conditions.map((row) => (
              <tr key={row.condition_id} className="text-gray-300">
                <td className="py-2 pr-3 font-mono text-xs">{row.condition_id}</td>
                <td className="py-2 pr-3 font-mono text-emerald-300">
                  {formatUnit(row.measurement_value, row.unit ?? conditions.unit ?? "")}
                </td>
                <td className="py-2 pr-3 font-mono text-xs">
                  {row.temperature_c ?? "—"}
                </td>
                <td className="py-2 pr-3 font-mono text-xs">{row.vdd_applied ?? "—"}</td>
                <td className="py-2 pr-3 text-xs">{row.test_mode ?? "—"}</td>
                <td className="py-2 font-mono text-xs">{row.pass_fail_condition ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-gray-500">
        Observed measurement · Synthetic dataset — values from Phase 10.11 conditions API.
      </p>
    </section>
  );
}
