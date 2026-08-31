import type { SafetyResult } from "@/api/types";

interface SafetyGateTraceProps {
  safety: SafetyResult | null | undefined;
}

function statusClass(status: string): string {
  if (status === "PASS") return "text-green-400";
  if (status === "SOFT_FAIL") return "text-amber-400";
  if (status === "HARD_FAIL") return "text-red-400";
  return "text-gray-400";
}

export function SafetyGateTrace({ safety }: SafetyGateTraceProps) {
  if (!safety) {
    return (
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-2">Safety Gate</h2>
        <p className="text-sm text-gray-500">No safety trace returned.</p>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-4" aria-label="Safety gate trace">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-200">Safety Gate</h2>
        <span className={`text-sm font-semibold ${statusClass(safety.status)}`}>
          {safety.status}
        </span>
      </div>
      <ul className="space-y-2">
        {safety.checks.map((check) => (
          <li
            key={`${check.layer}-${check.name}`}
            className="flex items-start gap-2 rounded border border-gray-800 bg-gray-950 p-2 text-xs"
          >
            <span
              className={check.passed ? "text-green-400" : "text-red-400"}
              aria-hidden="true"
            >
              {check.passed ? "✓" : "✗"}
            </span>
            <div>
              <p className="font-medium text-gray-200">
                {check.name}{" "}
                <span className="text-gray-500">(L{check.layer}, {check.severity})</span>
              </p>
              <p className="text-gray-500 mt-0.5">{check.message}</p>
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-gray-600">
        Safety evaluation is performed by the Phase 8 backend — not re-evaluated in the dashboard.
      </p>
    </section>
  );
}
