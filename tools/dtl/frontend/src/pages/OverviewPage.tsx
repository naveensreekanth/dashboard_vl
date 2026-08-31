import { useAppContext } from "@/state/useAppContext";

export function OverviewPage() {
  const { state } = useAppContext();
  const audit = state.recommendation?.audit;

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold text-gray-100">Overview</h1>
        <p className="text-sm text-gray-500 mt-1">
          Service status and provenance from the last recommendation response.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-500">Service readiness</p>
          <p
            className={`mt-1 text-lg font-semibold ${
              state.serviceReady ? "text-green-400" : "text-amber-400"
            }`}
          >
            {state.serviceReady === null
              ? "Checking…"
              : state.serviceReady
                ? "READY"
                : "NOT READY"}
          </p>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <p className="text-xs text-gray-500">API</p>
          <p className="mt-1 text-sm text-gray-300">Phase 9 FastAPI — /api/v1</p>
        </div>
        {audit?.dataset_version && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <p className="text-xs text-gray-500">Dataset version</p>
            <p className="mt-1 font-mono text-sm">{audit.dataset_version}</p>
          </div>
        )}
        {audit?.model_version && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <p className="text-xs text-gray-500">Model version</p>
            <p className="mt-1 font-mono text-sm">{audit.model_version}</p>
          </div>
        )}
      </div>
      {!state.recommendation && (
        <p className="text-sm text-gray-500">
          Run a recommendation to populate dataset and model version cards. Aggregate lot/die
          statistics require future inventory endpoints and are not fabricated here.
        </p>
      )}
    </div>
  );
}
