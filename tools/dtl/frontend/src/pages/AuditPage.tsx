import { AuditPanel } from "@/components/audit/AuditPanel";
import { useAppContext } from "@/state/useAppContext";

export function AuditPage() {
  const { state } = useAppContext();

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-semibold text-gray-100">Audit / Evidence</h1>
        <p className="text-sm text-gray-500 mt-1">
          Full provenance record from the Phase 8 recommendation engine.
        </p>
      </div>
      {state.recommendation ? (
        <AuditPanel result={state.recommendation} />
      ) : (
        <p className="text-sm text-gray-500">
          No audit record loaded. Run a recommendation on the Recommendation page first.
        </p>
      )}
    </div>
  );
}
