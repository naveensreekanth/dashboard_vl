import { AuditPanel } from "@/components/audit/AuditPanel";
import { AdvancedEvidence } from "@/components/common/AdvancedEvidence";
import { ErrorState } from "@/components/common/ErrorState";
import { LoadingState } from "@/components/common/LoadingState";
import { ServiceNotReadyBanner } from "@/components/common/ServiceNotReadyBanner";
import { SimulationEvidencePanel } from "@/components/evidence/SimulationEvidence";
import { LimitScale } from "@/components/limits/LimitScale";
import { CandidateTable } from "@/components/limits/CandidateTable";
import { DecisionContextPanel } from "@/components/measurement/DecisionContextPanel";
import { ModelPanel } from "@/components/model/ModelPanel";
import { DecisionCard } from "@/components/recommendation/DecisionCard";
import { ExplanationPanel } from "@/components/recommendation/ExplanationPanel";
import { RecommendationSummary } from "@/components/recommendation/RecommendationSummary";
import { SafetyGateTrace } from "@/components/safety/SafetyGateTrace";
import { SelectorPanel } from "@/components/selectors/SelectorPanel";
import { useAppContext } from "@/state/useAppContext";
import { getRecommendationForParameter } from "@/utils/recommendation";

export function RecommendationPage() {
  const { state } = useAppContext();
  const result = state.recommendation;
  const rec = result
    ? getRecommendationForParameter(result.recommendations, state.selectedParameter)
    : undefined;
  const candidates = result?.audit.candidate_set ?? [];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-gray-100">DTL Recommendation</h1>
        <p className="text-sm text-gray-500 mt-1">
          Lot → Die → Parameter → eligible candidates → maximum simulated yield → ML rank
          tie-breaker → final DTL
        </p>
      </div>

      {state.serviceReady === false && <ServiceNotReadyBanner />}

      <SelectorPanel />
      {state.loading && <LoadingState />}
      {state.error && <ErrorState message={state.error} code={state.errorCode} />}

      {rec && result && (
        <>
          <DecisionContextPanel rec={rec} candidates={candidates} />
          <DecisionCard decision={rec.decision} />
          <ExplanationPanel rec={rec} primary />
          <CandidateTable
            rec={rec}
            candidates={candidates}
            simulationRows={result.audit.simulation_evidence_rows ?? []}
          />
        </>
      )}

      {rec && result && (
        <AdvancedEvidence>
          <LimitScale rec={rec} candidates={candidates} />
          <SimulationEvidencePanel rec={rec} />
          <SafetyGateTrace safety={rec.safety_result} />
          <ModelPanel
            rec={rec}
            jointEnabled={result.audit.joint_enabled ?? false}
            treeDiagnostic={result.audit.include_tree_baseline_diagnostic ?? false}
          />
          <RecommendationSummary rec={rec} />
          <ExplanationPanel rec={rec} />
          <AuditPanel result={result} />
        </AdvancedEvidence>
      )}

      {!state.loading && !rec && !state.error && (
        <p className="text-sm text-gray-500">
          Select lot, die, and parameter, then run a recommendation to see Current DTL,
          Recommended DTL, and why the candidate was selected.
        </p>
      )}
    </div>
  );
}
